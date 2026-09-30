from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Projection_lrb_qwen3')]
from src.dager_qwen3.oracle_projection import defend_canonical_gradients, QwenColumnTransform
from src.qwen3_classifier import validate_trained_loading_info, Qwen3ClassifierError
from src.oracle_v2.calibration import choose_tau1, choose_tau2, choose_operating_points
from src.oracle_v2.protocol import fresh_final
from src.oracle_v2.attack import paired_spans, token_diagnostics, decode_observed_q_gradients
from src.dager_qwen3.layer1_filter import scan_qwen3_vocab_layer1_distances, Layer1DistanceScanResult


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))])
def test_exact_column_transform_and_holes(device):
    gen = torch.Generator().manual_seed(8)
    a = torch.randn(11, generator=gen).to(device)
    h = torch.randn(17, generator=gen).to(device)
    gradient = a[:, None] @ h[None]
    names = ('unused', 'model.layers.0.self_attn.q_proj.weight', 'model.layers.1.self_attn.q_proj.weight')
    result = defend_canonical_gradients((None, gradient, gradient), names, preset='proj_uniform', rho=.3)
    assert result.gradients[0] is None and result.layer_info[0]['active'] is False
    q0 = QwenColumnTransform(result.state_for(names[1]))
    q1 = QwenColumnTransform(result.state_for(names[2]))
    assert q0.state['projection_seed'] != q1.state['projection_seed']
    assert not torch.equal(q0(h[None]), q1(h[None]))
    _, _, vh = torch.linalg.svd(result.gradients[1].float(), full_matrices=False)
    transformed = q0(h[None])[0]
    residual = (transformed - transformed @ vh[:1].T @ vh[:1]).norm() / transformed.norm()
    assert residual < 2e-5
    wrong_residual = (h - h @ vh[:1].T @ vh[:1]).norm() / h.norm()
    assert wrong_residual > .1
    assert q0.metadata(verify_rank=True)['operator_rank'] == 5
    with pytest.raises(ValueError, match='wrong axis'):
        q0(torch.randn(1, 11, device=device))


def test_identity_and_proj_only_is_not_identity():
    gradient = torch.randn(11, 17)
    names = ('model.layers.0.self_attn.q_proj.weight',)
    result = defend_canonical_gradients((gradient,), names, preset='proj_uniform', rho=1)
    assert torch.equal(result.gradients[0], gradient)
    transform = QwenColumnTransform(result.state_for(names[0]))
    candidates = torch.randn(100, 17)
    assert torch.equal(transform(candidates), candidates)
    nonidentity = defend_canonical_gradients((gradient,), names, preset='proj_only', rho=1)
    assert nonidentity.layer_info[0]['keep_ratio'] < 1


def test_checkpoint_rejects_missing_and_mismatched_head():
    validate_trained_loading_info({'missing_keys': [], 'mismatched_keys': []})
    for info in ({'missing_keys': ['score.weight']}, {'mismatched_keys': [('score.weight', (1, 2), (2, 2))]}):
        with pytest.raises(Qwen3ClassifierError):
            validate_trained_loading_info(info)


def test_new_final_is_disjoint_and_deterministic():
    rows = [{'sample_key': f'{i:04d}'} for i in range(100)]
    old = {s['sample_key'] for s in rows[:45]}
    assert fresh_final(list(reversed(rows)), old) == rows[45:65]


def test_calibration_guards_and_tie_breaks():
    rows = []
    for rho, ratio in ((.1, .5), (.2, .7)):
        rows.extend({'stage': 'calibration', 'status': 'ok', 'preset': 'proj_uniform',
            'variant': 'oracle', 'rho': rho, 'capacity': [{'B_over_q': ratio}]} for _ in range(20))
    selected, matches = choose_operating_points(rows, [.6])
    assert selected == [.2]
    with pytest.raises(ValueError):
        choose_operating_points([{**rows[0], 'stage': 'final'}], [.6])
    token_rows = [{'stage': 'calibration', 'status': 'ok',
        'token_diagnostics': [{'active': True, 'is_eos': False, 'distance': .001}],
        'candidate_counts': {'0.001': 0, '0.002': 2}} for _ in range(20)]
    assert choose_tau1(token_rows, [.001, .002])['tau1'] == .002
    assert choose_tau1(token_rows[:19], [.001, .002])['status'] == 'calibration_failed'
    decode_rows = [{'stage': 'calibration', 'status': 'search_budget_exhausted', 'tau2': .001} for _ in range(20)]
    assert choose_tau2(decode_rows, [.001])['status'] == 'calibration_failed'


def test_ranks_ties_eos_and_duplicates():
    scan = Layer1DistanceScanResult(torch.arange(5), torch.tensor([.1, .1, .3, .4, .5]), 'l2', 5, ())
    result = token_diagnostics(scan, (1, 1, 4), 4, [True, True, False], [.2])
    assert result['top1'] == 0
    assert result['token_diagnostics'][0]['tie_rank_min'] == 1
    assert result['token_diagnostics'][0]['tie_rank_max'] == 2
    assert result['candidate_counts']['0.2'] == 2
    assert result['top10'] == 1


def test_default_scanner_unchanged_and_identity_rank():
    gradient = torch.randn(16, 16)
    spans = paired_spans((gradient, gradient), {'rank_atol': .001, 'rank_cutoff': 2})
    candidates = torch.randn(29, 16)
    adapter = SimpleNamespace(device=torch.device('cpu'), metadata=SimpleNamespace(vocab_size=29, hidden_size=16),
        layer0_qproj_inputs_for_token_ids=lambda ids: candidates[ids])
    standard = scan_qwen3_vocab_layer1_distances(adapter=adapter, span=spans[0], vocab_chunk_size=7)
    oracle = scan_qwen3_vocab_layer1_distances(adapter=adapter, span=spans[0], vocab_chunk_size=7,
                                             candidate_transform=lambda x: x.clone())
    assert torch.equal(standard.distances, oracle.distances)
    assert 'sample' not in __import__('inspect').signature(decode_observed_q_gradients).parameters


def test_native_qwen_padding_and_saved_head(tmp_path):
    from transformers import Qwen3Config, Qwen3ForSequenceClassification
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_labels=2, pad_token_id=0, eos_token_id=0, attention_dropout=0.)
    model = Qwen3ForSequenceClassification(config).eval()
    ids = torch.tensor([[3, 4, 0], [5, 0, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.no_grad():
        padded = model(input_ids=ids, attention_mask=mask).logits
        single = model(input_ids=ids[1:2, :2], attention_mask=torch.ones(1, 2)).logits
    torch.testing.assert_close(padded[1:2], single)
    model.save_pretrained(tmp_path)
    loaded, info = Qwen3ForSequenceClassification.from_pretrained(tmp_path, output_loading_info=True)
    validate_trained_loading_info(info)
    assert torch.equal(model.score.weight, loaded.score.weight)


def test_optional_layer2_transform_is_applied():
    from src.dager_qwen3.layer2_decoder import _evaluate_prefix_batch
    adapter = SimpleNamespace(device=torch.device('cpu'),
        layer1_qproj_inputs_from_prefixes=lambda **kwargs: torch.tensor([[[0., 1.]]]))
    span = SimpleNamespace(basis=torch.tensor([[1., 0.]]))
    standard, _ = _evaluate_prefix_batch(adapter=adapter, span=span, prefixes=[(1,)], threshold=.01, distance_norm='l2')
    oracle, _ = _evaluate_prefix_batch(adapter=adapter, span=span, prefixes=[(1,)], threshold=.01, distance_norm='l2',
                                      candidate_transform=lambda h: h.flip(-1))
    assert standard == [False] and oracle == [True]


def test_projection_bfloat16_algebra_tolerance():
    torch.manual_seed(77)
    g = (torch.randn(23, 1) @ torch.randn(1, 19)).bfloat16()
    update = defend_canonical_gradients((g,), ('q',), preset='proj_uniform', rho=.4)
    transform = QwenColumnTransform(update.state_for('q'))
    # Transform each row of G: left operation cannot enlarge its right-space.
    right = transform(g.float())
    _, _, vh = torch.linalg.svd(right, full_matrices=False)
    raw_rank = int(torch.linalg.matrix_rank(right, atol=1e-5, rtol=0))
    basis = vh[:raw_rank]
    defended = update.gradients[0].float()
    residual = (defended - defended @ basis.T @ basis).norm() / defended.norm()
    assert residual < .006


def test_no_candidate_metrics_and_exhaustion_are_distinct(monkeypatch):
    from src.oracle_v2 import attack
    selected = SimpleNamespace(token_ids=(7, 8), mean_span_distance=.001)
    decoded = SimpleNamespace(selected_token_ids=(7,), evaluated_prefix_count=2,
        termination_reason='search_budget_exhausted', per_length_survivor_counts=((1,1),(2,1)))
    fields = dict(token_recovery=.5, legacy_l1_token_membership=1., exact_recovery=False,
        rouge_1=.8, rouge_2=.7, empty_reconstruction=False,
        ground_truth_text='a b', reconstructed_text='a b',
        ground_truth_token_text=('a','b','eos'), reconstructed_token_text=('a','b'))
    from src.dager_qwen3.metrics import AttackMetrics
    monkeypatch.setattr(attack, 'compute_attack_metrics', lambda **kwargs: AttackMetrics(**fields))
    status = ['ok']
    monkeypatch.setattr(attack, 'decode_observed_q_gradients', lambda **kwargs:
        (SimpleNamespace(token_ids=(7,8), candidate_count=2), decoded, selected, status[0]))
    args = dict(adapter=SimpleNamespace(tokenizer=None), spans=None, transforms=None, scan=None,
        variant='oracle', sample=SimpleNamespace(input_ids=(7,8,0), eos_token_id=0),
        config={'selection_rule':'longest'}, tau1=.002, tau2=.001,
        rouge_backend=SimpleNamespace(metric=None, json_metadata=lambda: {}))
    result = attack.report_decode(**args)
    assert result['token_recovery'] == 1 and result['exact_recovery'] is True
    status[0] = 'search_budget_exhausted'
    result = attack.report_decode(**args)
    assert result['token_recovery'] is None and result['partial_metrics']['token_recovery'] == 1


def test_zero_absolute_rank_is_reported_as_empty_span_not_fake_recovery():
    from src.oracle_v2.attack import capacity
    gradients = (torch.zeros(25, 25), torch.zeros(25, 25))
    spans = paired_spans(gradients, {'rank_atol': .001, 'rank_cutoff': 20})
    assert spans[0].basis.shape == (0, 25)
    assert capacity(spans, (None, None))[0]['zero_rank_at_registered_atol'] is True
    candidates = torch.randn(3, 25)
    adapter = SimpleNamespace(device=torch.device('cpu'), metadata=SimpleNamespace(vocab_size=3, hidden_size=25),
        layer0_qproj_inputs_for_token_ids=lambda ids: candidates[ids])
    result = scan_qwen3_vocab_layer1_distances(adapter=adapter, span=spans[0], vocab_chunk_size=2)
    torch.testing.assert_close(result.distances, torch.ones(3))


def test_attack_timer_restores_signal_handler():
    import signal
    import time
    from src.oracle_v2.worker import attack_timer
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        with attack_timer(.01):
            time.sleep(.2)
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_source_inventory_hashes_runtime_inputs():
    from src.oracle_v2.protocol import code_hashes
    files = code_hashes(ROOT)
    assert 'utils/lrb_defense.py' in files
    assert 'Projection_lrb_qwen3/src/oracle_v2/worker.py' in files
    assert all(len(value) == 64 for value in files.values())


def test_summary_excludes_timeouts_from_recovery_mean(tmp_path):
    from src.oracle_v2.protocol import write_json, read_json
    from src.oracle_v2.summary import summarize
    directory = tmp_path / 'jobs/final_seed101_none/records'
    base = {'condition':'none', 'variant':'standard', 'token_recovery':.75,
            'exact_recovery':False, 'rouge_1':.6, 'rouge_2':.4}
    write_json(directory/'ok.json', {**base, 'status':'ok'})
    write_json(directory/'timeout.json', {**base, 'status':'timeout', 'token_recovery':None})
    summarize(tmp_path)
    row = read_json(tmp_path/'summary/privacy_summary.json')[0]
    assert row['n_total'] == 2 and row['n_completed'] == 1 and row['timeout'] == 1
    assert row['token_recovery'] == .75
    assert row['r1_r2_raw'] == 1.0 and row['r1_r2_pct'] == 100.0
    assert 'r1_r2' not in row
    import json
    records = [json.loads(line) for line in (tmp_path/'summary/per_sample.jsonl').read_text().splitlines()]
    assert len(records) == 2 and {r['status'] for r in records} == {'ok', 'timeout'}


def test_gpu_admission_waits_for_existing_compute_process(monkeypatch):
    from src.oracle_v2 import worker
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    snapshots = iter(['12345\n', ''])
    monkeypatch.setattr(worker.subprocess, 'check_output', lambda *a, **k: next(snapshots))
    waits = []
    monkeypatch.setattr(worker.time, 'sleep', waits.append)
    events = []
    worker.wait_for_idle_gpu(lambda **fields: events.append(fields))
    assert waits == [30]
    assert events[0]['kind'] == 'waiting_for_idle_gpu'
    assert events[-1]['kind'] == 'gpu_admission'


def test_shared_gpu_admission_requires_registered_free_memory(monkeypatch):
    from src.oracle_v2 import worker
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('QWEN_V2_SHARED_MIN_FREE_MIB', '50000')
    free = iter(['49000', '62000'])
    def query(command, **_kwargs):
        return next(free) if '--query-gpu=memory.free' in command else '12345\n'
    monkeypatch.setattr(worker.subprocess, 'check_output', query)
    waits = []
    monkeypatch.setattr(worker.time, 'sleep', waits.append)
    events = []
    worker.wait_for_idle_gpu(lambda **fields: events.append(fields))
    assert waits == [30]
    assert events[0]['kind'] == 'waiting_for_idle_gpu'
    assert events[-1]['kind'] == 'gpu_admission'
    assert events[-1]['free_mib'] == 62000
    assert events[-1]['other_compute_pids'] == [12345]
