#!/usr/bin/env python3
"""Run one preregistered Qwen3 standard DAGER attack through the shared core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.dager_qwen3 import ATTACK_NAME
from src.dager_qwen3.diagnostics import load_none_attack_controls, load_registered_sample, none_only_attack_metadata, registered_head_seed
from src.dager_qwen3.frozen_tau1_control import verify_frozen_tau1_control
from src.dager_qwen3.metrics import preflight_legacy_dager_rouge_backend
from src.dager_qwen3.none_attack_core import NoneAttackCoreControls, execute_none_only_dager
from src.hashing import sha256_file, sha256_text
from src.result_schema import ResultSchemaError, write_or_verify_jsonl

# Explicit patch surface for preflight-contract tests.  Model construction is
# intentionally owned by ``none_attack_core`` so this entrypoint and
# calibration cannot diverge into separate attacks.
load_local_qwen3_sequence_classifier: Any | None = None


class NoneAttackScriptError(RuntimeError):
    """Raised when the standalone none-only attack request is invalid."""


def _resolve(value: str, *, description: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise NoneAttackScriptError(f"{description} must remain under {PROJECT_ROOT}.") from error
    return resolved


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defense-unaware DAGER on one preregistered Qwen3 sample.")
    parser.add_argument("--config", default="Projection_lrb_qwen3/configs/experiment.json")
    parser.add_argument("--stage", choices=("calibration", "smoke", "final"), required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--head-seed", required=True, type=int)
    parser.add_argument("--tau1-control", required=True)
    parser.add_argument("--defense", choices=("none",), default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run_attack(args: argparse.Namespace) -> dict[str, Any]:
    if args.defense != "none":
        raise NoneAttackScriptError("This entrypoint only permits defense=none.")
    tau_path = _resolve(args.tau1_control, description="tau1 control")
    frozen_tau1 = verify_frozen_tau1_control(project_root=PROJECT_ROOT, control_path=tau_path)
    # All protocol controls are verified before ROUGE/model/CUDA work.
    config = load_experiment_config(_resolve(args.config, description="config"), require_dataset_path=False)
    registered_head_seed(config, stage=args.stage, requested_seed=args.head_seed)
    sample = load_registered_sample(config=config, stage=args.stage, sample_key=args.sample_key)
    controls = load_none_attack_controls(config, frozen_tau1=float(frozen_tau1["selected_tau1"]))
    output = _resolve(args.output, description="output")
    try:
        output.relative_to((PROJECT_ROOT / "outputs").resolve())
    except ValueError as error:
        raise NoneAttackScriptError("Output must remain under Projection_lrb_qwen3/outputs.") from error
    if output.suffix != ".jsonl":
        raise NoneAttackScriptError("Output must use .jsonl.")
    rouge_backend = preflight_legacy_dager_rouge_backend()
    core = execute_none_only_dager(
        model_path=config.model_path,
        sample=sample,
        controls=NoneAttackCoreControls(
            tau1=controls.l1_span_threshold, tau2=controls.l2_span_threshold,
            rank_tolerance=controls.rank_tolerance, rank_cutoff=controls.rank_cutoff,
            max_search_candidates=controls.max_search_candidates, max_candidate_ids=controls.max_candidate_ids,
            parallel=controls.decode_batch_size, max_sequence_length=controls.max_sequence_length,
        ),
        head_seed=args.head_seed, device=args.device, dtype=args.dtype, rouge_backend=rouge_backend,
    )
    identity = sha256_text(f"{ATTACK_NAME}|none|{sample.preregistration_sha256}|{sample.stage}|{sample.sample_key}|{args.head_seed}|{args.dtype}|{frozen_tau1['frozen_control_identity_sha256']}")
    record: dict[str, Any] = {
        "schema_version": 1, "record_type": "qwen3_dager_attack_result", "result_identity_sha256": identity,
        **none_only_attack_metadata(), "sample_id": sample.sample_key, "sample_key": sample.sample_key,
        "original_index": sample.original_index, "stage": sample.stage,
        "tau1_source": "frozen_tau1_control", "frozen_tau1_control_path": tau_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "frozen_tau1_control_sha256": sha256_file(tau_path), "frozen_tau1_control_identity_sha256": frozen_tau1["frozen_control_identity_sha256"],
        "aggregation_sha256": frozen_tau1["aggregation_sha256"], "bf16_gate": frozen_tau1["bfloat16_gate"],
        "bf16_gate_amendment_identity": frozen_tau1["bf16_gate_amendment_identity"],
        "preregistration_sha256": frozen_tau1["preregistration_sha256"], "calibration_sample_list_sha256": frozen_tau1["calibration_sample_list_sha256"],
        "head_seed": args.head_seed, "dtype": args.dtype, **core, "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    record["thresholds"]["tau1_source"] = "frozen_tau1_control"
    try:
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8").strip())
            old = dict(existing); new = dict(record); old.pop("created_at", None); new.pop("created_at", None)
            if old == new:
                return record
        write_or_verify_jsonl(output, [record])
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise NoneAttackScriptError(f"Unable to write immutable attack result: {error}") from error
    return record


def main() -> int:
    args = parse_args()
    try:
        record = run_attack(args)
    except Exception as error:
        print(json.dumps({"record_type": "qwen3_dager_attack_error", "attack_name": ATTACK_NAME, "defense": "none", "status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(record, sort_keys=True))
    return 0 if record["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
