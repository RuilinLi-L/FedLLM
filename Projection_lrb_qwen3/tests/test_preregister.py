"""Unit tests for deterministic, no-network preregistration semantics."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PreregistrationConfigError, load_experiment_config
from src.preregister import (
    allocate_stages,
    build_preregistration_document,
    load_official_sst2_validation,
    prepare_eligible_samples,
    preregister_experiment,
)


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        if kwargs.get("add_special_tokens") is not False:
            raise AssertionError("Protocol must disable tokenizer special tokens.")
        if kwargs.get("truncation") is not False or "max_length" in kwargs:
            raise AssertionError("Protocol must tokenize text before explicit truncation and EOS.")
        token_count = int(text.rsplit(" ", 1)[-1]) if text.startswith("sample ") else 1
        return {"input_ids": list(range(1, token_count + 1))}


def write_config(root: Path, *, calibration_seed: object = 404, smoke_seed: object = 505) -> Path:
    project = root / "Projection_lrb_qwen3"
    config_path = project / "configs" / "experiment.json"
    config_path.parent.mkdir(parents=True)
    document = {
        "model_path": "models/Qwen3-1.7B-Base",
        "max_length": 32,
        "min_effective_token_length": 1,
        "calibration_head_seed": calibration_seed,
        "smoke_head_seed": smoke_seed,
        "final_head_seeds": [101, 202, 303],
        "defense_base_seed": 700001,
        "calibration_parameter_grid": {"l1_span_thresh": [1e-5]},
        "attack_budget": {"parallel": 1000},
        "output_root": "Projection_lrb_qwen3/outputs",
    }
    config_path.write_text(json.dumps(document), encoding="utf-8")
    return config_path


class PreregisterTests(unittest.TestCase):
    def test_official_loader_uses_glue_sst2_validation(self) -> None:
        captured: dict[str, object] = {}

        def loader(*args: object, **kwargs: object) -> list[dict[str, object]]:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return [{"sentence": "sample 1", "label": 1}]

        rows = load_official_sst2_validation(loader=loader)
        self.assertEqual(rows, [{"sentence": "sample 1", "label": 1}])
        self.assertEqual(captured["args"], ("glue", "sst2"))
        self.assertEqual(captured["kwargs"], {"split": "validation"})

    def test_deterministic_allocation_eos_and_manifest_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_experiment_config(write_config(root))
            rows = [
                {"sentence": "", "label": 0},
                {"sentence": "   ", "label": 1},
                *({"sentence": f"sample {index}", "label": index % 2} for index in range(1, 56)),
            ]
            tokenizer = FakeTokenizer()
            eligible_a = prepare_eligible_samples(rows, tokenizer, config)
            eligible_b = prepare_eligible_samples(rows, tokenizer, config)
            stages_a = allocate_stages(eligible_a)
            stages_b = allocate_stages(eligible_b)

            self.assertEqual(stages_a, stages_b)
            self.assertEqual([len(stages_a[name]) for name in ("calibration", "smoke", "final")], [20, 5, 20])
            all_keys = [sample["sample_key"] for stage in stages_a.values() for sample in stage]
            self.assertEqual(len(set(all_keys)), 45)
            for sample in all_samples(stages_a):
                tokenization = sample["tokenization"]
                self.assertFalse(tokenization["add_special_tokens"])
                self.assertEqual(tokenization["input_ids"][-1], 99)
                self.assertLessEqual(tokenization["total_token_length"], 32)
            truncated = next(sample for sample in eligible_a if sample["sentence"] == "sample 55")
            self.assertTrue(truncated["tokenization"]["was_truncated"])

            first = build_preregistration_document(
                config=config,
                stages=stages_a,
                model_key_file_sha256={"config.json": "a" * 64},
                tokenizer_sha256="b" * 64,
                tokenizer_key_file_sha256={"tokenizer.json": "c" * 64},
                created_at="2026-07-28T00:00:00Z",
                commit="d" * 40,
                versions={"python": "3.11", "torch": "2.0", "transformers": "4.0"},
            )
            second = build_preregistration_document(
                config=config,
                stages=stages_b,
                model_key_file_sha256={"config.json": "a" * 64},
                tokenizer_sha256="b" * 64,
                tokenizer_key_file_sha256={"tokenizer.json": "c" * 64},
                created_at="2026-07-29T00:00:00Z",
                commit="d" * 40,
                versions={"python": "3.11", "torch": "2.0", "transformers": "4.0"},
            )
            self.assertEqual(first["sample_lists"], second["sample_lists"])
            self.assertEqual(first["sample_list_sha256"], second["sample_list_sha256"])
            self.assertEqual(first["preregistration_sha256"], second["preregistration_sha256"])
            self.assertNotEqual(first["created_at"], second["created_at"])

    def test_missing_head_seed_is_an_error(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaises(PreregistrationConfigError):
                load_experiment_config(write_config(Path(temporary), calibration_seed=None))

    def test_complete_preregistration_is_idempotent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root)
            rows = [
                {"sentence": f"sample {index}", "label": index % 2}
                for index in range(1, 56)
            ]
            with (
                patch("src.preregister.load_qwen3_tokenizer", return_value=FakeTokenizer()),
                patch("src.preregister.load_official_sst2_validation", return_value=rows),
                patch(
                    "src.preregister.collect_model_and_tokenizer_hashes",
                    return_value=(
                        {"config.json": "a" * 64},
                        "b" * 64,
                        {"tokenizer.json": "c" * 64},
                    ),
                ),
                patch("src.preregister.git_commit", return_value="d" * 40),
                patch(
                    "src.preregister.runtime_versions",
                    return_value={"python": "3.11", "torch": "2.0", "transformers": "4.0"},
                ),
            ):
                first = preregister_experiment(config_path)
                second = preregister_experiment(config_path)

            self.assertEqual(first["status"], "created")
            self.assertEqual(second["status"], "already_preregistered")
            self.assertEqual(first["preregistration_sha256"], second["preregistration_sha256"])
            self.assertEqual(first["sample_list_sha256"], second["sample_list_sha256"])
            manifests = root / "Projection_lrb_qwen3" / "manifests"
            self.assertTrue((manifests / "preregistration.json").is_file())
            self.assertEqual(len((manifests / "calibration.jsonl").read_text(encoding="utf-8").splitlines()), 20)
            self.assertEqual(len((manifests / "smoke.jsonl").read_text(encoding="utf-8").splitlines()), 5)
            self.assertEqual(len((manifests / "final.jsonl").read_text(encoding="utf-8").splitlines()), 20)


def all_samples(stages: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    return [sample for samples in stages.values() for sample in samples]


if __name__ == "__main__":
    unittest.main()
