"""No-network tests for local DatasetDict preregistration semantics."""

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
    PreregistrationError,
    allocate_stages,
    build_preregistration_document,
    load_local_sst2_validation,
    prepare_eligible_samples,
    preregister_experiment,
)

try:
    from datasets import Dataset, DatasetDict

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        if kwargs.get("add_special_tokens") is not False:
            raise AssertionError("Protocol must disable tokenizer special tokens.")
        if kwargs.get("truncation") is not False or "max_length" in kwargs:
            raise AssertionError("Protocol must tokenize text before explicit truncation and EOS.")
        token_count = int(text.rsplit(" ", 1)[-1]) if text.startswith("sample ") else 1
        return {"input_ids": list(range(1, token_count + 1))}


def write_local_dataset(root: Path, *, include_idx: bool = True) -> Path:
    """Create a real temporary DatasetDict artifact without any network access."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets is required by this test helper.")
    dataset_path = root / "data" / "glue_sst2"
    sentences = ["", "   ", *(f"sample {index}" for index in range(1, 56))]
    columns: dict[str, list[object]] = {
        "sentence": sentences,
        "label": [index % 2 for index in range(len(sentences))],
    }
    if include_idx:
        columns["idx"] = [1000 + index for index in range(len(sentences))]
    DatasetDict({"validation": Dataset.from_dict(columns)}).save_to_disk(str(dataset_path))
    return dataset_path


def write_config(root: Path, *, calibration_seed: object = 404, smoke_seed: object = 505) -> Path:
    project = root / "Projection_lrb_qwen3"
    config_path = project / "configs" / "experiment.json"
    config_path.parent.mkdir(parents=True)
    document = {
        "model_path": "models/Qwen3-1.7B-Base",
        "dataset_path": "data/glue_sst2",
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


class ConfigPathTests(unittest.TestCase):
    def test_dataset_path_must_be_an_existing_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PreregistrationConfigError, "dataset_path"):
                load_experiment_config(write_config(Path(temporary)))


@unittest.skipUnless(DATASETS_AVAILABLE, "datasets is required for local DatasetDict tests")
class PreregisterTests(unittest.TestCase):
    def test_loads_saved_validation_and_preserves_idx(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_local_dataset(root)
            config = load_experiment_config(write_config(root))
            local_dataset = load_local_sst2_validation(config)

            self.assertEqual(local_dataset.validation_row_count, 57)
            self.assertEqual(local_dataset.rows[2]["idx"], 1002)
            self.assertEqual(local_dataset.rows[2]["sentence"], "sample 1")
            self.assertTrue(local_dataset.dataset_fingerprint)
            self.assertEqual(len(local_dataset.directory_contents_sha256), 64)
            self.assertTrue(local_dataset.directory_file_sha256)

    def test_missing_idx_is_an_explicit_error(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_local_dataset(root, include_idx=False)
            config = load_experiment_config(write_config(root))
            with self.assertRaisesRegex(PreregistrationError, "idx"):
                load_local_sst2_validation(config)

    def test_deterministic_allocation_eos_and_manifest_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_local_dataset(root)
            config = load_experiment_config(write_config(root))
            local_dataset = load_local_sst2_validation(config)
            tokenizer = FakeTokenizer()
            eligible_a = prepare_eligible_samples(local_dataset.rows, tokenizer, config)
            eligible_b = prepare_eligible_samples(local_dataset.rows, tokenizer, config)
            stages_a = allocate_stages(eligible_a)
            stages_b = allocate_stages(eligible_b)

            self.assertEqual(stages_a, stages_b)
            self.assertEqual([len(stages_a[name]) for name in ("calibration", "smoke", "final")], [20, 5, 20])
            all_keys = [sample["sample_key"] for stage in stages_a.values() for sample in stage]
            self.assertEqual(len(set(all_keys)), 45)
            self.assertTrue(all(sample["original_index"] >= 1000 for sample in all_samples(stages_a)))
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
                local_dataset=local_dataset,
                created_at="2026-07-28T00:00:00Z",
                commit="d" * 40,
                versions={"python": "3.10", "torch": "2.0", "transformers": "4.0"},
            )
            second = build_preregistration_document(
                config=config,
                stages=stages_b,
                model_key_file_sha256={"config.json": "a" * 64},
                tokenizer_sha256="b" * 64,
                tokenizer_key_file_sha256={"tokenizer.json": "c" * 64},
                local_dataset=local_dataset,
                created_at="2026-07-29T00:00:00Z",
                commit="d" * 40,
                versions={"python": "3.10", "torch": "2.0", "transformers": "4.0"},
            )
            self.assertEqual(first["sample_lists"], second["sample_lists"])
            self.assertEqual(first["sample_list_sha256"], second["sample_list_sha256"])
            self.assertEqual(first["preregistration_sha256"], second["preregistration_sha256"])
            self.assertNotEqual(first["created_at"], second["created_at"])
            self.assertEqual(first["dataset_path"], "data/glue_sst2")
            self.assertEqual(first["validation_row_count"], 57)
            self.assertEqual(first["dataset_fingerprint"], local_dataset.dataset_fingerprint)

    def test_missing_head_seed_is_an_error(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_local_dataset(root)
            with self.assertRaises(PreregistrationConfigError):
                load_experiment_config(write_config(root, calibration_seed=None))

    def test_complete_preregistration_is_idempotent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_local_dataset(root)
            config_path = write_config(root)
            with (
                patch("src.preregister.load_qwen3_tokenizer", return_value=FakeTokenizer()),
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
                    return_value={"python": "3.10", "torch": "2.0", "transformers": "4.0"},
                ),
            ):
                first = preregister_experiment(config_path)
                second = preregister_experiment(config_path)

            self.assertEqual(first["status"], "created")
            self.assertEqual(second["status"], "already_preregistered")
            self.assertEqual(first["preregistration_sha256"], second["preregistration_sha256"])
            self.assertEqual(first["sample_list_sha256"], second["sample_list_sha256"])
            manifests = root / "Projection_lrb_qwen3" / "manifests"
            manifest = json.loads((manifests / "preregistration.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_path"], "data/glue_sst2")
            self.assertEqual(manifest["validation_row_count"], 57)
            self.assertEqual(len(manifest["dataset_directory_contents_sha256"]), 64)
            self.assertEqual(len((manifests / "calibration.jsonl").read_text(encoding="utf-8").splitlines()), 20)
            self.assertEqual(len((manifests / "smoke.jsonl").read_text(encoding="utf-8").splitlines()), 5)
            self.assertEqual(len((manifests / "final.jsonl").read_text(encoding="utf-8").splitlines()), 20)


def all_samples(stages: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    return [sample for samples in stages.values() for sample in samples]


if __name__ == "__main__":
    unittest.main()
