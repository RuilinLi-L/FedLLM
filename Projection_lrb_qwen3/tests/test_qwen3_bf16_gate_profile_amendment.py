"""Regression tests for the isolated 20-sample BF16 gate-profile amendment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dager_qwen3.bf16_gate_profile_amendment import (
    AMENDMENT_RELATIVE_PATH,
    Bfloat16GateProfileAmendmentError,
    FIXED_BFLOAT16_GATE_GRID,
    SELECTED_BFLOAT16_GATE,
    build_amendment_document,
    verify_amendment,
    write_or_verify_amendment,
)
from src.dager_qwen3.gradient_gate import diagnostic_thresholds


def _keys() -> list[str]:
    return [f"{index:064x}" for index in range(20)]


def _layer(*, residual: float, identity_passed: bool = True) -> dict[str, object]:
    return {
        "checks": {
            "gradient_identity": identity_passed,
            "relative_rank_within_theoretical_cap": True,
            "active_tokens_present": True,
            "active_token_residual": residual <= 3e-3,
            "all_numeric_values_finite": True,
            "orientation_is_fixed_gradient": True,
            "gradient_t_negative_control_worse": True,
        },
        "row_space_residual": {"active_tokens": {"max": residual}},
    }


def _write_profile(
    *,
    project_root: Path,
    q0_residuals: dict[int, float] | None = None,
    q1_residuals: dict[int, float] | None = None,
    identity_failure_ordinal: int | None = None,
) -> Path:
    keys = _keys()
    manifest = project_root / "manifests" / "calibration.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "\n".join(json.dumps({"sample": {"sample_key": key}}) for key in keys) + "\n",
        encoding="utf-8",
    )
    config = project_root / "configs" / "experiment.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"calibration_head_seed": 11}), encoding="utf-8")
    diagnostic_root = project_root / "outputs" / "calibration" / "bf16_gate_profile"
    diagnostic_root.mkdir(parents=True)
    for ordinal, key in enumerate(keys):
        q0 = (q0_residuals or {}).get(ordinal, 1e-3)
        q1 = (q1_residuals or {}).get(ordinal, 1e-3)
        document = {
            "schema_version": 1,
            "record_type": "qwen3_single_sample_gradient_diagnostic",
            "status": "ok" if max(q0, q1) <= 3e-3 else "failed_gradient_diagnostic",
            "diagnostic_sha256": f"{ordinal + 1:064x}",
            "compute_dtype": "torch.bfloat16",
            "head_seed": 11,
            "span_diagnostics": {
                "layers": {
                    "q0": _layer(
                        residual=q0,
                        identity_passed=identity_failure_ordinal != ordinal,
                    ),
                    "q1": _layer(residual=q1),
                }
            },
        }
        (diagnostic_root / f"{key}.json").write_text(
            json.dumps(document),
            encoding="utf-8",
        )
    return diagnostic_root


class Bfloat16GateProfileAmendmentTest(unittest.TestCase):
    def test_profile_selects_the_smallest_fixed_covering_gate_and_hashes_20_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagnostic_root = _write_profile(
                project_root=root,
                q1_residuals={2: 0.005206674337387085, 16: 0.005550340283662081},
            )
            document = build_amendment_document(
                project_root=root,
                diagnostic_root=diagnostic_root,
            )
            self.assertEqual(document["fixed_candidate_grid"], list(FIXED_BFLOAT16_GATE_GRID))
            self.assertEqual(document["diagnostic_sample_count"], 20)
            self.assertEqual(document["samples_exceeding_old_gate"], 2)
            self.assertEqual(
                document["maximum_observed_active_token_residual"],
                0.005550340283662081,
            )
            self.assertEqual(document["selected_bfloat16_gate"], SELECTED_BFLOAT16_GATE)
            self.assertTrue(document["all_non_residual_checks_passed"])
            self.assertEqual(len(document["diagnostic_json_files"]), 20)
            amendment = write_or_verify_amendment(
                project_root=root,
                diagnostic_root=diagnostic_root,
            )
            self.assertEqual(amendment, root / AMENDMENT_RELATIVE_PATH)
            self.assertEqual(
                verify_amendment(project_root=root)["selected_bfloat16_gate"],
                SELECTED_BFLOAT16_GATE,
            )

    def test_non_residual_failure_remains_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagnostic_root = _write_profile(
                project_root=root,
                identity_failure_ordinal=2,
            )
            with self.assertRaises(Bfloat16GateProfileAmendmentError):
                build_amendment_document(project_root=root, diagnostic_root=diagnostic_root)

    def test_residual_above_fixed_gate_fails_without_grid_expansion(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagnostic_root = _write_profile(
                project_root=root,
                q1_residuals={2: 0.0100001},
            )
            with self.assertRaises(Bfloat16GateProfileAmendmentError):
                build_amendment_document(project_root=root, diagnostic_root=diagnostic_root)

    def test_verification_detects_mutated_profile_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            diagnostic_root = _write_profile(
                project_root=root,
                q1_residuals={2: 0.005206674337387085, 16: 0.005550340283662081},
            )
            write_or_verify_amendment(project_root=root, diagnostic_root=diagnostic_root)
            key = _keys()[0]
            path = diagnostic_root / f"{key}.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["span_diagnostics"]["layers"]["q0"]["row_space_residual"]["active_tokens"]["max"] = 0.002
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(Bfloat16GateProfileAmendmentError):
                verify_amendment(project_root=root)

    def test_gate_boundary_and_fp32_policy_are_exact(self) -> None:
        self.assertEqual(
            diagnostic_thresholds("bfloat16")["max_active_relative_residual"],
            7.5e-3,
        )
        self.assertEqual(
            diagnostic_thresholds("float32")["max_active_relative_residual"],
            2e-4,
        )


if __name__ == "__main__":
    unittest.main()
