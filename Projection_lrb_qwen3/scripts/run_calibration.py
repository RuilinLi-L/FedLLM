#!/usr/bin/env python3
"""Run manifest-isolated, none-only Qwen3 DAGER calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QWEN_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from src.calibration import CalibrationError, run_calibration
from src.config import load_experiment_config


def _repository_path(value: str, *, description: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(QWEN_ROOT)
    except ValueError as error:
        raise CalibrationError(f"{description} must remain under {QWEN_ROOT}, got {resolved}.") from error
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every experiment.json calibration candidate on every preregistered calibration.jsonl "
            "sample. This entrypoint is permanently defense=none and exposes no threshold or budget overrides."
        )
    )
    parser.add_argument(
        "--config",
        default="Projection_lrb_qwen3/configs/experiment.json",
        help="Repository-relative Qwen3 experiment.json containing the complete candidate grid.",
    )
    parser.add_argument("--device", default="cuda", help="Explicit execution device, e.g. cuda or cuda:0.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = _repository_path(args.config, description="config path")
        expected_config = (QWEN_ROOT / "configs" / "experiment.json").resolve()
        if config_path != expected_config:
            raise CalibrationError(f"Calibration must read the registered experiment.json at {expected_config}.")
        config = load_experiment_config(config_path, require_dataset_path=False)
        result = run_calibration(
            config=config,
            manifest_path=QWEN_ROOT / "manifests" / "calibration.jsonl",
            output_root=QWEN_ROOT / "outputs" / "calibration",
            device=args.device,
            dtype=args.dtype,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "record_type": "qwen3_dager_calibration_error",
                    "status": "error",
                    "defense": "none",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "record_type": "qwen3_dager_calibration_complete",
                "status": result["status"],
                "defense": result["defense"],
                "all_results_path": result["all_results_path"].relative_to(REPOSITORY_ROOT).as_posix(),
                "summary_path": result["summary_path"].relative_to(REPOSITORY_ROOT).as_posix(),
                "frozen_attack_config_path": (
                    None
                    if result["frozen_attack_config_path"] is None
                    else result["frozen_attack_config_path"].relative_to(REPOSITORY_ROOT).as_posix()
                ),
                "selected_candidate_id": result["selected_candidate_id"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
