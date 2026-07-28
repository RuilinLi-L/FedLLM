#!/usr/bin/env python3
"""Run or plan the immutable Stage-5 Qwen3 none-only calibration grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import CalibrationError, run_calibration
from src.config import load_experiment_config


def _resolve(value: str, *, description: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise CalibrationError(f"{description} must remain under {PROJECT_ROOT}.") from error
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-5 none-only Qwen3 calibration from immutable controls.")
    parser.add_argument("--tau1-control", required=True, help="Repository-relative frozen tau1 control.")
    parser.add_argument("--calibration-grid-control", required=True, help="Repository-relative immutable calibration-grid control.")
    parser.add_argument("--config", default="Projection_lrb_qwen3/configs/experiment.json", help="Registered experiment config; its attack grid is not consumed.")
    parser.add_argument("--device", default="cuda", help="Execution device.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--plan-only", action="store_true", help="Validate controls and print the candidate plan without ROUGE/model/CUDA work.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = _resolve(args.config, description="config path")
        if config_path != (PROJECT_ROOT / "configs" / "experiment.json").resolve():
            raise CalibrationError("Stage-5 calibration must use the registered experiment.json path.")
        config = load_experiment_config(config_path, require_dataset_path=False)
        result = run_calibration(
            config=config,
            manifest_path=PROJECT_ROOT / "manifests" / "calibration.jsonl",
            tau1_control_path=_resolve(args.tau1_control, description="tau1 control"),
            calibration_grid_control_path=_resolve(args.calibration_grid_control, description="grid control"),
            output_root=PROJECT_ROOT / "outputs" / "calibration",
            device=args.device,
            dtype=args.dtype,
            plan_only=args.plan_only,
        )
    except Exception as error:
        print(json.dumps({"record_type": "qwen3_stage5_calibration_error", "status": "error", "defense": "none", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["status"] in {"ok", "planned"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
