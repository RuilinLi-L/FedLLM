#!/usr/bin/env python3
"""Create or verify the immutable Stage-5 Qwen3 none-only calibration grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dager_qwen3.calibration_grid_control import write_or_verify_calibration_grid_control
from src.hashing import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or verify the immutable Qwen3 Stage-5 calibration grid.")
    parser.add_argument(
        "--tau1-control",
        default="Projection_lrb_qwen3/frozen_controls/qwen3_none_tau1_calibration.json",
        help="Repository-relative frozen tau1 control.",
    )
    parser.add_argument(
        "--output",
        default="Projection_lrb_qwen3/frozen_controls/qwen3_none_attack_calibration_grid.json",
        help="Repository-relative immutable calibration-grid control output.",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("Control paths must be repository-relative.")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def main() -> int:
    args = parse_args()
    try:
        path, document, written = write_or_verify_calibration_grid_control(
            project_root=PROJECT_ROOT,
            tau1_control_path=_resolve(args.tau1_control),
            output_path=_resolve(args.output),
        )
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "created" if written else "verified_existing", "path": path.relative_to(REPOSITORY_ROOT).as_posix(), "sha256": sha256_file(path), "identity_sha256": document["identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
