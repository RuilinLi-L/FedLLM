#!/usr/bin/env python3
"""Freeze the audited Qwen3 Layer-1 tau1 selection into an immutable control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QWEN_ROOT.parent
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from src.dager_qwen3.frozen_tau1_control import write_or_verify_frozen_tau1_control
from src.hashing import sha256_file


def _resolve_project_path(value: str, *, description: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(QWEN_ROOT)
    except ValueError as error:
        raise ValueError(f"{description} must remain under {QWEN_ROOT}, got {resolved}.") from error
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify the immutable Qwen3 none-only frozen tau1 control from a completed "
            "Layer-1-only calibration aggregation.  This command loads no model, ROUGE, Layer-2, or LRB code."
        )
    )
    parser.add_argument(
        "--aggregation",
        required=True,
        help="Repository-relative aggregation.json under Projection_lrb_qwen3/outputs/calibration/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Repository-relative immutable control JSON under Projection_lrb_qwen3/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        aggregation_path = _resolve_project_path(args.aggregation, description="aggregation path")
        output_path = _resolve_project_path(args.output, description="output path")
        path, document, written = write_or_verify_frozen_tau1_control(
            project_root=QWEN_ROOT,
            aggregation_path=aggregation_path,
            output_path=output_path,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "record_type": "qwen3_frozen_tau1_control_error",
                    "status": "error",
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
                "record_type": "qwen3_frozen_tau1_control",
                "status": "created" if written else "verified_existing",
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "frozen_control_identity_sha256": document["frozen_control_identity_sha256"],
                "selected_tau1": document["selected_tau1"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
