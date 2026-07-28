#!/usr/bin/env python3
"""Create the required Layer-1-only calibration amendment from the real run6 log."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QWEN_ROOT.parent
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from src.dager_qwen3.layer1_calibration_amendment import (
    RUN6_LOG_RELATIVE_PATH,
    write_or_verify_amendment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed Qwen3 Layer-1 calibration amendment.")
    parser.add_argument(
        "--run6-log",
        default=str(Path("Projection_lrb_qwen3") / RUN6_LOG_RELATIVE_PATH),
        help="Repository-relative run6 failure log captured before any successful reconstruction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = Path(args.run6_log).expanduser()
    run6_log = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        path = write_or_verify_amendment(project_root=QWEN_ROOT, run6_log=run6_log)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(path.relative_to(REPOSITORY_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
