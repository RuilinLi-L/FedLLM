#!/usr/bin/env python3
"""Create the immutable 20-sample BF16 structural-gate amendment only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QWEN_ROOT.parent
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from src.dager_qwen3.bf16_gate_profile_amendment import (
    DEFAULT_DIAGNOSTIC_ROOT_RELATIVE_PATH,
    write_or_verify_amendment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the 20-sample BF16 numerical diagnostic-gate amendment. "
            "This command does not run Layer-1, Layer-2, ROUGE, reconstruction, or LRB."
        )
    )
    parser.add_argument(
        "--diagnostic-root",
        default=str(Path("Projection_lrb_qwen3") / DEFAULT_DIAGNOSTIC_ROOT_RELATIVE_PATH),
        help="Repository-relative directory containing <sample_key>.json structural diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = Path(args.diagnostic_root).expanduser()
    diagnostic_root = (
        candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    )
    try:
        path = write_or_verify_amendment(
            project_root=QWEN_ROOT,
            diagnostic_root=diagnostic_root,
        )
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(path.relative_to(REPOSITORY_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
