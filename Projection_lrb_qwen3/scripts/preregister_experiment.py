#!/usr/bin/env python3
"""CLI entry point for deterministic Qwen3 SST-2 preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PreregistrationConfigError
from src.preregister import PreregistrationError, preregister_experiment
from src.result_schema import ResultSchemaError


def parse_args() -> argparse.Namespace:
    """Parse the isolated preregistration CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically preregister official GLUE SST-2 validation samples "
            "for the isolated Qwen3 experiment. No DAGER attack is run."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.json",
        help="Path to the explicit Qwen3 experiment JSON configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Run preregistration and emit one structured JSON status object."""
    args = parse_args()
    try:
        result = preregister_experiment(args.config)
    except (PreregistrationConfigError, PreregistrationError, ResultSchemaError) as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
