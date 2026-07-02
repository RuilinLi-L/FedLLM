#!/bin/bash
# Run one PEFTLeak-style image shared-bin mechanism experiment (default mode: vit_adapter).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

python attack_peftleak_image.py "$@"
