#!/bin/bash
# Sets xblick-vjepa2 up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv)
# * the checkpoint in weights/ssv2, through xblick-vjepa2 --download-only (weights.py's CHECKPOINTS; only if
#   missing): Meta's V-JEPA 2 ViT-L with its Something-Something v2 head (Hugging Face
#   facebook/vjepa2-vitl-fpc16-256-ssv2, 1.4 GB, fixed revision). The node downloads it at its start too.
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git. Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }

"$UV" sync --frozen
.venv/bin/xblick-vjepa2 --download-only
echo "xblick-vjepa2: set up; checkpoints in weights/"
