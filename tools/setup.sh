#!/bin/bash
# Sets durchblick-vjepa2 up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv)
# * the checkpoints in weights/<name> (only if missing; the node runs offline): model.py's CHECKPOINTS, Meta's
#   V-JEPA 2 ViT-L with its Something-Something v2 head (Hugging Face facebook/vjepa2-vitl-fpc16-256-ssv2, 1.4 GB,
#   fixed revision)
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git. Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }

"$UV" sync --frozen
HF_HUB_OFFLINE=0 .venv/bin/python -c "
from huggingface_hub import hf_hub_download
from durchblick_vjepa2.model import CHECKPOINT_FILES, CHECKPOINTS, WEIGHTS
for name, (repo, revision) in CHECKPOINTS.items():
    for f in CHECKPOINT_FILES:
        if not (WEIGHTS / name / f).is_file():
            print(hf_hub_download(repo, f, revision=revision, local_dir=WEIGHTS / name))"
echo "durchblick-vjepa2: set up; checkpoints in weights/"
