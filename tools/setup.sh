#!/bin/bash
# Sets xblick-vjepa2 up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv)
# * the checkpoint in ~/.cache/xblick/vjepa2/ssv2 (the node's cache directory), through xblick-vjepa2 --download-only (weights.py's CHECKPOINTS; only if
#   missing): Meta's V-JEPA 2 ViT-L with its Something-Something v2 head (Hugging Face
#   facebook/vjepa2-vitl-fpc16-256-ssv2, 1.4 GB, fixed revision). The node downloads it at its start too.
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH) and git: without them it stops with the command that installs
# them. Run again, it does only what changed. It never asks and never uses sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
PROG=xblick-vjepa2

usage() {
    cat <<'EOF'
Usage: tools/setup.sh

Sets xblick-vjepa2 up in this clone, or updates it. It creates .venv exactly as uv.lock fixes it (Python 3.12,
PyTorch for CUDA 13, transformers, the core xblick from PyPI) and downloads the SSv2 checkpoint into the node's
cache directory (~/.cache/xblick/vjepa2). A second call does only what changed. It never asks and never uses sudo.
Needs git and uv (UV=/path/to/uv names another uv). The node needs an NVIDIA GPU with driver 580 or newer.

  -h, --help    this text
EOF
}

die() {
    echo "setup.sh: $*" >&2
    exit 1
}

for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    die "the node needs Linux on x86-64 (the core's wheels), this is $(uname -s) $(uname -m)"
command -v git > /dev/null || die "no git: sudo apt install git"
command -v "$UV" > /dev/null ||
    die "no uv ($UV): install it with  wget -qO- https://astral.sh/uv/install.sh | sh  and open a new terminal"
for arg in "$@"; do
    case $arg in
        -h | --help) ;;
        *) usage >&2; die "unknown option: $arg" ;;
    esac
done
MIN_DRIVER=580  # NVIDIA's minimum Linux driver for CUDA 13.0, which the cu130 torch is built for
if ! command -v nvidia-smi > /dev/null; then
    echo "$PROG: no NVIDIA driver found (no nvidia-smi). The node needs an NVIDIA GPU with driver $MIN_DRIVER or" \
        "newer to start (Ubuntu: sudo apt install nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
elif ! driver=$(timeout 30 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null) ||
        [[ -z $driver ]]; then
    echo "$PROG: nvidia-smi failed or lists no GPU. The node needs an NVIDIA GPU with driver $MIN_DRIVER or newer" \
        "to start. The setup goes on."
else
    driver=$(cut -d. -f1 <<< "$driver" | sort -n | head -1 | tr -d ' ')
    ((driver >= MIN_DRIVER)) ||
        echo "$PROG: NVIDIA driver $driver. PyTorch for CUDA 13 needs $MIN_DRIVER or newer (Ubuntu: sudo apt install" \
            "nvidia-driver-$MIN_DRIVER, then restart). The setup goes on."
fi

cat <<EOF
$PROG: setting up $PWD. It downloads what is missing of:
  * .venv as uv.lock fixes it: Python 3.12, the core xblick, and PyTorch for CUDA 13 with NVIDIA's CUDA libraries
    (NVIDIA's licence, License.txt in each nvidia package); about 3 GB, 6 GB on disk
  * the SSv2 checkpoint (1.4 GB, Hugging Face facebook/vjepa2-vitl-fpc16-256-ssv2, MIT) into the node's cache
    directory
EOF
"$UV" sync --frozen
.venv/bin/xblick-vjepa2 --download-only
echo "$PROG: set up. The commands are in .venv/bin/; the first start: README.md, Install."
