#!/bin/bash
# install.bash — Create the abb4_env conda environment and install all
# dependencies required to run ABB4-STEROIDS.
#
# The script detects the system CUDA version via nvidia-smi and selects the
# best matching PyTorch wheel (cu129 / cu128 / cu126). It prompts for
# confirmation before starting the GPU package downloads.
#
# Requirements:
#   - Conda (Miniconda or Anaconda)
#   - CUDA driver >= 12.6 (or no GPU for a CPU-only install)
#
# Usage:
#   bash scripts/install.bash
#
# If installation fails due to a CUDA mismatch, edit the CUDA_TAG variable or
# the two pip lines below directly:
#   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/<CUDA_TAG>
#   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+<CUDA_TAG>.html

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

conda create -y -n abb4_env python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate abb4_env

# Detect CUDA version and select best matching wheel tag
# Known supported CUDA tags for torch==2.8.0 (descending order)
SUPPORTED_TAGS=("cu129" "cu128" "cu126")

if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    DETECTED_NUM=$((CUDA_MAJOR * 100 + CUDA_MINOR))
    echo "Detected CUDA $CUDA_VERSION"

    CUDA_TAG=""
    for TAG in "${SUPPORTED_TAGS[@]}"; do
        TAG_NUM="${TAG#cu}"
        TAG_MAJOR=$((TAG_NUM / 10))
        TAG_MINOR=$((TAG_NUM % 10))
        TAG_CMP=$((TAG_MAJOR * 100 + TAG_MINOR))
        if [ "$DETECTED_NUM" -ge "$TAG_CMP" ]; then
            CUDA_TAG="$TAG"
            break
        fi
    done

    if [ -z "$CUDA_TAG" ]; then
        echo "ERROR: Detected CUDA $CUDA_VERSION is older than the minimum supported version (12.6). Aborting."
        exit 1
    fi
    echo "Selected wheel tag: $CUDA_TAG (best match for CUDA $CUDA_VERSION)"
else
    CUDA_TAG="cpu"
    echo "WARNING: nvidia-smi not found — installing CPU-only PyTorch"
fi

echo ""
echo "About to install:"
echo "  torch==2.8.0  from https://download.pytorch.org/whl/${CUDA_TAG}"
echo "  torch-scatter from https://data.pyg.org/whl/torch-2.8.0+${CUDA_TAG}.html"
echo ""
read -r -p "Proceed with CUDA tag '${CUDA_TAG}'? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted. Set CUDA_TAG manually and re-run if needed."
    exit 1
fi

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/${CUDA_TAG}
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+${CUDA_TAG}.html
conda install -y -c conda-forge biopython matplotlib numpy pandas pyyaml scipy seaborn dm-tree tqdm wandb
pip install mdtraj pytorch-lightning hydra-core GPUtil ml-collections lightning anarcii POT
pip install .
conda deactivate
