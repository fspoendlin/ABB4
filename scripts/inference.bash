#!/usr/bin/env bash
# inference.bash — Launch distributed ABB4 inference via torch.distributed.run.
#
# Wraps abb4/experiments/inference.py with multi-GPU DDP support. Inference
# settings (checkpoint path, input CSV, output directory, number of samples)
# are read from abb4/configs/inference.yaml; override any value on the command
# line by appending Hydra-style key=value arguments.
#
# Usage:
#   bash scripts/inference.bash [--nproc_per_node N] [--master_port PORT] \
#       [hydra_key=value ...]
#
# Options:
#   --nproc_per_node N    Number of GPUs to use (default: 1).
#   --master_port PORT    Port for the distributed process group (default: 29500).
#
# Examples:
#   bash scripts/inference.bash --nproc_per_node 4
#   bash scripts/inference.bash --nproc_per_node 1 \
#       data.dataset.predict.csv_path=/data/query.csv \
#       experiment.prediction.output_dir=/out/run1
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ------------------
# Defaults
# ------------------
NPROC_PER_NODE=1
MASTER_PORT=29500

# ------------------
# Parse arguments
# ------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --master_port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--nproc_per_node N] [--master_port PORT] [extra args]"
      exit 0
      ;;
    *)
      # Remaining args passed to inference.py
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# ------------------
# Activate conda environment
# ------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate abb4_env

# ------------------
# Run inference
# ------------------
python -W ignore -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  abb4/experiments/inference.py \
  -cn inference
