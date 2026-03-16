#!/bin/bash
#SBATCH -J ABB4_inference
#SBATCH --time=9-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:Quadro_RTX_6000:8

#SBATCH --mail-user=rebonto.haque@magd.ox.ac.uk
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=standard-opig-gpu
#SBATCH --clusters=swan
#SBATCH -w nagagpu03.cpu.stats.ox.ac.uk

#SBATCH --output=/ceph/opig-shared/users/haque/DPhil/ABB4/ABB4/logs/%x/%x_%j.out
#SBATCH --error=/ceph/opig-shared/users/haque/DPhil/ABB4/ABB4/logs/%x/%x_error_%j.out

cd /ceph/opig-shared/users/haque/DPhil/ABB4/ABB4
source ~/.bashrc
conda activate abb4_env
set -e

# ------------------
# Defaults
# ------------------
NPROC_PER_NODE=8
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
# Run inference
# ------------------
python -W ignore -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  abb4/experiments/inference.py \
  -cn inference