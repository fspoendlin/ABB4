#!/usr/bin/env bash
set -e

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
# Run inference
# ------------------
python -W ignore -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  abb4/experiments/inference.py \
  -cn inference
