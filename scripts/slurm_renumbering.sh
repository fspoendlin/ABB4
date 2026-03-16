#!/bin/bash
#SBATCH -J ABB4_renumbering
#SBATCH --time=9-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:Quadro_RTX_6000:9

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
set -euo pipefail

USE_TQDM=0
export PYTHONUNBUFFERED=1

echo "[$(date)] Starting job ${SLURM_JOB_ID:-unknown} on ${HOSTNAME}"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

CMD=(
	python -u scripts/renumber_predictions_optimised.py
	--pred_path predictions/gpu03_v2
	--out_path /scratch/fast/haque/ABB4/renumbered_predictions/gpu03_v2
	--gpu
	--gpu_ids 0,1,2,3,4,5,6,7,8
	--num_workers 9
	--task_batch_size 16
	--ncpu_per_worker 2
	--anarcii_batch_size 128
	--log_every 1
)

if [[ "${USE_TQDM}" == "1" ]]; then
	CMD+=(--use_tqdm)
fi

echo "[$(date)] Command: ${CMD[*]}"
"${CMD[@]}"