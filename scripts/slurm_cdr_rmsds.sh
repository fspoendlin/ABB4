#!/bin/bash
#SBATCH -J cdr_rmsds
#SBATCH --time=9-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=62
#SBATCH --mem=200G
#SBATCH --mail-user=rebonto.haque@magd.ox.ac.uk
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=standard-opig-gpu
#SBATCH --clusters=swan
#SBATCH -w nagagpu05.cpu.stats.ox.ac.uk

#SBATCH --output=/scratch/fast/haque/ABB4/logs/%x/%x_%j.out
#SBATCH --error=/scratch/fast/haque/ABB4/logs/%x/%x_error_%j.out

cd /ceph/opig-shared/users/haque/DPhil/ABB4/ABB4
source ~/.bashrc
conda activate abb4_env
set -e

python /ceph/opig-shared/users/haque/DPhil/ABB4/ABB4/scripts/calculate_cdr_rmsds.py \
    --pred_path /scratch/fast/haque/ABB4/renumbered_predictions/gpu03_v2 \
    --output_csv /scratch/fast/haque/ABB4/cdr_rmsds.csv \
    --n_jobs 62

