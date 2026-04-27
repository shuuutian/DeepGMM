#!/bin/bash -l
# PCI-DeepGMM compare runner — epoch sweep
#
# Job array of five tasks at max_epochs ∈ {10000, 14000, 18000, 22000, 26000}.
# All other params held at the [sets.compare] defaults from
# configs/experiment_params.toml. Outputs land under
# /data/projects/punim2738/wzzho2/dumps/ to keep home off quota.
#
# Wall-time per task scales ~ linearly with max_epochs against the n_rep=300
# 6000-epoch baseline (4h47m). Worst case (26000 epochs) ≈ 21h, the 30h time
# limit gives margin for fair-share scheduling delays.
#
# Submit:  sbatch slurm_job.sh
# Status:  squeue -u $USER ; sacct -j <jobid> --format=JobID,State,Elapsed
#SBATCH -J pci_compare_epochs
#SBATCH -A punim2738
#SBATCH -p sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --array=0-4
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
set -euo pipefail

EPOCHS=(10000 14000 18000 22000 26000)
MAX_EPOCHS=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

DUMP_ROOT=/data/projects/punim2738/wzzho2/dumps
mkdir -p "$DUMP_ROOT"

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[task $SLURM_ARRAY_TASK_ID] max_epochs=$MAX_EPOCHS dump_root=$DUMP_ROOT"

uv run python run_pci_compare.py \
  --config compare \
  --n-rep 300 \
  --num-cpus 16 \
  --max-epochs "$MAX_EPOCHS" \
  --dump-root "$DUMP_ROOT" \
  --no-cuda
