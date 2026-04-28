#!/bin/bash -l
# PCI-DeepGMM compare runner — epoch sweep (recovery for lost tasks 1-3)
#
# Original sweep (job 24389396_[0-4], 2026-04-28) had a dump-path collision:
# all 5 tasks called dt.datetime.now().strftime("%Y%m%d_%H%M%S") within the
# same second so tasks 1, 2, 3, 4 all wrote to the SAME folder, and "last
# writer wins" left only task 4's output. Task 0 (10k) survived (different
# second). This sweep re-runs the lost middle tasks at max_epochs ∈
# {14000, 18000, 22000} with the timestamp-microseconds + array-id-suffix
# fix in run_pci_compare.py.
#
# Submit:  sbatch slurm_job.sh
# Status:  squeue -u $USER ; sacct -j <jobid> --format=JobID,State,Elapsed
#SBATCH -J pci_compare_epochs
#SBATCH -A punim2738
#SBATCH -p sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
set -euo pipefail

EPOCHS=(14000 18000 22000)
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
