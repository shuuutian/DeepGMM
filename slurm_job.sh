#!/bin/bash -l
# PCI-DeepGMM compare runner — full 5-task epoch sweep WITH per-grid β̂(a)
#
# This re-run captures full per-treatment-grid β̂(a) data (predictions.csv)
# that prior runs did not save. Specifically:
#   - The 2026-04-28 sweep (job 24389396_[0-4]) only saved beta_a1 / beta_a0
#     (the two endpoints) per rep, so the 8 interior grid points were lost.
#   - Tasks 1-3 of that sweep were also overwritten by a dump-path timestamp
#     collision (since fixed at commit 567999d).
# This sweep re-runs all 5 epoch points {10k, 14k, 18k, 22k, 26k} on the
# code at tag run/M/20260510-01, which adds full-grid β̂(a) to predictions.csv.
#
# Output folders are uniquely named with microseconds + SLURM_ARRAY_TASK_ID,
# so concurrent array tasks NEVER overwrite each other and these new dumps
# do NOT touch the prior 2026-04-28 dumps either.
#
# Submit:  sbatch slurm_job.sh
# Status:  squeue -u $USER ; sacct -j <jobid> --format=JobID,State,Elapsed
#SBATCH -J pci_compare_epochs
#SBATCH -A punim2738
#SBATCH -p sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
