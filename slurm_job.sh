#!/bin/bash -l
#SBATCH -J pci_compare_300rep
#SBATCH -A punim2738
#SBATCH -p sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

uv run python run_pci_compare.py --config compare --n-rep 300 --num-cpus 16 --no-cuda
