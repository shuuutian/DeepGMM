#!/bin/bash
#SBATCH --job-name=pci_deepgmm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out

module load python/3.10
cd /home/wzzho2/DeepGMM
uv run python run_pci_compare.py --config compare --n-rep 100 --missing-rate 0.3 --num-cpus 8

