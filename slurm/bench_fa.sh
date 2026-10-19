#!/bin/bash -l
#SBATCH -A uppmax2026-1-224
#SBATCH -J bench_fa
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 0:20:00
#SBATCH -o results/%x_%j.log

# FlashAttention kernel: time, GFLOP/s, memory footprint, DRAM bandwidth
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh

run_gpu_sweep fa "results/fa_${SLURM_JOB_ID}.csv"
