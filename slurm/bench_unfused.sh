#!/bin/bash -l
#SBATCH -A uppmax2026-1-224
#SBATCH -J bench_unfused
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 0:20:00
#SBATCH -o results/%x_%j.log

# Unfused PyTorch attention on the GPU: time, GFLOP/s, memory footprint, DRAM bandwidth
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh

run_gpu_sweep unfused "results/unfused_${SLURM_JOB_ID}.csv"
