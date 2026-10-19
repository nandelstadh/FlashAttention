#!/bin/bash -l
#SBATCH -A uppmax2026-1-224
#SBATCH -J bench_cpu
#SBATCH -p pelle
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0:20:00
#SBATCH -o results/%x_%j.log

# Unfused PyTorch attention on the CPU: time, GFLOP/s and peak RSS
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CSV="results/cpu_${SLURM_JOB_ID}.csv"

cpu_row() {
    local cfg="$1,$2,$3,$4" row
    echo "== cpu $cfg (${SECONDS}s)"
    # Unfused attention peaks at about two B*nh*N^2 float buffers; skip what can't fit
    local need_mib=$(( 2 * $1 * $2 * $3 * $3 * 4 / 1048576 ))
    if (( need_mib > SLURM_MEM_PER_NODE * 3 / 4 )); then # SLURM_MEM_PER_NODE is in MB
        echo "$cfg,skipped,,," >> "$CSV"
        return
    fi
    if row=$(build/main cpu "$@"); then
        echo "$cfg,$row" >> "$CSV"
    else
        echo "$cfg,failed,,," >> "$CSV"
    fi
}

echo "B,nh,N,d,iters,ms,gflops,peak_rss_mib" > "$CSV"
for_each_config cpu_row
