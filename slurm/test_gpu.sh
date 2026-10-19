#!/bin/bash -l
#SBATCH -A uppmax2026-1-224
#SBATCH -J test_gpu
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 0:10:00
#SBATCH -o results/%x_%j.log

# Correctness: FA GPU kernel vs unfused PyTorch GPU attention, including
# sequence lengths that are smaller than or not multiples of the tile sizes
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh

pass=0; fail=0
for heads in "1,1" "2,4"; do
    IFS=, read -r B nh <<< "$heads"
    for d in 32 64 128; do
        for N in 1 63 64 65 1000 4096; do
            printf "B=%d nh=%d N=%d d=%d: " "$B" "$nh" "$N" "$d"
            if build/main test "$B" "$nh" "$N" "$d"; then
                pass=$((pass + 1))
            else
                fail=$((fail + 1))
            fi
        done
    done
done

echo "passed $pass, failed $fail"
[[ $fail -eq 0 ]]
