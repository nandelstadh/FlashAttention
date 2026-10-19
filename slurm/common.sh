# Shared setup for the SLURM jobs; sourced from the repo root

# Toolchain from the course's UPPMAX exercises
module load GCC/14.3.0
module load CUDA/13.2.0
module load CMake/3.31.8-GCCcore-14.3.0

# Set after the module loads, which can trip over unset variables
set -euo pipefail

# Unpacked prebuilt libtorch (libtorch-shared-with-deps-2.14.1+cu132.zip)
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
# Read by libtorch's CMake config; matches CMAKE_CUDA_ARCHITECTURES (L40s) and
# stops it from probing for a GPU on login nodes
export TORCH_CUDA_ARCH_LIST=8.9

# Sweep from the Measurements section of plan.pdf; HEADS holds B,nh pairs
NS="1024 2048 4096 8192 16384"
DS="64 128"
HEADS="1,1 4,8"

GPU_HEADER="B,nh,N,d,iters,ms,gflops,inputs_mib,peak_mib,extra_mib,dram_gb,dram_gbps,dram_pct_peak,occupancy_pct"
NCU_METRICS="dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

# Build once; the lock stops concurrent jobs from building over each other
build_if_needed() {
    if [[ ! -f "$LIBTORCH/share/cmake/Torch/TorchConfig.cmake" ]]; then
        echo "libtorch not found at $LIBTORCH; download it first (see README) or set LIBTORCH" >&2
        return 1
    fi
    mkdir -p build
    (
        flock 9
        if [[ ! -x build/main ]]; then
            cmake -S . -B build -DCMAKE_PREFIX_PATH="$LIBTORCH" -DCMAKE_BUILD_TYPE=Release
            cmake --build build -j
        fi
    ) 9> build/.lock
}

# Calls "$@" B nh N d for every sweep configuration
for_each_config() {
    local heads B nh d N
    for heads in $HEADS; do
        IFS=, read -r B nh <<< "$heads"
        for d in $DS; do
            for N in $NS; do
                "$@" "$B" "$nh" "$N" "$d"
            done
        done
    done
}

# Appends one row to $CSV for a GPU mode: timing and memory from a normal run,
# DRAM traffic from a separate ncu run (ncu's kernel replay would distort timing)
gpu_row() {
    local mode=$1; shift
    local cfg="$1,$2,$3,$4" row dram=",,,"
    echo "== $mode $cfg (${SECONDS}s)"
    if ! row=$(build/main "$mode" "$@"); then
        echo "$cfg,failed,,,,,,,,," >> "$CSV"
        return
    fi
    rm -f "$NCU_LOG"
    if ncu --profile-from-start off --clock-control none --metrics "$NCU_METRICS" \
           --csv --page raw --print-units base --log-file "$NCU_LOG" \
           build/main "$mode" "$@" > /dev/null; then
        dram=$(python3 slurm/sum_dram.py "$NCU_LOG") || dram=",,,"
    fi
    echo "$cfg,$row,$dram" >> "$CSV"
}

run_gpu_sweep() {
    CSV=$2
    NCU_LOG="results/.ncu_${SLURM_JOB_ID:-local}.csv"
    echo "$GPU_HEADER" > "$CSV"
    for_each_config gpu_row "$1"
    rm -f "$NCU_LOG"
}

mkdir -p results
build_if_needed
