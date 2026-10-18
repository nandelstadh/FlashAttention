Implementation of [the originial Flash Attention algorithm](https://arxiv.org/abs/2205.14135) in CUDA. The core CUDA kernel is an efficient implementation of the original paper. Scaffolding is based on [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal), with major modifications. 

## Building and running on UPPMAX Pelle

The kernel uses TF32 tensor cores and is built for the H100 (sm_90). It links against the prebuilt libtorch C++ distribution; nothing besides this project needs to be compiled.

One-time setup on a login node:

```bash
module load GCC/14.3.0 CUDA/13.2.0 CMake/3.31.8-GCCcore-14.3.0
cd ~
wget https://download.pytorch.org/libtorch/cu132/libtorch-shared-with-deps-2.14.1%2Bcu132.zip
unzip -q libtorch-shared-with-deps-2.14.1+cu132.zip   # creates ~/libtorch
rm libtorch-shared-with-deps-2.14.1+cu132.zip

cd <this repo>
source slurm/common.sh   # loads the modules and builds build/main
```

To keep libtorch elsewhere, `export LIBTORCH=/path/to/libtorch` before building or submitting.

Submit from the repo root; logs and CSVs go to `results/`:

```bash
sbatch slurm/test_gpu.sh       # correctness of the FA kernel against unfused PyTorch on the GPU
sbatch slurm/bench_cpu.sh      # unfused PyTorch on the CPU
sbatch slurm/bench_unfused.sh  # unfused PyTorch on the GPU
sbatch slurm/bench_fa.sh       # FlashAttention kernel
```

`build/main <cpu|unfused|fa|test> B nh N d` can also be run directly.
