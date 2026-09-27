# HPC for AI: CUDA Coursework

Coursework from a high-performance computing for AI course at Northeastern University (Fall 2025). It starts with a multithreaded matrix multiply on the CPU, then builds CUDA kernels for GEMM, softmax and attention, and finishes with FlashAttention-2 (forward and backward), multi-GPU sequence-parallel attention and an expert-parallel Mixture-of-Experts layer.

## Contents

| Week | What I built | Files |
|---|---|---|
| 1 | Multithreaded matrix multiply in C with pthreads: correctness tests against a single-threaded version, and speedup vs. thread count (up to 4.8× on 500×500) | [`week_1/`](week_1/) |
| 2 | Tiled CUDA GEMM (`D = αAB + βC`) using shared-memory tiles padded to avoid bank conflicts, checked against a CPU reference. Also includes a Java reader for llama2.c model checkpoints and exercise answers for PMPP chapters 3–5 | [`main.cu`](week_2/main.cu), [`gemm_kernel.cu`](week_2/gemm_kernel.cu), [`ReadCheckpoint.java`](week_2/ReadCheckpoint.java) |
| 3 | Faster GEMM with register blocking and double-buffered shared memory; GEMM with transpose options (`op(A)·op(B)`); one-pass online softmax in C | [`advanced_tiled_gemm.cu`](week_3/advanced_tiled_gemm.cu), [`gemm_op_transpose.cu`](week_3/gemm_op_transpose.cu), [`online_softmax.c`](week_3/online_softmax.c) |
| 4 | Causal FlashAttention forward pass in CUDA, compared with naive and tiled CPU implementations. Also includes PMPP chapter 6 exercise answers | [`flash_attn.cu`](week_4/flash_attn.cu) |
| 5 | FlashAttention-2 forward and backward kernels, with `O`, `L`, `dQ`, `dK` and `dV` checked against a CPU reference | [`flash_attention2.cu`](week_5/flash_attention2.cu) |
| 7 | Sequence-parallel FlashAttention-2 forward pass across 1–8 GPUs. Each GPU keeps its own Q shard and pulls K/V shards from the other GPUs with peer-to-peer copies, using an online softmax so the full attention matrix is never materialized | [`dist_fa2.cu`](week_7/dist_fa2.cu) |
| 8 | DeepSeek-V3-style MoE layer in PyTorch: grouped top-k routing, shared experts, and expert parallelism via all-to-all. Custom CUDA kernels, built as a PyTorch C++ extension, compute the expert histogram, token positions, scatter and gather | [`week_8/`](week_8/) |

## Build and run

The code was developed on Windows with an RTX 40-series GPU (`sm_89`). The `.bat` files are Windows helpers. On other GPUs, change `-arch` to match your card.

```bash
# Week 1 (CPU, Linux or macOS)
gcc -O2 -pthread week_1/matrix_st_updated.c -o matmul -lm && ./matmul

# Single-file CUDA programs (weeks 2-5, 7)
nvcc -O3 -arch=sm_89 week_2/main.cu -o gemm
nvcc -O3 -arch=sm_89 week_3/advanced_tiled_gemm.cu -o advanced_gemm
nvcc -O3 -arch=sm_89 week_4/flash_attn.cu -o flash_attn
nvcc -O3 -arch=sm_89 -std=c++17 week_5/flash_attention2.cu -o fa2
nvcc -O3 -arch=sm_89 week_7/dist_fa2.cu -lpthread -o dist_fa2 && ./dist_fa2 --seq 4096 --d 128 --ngpu 2

# Week 8 (PyTorch with CUDA; the extension compiles on first run)
cd week_8 && python demo.py
```

## Tech

CUDA C++, C (pthreads), PyTorch (`torch.distributed`, C++/CUDA extensions), Java
