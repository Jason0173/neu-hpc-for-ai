# Benchmarks

This folder measures the course kernels against vendor baselines on one GPU. It also checks the kernels' results against reference implementations. The latest numbers are in [`RESULTS.md`](RESULTS.md).

| Program | Compares | Baseline |
|---|---|---|
| `bench_gemm.cu` | Week 2 tiled GEMM, week 3 register-blocked GEMM | cuBLAS SGEMM |
| `bench_attention.cu` | Week 4 FlashAttention forward, week 5 FlashAttention-2 forward and backward | `bench_sdpa.py`: PyTorch `scaled_dot_product_attention` in FP32 and FP16 |

The programs `#include` the week files directly, with each file's `main()` renamed. This way the numbers always come from the committed kernels, not from copies.

## Run

Requirements:

- An NVIDIA GPU
- The CUDA toolkit (`nvcc`, cuBLAS)
- Python with a CUDA build of PyTorch, plus matplotlib

```bash
bash bench/run_all.sh
```

This builds everything for the local GPU architecture and runs all benchmarks, which takes a few minutes. It writes `results/*.csv`, `plots/*.png` and `RESULTS.md`.

**No local GPU?** `modal_run.py` runs the same script on a [Modal](https://modal.com) cloud GPU and copies the results back:

```bash
pip install modal && modal setup   # once
modal run bench/modal_run.py       # NVIDIA L4 by default; --gpu A10G etc. also work
```

The L4 has the same Ada Lovelace architecture (sm_89) as RTX 40-series cards.

## What is measured

- **Precision:** all course kernels are FP32. The GEMM baseline is cuBLAS SGEMM in plain FP32, not TF32. The attention baselines are PyTorch SDPA in FP32 (memory-efficient kernel) and in FP16 (FlashAttention kernel), the setup PyTorch would use in practice.
- **Timing:** CUDA events, median of 3–50 runs after two warm-up runs. For attention, a kernel is skipped at the remaining (larger) sizes once a single run is predicted to take more than 20 s.
- **Throughput accounting:** GEMM uses 2·M·N·K FLOPs. Attention forward uses 4·N²·d FLOPs and backward 2.5 times that, the usual FlashAttention accounting.
- **Correctness:**
  - GEMM results are compared with cuBLAS.
  - Attention results are compared with the CPU references in each week's file, for N ≤ 1,024.
  - The week 5 kernels compute softmax(QKᵀ)V without the 1/√d scale, and so does their CPU reference.
