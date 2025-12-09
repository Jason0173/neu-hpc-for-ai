#pragma once
#include <stdint.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// ---------------------------
// Symmetric tensor layout
// L \in R^{P x R x B x E x C x H}
// P: #PEs (GPUs)
// R: rounds (0 = dispatch, 1 = combine)
// B: buffers per round (0 = send, 1 = recv)
// E: local experts per GPU
// C: capacity (tokens per expert per GPU)
// H: hidden size
// ---------------------------

struct SymLayoutConfig {
    int P;   // number of PEs / GPUs
    int R;   // communication rounds, normally 2
    int B;   // buffers per round, normally 2 (send/recv)
    int E;   // local experts
    int C;   // capacity per expert
    int H;   // hidden dimension
};

enum LayoutRound : int {
    ROUND_DISPATCH = 0,
    ROUND_COMBINE  = 1
};

enum LayoutBuffer : int {
    BUF_SEND = 0,
    BUF_RECV = 1
};

// Flattened index: ((((p * R + r) * B + b) * E + e) * C + c) * H + h
__host__ __device__ inline
size_t layout_index(const SymLayoutConfig &cfg,
                    int p, int r, int b,
                    int e, int c, int h) {
    size_t idx = h;
    idx += (size_t)c * cfg.H;
    idx += (size_t)e * cfg.C * cfg.H;
    idx += (size_t)b * cfg.E * cfg.C * cfg.H;
    idx += (size_t)r * cfg.B * cfg.E * cfg.C * cfg.H;
    idx += (size_t)p * cfg.R * cfg.B * cfg.E * cfg.C * cfg.H;
    return idx;
}

__host__ __device__ inline
size_t layout_num_elems(const SymLayoutConfig &cfg) {
    return (size_t)cfg.P * cfg.R * cfg.B * cfg.E * cfg.C * cfg.H;
}

__device__ inline
float *layout_ptr(float *base,
                  const SymLayoutConfig &cfg,
                  int p, int r, int b,
                  int e, int c, int h) {
    return base + layout_index(cfg, p, r, b, e, c, h);
}

// host-side helper for symmetric allocation
inline float *flash_dmoe_alloc_layout(const SymLayoutConfig &cfg) {
    size_t n = layout_num_elems(cfg);
    float *L = (float *)nvshmem_malloc(n * sizeof(float));
    if (!L) {
        fprintf(stderr, "nvshmem_malloc failed in flash_dmoe_alloc_layout\n");
        nvshmem_global_exit(1);
    }
    // Ensure all PEs complete allocation (symmetric allocation requirement)
    nvshmem_barrier_all();
    return L;
}
