// gemm.cu
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

// Row-major indexing helper
__device__ __forceinline__
int idxRM(int row, int col, int ld_cols) { return row * ld_cols + col; }

// D = alpha * A(m×k) * B(k×n) + beta * C(m×n)
// All matrices row-major
__global__ void gemm_tiled_rowmajor(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ D,
    int m, int n, int k,
    float alpha, float beta)
{
    // Shared tiles (pad +1 to avoid bank conflicts)
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y; // output row
    const int col = blockIdx.x * TILE + threadIdx.x; // output col

    float acc = 0.0f; // register accumulator

    // Loop over k dimension in TILE-sized chunks
    for (int t = 0; t < k; t += TILE) {
        // Cooperative loads, guarded for edges
        if (row < m && (t + threadIdx.x) < k)
            As[threadIdx.y][threadIdx.x] = A[idxRM(row, t + threadIdx.x, k)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t + threadIdx.y) < k && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[idxRM(t + threadIdx.y, col, n)];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // ensure tiles are ready

        // Compute on tile
        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads(); // reuse shared memory next iteration
    }

    // Write back: D = α·acc + β·C
    if (row < m && col < n) {
        float out = alpha * acc;
        if (beta != 0.0f && C)
            out += beta * C[idxRM(row, col, n)];
        D[idxRM(row, col, n)] = out;
    }
}

// Host-side launcher
void gemm(const float* d_A, const float* d_B, const float* d_C, float* d_D,
          int m, int n, int k, float alpha, float beta, cudaStream_t stream = 0)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    gemm_tiled_rowmajor<<<grid, block, 0, stream>>>(d_A, d_B, d_C, d_D, m, n, k, alpha, beta);
}
