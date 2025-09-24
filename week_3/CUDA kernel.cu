// gemm_op.cu
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16     // try 16 or 32 depending on your GPU
#endif

// Row-major index helper
__device__ __forceinline__
int rm(int r, int c, int ld_cols) { return r * ld_cols + c; }

// Read op(A)[row, col] from base A stored row-major with shape (AR x AC).
// opA: 'N' (no-transpose) or 'T' (transpose).
__device__ __forceinline__
float getA(const float* A, int AR, int AC, char opA, int row, int col) {
    if (opA == 'N') return A[rm(row, col, AC)];   // A[row, col]
    // A^T[row, col] == A[col, row]
    return A[rm(col, row, AC)];
}

// Read op(B)[row, col] from base B stored row-major with shape (BR x BC).
__device__ __forceinline__
float getB(const float* B, int BR, int BC, char opB, int row, int col) {
    if (opB == 'N') return B[rm(row, col, BC)];   // B[row, col]
    // B^T[row, col] == B[col, row]
    return B[rm(col, row, BC)];
}

// C <- alpha * op(A) * op(B) + beta * C
// A_base shape: (mA x kA)  (usually m x k)
// B_base shape: (kB x nB)  (usually k x n)
// opA/opB in {'N','T'}
__global__ void gemm_op_inplace(
    const float* __restrict__ A, int mA, int kA, char opA,
    const float* __restrict__ B, int kB, int nB, char opB,
    float* __restrict__ C, int m, int n,  // C is m x n (in-place)
    int K,                                // common dim after op: K = (opA cols) = (opB rows)
    float alpha, float beta)
{
    __shared__ float As[TILE][TILE + 1];  // +1 to reduce bank conflicts
    __shared__ float Bs[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y; // [0..m)
    int col = blockIdx.x * TILE + threadIdx.x; // [0..n)

    float acc = 0.0f;

    // Loop over K dimension in chunks of TILE
    for (int t = 0; t < K; t += TILE) {

        // --- load a TILE from op(A) at rows=row, cols=t..t+TILE-1
        // Coalesced path when opA == 'N'.
        int kAcol = t + threadIdx.x;
        if (row < m && kAcol < K)
            As[threadIdx.y][threadIdx.x] =
                getA(A, mA, kA, opA, row, kAcol);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // --- load a TILE from op(B) at rows=t..t+TILE-1, cols=col
        // Coalesced path when opB == 'N'.
        int kBrow = t + threadIdx.y;
        if (kBrow < K && col < n)
            Bs[threadIdx.y][threadIdx.x] =
                getB(B, kB, nB, opB, kBrow, col);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute the partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    // In-place update of C (standard GEMM epilogue)
    if (row < m && col < n) {
        float out = alpha * acc;
        if (beta != 0.0f) out += beta * C[rm(row, col, n)];
        C[rm(row, col, n)] = out;
    }
}
