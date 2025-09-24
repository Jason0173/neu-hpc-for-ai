// gemm_tiled_ops.cu
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16            // try 16 or 32; tune for your GPU
#endif

// Row-major index helper
__device__ __forceinline__
int RM(int r, int c, int ld_cols) { return r * ld_cols + c; }

// C <- alpha * op(A) * op(B) + beta * C
// A_base has shape (mA x kA), B_base has shape (kB x nB), C is (m x n).
// opA/opB in {'N','T'}.
__global__ void gemm_tiled_ops(
    const float* __restrict__ A, int mA, int kA, char opA,
    const float* __restrict__ B, int kB, int nB, char opB,
    float* __restrict__ C, int m, int n,
    int K,                      // common dimension after ops
    float alpha, float beta)
{
    // +1 padding to mitigate shared-mem bank conflicts
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y; // output row in C
    const int col = blockIdx.x * TILE + threadIdx.x; // output col in C

    float acc = 0.0f;

    // Loop over K in tiles
    for (int t = 0; t < K; t += TILE) {

        // ----- Load tile of op(A) into As -----
        // We want op(A)[row, t..t+TILE-1]
        if (opA == 'N') {
            // Coalesced: threads in a warp vary threadIdx.x -> consecutive columns
            int kcol = t + threadIdx.x;
            As[threadIdx.y][threadIdx.x] =
                (row < m && kcol < K) ? A[RM(row, kcol, kA)] : 0.0f;
        } else { // 'T' : op(A)=A^T -> element is A[kcol, row]
            // Corner-turning: store transposed in shared memory so compute loop is the same.
            // Access pattern here is strided in global (column of A), but we pay it once and reuse TILE times.
            int krow = t + threadIdx.x;
            As[threadIdx.x][threadIdx.y] =
                (krow < K && row < mA) ? A[RM(krow, row, kA)] : 0.0f;
        }

        // ----- Load tile of op(B) into Bs -----
        // We want op(B)[t..t+TILE-1, col]
        if (opB == 'N') {
            // Coalesced: threads in a warp vary threadIdx.y -> but we map to x for coalescing via corner-turn
            int krow = t + threadIdx.y;
            Bs[threadIdx.y][threadIdx.x] =
                (krow < K && col < n) ? B[RM(krow, col, nB)] : 0.0f;
        } else { // 'T' : op(B)=B^T -> element is B[col, krow]
            // Corner-turning: put into Bs transposed so compute loop stays the same
            int kcol = t + threadIdx.y;
            Bs[threadIdx.x][threadIdx.y] =
                (col < kB && kcol < K) ? B[RM(col, kcol, nB)] : 0.0f;
        }

        __syncthreads(); // all tiles ready

        // ----- Compute on tiles -----
        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads(); // reuse shared mem next iteration
    }

    // In-place epilogue: C = alpha*acc + beta*C
    if (row < m && col < n) {
        float out = alpha * acc;
        if (beta != 0.0f) out += beta * C[RM(row, col, n)];
        C[RM(row, col, n)] = out;
    }
}

// Host launcher: computes output shapes and K automatically.
void gemm_tiled(
    const float* dA, int mA, int kA, char opA,
    const float* dB, int kB, int nB, char opB,
    float* dC, int m, int n, float alpha, float beta,
    cudaStream_t stream = 0)
{
    // After ops:
    // op(A): mA x kA if 'N', else kA x mA
    // op(B): kB x nB if 'N', else nB x kB
    int m_after = (opA == 'N') ? mA : kA;
    int n_after = (opB == 'N') ? nB : kB;
    int K_A     = (opA == 'N') ? kA : mA;
    int K_B     = (opB == 'N') ? kB : nB;

    // (Optional) sanity checks for debug builds:
    // assert(m_after == m && n_after == n && K_A == K_B);

    int K = K_A;

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    gemm_tiled_ops<<<grid, block, 0, stream>>>(
        dA, mA, kA, opA,
        dB, kB, nB, opB,
        dC, m, n, K, alpha, beta
    );
}
