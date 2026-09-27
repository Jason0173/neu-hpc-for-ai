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

// Host launcher function
void launch_gemm_op(const float* d_A, int mA, int kA, char opA,
                    const float* d_B, int kB, int nB, char opB,
                    float* d_C, int m, int n, int K,
                    float alpha, float beta, cudaStream_t stream = 0) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    gemm_op_inplace<<<grid, block, 0, stream>>>(
        d_A, mA, kA, opA, d_B, kB, nB, opB, d_C, m, n, K, alpha, beta);
}

// CPU reference implementation for validation
void gemm_cpu_ref(const float* A, int mA, int kA, char opA,
                  const float* B, int kB, int nB, char opB,
                  float* C, int m, int n, int K,
                  float alpha, float beta) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = (opA == 'N') ? A[i * kA + k] : A[k * mA + i];
                float b_val = (opB == 'N') ? B[k * nB + j] : B[j * kB + k];
                acc += a_val * b_val;
            }
            C[i * n + j] = alpha * acc + beta * C[i * n + j];
        }
    }
}

// Test program
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    // Test parameters
    int m = 256, n = 256, k = 256;
    char opA = 'N', opB = 'N';
    float alpha = 1.25f, beta = 0.5f;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tA") == 0) opA = 'T';
        if (strcmp(argv[i], "--tB") == 0) opB = 'T';
    }
    
    // Determine matrix dimensions
    int mA = (opA == 'N') ? m : k;
    int kA = (opA == 'N') ? k : m;
    int kB = (opB == 'N') ? k : n;
    int nB = (opB == 'N') ? n : k;
    
    printf("Testing GEMM: C = %.2f * op(A) * op(B) + %.2f * C\n", alpha, beta);
    printf("op(A) = %s, op(B) = %s\n", (opA == 'N') ? "A" : "A^T", (opB == 'N') ? "B" : "B^T");
    printf("Matrix sizes: A(%dx%d), B(%dx%d), C(%dx%d)\n", mA, kA, kB, nB, m, n);
    
    // Allocate host memory
    size_t bytesA = mA * kA * sizeof(float);
    size_t bytesB = kB * nB * sizeof(float);
    size_t bytesC = m * n * sizeof(float);
    
    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);
    float* h_C_ref = (float*)malloc(bytesC);
    
    // Initialize data
    for (int i = 0; i < mA * kA; ++i) h_A[i] = (float)((i % 13) - 6) / 7.0f;
    for (int i = 0; i < kB * nB; ++i) h_B[i] = (float)((i % 17) - 8) / 9.0f;
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = (float)((i % 7) - 3) / 5.0f;
        h_C_ref[i] = h_C[i];
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));
    
    // Run CPU reference
    gemm_cpu_ref(h_A, mA, kA, opA, h_B, kB, nB, opB, h_C_ref, m, n, k, alpha, beta);
    
    // Run GPU kernel
    launch_gemm_op(d_A, mA, kA, opA, d_B, kB, nB, opB, d_C, m, n, k, alpha, beta);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    
    // Validate results
    double max_err = 0.0, sum_ref = 0.0, sum_gpu = 0.0;
    for (int i = 0; i < m * n; ++i) {
        double err = fabs(h_C[i] - h_C_ref[i]);
        max_err = std::max(max_err, err);
        sum_ref += h_C_ref[i];
        sum_gpu += h_C[i];
    }
    
    printf("Results:\n");
    printf("  Max absolute error: %.2e\n", max_err);
    printf("  Sum reference: %.6f\n", sum_ref);
    printf("  Sum GPU result: %.6f\n", sum_gpu);
    printf("  %s\n", (max_err < 1e-4) ? "PASS" : "FAIL");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    return 0;
}