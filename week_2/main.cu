#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

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

// CPU reference implementation for verification
void gemm_cpu(const float* A, const float* B, const float* C, float* D,
              int m, int n, int k, float alpha, float beta) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            D[i * n + j] = alpha * sum + beta * (C ? C[i * n + j] : 0.0f);
        }
    }
}

// Initialize matrix with random values
void init_matrix(float* matrix, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

// Check if two matrices are approximately equal
bool check_result(const float* result, const float* expected, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(result[i] - expected[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << result[i] << " vs " << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions
    const int m = 512;  // rows of A and C
    const int n = 512;  // cols of B and C  
    const int k = 512;    // cols of A and rows of B
    
    const float alpha = 1.0f;
    const float beta = 0.0f;  // C = 0, so D = alpha * A * B
    
    std::cout << "CUDA GEMM Test" << std::endl;
    std::cout << "Matrix dimensions: " << m << " x " << k << " * " << k << " x " << n << " = " << m << " x " << n << std::endl;
    std::cout << "Tile size: " << TILE << " x " << TILE << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C(m * n, 0.0f);  // Initialize to 0
    std::vector<float> h_D_gpu(m * n);
    std::vector<float> h_D_cpu(m * n);
    
    // Initialize matrices
    init_matrix(h_A.data(), m * k);
    init_matrix(h_B.data(), k * n);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMalloc(&d_D, m * n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warm up
    gemm(d_A, d_B, d_C, d_D, m, n, k, alpha, beta);
    cudaDeviceSynchronize();
    
    // Time GPU computation
    auto start = std::chrono::high_resolution_clock::now();
    gemm(d_A, d_B, d_C, d_D, m, n, k, alpha, beta);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy result back
    cudaMemcpy(h_D_gpu.data(), d_D, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU reference computation
    start = std::chrono::high_resolution_clock::now();
    gemm_cpu(h_A.data(), h_B.data(), h_C.data(), h_D_cpu.data(), m, n, k, alpha, beta);
    end = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify results
    bool correct = check_result(h_D_gpu.data(), h_D_cpu.data(), m * n);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "GPU time: " << gpu_time.count() << " microseconds" << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " microseconds" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time.count() / gpu_time.count() << "x" << std::endl;
    std::cout << "Result correct: " << (correct ? "YES" : "NO") << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    
    return correct ? 0 : 1;
}
