// Advanced Tiled GEMM CUDA Kernel with Multiple Optimization Levels
// This implementation uses register blocking, double buffering, and optimized memory access patterns
// to minimize HBM (High Bandwidth Memory) access and maximize performance

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>

// Configuration parameters for different optimization levels
#ifndef TILE_M
#define TILE_M 64    // Thread block tile size in M dimension
#endif

#ifndef TILE_N  
#define TILE_N 64    // Thread block tile size in N dimension
#endif

#ifndef TILE_K
#define TILE_K 16    // Thread block tile size in K dimension
#endif

#ifndef REG_TILE_M
#define REG_TILE_M 4  // Register tile size in M dimension
#endif

#ifndef REG_TILE_N
#define REG_TILE_N 4  // Register tile size in N dimension  
#endif

// Row-major indexing helper
__device__ __forceinline__
int rm(int row, int col, int ld_cols) { 
    return row * ld_cols + col; 
}

// Advanced tiled GEMM kernel with register blocking and double buffering
// C = alpha * A * B + beta * C
// A: m x k, B: k x n, C: m x n
__global__ void advanced_tiled_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    const float* __restrict__ C,
    float* __restrict__ D,
    int m, int n, int k,
    float alpha, float beta)
{
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];
    
    // Double buffering for shared memory
    __shared__ float As_next[TILE_M][TILE_K + 1];
    __shared__ float Bs_next[TILE_K][TILE_N + 1];
    
    // Thread block indices
    const int blockM = blockIdx.y * TILE_M;
    const int blockN = blockIdx.x * TILE_N;
    const int threadM = threadIdx.y;
    const int threadN = threadIdx.x;
    
    // Register arrays for accumulation
    float regA[REG_TILE_M];
    float regB[REG_TILE_N];
    float regC[REG_TILE_M][REG_TILE_N];
    
    // Initialize register accumulators
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) {
            regC[i][j] = 0.0f;
        }
    }
    
    // Prefetch first tiles - each thread loads multiple elements
    // Load As: need to load TILE_M x TILE_K elements with blockDim.x * blockDim.y threads
    for (int i = threadM; i < TILE_M; i += blockDim.y) {
        for (int j = threadN; j < TILE_K; j += blockDim.x) {
            if (blockM + i < m && j < k) {
                As[i][j] = A[rm(blockM + i, j, k)];
            } else {
                As[i][j] = 0.0f;
            }
        }
    }
    
    // Load Bs: need to load TILE_K x TILE_N elements
    for (int i = threadM; i < TILE_K; i += blockDim.y) {
        for (int j = threadN; j < TILE_N; j += blockDim.x) {
            if (i < k && blockN + j < n) {
                Bs[i][j] = B[rm(i, blockN + j, n)];
            } else {
                Bs[i][j] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // Main computation loop with double buffering
    for (int k_tile = 0; k_tile < k; k_tile += TILE_K) {
        // Prefetch next tiles (double buffering)
        for (int i = threadM; i < TILE_M; i += blockDim.y) {
            for (int j = threadN; j < TILE_K; j += blockDim.x) {
                if (blockM + i < m && k_tile + TILE_K + j < k) {
                    As_next[i][j] = A[rm(blockM + i, k_tile + TILE_K + j, k)];
                } else {
                    As_next[i][j] = 0.0f;
                }
            }
        }
        
        for (int i = threadM; i < TILE_K; i += blockDim.y) {
            for (int j = threadN; j < TILE_N; j += blockDim.x) {
                if (k_tile + TILE_K + i < k && blockN + j < n) {
                    Bs_next[i][j] = B[rm(k_tile + TILE_K + i, blockN + j, n)];
                } else {
                    Bs_next[i][j] = 0.0f;
                }
            }
        }
        
        // Register blocking computation
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            // Load into registers with coalesced access
            #pragma unroll
            for (int i = 0; i < REG_TILE_M; ++i) {
                int regM = threadM + i * (TILE_M / REG_TILE_M);
                if (regM < TILE_M) {
                    regA[i] = As[regM][kk];
                } else {
                    regA[i] = 0.0f;
                }
            }
            
            #pragma unroll
            for (int j = 0; j < REG_TILE_N; ++j) {
                int regN = threadN + j * (TILE_N / REG_TILE_N);
                if (regN < TILE_N) {
                    regB[j] = Bs[kk][regN];
                } else {
                    regB[j] = 0.0f;
                }
            }
            
            // Fused multiply-add operations
            #pragma unroll
            for (int i = 0; i < REG_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_TILE_N; ++j) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }
        
        // Swap buffers for next iteration
        __syncthreads();
        if (k_tile + TILE_K < k) {
            // Copy next tiles to current tiles
            for (int i = threadM; i < TILE_M; i += blockDim.y) {
                for (int j = threadN; j < TILE_K; j += blockDim.x) {
                    As[i][j] = As_next[i][j];
                }
            }
            
            for (int i = threadM; i < TILE_K; i += blockDim.y) {
                for (int j = threadN; j < TILE_N; j += blockDim.x) {
                    Bs[i][j] = Bs_next[i][j];
                }
            }
        }
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) {
            int outM = blockM + threadM + i * (TILE_M / REG_TILE_M);
            int outN = blockN + threadN + j * (TILE_N / REG_TILE_N);
            
            if (outM < m && outN < n) {
                float result = alpha * regC[i][j];
                if (beta != 0.0f && C) {
                    result += beta * C[rm(outM, outN, n)];
                }
                D[rm(outM, outN, n)] = result;
            }
        }
    }
}

// Optimized kernel launcher with automatic configuration
void launch_advanced_tiled_gemm(
    const float* d_A, const float* d_B, const float* d_C, float* d_D,
    int m, int n, int k, float alpha, float beta, cudaStream_t stream = 0) {
    
    // Calculate optimal grid dimensions
    // Each thread handles REG_TILE_M x REG_TILE_N elements
    dim3 block(TILE_N / REG_TILE_N, TILE_M / REG_TILE_M);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    
    // Launch kernel
    advanced_tiled_gemm<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, d_D, m, n, k, alpha, beta);
}

// CPU reference implementation for validation
void gemm_cpu_reference(
    const float* A, const float* B, const float* C, float* D,
    int m, int n, int k, float alpha, float beta) {
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
            }
            D[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

// Performance measurement utilities
class PerformanceTimer {
private:
    cudaEvent_t start_, stop_;
    bool started_;
    
public:
    PerformanceTimer() : started_(false) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~PerformanceTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        cudaEventRecord(start_);
        started_ = true;
    }
    
    float stop() {
        if (!started_) return 0.0f;
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        started_ = false;
        return milliseconds;
    }
};

// Calculate theoretical peak performance
double calculate_theoretical_peak(int m, int n, int k, float milliseconds) {
    // FLOPS = 2 * m * n * k (multiply-add operations)
    double flops = 2.0 * m * n * k;
    double gflops = flops / (milliseconds * 1e6);
    return gflops;
}

// Memory bandwidth calculation
double calculate_memory_bandwidth(int m, int n, int k, float milliseconds) {
    // Bytes = (A + B + C + D) * sizeof(float)
    double bytes = (m * k + k * n + m * n + m * n) * sizeof(float);
    double gbps = bytes / (milliseconds * 1e6);
    return gbps;
}

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Validation function
bool validate_results(const float* gpu_result, const float* cpu_result, int size, float tolerance = 1e-4) {
    double max_error = 0.0;
    double sum_gpu = 0.0, sum_cpu = 0.0;
    
    for (int i = 0; i < size; ++i) {
        double error = fabs(gpu_result[i] - cpu_result[i]);
        max_error = std::max(max_error, error);
        sum_gpu += gpu_result[i];
        sum_cpu += cpu_result[i];
    }
    
    printf("Validation Results:\n");
    printf("  Max absolute error: %.2e\n", max_error);
    printf("  Sum GPU: %.6f, Sum CPU: %.6f\n", sum_gpu, sum_cpu);
    printf("  %s\n", (max_error < tolerance) ? "PASS" : "FAIL");
    
    return max_error < tolerance;
}

// Main test program
int main(int argc, char** argv) {
    // Default test parameters
    int m = 1024, n = 1024, k = 1024;
    float alpha = 1.0f, beta = 0.0f;
    int num_iterations = 10;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) m = atoi(argv[++i]);
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = atoi(argv[++i]);
        if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) alpha = atof(argv[++i]);
        if (strcmp(argv[i], "--beta") == 0 && i + 1 < argc) beta = atof(argv[++i]);
        if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) num_iterations = atoi(argv[++i]);
    }
    
    printf("Advanced Tiled GEMM CUDA Kernel Test\n");
    printf("====================================\n");
    printf("Matrix dimensions: A(%dx%d), B(%dx%d), C(%dx%d)\n", m, k, k, n, m, n);
    printf("Parameters: alpha=%.2f, beta=%.2f\n", alpha, beta);
    printf("Tile sizes: M=%d, N=%d, K=%d\n", TILE_M, TILE_N, TILE_K);
    printf("Register tile sizes: M=%d, N=%d\n", REG_TILE_M, REG_TILE_N);
    printf("Number of iterations: %d\n\n", num_iterations);
    
    // Allocate host memory
    size_t bytesA = m * k * sizeof(float);
    size_t bytesB = k * n * sizeof(float);
    size_t bytesC = m * n * sizeof(float);
    size_t bytesD = m * n * sizeof(float);
    
    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);
    float* h_D_gpu = (float*)malloc(bytesD);
    float* h_D_cpu = (float*)malloc(bytesD);
    
    // Initialize matrices with random data
    srand(42);
    for (int i = 0; i < m * k; ++i) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < k * n; ++i) h_B[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < m * n; ++i) h_C[i] = (float)rand() / RAND_MAX - 0.5f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));
    CHECK_CUDA(cudaMalloc(&d_D, bytesD));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));
    
    // Run CPU reference
    printf("Running CPU reference...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    gemm_cpu_reference(h_A, h_B, h_C, h_D_cpu, m, n, k, alpha, beta);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    printf("CPU time: %ld ms\n\n", cpu_duration.count());
    
    // Warm up GPU
    printf("Warming up GPU...\n");
    launch_advanced_tiled_gemm(d_A, d_B, d_C, d_D, m, n, k, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Performance measurement
    printf("Running GPU kernel %d times...\n", num_iterations);
    PerformanceTimer timer;
    timer.start();
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        launch_advanced_tiled_gemm(d_A, d_B, d_C, d_D, m, n, k, alpha, beta);
    }
    
    float gpu_time = timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_D_gpu, d_D, bytesD, cudaMemcpyDeviceToHost));
    
    // Validate results
    printf("\nValidation:\n");
    bool passed = validate_results(h_D_gpu, h_D_cpu, m * n);
    
    // Performance analysis
    printf("\nPerformance Analysis:\n");
    printf("  GPU time (avg over %d runs): %.3f ms\n", num_iterations, gpu_time / num_iterations);
    printf("  Theoretical peak GFLOPS: %.2f\n", calculate_theoretical_peak(m, n, k, gpu_time / num_iterations));
    printf("  Memory bandwidth: %.2f GB/s\n", calculate_memory_bandwidth(m, n, k, gpu_time / num_iterations));
    
    // Calculate speedup
    double speedup = (double)cpu_duration.count() / (gpu_time / num_iterations);
    printf("  Speedup over CPU: %.2fx\n", speedup);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_D_gpu); free(h_D_cpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    
    printf("\nTest %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
