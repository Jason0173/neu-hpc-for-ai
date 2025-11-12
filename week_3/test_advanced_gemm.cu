// test_advanced_gemm.cu - Comprehensive test suite for advanced tiled GEMM
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#ifndef TILE
#define TILE 16
#endif

// Forward declarations for functions from advanced_tiled_gemm.cu
void launch_advanced_tiled_gemm(const float* d_A, const float* d_B, const float* d_C, float* d_D,
                                int m, int n, int k, float alpha, float beta, cudaStream_t stream = 0);

// CPU reference implementation for validation
void gemm_cpu_reference(const float* A, int mA, int kA, char opA,
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

// Test configuration structure
struct TestConfig {
    int m, n, k;
    char opA, opB;
    float alpha, beta;
    const char* name;
};

// Test suite with various configurations
std::vector<TestConfig> get_test_configs() {
    return {
        {256, 256, 256, 'N', 'N', 1.0f, 0.0f, "Small Square"},
        {512, 512, 512, 'N', 'N', 1.0f, 0.0f, "Medium Square"},
        {1024, 1024, 1024, 'N', 'N', 1.0f, 0.0f, "Large Square"},
        {2048, 2048, 2048, 'N', 'N', 1.0f, 0.0f, "Very Large Square"},
        {1024, 512, 256, 'N', 'N', 1.0f, 0.0f, "Rectangular 1"},
        {512, 1024, 256, 'N', 'N', 1.0f, 0.0f, "Rectangular 2"},
        {1024, 1024, 1024, 'T', 'N', 1.0f, 0.0f, "Transpose A"},
        {1024, 1024, 1024, 'N', 'T', 1.0f, 0.0f, "Transpose B"},
        {1024, 1024, 1024, 'T', 'T', 1.0f, 0.0f, "Transpose Both"},
        {1024, 1024, 1024, 'N', 'N', 2.5f, 0.5f, "Alpha Beta Test"},
        {1024, 1024, 1024, 'N', 'N', 1.0f, 1.0f, "Beta = 1.0"},
        {1024, 1024, 1024, 'N', 'N', 0.0f, 1.0f, "Alpha = 0.0"},
    };
}

// Validation function
bool validate_results(const float* gpu_result, const float* cpu_result, 
                     int size, double tolerance = 1e-4) {
    double max_error = 0.0;
    double sum_gpu = 0.0, sum_cpu = 0.0;
    
    for (int i = 0; i < size; ++i) {
        double error = fabs(gpu_result[i] - cpu_result[i]);
        max_error = std::max(max_error, error);
        sum_gpu += gpu_result[i];
        sum_cpu += cpu_result[i];
    }
    
    printf("  Max absolute error: %.2e\n", max_error);
    printf("  Sum GPU: %.6f, Sum CPU: %.6f\n", sum_gpu, sum_cpu);
    
    return max_error < tolerance;
}

// Performance measurement
struct PerformanceResult {
    float time_ms;
    double gflops;
    double bandwidth_util;
    bool passed;
};

PerformanceResult run_test(const TestConfig& config) {
    printf("\n=== Testing %s ===\n", config.name);
    printf("Matrix sizes: %dx%d, %dx%d -> %dx%d\n", 
           (config.opA == 'N') ? config.m : config.k,
           (config.opA == 'N') ? config.k : config.m,
           (config.opB == 'N') ? config.k : config.n,
           (config.opB == 'N') ? config.n : config.k,
           config.m, config.n);
    
    // Determine matrix dimensions
    int mA = (config.opA == 'N') ? config.m : config.k;
    int kA = (config.opA == 'N') ? config.k : config.m;
    int kB = (config.opB == 'N') ? config.k : config.n;
    int nB = (config.opB == 'N') ? config.n : config.k;
    int K = (config.opA == 'N') ? config.k : config.m;
    
    // Allocate host memory
    size_t bytesA = mA * kA * sizeof(float);
    size_t bytesB = kB * nB * sizeof(float);
    size_t bytesC = config.m * config.n * sizeof(float);
    
    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);
    float* h_C_ref = (float*)malloc(bytesC);
    
    // Initialize data
    for (int i = 0; i < mA * kA; ++i) {
        h_A[i] = (float)((i % 13) - 6) / 7.0f;
    }
    for (int i = 0; i < kB * nB; ++i) {
        h_B[i] = (float)((i % 17) - 8) / 9.0f;
    }
    for (int i = 0; i < config.m * config.n; ++i) {
        h_C[i] = (float)((i % 7) - 3) / 5.0f;
        h_C_ref[i] = h_C[i];
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice);
    
    // Run CPU reference
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gemm_cpu_reference(h_A, mA, kA, config.opA, h_B, kB, nB, config.opB, 
                      h_C_ref, config.m, config.n, K, config.alpha, config.beta);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
    
    // Warm up GPU
    for (int i = 0; i < 3; ++i) {
        launch_advanced_tiled_gemm(d_A, d_B, d_C, d_C, config.m, config.n, config.k, 
                                  config.alpha, config.beta);
    }
    cudaDeviceSynchronize();
    
    // Benchmark GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        launch_advanced_tiled_gemm(d_A, d_B, d_C, d_C, config.m, config.n, config.k, 
                                  config.alpha, config.beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time /= 10.0f;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    
    // Validate results
    bool passed = validate_results(h_C, h_C_ref, config.m * config.n);
    
    // Calculate performance metrics
    double flops = 2.0 * config.m * config.n * config.k;
    double gflops = flops / (gpu_time * 1e6);
    
    double bytes = (mA * kA + kB * nB + config.m * config.n) * sizeof(float);
    double bandwidth_gbps = bytes / (gpu_time * 1e6);
    double bandwidth_util = (bandwidth_gbps / 900.0) * 100.0;  // Assuming 900 GB/s peak bandwidth
    
    printf("  CPU time: %.3f ms\n", cpu_time);
    printf("  GPU time: %.3f ms (%.1fx speedup)\n", gpu_time, cpu_time / gpu_time);
    printf("  Performance: %.2f GFLOPS\n", gflops);
    printf("  Bandwidth utilization: %.1f%%\n", bandwidth_util);
    printf("  Result: %s\n", passed ? "PASS" : "FAIL");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return {gpu_time, gflops, bandwidth_util, passed};
}

// Memory access pattern analysis
void analyze_memory_access(int m, int n, int k) {
    printf("\n=== Memory Access Analysis ===\n");
    
    // Calculate memory access patterns
    size_t A_bytes = m * k * sizeof(float);
    size_t B_bytes = k * n * sizeof(float);
    size_t C_bytes = m * n * sizeof(float);
    size_t total_bytes = A_bytes + B_bytes + C_bytes;
    
    printf("Matrix A: %zu bytes (%.2f MB)\n", A_bytes, A_bytes / (1024.0 * 1024.0));
    printf("Matrix B: %zu bytes (%.2f MB)\n", B_bytes, B_bytes / (1024.0 * 1024.0));
    printf("Matrix C: %zu bytes (%.2f MB)\n", C_bytes, C_bytes / (1024.0 * 1024.0));
    printf("Total memory: %zu bytes (%.2f MB)\n", total_bytes, total_bytes / (1024.0 * 1024.0));
    
    // Calculate arithmetic intensity
    double operations = 2.0 * m * n * k;  // 2 FLOPs per multiply-add
    double arithmetic_intensity = operations / total_bytes;
    printf("Arithmetic intensity: %.2f FLOPs/byte\n", arithmetic_intensity);
    
    // Memory access efficiency
    double A_reuse = (double)(m * n) / (m * k);  // How many times each A element is reused
    double B_reuse = (double)(m * n) / (k * n);  // How many times each B element is reused
    printf("Matrix A reuse factor: %.2f\n", A_reuse);
    printf("Matrix B reuse factor: %.2f\n", B_reuse);
}

int main(int argc, char** argv) {
    printf("=== Advanced Tiled GEMM Test Suite ===\n");
    printf("Tile size: %dx%d\n", TILE, TILE);
    printf("Register tile size: 4x4\n");
    
    // Get test configurations
    auto configs = get_test_configs();
    
    // Run tests
    std::vector<PerformanceResult> results;
    int passed_tests = 0;
    
    for (const auto& config : configs) {
        results.push_back(run_test(config));
        if (results.back().passed) {
            passed_tests++;
        }
    }
    
    // Summary
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d/%zu tests\n", passed_tests, configs.size());
    
    // Performance summary
    printf("\n=== Performance Summary ===\n");
    double avg_gflops = 0.0;
    double avg_bandwidth = 0.0;
    
    for (const auto& result : results) {
        avg_gflops += result.gflops;
        avg_bandwidth += result.bandwidth_util;
    }
    
    avg_gflops /= results.size();
    avg_bandwidth /= results.size();
    
    printf("Average performance: %.2f GFLOPS\n", avg_gflops);
    printf("Average bandwidth utilization: %.1f%%\n", avg_bandwidth);
    
    // Memory analysis for largest test
    if (!configs.empty()) {
        auto largest = std::max_element(configs.begin(), configs.end(),
            [](const TestConfig& a, const TestConfig& b) {
                return (a.m * a.n * a.k) < (b.m * b.n * b.k);
            });
        analyze_memory_access(largest->m, largest->n, largest->k);
    }
    
    printf("\n=== Optimization Features ===\n");
    printf("✓ Double buffering for memory-compute overlap\n");
    printf("✓ Register-level tiling (4x4) for data reuse\n");
    printf("✓ Software prefetching for cache optimization\n");
    printf("✓ Bank conflict avoidance with padding\n");
    printf("✓ Coalesced memory access patterns\n");
    printf("✓ Comprehensive test suite with validation\n");
    
    return (passed_tests == configs.size()) ? 0 : 1;
}
