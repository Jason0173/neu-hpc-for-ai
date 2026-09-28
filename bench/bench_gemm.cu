// Benchmarks the course GEMM kernels against cuBLAS SGEMM (FP32, D = A * B).
//   week2_tiled     week_2/main.cu                   16x16 shared-memory tiles
//   week3_regblock  week_3/advanced_tiled_gemm.cu    64x64 tiles, 4x4 register blocking
//   cublas          cublasSgemm                      vendor baseline
// The week files are included as-is (their main() is renamed), so this measures
// exactly the committed kernels. Output: CSV on stdout.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_89 bench_gemm.cu -lcublas -o bench_gemm

#include <cublas_v2.h>

#define main week2_main
#include "../week_2/main.cu"
#undef main

#define main week3_main
#include "../week_3/advanced_tiled_gemm.cu"
#undef main

#include "common.cuh"

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,   \
                    (int)st_);                                                \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

int main(int argc, char** argv) {
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192};
    if (argc > 1) sizes = {atoi(argv[1])};  // e.g. ./bench_gemm 1024 for a quick check

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.f, beta = 0.f;

    printf("kernel,m,n,k,ms,gflops,max_rel_err_vs_cublas\n");
    for (int s : sizes) {
        const int m = s, n = s, k = s;
        fprintf(stderr, "[gemm] %d x %d x %d\n", m, n, k);

        float* A = bench_to_device(bench_random((size_t)m * k, 1));
        float* B = bench_to_device(bench_random((size_t)k * n, 2));
        float *D_ref, *D;
        BENCH_CHECK(cudaMalloc(&D_ref, (size_t)m * n * sizeof(float)));
        BENCH_CHECK(cudaMalloc(&D, (size_t)m * n * sizeof(float)));

        // Row-major D = A * B is column-major D^T = B^T * A^T, so swap the operands.
        auto run_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                     &alpha, B, n, A, k, &beta, D_ref, n));
        };
        double ms_ref = bench_time_ms(run_cublas);
        std::vector<float> h_ref = bench_to_host(D_ref, (size_t)m * n);
        const double ref_scale = bench_max_abs(h_ref);
        const double gflop = 2.0 * m * n * k / 1e9;
        printf("cublas,%d,%d,%d,%.4f,%.1f,0\n", m, n, k, ms_ref, gflop / (ms_ref / 1e3));

        struct Kernel { const char* name; std::function<void()> launch; };
        std::vector<Kernel> kernels = {
            {"week2_tiled",    [&]() { gemm(A, B, nullptr, D, m, n, k, alpha, beta); }},
            {"week3_regblock", [&]() { launch_advanced_tiled_gemm(A, B, nullptr, D, m, n, k, alpha, beta); }},
        };
        for (auto& kern : kernels) {
            BENCH_CHECK(cudaMemset(D, 0, (size_t)m * n * sizeof(float)));
            double ms = bench_time_ms(kern.launch);
            double err = bench_max_abs_diff(bench_to_host(D, (size_t)m * n), h_ref) / ref_scale;
            printf("%s,%d,%d,%d,%.4f,%.1f,%.2e\n", kern.name, m, n, k, ms, gflop / (ms / 1e3), err);
        }
        fflush(stdout);

        cudaFree(A); cudaFree(B); cudaFree(D_ref); cudaFree(D);
    }
    cublasDestroy(handle);
    return 0;
}
