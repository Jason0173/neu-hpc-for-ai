// Benchmarks the course attention kernels (FP32, one head, head dim 64, no mask).
//   week4_flash      week_4/flash_attn.cu            FlashAttention-1 style forward
//   week5_fa2_fwd    week_5/flash_attention2.cu      FlashAttention-2 forward
//   week5_fa2_bwd    week_5/flash_attention2.cu      FlashAttention-2 backward
// Correctness is checked against the CPU references in the same files for
// N <= 1024 (they are O(N^2 d) on one CPU core). Note the week-5 kernels compute
// softmax(Q K^T) V without the 1/sqrt(d) scale; timing is unaffected.
// Output: CSV on stdout. PyTorch baselines come from bench_sdpa.py.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_89 bench_attention.cu -o bench_attention

#define main week4_main
#include "../week_4/flash_attn.cu"
#undef main

#define main week5_main
#include "../week_5/flash_attention2.cu"
#undef main

#include "common.cuh"

#include <map>
#include <string>

int main(int argc, char** argv) {
    std::vector<int> seq_lens = {512, 1024, 2048, 4096, 8192, 16384};
    if (argc > 1) seq_lens = {atoi(argv[1])};
    const int d = 64;             // head dim; week 4 kernel supports d <= 64 (48 KB shared memory)
    const int Br = 16, Bc = 16;   // week 5 tile sizes, as committed
    const int check_max_n = 1024; // run CPU references up to this length
    const double skip_after_ms = 20000.0;  // skip a kernel once one run is predicted to take > 20 s

    printf("kernel,pass,N,d,ms,tflops,max_abs_err,err_detail\n");

    // Attention cost grows with N^2: predict the next size from the last one and
    // skip a kernel once a single run would take longer than skip_after_ms.
    std::map<std::string, std::pair<int, double>> last;  // kernel -> (N, ms)
    auto predicted_ms = [&](const std::string& key, int N) {
        auto it = last.find(key);
        if (it == last.end()) return 0.0;
        double r = (double)N / it->second.first;
        return it->second.second * r * r;
    };

    for (int N : seq_lens) {
        fprintf(stderr, "[attention] N=%d d=%d\n", N, d);
        const size_t nd = (size_t)N * d;
        std::vector<float> hQ = bench_random(nd, 11, -0.5f, 0.5f);
        std::vector<float> hK = bench_random(nd, 12, -0.5f, 0.5f);
        std::vector<float> hV = bench_random(nd, 13, -0.5f, 0.5f);
        std::vector<float> hdO = bench_random(nd, 14, -0.5f, 0.5f);
        float *Q = bench_to_device(hQ), *K = bench_to_device(hK), *V = bench_to_device(hV), *dO = bench_to_device(hdO);
        float *O, *L, *dQ, *dK, *dV;
        BENCH_CHECK(cudaMalloc(&O, nd * sizeof(float)));
        BENCH_CHECK(cudaMalloc(&L, (size_t)N * sizeof(float)));
        BENCH_CHECK(cudaMalloc(&dQ, nd * sizeof(float)));
        BENCH_CHECK(cudaMalloc(&dK, nd * sizeof(float)));
        BENCH_CHECK(cudaMalloc(&dV, nd * sizeof(float)));

        const bool check = N <= check_max_n;
        const double fwd_tflop = 4.0 * N * (double)N * d / 1e12;
        const double bwd_tflop = 2.5 * fwd_tflop;  // usual FlashAttention accounting

        // ---- week 4: FlashAttention-1 style forward ----
        if (double p = predicted_ms("week4", N); p > skip_after_ms) {
            printf("week4_flash,fwd,%d,%d,,,,skipped (predicted %.0f s per run)\n", N, d, p / 1e3);
        } else {
            double ms = bench_time_ms([&]() { flash_attention_cuda(Q, K, V, O, N, d, false); });
            last["week4"] = {N, ms};
            double err = -1; char detail[128] = "";
            if (check) {
                std::vector<float> ref(nd);
                attention_naive_cpu(hQ.data(), hK.data(), hV.data(), ref.data(), N, d, false);
                err = bench_max_abs_diff(bench_to_host(O, nd), ref);
                snprintf(detail, sizeof detail, "O=%.1e", err);
            }
            printf("week4_flash,fwd,%d,%d,%.4f,%.3f,%s,%s\n", N, d, ms, fwd_tflop / (ms / 1e3),
                   bench_err_str(check, err).c_str(), detail);
        }
        fflush(stdout);

        // ---- week 5: FlashAttention-2 forward ----
        std::vector<float> refO, refL;
        if (double p = predicted_ms("fa2_fwd", N); p > skip_after_ms) {
            printf("week5_fa2,fwd,%d,%d,,,,skipped (predicted %.0f s per run)\n", N, d, p / 1e3);
        } else {
            double ms_f = bench_time_ms([&]() { run_forward_gpu(Q, K, V, O, L, 1, 1, N, d, Br, Bc); });
            last["fa2_fwd"] = {N, ms_f};
            double err = -1; char detail[128] = "";
            if (check) {
                refO.resize(nd); refL.resize(N);
                forward_cpu_ref(hQ.data(), hK.data(), hV.data(), refO.data(), refL.data(), 1, 1, N, d);
                double eO = bench_max_abs_diff(bench_to_host(O, nd), refO);
                double eL = bench_max_abs_diff(bench_to_host(L, N), refL);
                err = std::max(eO, eL);
                snprintf(detail, sizeof detail, "O=%.1e;L=%.1e", eO, eL);
            }
            printf("week5_fa2,fwd,%d,%d,%.4f,%.3f,%s,%s\n", N, d, ms_f, fwd_tflop / (ms_f / 1e3),
                   bench_err_str(check, err).c_str(), detail);
        }
        fflush(stdout);

        // ---- week 5: FlashAttention-2 backward (uses O and L from a fresh forward) ----
        auto zero_grads = [&]() {
            BENCH_CHECK(cudaMemset(dQ, 0, nd * sizeof(float)));
            BENCH_CHECK(cudaMemset(dK, 0, nd * sizeof(float)));
            BENCH_CHECK(cudaMemset(dV, 0, nd * sizeof(float)));
        };
        if (double p = predicted_ms("fa2_bwd", N); p > skip_after_ms) {
            printf("week5_fa2,bwd,%d,%d,,,,skipped (predicted %.0f s per run)\n", N, d, p / 1e3);
        } else {
            run_forward_gpu(Q, K, V, O, L, 1, 1, N, d, Br, Bc);
            BENCH_CHECK(cudaDeviceSynchronize());
            double ms_b = bench_time_ms([&]() { run_backward_gpu(Q, K, V, O, L, dO, dQ, dK, dV, 1, 1, N, d, Br, Bc); },
                                        zero_grads);
            last["fa2_bwd"] = {N, ms_b};
            double err = -1; char detail[160] = "";
            if (check) {
                zero_grads();
                run_backward_gpu(Q, K, V, O, L, dO, dQ, dK, dV, 1, 1, N, d, Br, Bc);
                BENCH_CHECK(cudaDeviceSynchronize());
                if (refO.empty()) {  // forward was skipped above; the CPU backward needs O
                    refO.resize(nd); refL.resize(N);
                    forward_cpu_ref(hQ.data(), hK.data(), hV.data(), refO.data(), refL.data(), 1, 1, N, d);
                }
                std::vector<float> rdQ(nd), rdK(nd), rdV(nd);
                backward_cpu_ref(hQ.data(), hK.data(), hV.data(), refO.data(), hdO.data(),
                                 rdQ.data(), rdK.data(), rdV.data(), 1, 1, N, d);
                double eQ = bench_max_abs_diff(bench_to_host(dQ, nd), rdQ);
                double eK = bench_max_abs_diff(bench_to_host(dK, nd), rdK);
                double eV = bench_max_abs_diff(bench_to_host(dV, nd), rdV);
                err = std::max(eQ, std::max(eK, eV));
                snprintf(detail, sizeof detail, "dQ=%.1e;dK=%.1e;dV=%.1e", eQ, eK, eV);
            }
            printf("week5_fa2,bwd,%d,%d,%.4f,%.3f,%s,%s\n", N, d, ms_b, bwd_tflop / (ms_b / 1e3),
                   bench_err_str(check, err).c_str(), detail);
        }
        fflush(stdout);

        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(dO);
        cudaFree(O); cudaFree(L); cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    }
    return 0;
}
