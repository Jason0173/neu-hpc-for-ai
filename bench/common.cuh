// Shared helpers for the benchmark programs: error checking, random data and
// CUDA-event timing. Progress goes to stderr so stdout stays clean CSV.
#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define BENCH_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err_));                                   \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// Uniform random values in [lo, hi), fixed seed so every run sees the same data.
inline std::vector<float> bench_random(size_t n, unsigned seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

inline float* bench_to_device(const std::vector<float>& h) {
    float* d = nullptr;
    BENCH_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    BENCH_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

inline std::vector<float> bench_to_host(const float* d, size_t n) {
    std::vector<float> h(n);
    BENCH_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

inline double bench_max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

inline double bench_max_abs(const std::vector<float>& a) {
    double m = 0.0;
    for (float x : a) m = std::max(m, (double)std::fabs(x));
    return m;
}

// Error as "1.23e-06", or "" when the correctness check was skipped.
inline std::string bench_err_str(bool checked, double err) {
    if (!checked) return "";
    char buf[32];
    snprintf(buf, sizeof buf, "%.2e", err);
    return buf;
}

// Median time in ms of `launch`. `before_each` runs outside the timed region
// (for example to zero gradient buffers). Runs 2 warm-up calls, then enough
// timed calls to fill about `budget_ms` (between 3 and 50).
inline double bench_time_ms(const std::function<void()>& launch,
                            const std::function<void()>& before_each = nullptr,
                            double budget_ms = 300.0) {
    cudaEvent_t start, stop;
    BENCH_CHECK(cudaEventCreate(&start));
    BENCH_CHECK(cudaEventCreate(&stop));

    auto timed_once = [&]() {
        if (before_each) before_each();
        BENCH_CHECK(cudaEventRecord(start));
        launch();
        BENCH_CHECK(cudaEventRecord(stop));
        BENCH_CHECK(cudaEventSynchronize(stop));
        BENCH_CHECK(cudaGetLastError());
        float ms = 0.f;
        BENCH_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return (double)ms;
    };

    double first = timed_once();          // warm-up 1, also sizes the run
    if (first < budget_ms) timed_once();  // warm-up 2
    int iters = (int)std::min(50.0, std::max(3.0, budget_ms / std::max(first, 1e-3)));
    if (first > 2000.0) iters = 1;        // very slow kernels: one timed run is enough

    std::vector<double> times;
    for (int i = 0; i < iters; ++i) times.push_back(timed_once());
    std::sort(times.begin(), times.end());

    BENCH_CHECK(cudaEventDestroy(start));
    BENCH_CHECK(cudaEventDestroy(stop));
    return times[times.size() / 2];
}
