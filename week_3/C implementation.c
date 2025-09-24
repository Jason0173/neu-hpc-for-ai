// online_softmax.c
#include <math.h>
#include <float.h>
#include <stddef.h>

#ifndef SOFTMAX_INLINE
#define SOFTMAX_INLINE static inline
#endif

// Online normalizer softmax (Algorithm 3, fp32)
// y[i] = exp(x[i] - mV) / dV, where mV and dV are computed online in one pass.
SOFTMAX_INLINE void softmax_online_f32(const float *x, float *y, size_t V) {
    if (V == 0) return;

    // Lines 1–2 in Alg. 3: initialize running max m and normalizer d.
    float m = -INFINITY;   // running maximum mj
    float d = 0.0f;        // running normalizer dj

    // Lines 3–6: single pass to update (m, d)
    for (size_t j = 0; j < V; ++j) {
        const float xj = x[j];
        const float mj = fmaxf(m, xj);                  // mj = max(m_{j-1}, x_j)
        // dj = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
        d = d * expf(m - mj) + expf(xj - mj);
        m = mj;
    }

    // Lines 7–9: compute outputs using final mV, dV
    for (size_t i = 0; i < V; ++i) {
        y[i] = expf(x[i] - m) / d;
    }
}

// (Optional) double-precision version for very long vectors
SOFTMAX_INLINE void softmax_online_f64(const double *x, double *y, size_t V) {
    if (V == 0) return;
    double m = -INFINITY, d = 0.0;
    for (size_t j = 0; j < V; ++j) {
        const double xj = x[j];
        const double mj = fmax(m, xj);
        d = d * exp(m - mj) + exp(xj - mj);
        m = mj;
    }
    for (size_t i = 0; i < V; ++i) y[i] = exp(x[i] - m) / d;
}
