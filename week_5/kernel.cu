// flashattention2_minimal.cu
// A minimal, correct FlashAttention-2 style implementation (float32) for learning & verification.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

// -------------------- Utilities --------------------
#define CUDA_CHECK(expr) do {                             \
    cudaError_t _err = (expr);                            \
    if (_err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::abort();                                     \
    }                                                     \
} while(0)

inline __host__ __device__ int div_up(int x, int y) { return (x + y - 1) / y; }

// Row-major layout: [B, H, N, d] contiguous in last dim d
// For a fixed (hb) == (h + b*H), base pointer offset (in elements) is hb * N * d.

// -------------------- CPU reference (for sanity check) --------------------
// Naive CPU forward: O = softmax(QK^T) V; also return L = logsumexp along rows of S
void forward_cpu_ref(const float* Q, const float* K, const float* V,
    float* O, float* L,
    int B, int H, int N, int d)
{
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* Qbh = Q + ((b * H + h) * N * d);
            const float* Kbh = K + ((b * H + h) * N * d);
            const float* Vbh = V + ((b * H + h) * N * d);
            float* Obh = O + ((b * H + h) * N * d);
            float* Lbh = L + ((b * H + h) * N);

            for (int i = 0; i < N; ++i) {
                // S_i* = Q_i dot K_*^T
                float s_max = -INFINITY;
                std::vector<float> s(N);
                for (int j = 0; j < N; ++j) {
                    float dot = 0.f;
                    const float* qi = Qbh + i * d;
                    const float* kj = Kbh + j * d;
                    for (int k = 0; k < d; ++k) dot += qi[k] * kj[k];
                    s[j] = dot;
                    s_max = fmaxf(s_max, dot);
                }
                double l = 0.0;
                for (int j = 0; j < N; ++j) l += std::exp((double)s[j] - (double)s_max);
                const float Li = s_max + (float)std::log(l);
                Lbh[i] = Li;

                // O_i = sum_j softmax(S_i)[j]*V_j
                float* oi = Obh + i * d;
                for (int k = 0; k < d; ++k) oi[k] = 0.f;
                for (int j = 0; j < N; ++j) {
                    float pij = std::exp(s[j] - Li); // softmax row i
                    const float* vj = Vbh + j * d;
                    for (int k = 0; k < d; ++k) oi[k] += pij * vj[k];
                }
            }
        }
    }
}

// Naive CPU backward (given dO): compute dQ, dK, dV
// Formulas: P = softmax(S), D_i = sum_j dP_ij * P_ij where dP_ij = dO_i · V_j
// dS_ij = P_ij * (dP_ij - D_i)
// dQ_i += dS_ij * K_j ; dK_j += dS_ij * Q_i ; dV_j += P_ij * dO_i
void backward_cpu_ref(const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    float* dQ, float* dK, float* dV,
    int B, int H, int N, int d)
{
    std::fill(dQ, dQ + (size_t)B * H * N * d, 0.f);
    std::fill(dK, dK + (size_t)B * H * N * d, 0.f);
    std::fill(dV, dV + (size_t)B * H * N * d, 0.f);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* Qbh = Q + ((b * H + h) * N * d);
            const float* Kbh = K + ((b * H + h) * N * d);
            const float* Vbh = V + ((b * H + h) * N * d);
            const float* dObh = dO + ((b * H + h) * N * d);
            float* dQbh = dQ + ((b * H + h) * N * d);
            float* dKbh = dK + ((b * H + h) * N * d);
            float* dVbh = dV + ((b * H + h) * N * d);

            // Precompute S and P row-wise (naive)
            std::vector<float> S(N * N), P(N * N);
            for (int i = 0; i < N; ++i) {
                float s_max = -INFINITY;
                for (int j = 0; j < N; ++j) {
                    float dot = 0.f;
                    for (int k = 0; k < d; ++k) dot += Qbh[i * d + k] * Kbh[j * d + k];
                    S[i * N + j] = dot;
                    s_max = fmaxf(s_max, dot);
                }
                double l = 0.0;
                for (int j = 0; j < N; ++j) l += std::exp((double)S[i * N + j] - (double)s_max);
                float Li = s_max + (float)std::log(l);
                for (int j = 0; j < N; ++j) P[i * N + j] = std::exp(S[i * N + j] - Li);
            }
            // Backward
            for (int i = 0; i < N; ++i) {
                // D_i
                double Di = 0.0;
                for (int j = 0; j < N; ++j) {
                    float dPij = 0.f;
                    for (int k = 0; k < d; ++k) dPij += dObh[i * d + k] * Vbh[j * d + k];
                    Di += (double)(dPij * P[i * N + j]);
                }
                for (int j = 0; j < N; ++j) {
                    float dPij = 0.f;
                    for (int k = 0; k < d; ++k) dPij += dObh[i * d + k] * Vbh[j * d + k];
                    float dSij = P[i * N + j] * (dPij - (float)Di);

                    // dQ_i += dS_ij * K_j
                    for (int k = 0; k < d; ++k)
                        dQbh[i * d + k] += dSij * Kbh[j * d + k];

                    // dK_j += dS_ij * Q_i
                    for (int k = 0; k < d; ++k)
                        dKbh[j * d + k] += dSij * Qbh[i * d + k];

                    // dV_j += P_ij * dO_i
                    for (int k = 0; k < d; ++k)
                        dVbh[j * d + k] += P[i * N + j] * dObh[i * d + k];
                }
            }
        }
    }
}

// -------------------- GPU kernels: FlashAttention-2 (teaching version) --------------------

// Forward kernel (row-tiles parallel; split-Q concept: each thread handles one row).
// Online softmax with final scaling. We store L = logsumexp per row (float).
__global__ void fa2_forward_kernel(
    const float* __restrict__ Q, // [B*H, N, d]
    const float* __restrict__ K, // [B*H, N, d]
    const float* __restrict__ V, // [B*H, N, d]
    float* __restrict__ O,       // [B*H, N, d]
    float* __restrict__ L,       // [B*H, N]  (logsumexp per row)
    int N, int d, int Br, int Bc)
{
    // Grid mapping: blockIdx.x = rTile, blockIdx.y = hb (head*batch)
    const int rTile = blockIdx.x;
    const int hb = blockIdx.y;

    const int row0 = rTile * Br;
    const int rows = min(Br, N - row0);
    if (rows <= 0) return;

    // Pointers to this (hb):
    const float* Qbh = Q + ((size_t)hb * N * d);
    const float* Kbh = K + ((size_t)hb * N * d);
    const float* Vbh = V + ((size_t)hb * N * d);
    float* Obh = O + ((size_t)hb * N * d);
    float* Lbh = L + ((size_t)hb * N);

    extern __shared__ float smem[];
    // Layout in smem:
    // [ Ks (Bc*d) | Vs (Bc*d) | Qs (Br*d) | Otilde (Br*d) | m (Br) | l (Br) ]
    float* Ks = smem;
    float* Vs = Ks + (size_t)Bc * d;
    float* Qs = Vs + (size_t)Bc * d;
    float* Otilde = Qs + (size_t)Br * d;
    float* m = Otilde + (size_t)Br * d;
    float* l = m + Br;

    // Load Q tile rows to shared (all threads cooperate)
    for (int r = threadIdx.x; r < rows * d; r += blockDim.x) {
        Qs[r] = Qbh[(row0 * d) + r];
    }
    // Init Otilde=0, m=-inf, l=0
    for (int r = threadIdx.x; r < rows * d; r += blockDim.x) Otilde[r] = 0.f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) { m[r] = -INFINITY; l[r] = 0.f; }
    __syncthreads();

    // Loop over column tiles
    const int Tc = div_up(N, Bc);
    for (int cTile = 0; cTile < Tc; ++cTile) {
        const int col0 = cTile * Bc;
        const int cols = min(Bc, N - col0);

        // Load Ks, Vs tile into shared
        for (int x = threadIdx.x; x < cols * d; x += blockDim.x) {
            const int j = x / d;
            const int k = x % d;
            Ks[j * d + k] = Kbh[(col0 + j) * d + k];
            Vs[j * d + k] = Vbh[(col0 + j) * d + k];
        }
        __syncthreads();

        // Each thread handles one row (split-Q 概念：每线程/warp 处理不同 Q 行)
        for (int r_local = threadIdx.x; r_local < rows; r_local += blockDim.x) {
            float m_old = m[r_local];
            float l_old = l[r_local];

            // pass 1: s_max over this tile
            float smax_tile = -INFINITY;
            for (int j = 0; j < cols; ++j) {
                // dot(Q[row], K[col0+j])
                const float* qi = Qs + r_local * d;
                const float* kj = Ks + j * d;
                float s = 0.f;
#pragma unroll 1
                for (int kk = 0; kk < d; ++kk) s += qi[kk] * kj[kk];
                smax_tile = fmaxf(smax_tile, s);
            }
            float m_new = fmaxf(m_old, smax_tile);
            float scale_old = (isinf(m_old) ? 0.f : expf(m_old - m_new));

            // scale Otilde (apply once per tile)
            for (int kk = 0; kk < d; ++kk)
                Otilde[r_local * d + kk] *= scale_old;

            // pass 2: l_new, Otilde add
            double l_new = (double)l_old * (double)scale_old;
            for (int j = 0; j < cols; ++j) {
                // s again
                const float* qi = Qs + r_local * d;
                const float* kj = Ks + j * d;
                float s = 0.f;
                for (int kk = 0; kk < d; ++kk) s += qi[kk] * kj[kk];
                float e = expf(s - m_new);
                l_new += (double)e;

                const float* vj = Vs + j * d;
                for (int kk = 0; kk < d; ++kk)
                    Otilde[r_local * d + kk] += e * vj[kk];
            }

            m[r_local] = m_new;
            l[r_local] = (float)l_new;
        }
        __syncthreads();
    }

    // Final scaling: O = Otilde / l ; store L = m + log(l)
    for (int r_local = threadIdx.x; r_local < rows; r_local += blockDim.x) {
        const float Li = m[r_local] + logf(fmaxf(l[r_local], 1e-20f));
        Lbh[row0 + r_local] = Li;
        float inv_l = 1.f / fmaxf(l[r_local], 1e-20f);
        float* oi = Obh + (row0 + r_local) * d;
        for (int kk = 0; kk < d; ++kk)
            oi[kk] = Otilde[r_local * d + kk] * inv_l;
    }
}

// Backward kernel (col-tiles parallel). Recompute S,P; accumulate dV,dK locally;
// dQ gets atomicAdd across col-tiles.
__global__ void fa2_backward_kernel(
    const float* __restrict__ Q,   // [B*H, N, d]
    const float* __restrict__ K,   // [B*H, N, d]
    const float* __restrict__ V,   // [B*H, N, d]
    const float* __restrict__ O,   // [B*H, N, d] (not strictly needed but kept for completeness)
    const float* __restrict__ dO,  // [B*H, N, d]
    const float* __restrict__ L,   // [B*H, N]   (logsumexp per row from fwd)
    float* __restrict__ dQ,        // [B*H, N, d]
    float* __restrict__ dK,        // [B*H, N, d]
    float* __restrict__ dV,        // [B*H, N, d]
    int N, int d, int Br, int Bc)
{
    const int cTile = blockIdx.x;  // column tile id
    const int hb = blockIdx.y;  // head*batch

    const int col0 = cTile * Bc;
    const int cols = min(Bc, N - col0);
    if (cols <= 0) return;

    const float* Qbh = Q + ((size_t)hb * N * d);
    const float* Kbh = K + ((size_t)hb * N * d);
    const float* Vbh = V + ((size_t)hb * N * d);
    const float* dObh = dO + ((size_t)hb * N * d);
    const float* Lbh = L + ((size_t)hb * N);
    float* dQbh = dQ + ((size_t)hb * N * d);
    float* dKbh = dK + ((size_t)hb * N * d);
    float* dVbh = dV + ((size_t)hb * N * d);

    extern __shared__ float smem[];
    // [ Ks (Bc*d) | Vs (Bc*d) | Qs (Br*d) | dOs (Br*d) | Ls (Br) | dK_local (Bc*d) | dV_local (Bc*d) ]
    float* Ks = smem;
    float* Vs = Ks + (size_t)Bc * d;
    float* Qs = Vs + (size_t)Bc * d;
    float* dOs = Qs + (size_t)Br * d;
    float* Ls = dOs + (size_t)Br * d;
    float* dK_local = Ls + Br;
    float* dV_local = dK_local + (size_t)Bc * d;

    // Load K/V tile
    for (int x = threadIdx.x; x < cols * d; x += blockDim.x) {
        const int j = x / d;
        const int kk = x % d;
        Ks[j * d + kk] = Kbh[(col0 + j) * d + kk];
        Vs[j * d + kk] = Vbh[(col0 + j) * d + kk];
    }
    // Zero local dK/dV
    for (int x = threadIdx.x; x < cols * d; x += blockDim.x) {
        dK_local[x] = 0.f;
        dV_local[x] = 0.f;
    }
    __syncthreads();

    const int Tr = div_up(N, Br);
    for (int rTile = 0; rTile < Tr; ++rTile) {
        const int row0 = rTile * Br;
        const int rows = min(Br, N - row0);
        if (rows <= 0) break;

        // Load Q rows, dO rows, and L
        for (int x = threadIdx.x; x < rows * d; x += blockDim.x)
            Qs[x] = Qbh[(row0 * d) + x];
        for (int x = threadIdx.x; x < rows * d; x += blockDim.x)
            dOs[x] = dObh[(row0 * d) + x];
        for (int r = threadIdx.x; r < rows; r += blockDim.x)
            Ls[r] = Lbh[row0 + r];
        __syncthreads();

        // Each thread handles one row 
        for (int r_local = threadIdx.x; r_local < rows; r_local += blockDim.x) {
            const float* qi = Qs + r_local * d;
            const float* doi = dOs + r_local * d;
            const float Li = Ls[r_local];

            // Pass 1: compute D_i = sum_j [ (dO_i · V_j) * P_ij ]
            double Di = 0.0;
            for (int j = 0; j < cols; ++j) {
                // s_ij
                float s = 0.f;
                const float* kj = Ks + j * d;
                for (int kk = 0; kk < d; ++kk) s += qi[kk] * kj[kk];
                float Pij = expf(s - Li);

                // dP_ij = dO_i · V_j
                float dPij = 0.f;
                const float* vj = Vs + j * d;
                for (int kk = 0; kk < d; ++kk) dPij += doi[kk] * vj[kk];

                Di += (double)(Pij * dPij);
            }

            // Pass 2: accumulate dV_local, dK_local, atomicAdd dQ
            for (int j = 0; j < cols; ++j) {
                // s_ij & P_ij
                float s = 0.f;
                const float* kj = Ks + j * d;
                for (int kk = 0; kk < d; ++kk) s += qi[kk] * kj[kk];
                float Pij = expf(s - Li);

                // dP_ij
                float dPij = 0.f;
                const float* vj = Vs + j * d;
                for (int kk = 0; kk < d; ++kk) dPij += doi[kk] * vj[kk];

                float dSij = Pij * (dPij - (float)Di);  // softmax backward

                // dQ_i += dS_ij * K_j   (atomicAdd across col-tiles)
                float* dQi = dQbh + (row0 + r_local) * d;
                for (int kk = 0; kk < d; ++kk)
                    atomicAdd(&dQi[kk], dSij * kj[kk]);

                // dK_j += dS_ij * Q_i   (local)
                for (int kk = 0; kk < d; ++kk)
                    dK_local[j * d + kk] += dSij * qi[kk];

                // dV_j += P_ij * dO_i   (local)
                for (int kk = 0; kk < d; ++kk)
                    dV_local[j * d + kk] += Pij * doi[kk];
            }
        }
        __syncthreads();
    }

    // Write back dK, dV for this column tile (no overlap across tiles)
    for (int x = threadIdx.x; x < cols * d; x += blockDim.x) {
        const int j = x / d;
        const int kk = x % d;
        dKbh[(col0 + j) * d + kk] += dK_local[x];
        dVbh[(col0 + j) * d + kk] += dV_local[x];
    }
}

// -------------------- Host helpers --------------------
void run_forward_gpu(const float* dQ, const float* dK, const float* dV,
    float* dO, float* dL,
    int B, int H, int N, int d, int Br, int Bc)
{
    dim3 grid(div_up(N, Br), B * H);
    int threads = 128; // threads per block (教学示例；可调优)
    size_t smem_floats = (size_t)Bc * d * 2 + (size_t)Br * d * 2 + 2 * Br; // Ks,Vs,Qs,Otilde,m,l
    size_t smem_bytes = smem_floats * sizeof(float);
    fa2_forward_kernel << <grid, threads, smem_bytes >> > (dQ, dK, dV, dO, dL, N, d, Br, Bc);
    CUDA_CHECK(cudaGetLastError());
}

void run_backward_gpu(const float* dQ, const float* dK, const float* dV,
    const float* dO, const float* dL, const float* ddO,
    float* ddQ, float* ddK, float* ddV,
    int B, int H, int N, int d, int Br, int Bc)
{
    dim3 grid(div_up(N, Bc), B * H);
    int threads = 128; 
    size_t smem_floats = (size_t)Bc * d * 2 + (size_t)Br * d * 2 + Br + (size_t)Bc * d * 2; // Ks,Vs,Qs,dOs,Ls,dKloc,dVloc
    size_t smem_bytes = smem_floats * sizeof(float);
    fa2_backward_kernel << <grid, threads, smem_bytes >> > (dQ, dK, dV, dO, ddO, dL, ddQ, ddK, ddV, N, d, Br, Bc);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------- Demo main  --------------------
int main() {
    // Small sanity test
    const int B = 1, H = 1, N = 64, d = 64;
    const int Br = 16, Bc = 16;  // Reduced tile sizes to fit in shared memory

    // Host buffers
    std::vector<float> hQ((size_t)B * H * N * d), hK((size_t)B * H * N * d), hV((size_t)B * H * N * d);
    std::vector<float> hO((size_t)B * H * N * d), hL((size_t)B * H * N);
    std::vector<float> hO_ref((size_t)B * H * N * d), hL_ref((size_t)B * H * N);
    std::vector<float> hdO((size_t)B * H * N * d), hdQ((size_t)B * H * N * d), hdK((size_t)B * H * N * d), hdV((size_t)B * H * N * d);
    std::vector<float> hdQ_ref((size_t)B * H * N * d), hdK_ref((size_t)B * H * N * d), hdV_ref((size_t)B * H * N * d);

    // Init random-ish data (deterministic)
    for (size_t i = 0; i < hQ.size(); ++i) hQ[i] = (float)((i % 31) - 15) / 32.f;
    for (size_t i = 0; i < hK.size(); ++i) hK[i] = (float)(((i * 7) % 29) - 14) / 29.f;
    for (size_t i = 0; i < hV.size(); ++i) hV[i] = (float)(((i * 13) % 37) - 18) / 37.f;
    for (size_t i = 0; i < hdO.size(); ++i) hdO[i] = (float)(((i * 5) % 23) - 11) / 23.f;

    // CPU reference
    forward_cpu_ref(hQ.data(), hK.data(), hV.data(), hO_ref.data(), hL_ref.data(), B, H, N, d);
    backward_cpu_ref(hQ.data(), hK.data(), hV.data(), hO_ref.data(), hdO.data(),
        hdQ_ref.data(), hdK_ref.data(), hdV_ref.data(), B, H, N, d);

    // Device buffers
    float* dQ = nullptr, * dK = nullptr, * dV = nullptr, * dO = nullptr, * dL = nullptr, * ddO = nullptr, * ddQ = nullptr, * ddK = nullptr, * ddV = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, hO.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL, hL.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddO, hdO.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddQ, hdQ.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddK, hdK.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddV, hdV.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(ddQ, 0, hdQ.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(ddK, 0, hdK.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(ddV, 0, hdV.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ddO, hdO.data(), hdO.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Forward
    run_forward_gpu(dQ, dK, dV, dO, dL, B, H, N, d, Br, Bc);

    // Backward 
    run_backward_gpu(dQ, dK, dV, dO, dL, ddO, ddQ, ddK, ddV, B, H, N, d, Br, Bc);

    // Copy back
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hL.data(), dL, hL.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdQ.data(), ddQ, hdQ.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdK.data(), ddK, hdK.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdV.data(), ddV, hdV.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    auto max_abs_err = [](const std::vector<float>& a, const std::vector<float>& b) {
        double e = 0.0;
        for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs((double)a[i] - (double)b[i]));
        return e;
        };
    double err_O = max_abs_err(hO, hO_ref);
    double err_L = max_abs_err(hL, hL_ref);
    double err_dQ = max_abs_err(hdQ, hdQ_ref);
    double err_dK = max_abs_err(hdK, hdK_ref);
    double err_dV = max_abs_err(hdV, hdV_ref);

    printf("[Forward] max|O-O_ref|=%.3e, max|L-L_ref|=%.3e\n", err_O, err_L);
    printf("[Backward] max|dQ-dQ_ref|=%.3e, |dK-dK_ref|=%.3e, |dV-dV_ref|=%.3e\n", err_dQ, err_dK, err_dV);

    // Cleanup
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
    cudaFree(ddO); cudaFree(ddQ); cudaFree(ddK); cudaFree(ddV);
    return 0;
}
