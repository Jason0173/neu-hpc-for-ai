// dist_fa2.cu
// Multi-GPU FlashAttention-v2 (forward, single head) in pure CUDA C.
// Sequence-parallel across N GPUs (1..8). Each GPU holds its local Q shard,
// and pulls K/V shards from owners (cudaMemcpyPeerAsync across devices, or
// cudaMemcpyAsync device-to-device when owner==self).
// Online softmax (row-wise m,l). No SxS materialization.
//
// Windows (RTX 40): nvcc -O3 -arch=sm_89 dist_fa2.cu -o dist_fa2.exe
// Linux:            nvcc -O3 -arch=sm_80 dist_fa2.cu -lpthread -o dist_fa2
// Example:          ./dist_fa2 --seq 4096 --d 128 --ngpu 1
//
// You can further reduce per-block resources if needed by compiling with:
//   -DTX=64 -DBK_CAP=32

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

struct Args { int S{ 4096 }; int d{ 128 }; int N{ 1 }; int iters{ 1 }; };

// Tunables (caps). You may override at compile time, e.g. -DTX=64 -DBK_CAP=32
#ifndef BK_CAP
#define BK_CAP 64     // max K/V rows per streamed tile; runtime chooses <= this
#endif
#ifndef TX
#define TX 128        // threads along d per CTA (<=1024), multiple of 32
#endif

// ---------------- Kernels ----------------

// One CTA per row in current batch: grid.y = rows, block=(TX,1).
// Dynamic SMEM layout: Ks | Vs (bk*d floats each).
__global__ void fa2_tile_fwd_kernel(
    const float* __restrict__ Q,     // [rows, d]
    const float* __restrict__ K,     // [bk,   d]
    const float* __restrict__ V,     // [bk,   d]
    float* __restrict__ O,           // [rows, d] (accumulator)
    float* __restrict__ m,           // [rows]
    float* __restrict__ l,           // [rows]
    int rows, int d, int bk, float inv_sqrt_d) {

    const int row = blockIdx.y;
    const int tx = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];   // Ks | Vs
    float* Ks = smem;
    float* Vs = Ks + (size_t)bk * d;
    __shared__ float warp_sums[32];   // TX up to 1024 -> <=32 warps

    // Load K,V into smem
    for (size_t idx = tx; idx < (size_t)bk * d; idx += blockDim.x) Ks[idx] = K[idx];
    for (size_t idx = tx; idx < (size_t)bk * d; idx += blockDim.x) Vs[idx] = V[idx];
    __syncthreads();

    // Per-row m,l
    float mi = (tx == 0 ? m[row] : 0.f);
    float li = (tx == 0 ? l[row] : 0.f);
    __shared__ float s_mi, s_li;
    if (tx == 0) { s_mi = mi; s_li = li; }
    __syncthreads();
    mi = s_mi; li = s_li;

    // Iterate over bk rows
    for (int j = 0; j < bk; ++j) {
        // dot = Q[row,:] ¡¤ K[j,:]
        float part = 0.f;
        for (int c = tx; c < d; c += blockDim.x) {
            part += Q[(size_t)row * d + c] * Ks[(size_t)j * d + c];
        }
        // warp reduction
        for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffff, part, off);
        const int warp_id = (tx >> 5);
        if ((tx & 31) == 0) warp_sums[warp_id] = part;
        __syncthreads();
        float dot = 0.f;
        if (warp_id == 0) {
            float v = (tx < (blockDim.x >> 5)) ? warp_sums[tx] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (tx == 0) dot = v;
        }
        dot = __shfl_sync(0xffffffff, dot, 0);
        float s = dot * inv_sqrt_d;

        // online softmax scalars
        float m_new = fmaxf(mi, s);
        float alpha = expf(mi - m_new);
        float p = expf(s - m_new);
        mi = m_new;
        li = li * alpha + p;

        // vector update: O[row,:] = O[row,:]*alpha + p*V[j,:]
        for (int c = tx; c < d; c += blockDim.x) {
            float old = O[(size_t)row * d + c];
            float add = Vs[(size_t)j * d + c];
            O[(size_t)row * d + c] = old * alpha + p * add;
        }
        __syncthreads();
    }

    if (tx == 0) { m[row] = mi; l[row] = li; }
}

// Normalize rows: O[i,:] /= l[i]
__global__ void fa2_row_normalize(float* __restrict__ O,
    const float* __restrict__ l,
    int rows, int d) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float li = l[row];
    if (li <= 0.f) return;
    for (int c = col; c < d; c += blockDim.x * gridDim.x) {
        O[(size_t)row * d + c] /= li;
    }
}

// ------------- Host orchestration -------------

struct GpuCtx {
    int dev{ -1 }, rank{ 0 }, N{ 1 };
    int S{ 0 }, d{ 0 }, S_local{ 0 };
    float inv_sqrt_d{ 1.f };

    float* d_Q{ nullptr }, * d_K_local{ nullptr }, * d_V_local{ nullptr };
    float* d_O{ nullptr }, * d_m{ nullptr }, * d_l{ nullptr };
    float* d_K_buf{ nullptr }, * d_V_buf{ nullptr };

    cudaStream_t compute{ nullptr }, comm{ nullptr };

    int smem_limit_bytes{ 0 }; // legacy per-block dynamic smem limit
    int bk_effective{ 0 };     // runtime BK chosen
};

static float* g_K_owner[8];
static float* g_V_owner[8];
static int     g_owner_dev[8];

static void enableP2PAll(int N) {
    int ng;
    CUDA_CHECK(cudaGetDeviceCount(&ng));
    assert(N <= ng);
    for (int i = 0; i < N; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, i, j);
            if (can) (void)cudaDeviceEnablePeerAccess(j, 0);
        }
    }
}

static int round_down_multiple(int x, int m) {
    return (x / m) * m;
}

void gpu_worker(GpuCtx ctx) {
    CUDA_CHECK(cudaSetDevice(ctx.dev));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.comm, cudaStreamNonBlocking));

    // Query legacy dynamic smem limit (no opt-in needed). Often 48KB on Windows/Ada.
    int legacy_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&legacy_limit, cudaDevAttrMaxSharedMemoryPerBlock, ctx.dev));
    ctx.smem_limit_bytes = legacy_limit;

    // choose bk_effective so that 2*bk*d*sizeof(float) <= smem_limit
    const int bytes_per_row = ctx.d * (int)sizeof(float);
    int bk_lim = std::max(1, ctx.smem_limit_bytes / (2 * bytes_per_row));
    bk_lim = std::min(bk_lim, BK_CAP);
    ctx.bk_effective = std::max(1, round_down_multiple(bk_lim, 16)); // align to 16

    // Init O=0, m=-inf, l=0
    CUDA_CHECK(cudaMemsetAsync(ctx.d_O, 0, (size_t)ctx.S_local * ctx.d * sizeof(float), ctx.compute));
    for (int q0 = 0; q0 < ctx.S_local; q0 += 1024) {
        int rows = std::min(1024, ctx.S_local - q0);
        std::vector<float> h_m(rows, -INFINITY), h_l(rows, 0.f);
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_m + q0, h_m.data(), rows * sizeof(float), cudaMemcpyHostToDevice, ctx.compute));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_l + q0, h_l.data(), rows * sizeof(float), cudaMemcpyHostToDevice, ctx.compute));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.compute));

    dim3 block(TX, 1);

    // Visit all owner shards
    for (int step = 0; step < ctx.N; ++step) {
        int owner = step % ctx.N;
        size_t bytes = (size_t)ctx.S_local * ctx.d * sizeof(float);

        // Pull K,V from owner to local buffers
        const float* srcK = g_K_owner[owner];
        const float* srcV = g_V_owner[owner];
        int srcDev = g_owner_dev[owner];

        if (srcDev == ctx.dev) {
            CUDA_CHECK(cudaMemcpyAsync(ctx.d_K_buf, srcK, bytes, cudaMemcpyDeviceToDevice, ctx.comm));
            CUDA_CHECK(cudaMemcpyAsync(ctx.d_V_buf, srcV, bytes, cudaMemcpyDeviceToDevice, ctx.comm));
        }
        else {
            CUDA_CHECK(cudaMemcpyPeerAsync(ctx.d_K_buf, ctx.dev, srcK, srcDev, bytes, ctx.comm));
            CUDA_CHECK(cudaMemcpyPeerAsync(ctx.d_V_buf, ctx.dev, srcV, srcDev, bytes, ctx.comm));
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.comm));

        // Process Q rows in batches (grid.y limited to <=512 for safety)
        for (int q0 = 0; q0 < ctx.S_local; ) {
            int rows = std::min(512, ctx.S_local - q0);
            dim3 grid(1, rows);

            const float* Q_tile = ctx.d_Q + (size_t)q0 * ctx.d;
            float* O_tile = ctx.d_O + (size_t)q0 * ctx.d;
            float* m_tile = ctx.d_m + q0;
            float* l_tile = ctx.d_l + q0;

            // Stream K/V shard in chunks of bk_effective
            for (int k0 = 0; k0 < ctx.S_local; k0 += ctx.bk_effective) {
                const float* K_tile = ctx.d_K_buf + (size_t)k0 * ctx.d;
                const float* V_tile = ctx.d_V_buf + (size_t)k0 * ctx.d;
                int bk_chunk = std::min(ctx.bk_effective, ctx.S_local - k0);
                size_t smem_chunk = (size_t)bk_chunk * ctx.d * 2 * sizeof(float); // exact size

                fa2_tile_fwd_kernel << <grid, block, smem_chunk, ctx.compute >> > (
                    Q_tile, K_tile, V_tile, O_tile, m_tile, l_tile, rows, ctx.d, bk_chunk, ctx.inv_sqrt_d);
                CUDA_CHECK(cudaGetLastError());
            }
            q0 += rows;
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.compute));
    }

    // Normalize rows
    for (int q0 = 0; q0 < ctx.S_local; ) {
        int rows = std::min(512, ctx.S_local - q0);
        dim3 nth(128, 1), nbl((ctx.d + nth.x - 1) / nth.x, rows);
        fa2_row_normalize << <nbl, nth, 0, ctx.compute >> > (
            ctx.d_O + (size_t)q0 * ctx.d,
            ctx.d_l + q0,
            rows, ctx.d);
        CUDA_CHECK(cudaGetLastError());
        q0 += rows;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.compute));
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seq") && i + 1 < argc) a.S = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d") && i + 1 < argc) a.d = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ngpu") && i + 1 < argc) a.N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) a.iters = atoi(argv[++i]);
    }

    int ng = 0; CUDA_CHECK(cudaGetDeviceCount(&ng));
    if (a.N < 1 || a.N > 8 || a.N > ng) {
        fprintf(stderr, "Invalid --ngpu (available=%d, asked=%d)\n", ng, a.N);
        return 1;
    }
    if (a.S % a.N) { fprintf(stderr, "Require --seq divisible by --ngpu.\n"); return 1; }

    enableP2PAll(a.N);

    int S_local = a.S / a.N;
    float inv_sqrt_d = 1.f / sqrtf((float)a.d);

    std::vector<GpuCtx> ctxs(a.N);

    // Allocate per GPU
    for (int r = 0; r < a.N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        ctxs[r].dev = r; ctxs[r].rank = r; ctxs[r].N = a.N;
        ctxs[r].S = a.S; ctxs[r].d = a.d; ctxs[r].S_local = S_local;
        ctxs[r].inv_sqrt_d = inv_sqrt_d;

        size_t vec = (size_t)S_local * a.d;

        CUDA_CHECK(cudaMalloc(&ctxs[r].d_Q, vec * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctxs[r].d_K_local, vec * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctxs[r].d_V_local, vec * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctxs[r].d_O, vec * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctxs[r].d_m, S_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctxs[r].d_l, S_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctxs[r].d_K_buf, vec * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctxs[r].d_V_buf, vec * sizeof(float)));

        // Initialize Q,K,V deterministically
        std::vector<float> hQ(vec), hK(vec), hV(vec);
        for (size_t i = 0; i < vec; i++) {
            hQ[i] = sinf(0.001f * (float)(i + r * 13));
            hK[i] = cosf(0.002f * (float)(i + r * 7));
            hV[i] = sinf(0.003f * (float)(i + r * 5));
        }
        CUDA_CHECK(cudaMemcpy(ctxs[r].d_Q, hQ.data(), vec * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctxs[r].d_K_local, hK.data(), vec * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctxs[r].d_V_local, hV.data(), vec * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Publish owners
    for (int r = 0; r < a.N; ++r) {
        g_K_owner[r] = ctxs[r].d_K_local;
        g_V_owner[r] = ctxs[r].d_V_local;
        g_owner_dev[r] = r;
    }

    // Launch workers
    std::vector<std::thread> th;
    for (int r = 0; r < a.N; ++r) th.emplace_back(gpu_worker, ctxs[r]);
    for (auto& t : th) t.join();

    // Checksum from GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<float> Ohost((size_t)S_local * a.d);
    CUDA_CHECK(cudaMemcpy(Ohost.data(), ctxs[0].d_O, Ohost.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0; for (auto v : Ohost) checksum += v;
    printf("Done. O[GPU0] checksum = %.6f\n", checksum);

    // Cleanup
    for (int r = 0; r < a.N; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaFree(ctxs[r].d_Q));
        CUDA_CHECK(cudaFree(ctxs[r].d_K_local));
        CUDA_CHECK(cudaFree(ctxs[r].d_V_local));
        CUDA_CHECK(cudaFree(ctxs[r].d_O));
        CUDA_CHECK(cudaFree(ctxs[r].d_m));
        CUDA_CHECK(cudaFree(ctxs[r].d_l));
        CUDA_CHECK(cudaFree(ctxs[r].d_K_buf));
        CUDA_CHECK(cudaFree(ctxs[r].d_V_buf));
    }
    return 0;
}


