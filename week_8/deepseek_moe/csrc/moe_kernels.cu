// deepseek_moe/csrc/moe_kernels.cu
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

template <typename T>
__device__ __forceinline__ T ldg(const T* p) { return *p; }

//////////////////////////////////////////////////////////////
// Histogram (sizes[e] = #occurrences in eids)
template <typename idx_t>
__global__ void histo_kernel(const idx_t* __restrict__ eids, int64_t N, int32_t* __restrict__ sizes, int64_t nE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int32_t e = static_cast<int32_t>(eids[i]);
        if (0 <= e && e < nE) atomicAdd(&sizes[e], 1);
    }
}

void expert_histogram_cuda(torch::Tensor eids, int64_t n_experts, torch::Tensor sizes) {
    TORCH_CHECK(eids.is_cuda() && sizes.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(eids.scalar_type() == at::kInt, "eids must be int32");
    TORCH_CHECK(sizes.scalar_type() == at::kInt, "sizes must be int32");
    const int64_t N = eids.numel();
    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);
    sizes.zero_();
    histo_kernel<<<blocks, threads>>>(eids.data_ptr<int32_t>(), N, sizes.data_ptr<int32_t>(), n_experts);
    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////////////////
// Positions within expert: pos[i] = atomicAdd(counter[eids[i]], 1)
__global__ void positions_kernel(const int32_t* __restrict__ eids, int64_t N, int32_t* __restrict__ counters, int32_t* __restrict__ pos, int64_t nE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int32_t e = eids[i];
        if (0 <= e && e < nE) {
            int32_t ticket = atomicAdd(&counters[e], 1);
            pos[i] = ticket;
        } else {
            pos[i] = 0;
        }
    }
}

void expert_positions_cuda(torch::Tensor eids, int64_t n_experts, torch::Tensor pos) {
    TORCH_CHECK(eids.is_cuda() && pos.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(eids.scalar_type() == at::kInt, "eids must be int32");
    TORCH_CHECK(pos.scalar_type() == at::kInt, "pos must be int32");
    const int64_t N = eids.numel();
    auto counters = torch::zeros({(long)n_experts}, torch::dtype(at::kInt).device(eids.device()));
    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);
    positions_kernel<<<blocks, threads>>>(
        eids.data_ptr<int32_t>(), N, counters.data_ptr<int32_t>(), pos.data_ptr<int32_t>(), n_experts);
    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////////////////
// Scatter/Gather (float, half, bfloat16)
template <typename scalar_t>
__global__ void scatter_kernel(
    const scalar_t* __restrict__ tokens,  // [N,H]
    const int32_t*  __restrict__ eids,    // [N]
    const int32_t*  __restrict__ offsets, // [E+1]
    const int32_t*  __restrict__ pos,     // [N]
    scalar_t* __restrict__ out,           // [N,H] expert-bucketed
    int64_t N, int64_t H) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H) return;
    int64_t r = idx / H;      // row in N
    int64_t h = idx % H;
    int32_t e = eids[r];
    int64_t dst = (int64_t)offsets[e] + pos[r];
    out[dst * H + h] = tokens[r * H + h];
}

template <typename scalar_t>
__global__ void gather_kernel(
    const scalar_t* __restrict__ routed,  // [N,H] expert-bucketed
    const int32_t*  __restrict__ eids,    // [N]
    const int32_t*  __restrict__ offsets, // [E+1]
    const int32_t*  __restrict__ pos,     // [N]
    scalar_t* __restrict__ dst,           // [N,H] original order
    int64_t N, int64_t H) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H) return;
    int64_t r = idx / H;
    int64_t h = idx % H;
    int32_t e = eids[r];
    int64_t src = (int64_t)offsets[e] + pos[r];
    dst[r * H + h] = routed[src * H + h];
}

void expert_scatter_cuda(torch::Tensor tokens, torch::Tensor eids, torch::Tensor offsets, torch::Tensor pos, torch::Tensor out) {
    TORCH_CHECK(tokens.is_cuda() && eids.is_cuda() && offsets.is_cuda() && pos.is_cuda() && out.is_cuda(), "CUDA only");
    TORCH_CHECK(eids.scalar_type() == at::kInt, "eids int32");
    TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets int32");
    TORCH_CHECK(pos.scalar_type() == at::kInt, "pos int32");
    const int64_t N = tokens.size(0);
    const int64_t H = tokens.size(1);
    const int threads = 256;
    const int blocks = (int)((N * H + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, tokens.scalar_type(), "expert_scatter", [&]{
        scatter_kernel<scalar_t><<<blocks, threads>>>(
            tokens.data_ptr<scalar_t>(), eids.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
            pos.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), N, H);
    });
    cudaDeviceSynchronize();
}

void expert_gather_cuda(torch::Tensor routed, torch::Tensor eids, torch::Tensor offsets, torch::Tensor pos, torch::Tensor dst) {
    TORCH_CHECK(routed.is_cuda() && eids.is_cuda() && offsets.is_cuda() && pos.is_cuda() && dst.is_cuda(), "CUDA only");
    TORCH_CHECK(eids.scalar_type() == at::kInt, "eids int32");
    TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets int32");
    TORCH_CHECK(pos.scalar_type() == at::kInt, "pos int32");
    const int64_t N = dst.size(0);
    const int64_t H = dst.size(1);
    const int threads = 256;
    const int blocks = (int)((N * H + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dst.scalar_type(), "expert_gather", [&]{
        gather_kernel<scalar_t><<<blocks, threads>>>(
            routed.data_ptr<scalar_t>(), eids.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
            pos.data_ptr<int32_t>(), dst.data_ptr<scalar_t>(), N, H);
    });
    cudaDeviceSynchronize();
}
