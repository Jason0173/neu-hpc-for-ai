// deepseek_moe/csrc/moe_bindings.cpp
#include <torch/extension.h>

void expert_histogram_cuda(torch::Tensor eids, int64_t n_experts, torch::Tensor sizes);
void expert_positions_cuda(torch::Tensor eids, int64_t n_experts, torch::Tensor pos);
void expert_scatter_cuda(torch::Tensor tokens, torch::Tensor eids, torch::Tensor offsets, torch::Tensor pos, torch::Tensor out);
void expert_gather_cuda(torch::Tensor routed, torch::Tensor eids, torch::Tensor offsets, torch::Tensor pos, torch::Tensor dst);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert_histogram", &expert_histogram_cuda, "Count tokens per expert (int32 eids -> int32 sizes)");
  m.def("expert_positions", &expert_positions_cuda, "Compute per-token position within its expert (atomic)");
  m.def("expert_scatter", &expert_scatter_cuda, "Scatter tokens -> expert buckets");
  m.def("expert_gather", &expert_gather_cuda, "Gather expert outputs -> token order");
}
