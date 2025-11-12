# deepseek_moe/config.py
from dataclasses import dataclass

@dataclass
class DeepseekConfig:
    d_model:          int
    moe_intermediate_size: int
    n_shared_experts: int
    n_routed_experts: int
    n_group:          int
    topk_group:       int
    num_experts_per_tok: int
    norm_topk_prob:   bool
    routed_scaling_factor: float
    # parallel
    ep_size:          int
    dp_size:          int = 1
    # misc
    bias:             bool = False
