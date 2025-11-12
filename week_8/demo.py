# torchrun --nproc_per_node=<EP_SIZE> your_script.py
import os
import torch, torch.distributed as dist
from deepseek_moe.config import DeepseekConfig
from deepseek_moe.deepseek_moe import DeepseekV3MoE

def main():
    # Windows 上使用 gloo 后端，并设置必要的环境变量
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['USE_LIBUV'] = '0'
        # Windows 上使用 gloo 后端，Linux 上可以使用 nccl
        backend = "gloo" if os.name == 'nt' else "nccl"
        dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)
    cfg = DeepseekConfig(
        d_model=2048,
        moe_intermediate_size=2048,    # example
        n_shared_experts=2,
        n_routed_experts=32,           # must be divisible by ep_size
        n_group=8,
        topk_group=2,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        ep_size=dist.get_world_size(),
    )
    moe = DeepseekV3MoE(cfg).cuda().half()
    B, T, H = 2, 128, cfg.d_model
    x = torch.randn(B, T, H, device="cuda", dtype=torch.float16)
    y = moe(x)                         # [B,T,H]
    print("OK", y.shape)

if __name__ == "__main__":
    main()
