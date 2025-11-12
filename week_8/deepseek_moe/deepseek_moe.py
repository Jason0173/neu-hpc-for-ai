# deepseek_moe/deepseek_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DeepseekConfig
from .parallel import EPContext
from .build import moe_cuda

def gelu(x):  # unfused on purpose
    return F.gelu(x, approximate="tanh")

class DeepseekV3MLP(nn.Module):
    def __init__(self, config: DeepseekConfig, d_model=None, intermediate_size=None):
        super().__init__()
        d_model = d_model or config.d_model
        inter   = intermediate_size or config.moe_intermediate_size
        self.fc1 = nn.Linear(d_model, inter, bias=config.bias)
        self.fc2 = nn.Linear(inter, d_model, bias=config.bias)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.d_model, config.n_routed_experts, bias=True)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))
    def forward(self, hidden_states):                 # [B,T,H] -> [B,T,E]
        return self.proj(hidden_states)

class DeepseekV3NaiveMoe(nn.Module):
    """
    Expert-parallel routed experts (unfused). One A2A out, local compute, one A2A back.
    """
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.cfg = config
        assert config.n_routed_experts % config.ep_size == 0, "n_routed_experts must be divisible by ep_size"
        self.local_n = config.n_routed_experts // config.ep_size
        self.experts = nn.ModuleList(
            [DeepseekV3MLP(config) for _ in range(self.local_n)]
        )
        self._ep = None

    def _ensure_ep(self):
        if self._ep is None:
            self._ep = EPContext(self.cfg.ep_size)

    def _bucket_plan(self, eids_flat: torch.Tensor):
        """Compute (sizes, offsets, pos) for local expert ids (int32)."""
        E = self.local_n
        dev = eids_flat.device
        sizes = torch.zeros(E, dtype=torch.int32, device=dev)
        moe_cuda.expert_histogram(eids_flat, E, sizes)              # sizes[e]
        # offsets on CPU (E small, deterministic):
        off_cpu = torch.zeros(E + 1, dtype=torch.int32)
        torch.cumsum(sizes.cpu(), 0, out=off_cpu[1:])
        offsets = off_cpu.to(dev)                                   # [E+1]
        pos = torch.empty_like(eids_flat)
        moe_cuda.expert_positions(eids_flat, E, pos)                # 0..(sizes[e]-1) within expert
        return sizes, offsets, pos

    def forward(self, tokens_2d: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        """
        tokens_2d:     [BT, H]
        topk_indices:  [BT, K]  (global expert ids)
        topk_weights:  [BT, K]
        returns:       [BT, H]
        """
        self._ensure_ep()
        cfg = self.cfg
        BT, H = tokens_2d.shape
        K = topk_indices.size(1)
        E_global = cfg.n_routed_experts
        ep = self._ep

        # 1) Expand tokens and flatten expert ids/weights
        x_rep = tokens_2d.repeat_interleave(K, dim=0)               # [BT*K, H]
        w_rep = topk_weights.reshape(-1, 1).to(x_rep.dtype)         # [BT*K, 1]
        eids_global = topk_indices.reshape(-1).to(torch.int32)      # [BT*K]

        # 2) Map global expert -> (owner rank, local id)
        experts_per = E_global // ep.ep_size
        owner = (eids_global // experts_per).to(torch.int32)        # [BT*K]
        leids = (eids_global % experts_per).to(torch.int32)         # [BT*K]

        # 3) Stable sort by owner; build split sizes for A2A
        ord_owner = torch.argsort(owner, stable=True)
        x_send = x_rep.index_select(0, ord_owner).contiguous()
        w_send = w_rep.index_select(0, ord_owner).contiguous()
        leids_send = leids.index_select(0, ord_owner).contiguous()

        counts = torch.bincount(owner.index_select(0, ord_owner), minlength=ep.ep_size)
        in_splits = counts.tolist()
        # For single process, out_splits must match in_splits
        # For multi-process, this would be inferred, but we need to provide it
        out_splits = in_splits.copy() if ep.ep_size == 1 else [0] * ep.ep_size

        # 4) All-to-all (3 streams of data kept separate for clarity)
        recv_x = torch.empty_like(x_send)     # size matches total tokens; splits define per-rank segments
        recv_w = torch.empty_like(w_send)
        recv_e = torch.empty_like(leids_send)
        ep.all_to_all(recv_x, x_send, out_splits, in_splits)
        ep.all_to_all(recv_w, w_send, out_splits, in_splits)
        ep.all_to_all(recv_e, leids_send, out_splits, in_splits)

        # 5) Locally bucket by local expert id
        sizes, offsets, pos = self._bucket_plan(recv_e)
        routed_in = torch.empty_like(recv_x)
        moe_cuda.expert_scatter(recv_x, recv_e, offsets, pos, routed_in)

        # 6) Run local experts segment-by-segment
        out_buf = torch.empty_like(recv_x)
        start = 0
        for i, sz in enumerate(sizes.tolist()):
            if sz == 0:
                continue
            s, t = start, start + sz
            out_buf[s:t] = self.experts[i](routed_in[s:t])
            start = t

        # 7) Inverse bucket order
        #    We can just gather using the same (eids, offsets, pos) back to "recv_x" order:
        unbucket = torch.empty_like(recv_x)
        moe_cuda.expert_gather(out_buf, recv_e, offsets, pos, unbucket)

        # 8) Apply gate weights, all_to_all back
        unbucket.mul_(recv_w)                                         # [recvN,H]
        send_back = unbucket.contiguous()
        recv_back = torch.empty_like(send_back)
        ep.all_to_all(recv_back, send_back, out_splits, in_splits)

        # 9) Undo owner-sort and sum contributions across K
        inv = torch.empty_like(ord_owner)
        inv[ord_owner] = torch.arange(ord_owner.numel(), device=ord_owner.device, dtype=ord_owner.dtype)
        y_perm = recv_back.index_select(0, inv)                       # [BT*K,H]
        y = y_perm.view(BT, K, H).sum(dim=1)                          # [BT,H]
        return y

class DeepseekV3MoE(nn.Module):
    """
    IO/behavior matches the reference you provided.
    """
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        self.experts = DeepseekV3NaiveMoe(config)
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    @torch.no_grad()
    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        B, T, E = router_logits_for_choice.shape
        G = self.n_group
        assert E % G == 0
        per_group = E // G
        x = router_logits_for_choice.view(B*T, G, per_group)
        group_scores = x.topk(2, dim=-1).values.sum(dim=-1)                   # [BT,G]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, G, per_group).reshape(B*T, E)
        scores_for_choice = router_logits_for_choice.view(B*T, E).masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False).indices  # [BT,K]
        topk_weights = router_logits.view(B*T, E).gather(1, topk_indices)                          # [BT,K]
        if self.norm_topk_prob:
            denom = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denom
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig = hidden_states.shape
        if hidden_states.dim() == 2:                    # [T,H] -> [1,T,H]
            x = hidden_states.unsqueeze(0)
        else:
            x = hidden_states
        router_logits = self.gate(x)                    # [B,T,E]
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        y = self.experts(x.view(-1, x.size(-1)), topk_indices, topk_weights).view_as(x)
        y = y + self.shared_experts(residuals.view_as(x))
        return y.view(orig)
