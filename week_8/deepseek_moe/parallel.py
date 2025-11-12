# deepseek_moe/parallel.py
import torch.distributed as dist

class EPContext:
    def __init__(self, ep_size: int):
        assert dist.is_initialized(), "torch.distributed must be initialized"
        world = dist.get_world_size()
        rank  = dist.get_rank()
        assert world % ep_size == 0, "world_size must be divisible by ep_size"
        base = (rank // ep_size) * ep_size
        self.ranks = list(range(base, base + ep_size))
        self.group = dist.new_group(self.ranks)
        self.ep_size = ep_size
        self.rank = rank
        self.local_ep_rank = self.ranks.index(rank)

    def all_to_all(self, out, inp, out_splits, in_splits):
        return dist.all_to_all_single(out, inp, output_split_sizes=out_splits, input_split_sizes=in_splits, group=self.group)
