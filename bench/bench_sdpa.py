"""PyTorch scaled_dot_product_attention baselines for bench_attention.cu.

Same shapes as the CUDA benchmark: one head, head dim 64, no mask, N from 512
to 16384. Two baselines:
  sdpa_fp32   FP32, memory-efficient kernel (same precision as the course kernels)
  sdpa_fp16   FP16, FlashAttention kernel (what PyTorch would use in practice)
Prints CSV on stdout with the same columns as bench_attention.cu.
"""

import sys

import torch
import torch.nn.functional as F

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]
D = 64


def time_ms(fn, budget_ms=300.0):
    """Median time of fn() in ms using CUDA events (2 warm-ups, 3-50 timed runs)."""
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def once():
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop)

    first = once()
    once()
    iters = int(min(50, max(3, budget_ms / max(first, 1e-3))))
    times = sorted(once() for _ in range(iters))
    return times[len(times) // 2]


def backend_ctx(name):
    """Force one SDPA backend (API differs across PyTorch versions)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel({"flash": SDPBackend.FLASH_ATTENTION,
                            "efficient": SDPBackend.EFFICIENT_ATTENTION}[name])
    except ImportError:  # PyTorch < 2.3
        return torch.backends.cuda.sdp_kernel(enable_flash=name == "flash",
                                              enable_mem_efficient=name == "efficient",
                                              enable_math=False)


def main():
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see a CUDA GPU.")
    print("kernel,pass,N,d,ms,tflops,max_abs_err,err_detail")
    for label, dtype, backend in [("sdpa_fp32", torch.float32, "efficient"),
                                  ("sdpa_fp16", torch.float16, "flash")]:
        for n in SEQ_LENS:
            print(f"[sdpa] {label} N={n}", file=sys.stderr)
            q, k, v = (torch.randn(1, 1, n, D, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
            grad_out = torch.randn(1, 1, n, D, device="cuda", dtype=dtype)
            fwd_tflop = 4.0 * n * n * D / 1e12
            try:
                with backend_ctx(backend):
                    fwd = time_ms(lambda: F.scaled_dot_product_attention(q, k, v))

                    def fwd_bwd():
                        out = F.scaled_dot_product_attention(q, k, v)
                        torch.autograd.grad(out, (q, k, v), grad_out)

                    bwd = max(time_ms(fwd_bwd) - fwd, 1e-6)
            except RuntimeError as e:  # backend not available for this GPU / dtype
                reason = str(e).splitlines()[0][:60].replace(",", ";")
                print(f"{label},fwd,{n},{D},,,,unavailable: {reason}")
                continue
            print(f"{label},fwd,{n},{D},{fwd:.4f},{fwd_tflop / (fwd / 1e3):.3f},,")
            print(f"{label},bwd,{n},{D},{bwd:.4f},{2.5 * fwd_tflop / (bwd / 1e3):.3f},,")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
