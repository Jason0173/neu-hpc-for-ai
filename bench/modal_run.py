"""Run bench/run_all.sh on a cloud GPU with Modal and copy the results back here.

Usage, from the repository root on any machine (no local GPU needed):
    pip install modal && modal setup          # once
    modal run bench/modal_run.py              # L4 by default
    modal run bench/modal_run.py --gpu A10G   # or another Modal GPU type

Afterwards bench/RESULTS.md, bench/results/ and bench/plots/ hold the new numbers.
The L4 is an Ada Lovelace GPU (sm_89), the same architecture as an RTX 40-series card.
"""

import subprocess
from pathlib import Path

import modal

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent

image = (
    # CUDA 12.6 toolkit (nvcc, cuBLAS) + Python 3.11
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("matplotlib")
    .add_local_dir(REPO, remote_path="/repo",
                   ignore=[".git", "**/.git", "bench/build", "bench/results", "bench/plots", "**/__pycache__"])
)

app = modal.App("neu-hpc-bench", image=image)


@app.function(gpu="L4", cpu=4.0, memory=8192, timeout=60 * 60)
def run_benchmarks() -> dict:
    """Copy the repo to a writable folder, run every benchmark, return the output files."""
    subprocess.run(["cp", "-r", "/repo", "/work"], check=True)
    subprocess.run(["bash", "/work/bench/run_all.sh"], check=True)
    out = {}
    bench = Path("/work/bench")
    for path in [bench / "RESULTS.md", *bench.glob("results/*"), *bench.glob("plots/*")]:
        out[str(path.relative_to(bench))] = path.read_bytes()
    return out


@app.local_entrypoint()
def main(gpu: str = "L4"):
    fn = run_benchmarks if gpu == "L4" else run_benchmarks.with_options(gpu=gpu)
    files = fn.remote()
    for rel, data in files.items():
        dest = BENCH / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"wrote bench/{rel}")
    print("\nDone. Review bench/RESULTS.md, then: git add bench && git commit -m 'Add benchmark results' && git push")
