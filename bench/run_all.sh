#!/usr/bin/env bash
# Builds and runs every benchmark, then writes bench/RESULTS.md and bench/plots/.
# Needs: an NVIDIA GPU, the CUDA toolkit (nvcc + cuBLAS), and Python with torch (CUDA) and matplotlib.
# Usage (from anywhere):  bash bench/run_all.sh
# Use another Python:      PYTHON=~/bench-venv/bin/python bash bench/run_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"

# ---- preflight ----
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver visible."; exit 1; }
if ! command -v nvcc >/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then export PATH=/usr/local/cuda/bin:$PATH; fi
command -v nvcc >/dev/null || {
  echo "nvcc not found. Install the CUDA toolkit (on WSL2 use NVIDIA's 'WSL-Ubuntu' toolkit package, not a driver)."
  exit 1; }
"$PYTHON" - <<'EOF' || {
import importlib.util, sys
missing = [m for m in ("torch", "matplotlib") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing Python packages: " + ", ".join(missing))
import torch
if not torch.cuda.is_available():
    sys.exit("PyTorch is installed but cannot see the GPU (CPU-only build?)")
EOF
  echo
  echo "Fix: python3 -m venv ~/bench-venv && ~/bench-venv/bin/pip install torch matplotlib"
  echo "then: PYTHON=~/bench-venv/bin/python bash bench/run_all.sh"
  exit 1; }

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
ARCH="sm_${CC:-89}"
mkdir -p build results

{
  echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  echo "CUDA:   $(nvcc --version | grep -o 'release [0-9.]*')  (nvcc, -arch=$ARCH)"
  echo "PyTorch: $("$PYTHON" -c 'import torch; print(torch.__version__)')"
  echo "Date:   $(date +%Y-%m-%d)"
} > results/env.txt
cat results/env.txt

echo "== build"
nvcc -O3 -std=c++17 -arch="$ARCH" bench_gemm.cu -lcublas -o build/bench_gemm
nvcc -O3 -std=c++17 -arch="$ARCH" bench_attention.cu -o build/bench_attention

echo "== GEMM (about 1 minute)"
./build/bench_gemm > results/gemm.csv
echo "== attention kernels (a few minutes; slow sizes are skipped automatically)"
./build/bench_attention > results/attention.csv
echo "== PyTorch SDPA baselines"
"$PYTHON" bench_sdpa.py > results/sdpa.csv
echo "== report"
"$PYTHON" make_report.py
echo
echo "Done. Open bench/RESULTS.md. To publish: git add bench && git commit -m 'Add benchmark results' && git push"
