"""
NVSHMEM diagnostic script
Check NVSHMEM status in Modal environment
Run: modal run diagnose_nvshmem.py
"""
import modal
import os
import subprocess
import shutil

app = modal.App("nvshmem-diagnose")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-devel-ubuntu22.04",
        add_python="3.10"
    )
    .run_commands(
        "apt-get update && apt-get install -y libopenmpi-dev openmpi-bin build-essential tar xz-utils",
        "mkdir -p /usr/local/nvshmem",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600
)
def diagnose():
    """Diagnose NVSHMEM environment"""
    print("=" * 60)
    print("NVSHMEM Diagnostic Report")
    print("=" * 60)
    
    # 1. Check commands in PATH
    print("\n1. Checking NVSHMEM commands in PATH...")
    nvshmem_info = shutil.which("nvshmem_info")
    nvshmemrun = shutil.which("nvshmemrun")
    
    if nvshmem_info:
        print(f"   ✅ nvshmem_info found: {nvshmem_info}")
        nvshmem_bin = os.path.dirname(nvshmem_info)
        nvshmem_home = os.path.dirname(nvshmem_bin)
        print(f"   ✅ NVSHMEM_HOME: {nvshmem_home}")
    else:
        print("   ❌ nvshmem_info not found")
    
    if nvshmemrun:
        print(f"   ✅ nvshmemrun found: {nvshmemrun}")
    else:
        print("   ❌ nvshmemrun not found")
    
    # 2. Check common installation paths
    print("\n2. Checking common installation paths...")
    common_paths = [
        "/usr/local/nvshmem",
        "/opt/nvshmem",
        "/usr/nvshmem",
        "/usr/local/cuda/nvshmem",
    ]
    for path in common_paths:
        if os.path.exists(path):
            print(f"   📁 {path} exists")
            if os.path.exists(f"{path}/bin/nvshmem_info"):
                print(f"      ✅ bin/nvshmem_info exists")
            if os.path.exists(f"{path}/include/nvshmem.h"):
                print(f"      ✅ include/nvshmem.h exists")
            if os.path.exists(f"{path}/lib"):
                libs = os.listdir(f"{path}/lib")
                nvshmem_libs = [l for l in libs if "nvshmem" in l]
                if nvshmem_libs:
                    print(f"      ✅ Found library files: {', '.join(nvshmem_libs[:3])}")
        else:
            print(f"   ❌ {path} does not exist")
    
    # 3. Check environment variables
    print("\n3. Checking environment variables...")
    env_vars = ["NVSHMEM_HOME", "CUDA_HOME", "LD_LIBRARY_PATH", "PATH"]
    for var in env_vars:
        value = os.environ.get(var, "")
        if value:
            print(f"   {var}: {value[:100]}...")
        else:
            print(f"   {var}: (not set)")
    
    # 4. Check CUDA
    print("\n4. Checking CUDA...")
    result = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✅ nvcc found: {result.stdout.strip()}")
        nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if nvcc_result.returncode == 0:
            version_line = nvcc_result.stdout.split('\n')[1] if len(nvcc_result.stdout.split('\n')) > 1 else ""
            print(f"   {version_line}")
    else:
        print("   ❌ nvcc not found")
    
    # 5. Check MPI
    print("\n5. Checking MPI...")
    result = subprocess.run(["which", "mpicxx"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✅ mpicxx found: {result.stdout.strip()}")
    else:
        print("   ❌ mpicxx not found")
    
    # 6. Check files in project directory
    print("\n6. Checking project directory...")
    project_dir = "/project"
    if os.path.exists(project_dir):
        files = os.listdir(project_dir)
        nvshmem_files = [f for f in files if "nvshmem" in f.lower()]
        if nvshmem_files:
            print(f"   ✅ Found NVSHMEM related files: {', '.join(nvshmem_files)}")
        else:
            print("   ❌ No NVSHMEM files found")
    else:
        print("   ❌ /project directory does not exist")
    
    # 7. Check /tmp directory
    print("\n7. Checking /tmp directory...")
    if os.path.exists("/tmp"):
        tmp_files = os.listdir("/tmp")
        nvshmem_files = [f for f in tmp_files if "nvshmem" in f.lower()]
        if nvshmem_files:
            print(f"   ✅ Found NVSHMEM related files: {', '.join(nvshmem_files)}")
        else:
            print("   ❌ No NVSHMEM files found")
    
    print("\n" + "=" * 60)
    print("Diagnosis complete")
    print("=" * 60)
    
    # Summary
    if nvshmem_info or nvshmemrun:
        print("\n✅ NVSHMEM available in PATH")
        return 0
    elif os.path.exists("/usr/local/nvshmem/bin/nvshmem_info"):
        print("\n⚠️  NVSHMEM installed but not in PATH")
        print("   Need to set environment variables:")
        print("   export NVSHMEM_HOME=/usr/local/nvshmem")
        print("   export PATH=$NVSHMEM_HOME/bin:$PATH")
        return 0
    else:
        print("\n❌ NVSHMEM not found")
        print("\nSolution:")
        print("1. Download NVSHMEM from https://developer.nvidia.com/nvshmem")
        print("2. Place .txz or .tar.xz file in project root directory")
        print("3. Run: modal run run_modal.py")
        return 1

@app.local_entrypoint()
def main():
    result = diagnose.remote()
    return result
