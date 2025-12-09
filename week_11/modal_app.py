"""
Modal deployment script for FlashDMoE project
"""
import modal
import os
import glob
import shutil
import subprocess

# Use NVIDIA CUDA base image, ensure CUDA is pre-installed
# Using CUDA 12.3.2 to support NVSHMEM 3.4.5
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.3.2-devel-ubuntu22.04",
    add_python="3.10"
)

# Define image with CUDA, MPI and NVSHMEM
image = (
    base_image
    .apt_install(
        "build-essential",
        "wget",
        "curl",
        "git",
        "cmake",
        "openmpi-bin",
        "libopenmpi-dev",
        "openmpi-common",
        "libnuma-dev",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .run_commands(
        # Set CUDA environment variables
        "echo 'export CUDA_HOME=/usr/local/cuda' >> /etc/profile",
        "echo 'export PATH=$CUDA_HOME/bin:$PATH' >> /etc/profile",
        "echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> /etc/profile",
    )
)

app = modal.App("flash-dmoe", image=image)


def find_nvshmem_tarball(project_dir="/project"):
    """Find NVSHMEM tarball in current directory"""
    patterns = ["nvshmem*.txz", "nvshmem*.tar.xz", "*.txz", "*.tar.xz"]
    candidates = []
    for p in patterns:
        # Search in project directory
        full_pattern = os.path.join(project_dir, p)
        candidates.extend(glob.glob(full_pattern))
    candidates = sorted(set(candidates))
    if candidates:
        print("✓ Found NVSHMEM tarball in project directory:", candidates[0])
        return candidates[0]
    else:
        print("⚠ No .txz / .tar.xz files found in project directory")
        return None


def check_nvshmem(project_dir="/project"):
    """Check if NVSHMEM is available (NVSHMEM availability check)"""
    print("Checking for NVSHMEM in Modal environment (NVSHMEM in PATH)...")
    
    nvshmem_info = shutil.which("nvshmem_info")
    nvshmemrun = shutil.which("nvshmemrun")
    
    if nvshmem_info or nvshmemrun:
        print("✅ Found NVSHMEM commands in PATH:")
        if nvshmem_info:
            print("   nvshmem_info:", nvshmem_info)
        if nvshmemrun:
            print("   nvshmemrun   :", nvshmemrun)
        return "modal"
    
    print("⚠ NVSHMEM not found in PATH, checking for tarball in project directory...")
    
    tarball = find_nvshmem_tarball(project_dir)
    if tarball:
        print("👉 Will install NVSHMEM from this tarball in container:", tarball)
        return tarball
    
    # If neither found, exit with error
    print("❌ Execution failed, exit code: 1")
    print("=" * 60)
    print("ERROR: NVSHMEM not found!")
    print("=" * 60)
    print("Modal environment should have NVSHMEM pre-installed, or you need to provide nvshmem*.txz / .tar.xz")
    print("=" * 60)
    return None


@app.function(
    gpu=modal.gpu.A10G(count=1),  # Use A10G GPU, adjust as needed
    timeout=3600,
    image=image,
    mounts=[
        # Mount project files
        modal.Mount.from_local_dir(".", remote_path="/project")
    ],
    network_file_systems={
        "/workspace": modal.NetworkFileSystem.from_name("flash-dmoe-workspace", create=True)
    }
)
def build_and_run():
    """Build and run FlashDMoE project"""
    print("🚀 Running FlashDMoE on Modal...")
    print("Current working directory (cwd):", os.getcwd())
    print("Files in current directory:", os.listdir("."))
    
    # Set working directory
    project_dir = "/project"
    workspace = "/workspace"
    
    # Copy project to workspace (if persistence needed)
    if os.path.exists(workspace):
        shutil.copytree(project_dir, f"{workspace}/flash_dmoe", dirs_exist_ok=True)
        project_dir = f"{workspace}/flash_dmoe"
    
    os.chdir(project_dir)
    print("Changed to project directory:", project_dir)
    print("Files in project directory:", os.listdir("."))
    
    # Check NVSHMEM
    nvshmem_source = check_nvshmem(project_dir)
    if nvshmem_source is None:
        return 1
    
    # Set environment variables
    env = os.environ.copy()
    
    # Find CUDA path
    cuda_path = "/usr/local/cuda"
    if not os.path.exists(cuda_path):
        # Try other common paths
        for path in ["/usr/lib/cuda", "/opt/cuda"]:
            if os.path.exists(path):
                cuda_path = path
                break
    
    env["PATH"] = f"{cuda_path}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_HOME"] = cuda_path
    
    # If Modal's pre-installed NVSHMEM found, set environment variables
    if nvshmem_source == "modal":
        nvshmem_info = shutil.which("nvshmem_info")
        if nvshmem_info:
            nvshmem_bin = os.path.dirname(nvshmem_info)
            nvshmem_home = os.path.dirname(nvshmem_bin)
            env["NVSHMEM_HOME"] = nvshmem_home
            env["PATH"] = f"{nvshmem_home}/bin:{env.get('PATH', '')}"
            env["LD_LIBRARY_PATH"] = f"{nvshmem_home}/lib:{env.get('LD_LIBRARY_PATH', '')}"
            print(f"✓ Set NVSHMEM_HOME: {nvshmem_home}")
    elif nvshmem_source and nvshmem_source != "modal":
        # Install NVSHMEM from tarball
        print(f"📦 Installing NVSHMEM from tarball: {nvshmem_source}")
        nvshmem_home = "/usr/local/nvshmem"
        
        # Use install_nvshmem.sh script to install
        install_script_path = os.path.join(project_dir, "install_nvshmem.sh")
        if os.path.exists(install_script_path):
            print(f"Using install script: {install_script_path}")
            result = subprocess.run(
                ["bash", install_script_path],
                capture_output=True,
                text=True,
                check=False
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode == 0:
                print("✓ NVSHMEM installed successfully from project directory")
                env["NVSHMEM_HOME"] = nvshmem_home
                env["PATH"] = f"{nvshmem_home}/bin:{env.get('PATH', '')}"
                env["LD_LIBRARY_PATH"] = f"{nvshmem_home}/lib:{env.get('LD_LIBRARY_PATH', '')}"
            else:
                print("❌ NVSHMEM installation failed")
                return 1
        else:
            # If script doesn't exist, use inline installation (fallback)
            print(f"⚠ Install script not found, using inline installation: {install_script_path}")
            install_cmd = f"""
            cd {project_dir}
            mkdir -p {nvshmem_home}
            cd /tmp
            tar -xf {project_dir}/{os.path.basename(nvshmem_source)}
            if [ -d nvshmem_* ]; then
                mv nvshmem_*/* {nvshmem_home}/
                rmdir nvshmem_*
            elif [ -d nvshmem ]; then
                mv nvshmem/* {nvshmem_home}/
                rmdir nvshmem
            fi
            chmod +x {nvshmem_home}/bin/* 2>/dev/null || true
            """
            result = subprocess.run(["bash", "-c", install_cmd], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ NVSHMEM installed successfully from project directory")
                env["NVSHMEM_HOME"] = nvshmem_home
                env["PATH"] = f"{nvshmem_home}/bin:{env.get('PATH', '')}"
                env["LD_LIBRARY_PATH"] = f"{nvshmem_home}/lib:{env.get('LD_LIBRARY_PATH', '')}"
            else:
                print("❌ NVSHMEM installation failed")
                print(result.stderr)
                return 1
    
    # Check required tools
    print("Checking build tools...")
    result = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: nvcc not found. CUDA toolkit may not be installed correctly.")
        return
    
    result = subprocess.run(["which", "mpicxx"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: mpicxx not found. OpenMPI may not be installed correctly.")
        return
    
    # Build project
    print("Building project...")
    result = subprocess.run(["make", "clean"], env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("WARNING:", result.stderr)
    
    result = subprocess.run(["make"], env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)
    
    if result.returncode != 0:
        print("Build failed!")
        return
    
    # Check executable
    if not os.path.exists("flash_dmoe_demo"):
        print("ERROR: Executable not found after build!")
        return
    
    print("Build successful!")
    
    # Run program
    # Use mpirun to run, even single process needs MPI environment
    print("Running flash_dmoe_demo with mpirun...")
    
    # For multi-GPU, can use: mpirun -np 2 ./flash_dmoe_demo
    # But Modal single function instance usually has only one GPU, so use single process mode here
    result = subprocess.run(
        ["mpirun", "-np", "1", "--allow-run-as-root", "./flash_dmoe_demo"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"Exit code: {result.returncode}")
    return result.returncode


@app.local_entrypoint()
def main():
    """Local entry point, triggers build and run"""
    result = build_and_run.remote()
    print(f"Execution completed with exit code: {result}")
