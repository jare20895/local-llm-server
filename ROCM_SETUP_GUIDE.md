# Complete ROCm Setup Guide for WSL2

This comprehensive guide covers all aspects of setting up ROCm environments for running LLM models on AMD GPUs in WSL2.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Python Virtual Environment Setup](#python-virtual-environment-setup)
3. [Docker Setup](#docker-setup)
4. [Troubleshooting](#troubleshooting)
5. [Environment Comparison](#environment-comparison)

---

## Prerequisites

### System Requirements
- **GPU:** AMD Radeon RX 6000/7000 series (RDNA 2/3 architecture)
- **OS:** Windows 11 with WSL2
- **RAM:** 16GB+ recommended
- **VRAM:** 8GB+ (16GB recommended for larger models)

### Windows Side Setup

#### 1. Install AMD Drivers
Download and install the latest AMD Adrenalin drivers:
- Visit: https://www.amd.com/en/support
- Download driver for your GPU (e.g., RX 7900 series)
- **IMPORTANT:** Ensure ROCm components are selected during installation
- Restart Windows after installation

#### 2. Update WSL2
```powershell
# In PowerShell as Administrator
wsl --update
wsl --shutdown
```

#### 3. Verify GPU Passthrough
```bash
# In WSL2
ls -la /dev/dxg
# Should show: crw-rw-rw- 1 root root 10, 62 ...

# If permission denied:
sudo chmod 666 /dev/dxg
```

#### 4. Verify ROCm Installation (Optional)
```bash
# Check if ROCm tools are available
rocm-smi
rocminfo | grep "Marketing Name"
# Should show your AMD GPU name
```

---

## Python Virtual Environment Setup

This is the recommended approach for local development. It gives you full control over dependencies and is easier to debug.

### Step 1: Create Virtual Environment

```bash
cd /home/jare16/LLM  # Or your project directory

# Create venv (Python 3.12 recommended)
python3.12 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install ROCm PyTorch (CRITICAL)

**The Problem:** Using wrong PyTorch source is the #1 cause of "GPU not detected" errors.

#### Why AMD Wheels Matter

AMD provides official wheels at repo.radeon.com that include:
- WSL-specific compatibility fixes
- Tested configurations for AMD GPUs
- Proper ROCm version matching

PyTorch.org wheels:
- Are NOT extensively tested by AMD
- May be missing WSL compatibility fixes
- Change frequently with nightly builds

#### Automated Installation (Recommended)

```bash
# Use the provided script
./setup_rocm.sh
```

This script:
1. Uninstalls any existing PyTorch installations
2. Downloads AMD-recommended wheels from repo.radeon.com
3. Installs PyTorch with proper ROCm version (6.4.2)
4. **Applies WSL runtime library fix** (removes conflicting `libhsa-runtime64.so`)
5. Installs project dependencies
6. Verifies GPU detection

#### Manual Installation

```bash
# Uninstall any existing PyTorch
pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm

# Download AMD wheels (Python 3.12, ROCm 6.4.2)
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

# Install AMD wheels
pip install torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl \
            torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl \
            torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl \
            pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

# CRITICAL WSL FIX: Remove bundled libhsa-runtime64.so
# This library conflicts with WSL's GPU passthrough
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*

# Install other dependencies
pip install -r requirements.txt
```

**For other Python versions:**
- Python 3.10: Change `cp312` to `cp310` in wheel URLs
- Python 3.11: Change `cp312` to `cp311` in wheel URLs

**For other ROCm versions:**
- Check available wheels: https://repo.radeon.com/rocm/manylinux/
- ROCm 6.4.2 is recommended for RX 7900 series

### Step 3: Verify GPU Detection

```bash
# Run the verification script
python verify_gpu.py
```

Expected output:
```
==============================================================
PyTorch Information
==============================================================
  PyTorch Version: 2.6.0+rocm6.4.2.git76481f7c
  ROCm Version: 6.4.43484-123eb5128
  GPU Available: True
  GPU Count: 1
  GPU 0: AMD Radeon RX 7900 GRE
    - Total Memory: 15.98 GB
    - Compute Capability: 11.0

==============================================================
GPU Computation Test
==============================================================
  Using device: cuda:0
  Creating tensor on GPU...
  Performing matrix multiplication...
  ✅ Success! Result shape: torch.Size([1000, 1000])
```

### Step 4: Run the Server

```bash
# Start the server
./start.sh

# Or manually:
source venv/bin/activate
python main.py
```

Access at: http://localhost:8000

---

## Docker Setup

Docker provides a consistent, isolated environment and is recommended for production deployments.

### Prerequisites

```bash
# Verify Docker is installed
docker --version
docker-compose --version

# Verify WSL GPU libraries exist (CRITICAL for Docker GPU detection)
ls -la /usr/lib/wsl/lib/libdxcore.so
ls -la /opt/rocm/lib/libhsa-runtime64.so.1

# Both should exist. If not, reinstall AMD drivers on Windows
```

### Why Docker GPU Setup is Different

Docker containers are isolated from the host system. To access the GPU, we need to:
1. Pass through the `/dev/dxg` device
2. Mount WSL GPU libraries from host into container
3. Use AMD's official ROCm PyTorch image

**Without these library mounts, GPU won't be detected even with correct image!**

### Option 1: Pre-built AMD Image (Recommended)

Uses AMD's official Docker image directly - fastest and most reliable.

**File:** `docker-compose.yml`

```yaml
services:
  llm-server:
    image: rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0
    container_name: homelab-llm-server
    ports:
      - "8000:8000"
    devices:
      - /dev/dxg:/dev/dxg  # WSL GPU device
    volumes:
      # Code and data
      - .:/app
      - ./data:/app/data
      - hf-cache:/root/.cache/huggingface

      # WSL GPU libraries (CRITICAL!)
      - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
      - /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1

    working_dir: /app
    command: >
      bash -c "pip install -r requirements.txt && python main.py"

    environment:
      - PYTHONUNBUFFERED=1
      # Optional: Hugging Face token for gated models
      # - HF_TOKEN=your_token_here

volumes:
  hf-cache:
```

**Start the server:**
```bash
docker-compose up -d

# View logs
docker-compose logs -f llm-server

# Verify GPU in container
docker exec -it homelab-llm-server python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**Advantages:**
- ✅ No build time (starts immediately)
- ✅ AMD-tested and verified
- ✅ Includes all WSL compatibility fixes
- ✅ Most reliable option

### Option 2: Custom Build

Builds a custom image with dependencies baked in.

**File:** `docker-compose-build.yml`

```yaml
services:
  llm-server-custom:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: homelab-llm-server-custom
    ports:
      - "8000:8000"
    devices:
      - /dev/dxg:/dev/dxg
    volumes:
      - .:/app
      - ./data:/app/data
      - hf-cache:/root/.cache/huggingface
      - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
      - /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1

    working_dir: /app
    command: python main.py

    environment:
      - PYTHONUNBUFFERED=1

volumes:
  hf-cache:
```

**Dockerfile:**
```dockerfile
FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
```

**Build and start:**
```bash
docker-compose -f docker-compose-build.yml build
docker-compose -f docker-compose-build.yml up -d

# View logs
docker-compose -f docker-compose-build.yml logs -f llm-server-custom
```

**Advantages:**
- ✅ Dependencies baked into image (faster restarts)
- ✅ Good for production deployments
- ✅ Can customize Dockerfile

**Disadvantages:**
- ⚠️  Initial build takes 5-10 minutes
- ⚠️  Requires rebuild after dependency changes

### Docker GPU Verification

```bash
# Check GPU access in container
docker exec -it homelab-llm-server bash

# Inside container:
rocm-smi
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Data Persistence

Docker setup persists data in two locations:
1. **Database:** `./data/models.db` - Model registry and performance logs
2. **Model Cache:** Docker volume `hf-cache` - Downloaded Hugging Face models

---

## Troubleshooting

### Issue 1: "No GPU detected" in Python venv

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- Models run on CPU (very slow)

**Diagnosis:**
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Wrong outputs:
# - 2.x.x+cu128 (CUDA version - completely wrong for AMD GPUs)
# - 2.x.x+rocm6.2 (from pytorch.org - may not work in WSL)

# Correct output:
# - 2.6.0+rocm6.4.2.git76481f7c (from repo.radeon.com - works!)
```

**Solution:**
```bash
# Run the setup script
./setup_rocm.sh

# Or manually reinstall using AMD wheels (see Python venv section above)
```

### Issue 2: "No GPU detected" in Docker

**Symptoms:**
- GPU works in Python venv but not in Docker
- Container shows `GPU Available: False`

**Diagnosis:**
```bash
# Check if WSL GPU libraries exist on host
ls -la /usr/lib/wsl/lib/libdxcore.so
ls -la /opt/rocm/lib/libhsa-runtime64.so.1

# If either is missing, reinstall AMD drivers on Windows
```

**Solution:**

1. **Verify library mounts in docker-compose.yml:**
```yaml
volumes:
  - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
  - /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1
```

2. **If libraries don't exist on host:**
   - Reinstall AMD drivers on Windows
   - Ensure ROCm components are selected
   - Run `wsl --shutdown` and restart WSL

3. **Restart Docker:**
```bash
docker-compose down
docker-compose up -d
```

### Issue 3: Permission denied on /dev/dxg

**Symptoms:**
- Error: "Permission denied: /dev/dxg"

**Solution:**
```bash
sudo chmod 666 /dev/dxg

# Make permanent (add to ~/.bashrc):
echo 'sudo chmod 666 /dev/dxg' >> ~/.bashrc
```

### Issue 4: Wrong PyTorch version after pip install

**Symptoms:**
- Ran `pip install -r requirements.txt` but got CUDA version

**Cause:**
- `requirements.txt` has generic PyTorch requirement
- pip installs CUDA version by default

**Solution:**
```bash
# Always use AMD wheels for WSL
./setup_rocm.sh
```

### Issue 5: GPU detected but models run slowly

**Symptoms:**
- `torch.cuda.is_available()` returns `True`
- Models still run slowly
- Low GPU utilization

**Possible Causes:**

1. **Model loaded on CPU:**
```python
# Check where model is loaded
# In Python console:
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("model_name")
print(model.device)  # Should show cuda:0, not cpu
```

2. **Performance logging enabled:**
- Disable in Settings tab for 5-10% speed boost

3. **VRAM limitations:**
```bash
# Check VRAM usage
rocm-smi
# If >90% full, model may be swapping to RAM
```

### Issue 6: amdgpu module not loaded

**Symptoms:**
- `lsmod | grep amdgpu` returns empty

**Is this a problem?**
**NO!** This is normal in WSL2 and not the cause of GPU issues.

**Explanation:**
- WSL2 uses DirectX GPU passthrough via `/dev/dxg`
- Does NOT use traditional Linux kernel module (`amdgpu`)
- GPU works fine without `amdgpu` module loaded
- The real issue is almost always wrong PyTorch wheel source

---

## Environment Comparison

### Python Virtual Environment

**Pros:**
- ✅ Faster development iteration
- ✅ Easier to debug
- ✅ Direct access to host GPU
- ✅ No container overhead
- ✅ Simpler dependency management

**Cons:**
- ❌ Requires manual PyTorch installation
- ❌ Potential conflicts with other Python projects
- ❌ Less reproducible across different machines

**Best for:**
- Local development
- Testing and debugging
- Rapid iteration

**Database location:** `./models.db`
**Model cache:** `~/.cache/huggingface/`

### Docker Environment

**Pros:**
- ✅ Consistent across all machines
- ✅ Isolated from host system
- ✅ Production-ready
- ✅ Easy deployment
- ✅ Version-locked dependencies

**Cons:**
- ❌ Requires WSL library mounts
- ❌ Longer initial build time (custom build)
- ❌ More complex debugging
- ❌ Slight overhead

**Best for:**
- Production deployment
- Sharing with team
- Consistent environments

**Database location:** `./data/models.db`
**Model cache:** Docker volume `hf-cache`

### Switching Between Environments

**Important:** Database and model cache locations are different!

**From venv to Docker:**
```bash
# Copy database
cp ./models.db ./data/models.db

# Models will be re-downloaded (different cache location)
```

**From Docker to venv:**
```bash
# Copy database
cp ./data/models.db ./models.db

# Models will be re-downloaded (different cache location)
```

**Alternative:** Use same paths by configuring environment variables

---

## Performance Optimization

### Environment Variables

**For Python venv** (add to `~/.bashrc` or `.env`):
```bash
# GPU selection (if multiple GPUs)
export HIP_VISIBLE_DEVICES=0

# Memory management
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# ROCm path
export ROCM_PATH=/opt/rocm
```

**For Docker** (add to `docker-compose.yml`):
```yaml
environment:
  - HIP_VISIBLE_DEVICES=0
  - PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
  - ROCM_PATH=/opt/rocm
```

### Model Selection for 16GB VRAM

| Model Size | Quantization | Performance | VRAM Usage |
|------------|--------------|-------------|------------|
| 3B params  | None         | Very Fast   | ~6GB       |
| 7B params  | None         | Fast        | ~14GB      |
| 13B params | None         | Medium      | ~26GB (won't fit) |
| 13B params | 4-bit        | Medium      | ~8GB       |
| 30B params | 4-bit        | Slower      | ~15GB      |
| 70B params | 4-bit        | Slow        | ~35GB (won't fit) |

**Recommended models:**
- `Qwen/Qwen2.5-3B-Instruct` - Fast, great for testing
- `meta-llama/Llama-3.1-8B-Instruct` - Balanced performance
- `mistralai/Mistral-7B-Instruct-v0.3` - Popular, well-tested

---

## Additional Resources

### Official Documentation
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD ROCm PyTorch Guide](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/install-pytorch.html)
- [WSL GPU Support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)

### AMD Resources
- [AMD Drivers](https://www.amd.com/en/support)
- [AMD PyTorch Wheels](https://repo.radeon.com/rocm/manylinux/)
- [ROCm Docker Images](https://hub.docker.com/r/rocm/pytorch/tags)

### Project Files
- `README.md` - Project overview
- `QUICKSTART.md` - 3-step quick start
- `SETUP_LOCAL.md` - Detailed local setup
- `DOCKER.md` - Docker deployment guide
- `FIX_WSL_GPU.md` - Troubleshooting GPU issues
- `WSL_GPU_FIX_SUMMARY.md` - GPU fix summary

---

## Common Questions

**Q: Can I use PyTorch from pytorch.org?**
A: Not recommended for WSL. AMD specifically recommends using wheels from repo.radeon.com as they are tested for WSL compatibility.

**Q: Do I need to install ROCm on WSL?**
A: No! ROCm drivers are installed on Windows and pass through to WSL. Installing ROCm inside WSL can cause conflicts.

**Q: Why does GPU work in one venv but not another?**
A: Different PyTorch sources. One has AMD wheels (works), the other has pytorch.org wheels (doesn't work in WSL).

**Q: Should I set HSA_OVERRIDE_GFX_VERSION?**
A: Not needed with proper setup. The AMD wheels and library mounts auto-detect GPU version.

**Q: Can I use CUDA?**
A: No, AMD GPUs don't support CUDA. Use ROCm (AMD's equivalent).

**Q: Why is amdgpu module not loaded?**
A: This is normal in WSL2. GPU works via DirectX passthrough, not kernel module.

---

## Getting Help

If you're still having issues:

1. **Verify basics:**
   ```bash
   ls -la /dev/dxg  # Device exists
   rocminfo | grep "Marketing Name"  # ROCm sees GPU
   python -c "import torch; print(torch.__version__)"  # Check version
   ```

2. **Run diagnostics:**
   ```bash
   python verify_gpu.py  # Full GPU test
   ```

3. **Check documentation:**
   - This guide
   - FIX_WSL_GPU.md
   - WSL_GPU_FIX_SUMMARY.md

4. **Community support:**
   - [PyTorch Forums](https://discuss.pytorch.org/)
   - [AMD ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
   - [WSL GitHub](https://github.com/microsoft/WSL)
