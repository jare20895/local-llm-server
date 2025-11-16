# WSL GPU Detection Fix - Summary

## Problem
GPU was detected in one venv but not another, despite both having "ROCm" PyTorch installed.

## Root Cause Analysis

The issue had **TWO critical components**:

### 1. PyTorch Wheel Source
- **Wrong:** PyTorch.org wheels (`--index-url https://download.pytorch.org/whl/rocm6.4`)
- **Correct:** AMD repo.radeon.com wheels

AMD specifically states in their documentation:
> "AMD recommends proceeding with ROCm WHLs available at repo.radeon.com. The ROCm WHLs available at PyTorch.org are not tested extensively by AMD as the WHLs change regularly when the nightly builds are updated."

### 2. WSL Runtime Library Fix
Even with correct wheels, WSL requires removing the bundled HSA runtime library:
```bash
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*
```

**Why this is needed:** WSL uses the host system's HSA runtime library (via `/dev/dxg`), and the bundled library conflicts with this.

## The Solution

### Automated (Recommended)
```bash
cd /home/jare16/LLM
source venv/bin/activate
./setup_rocm.sh
```

This script:
1. Uninstalls any existing PyTorch
2. Downloads AMD-recommended wheels from repo.radeon.com
3. Installs the wheels
4. **Applies WSL runtime lib fix** (removes libhsa-runtime64.so)
5. Installs other dependencies
6. Verifies GPU detection

### Manual Installation
```bash
# 1. Download AMD wheels (Python 3.12)
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

# 2. Install wheels
pip install torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl \
            torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl \
            torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl \
            pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

# 3. CRITICAL WSL FIX
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*

# 4. Verify
python verify_gpu.py
```

## Version Comparison

| Source | Version | WSL Fix | GPU Detected? |
|--------|---------|---------|---------------|
| pytorch.org | 2.5.1+rocm6.2 | ❌ | ❌ No |
| pytorch.org | 2.9.1+rocm6.4 | ❌ | ❌ No |
| repo.radeon.com | 2.6.0+rocm6.4.2 | ❌ | ❌ No |
| repo.radeon.com | 2.6.0+rocm6.4.2 | ✅ | ✅ **YES!** |

## Expected Output After Fix

```
============================================================
PyTorch Information
============================================================
  PyTorch Version: 2.6.0+rocm6.4.2.git76481f7c
  ROCm Version: 6.4.43484-123eb5128
  GPU Available: True
  GPU Count: 1
  GPU 0: AMD Radeon RX 7900 GRE
    - Total Memory: 15.98 GB
    - Compute Capability: 11.0

============================================================
GPU Computation Test
============================================================
  ✅ Success! Result shape: torch.Size([1000, 1000])
```

## Docker Setup

### Critical Docker Requirements for WSL

Docker requires **additional volume mounts** beyond just `/dev/dxg`:

```yaml
devices:
  - /dev/dxg:/dev/dxg

volumes:
  # WSL GPU libraries (CRITICAL!)
  - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
  - /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1
```

**Why these are needed:**
- `libdxcore.so` - DirectX core library for WSL GPU passthrough
- `libhsa-runtime64.so.1` - HSA runtime from host system

**Critical:** When using these library mounts, do **NOT** set `HSA_OVERRIDE_GFX_VERSION` in the environment. The mounted `libhsa-runtime64.so.1` automatically detects the correct GPU version.

**Without these mounts, GPU won't be detected in Docker!**

### Running Docker

Two options available:

#### Option 1: Pre-built AMD Image (Recommended)
```bash
docker-compose up -d
```
Uses `rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0`

#### Option 2: Custom Build
```bash
docker-compose -f docker-compose-build.yml up -d
```
Builds from Dockerfile (based on same AMD image)

Both compose files now include the required WSL library mounts.

## Updated Files

All documentation and setup files updated:
- ✅ `setup_rocm.sh` - Uses AMD wheels + WSL fix
- ✅ `requirements.txt` - AMD wheel instructions
- ✅ `requirements-rocm.txt` - Detailed AMD wheel setup
- ✅ `SETUP_LOCAL.md` - Updated with AMD wheels
- ✅ `QUICKSTART.md` - Clear explanation of the issue
- ✅ `README.md` - Troubleshooting section updated
- ✅ `DOCKER.md` - Two compose options documented
- ✅ `Dockerfile` - Uses AMD-recommended base image
- ✅ `docker-compose.yml` - Pre-built AMD image
- ✅ `docker-compose-build.yml` - Custom build option

## References

- [AMD ROCm PyTorch Installation Guide](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/install-pytorch.html)
- [AMD PyTorch Wheels](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/)
- [WSL GPU Support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

## System Info

- **GPU:** AMD Radeon RX 7900 GRE (gfx1100, RDNA 3)
- **VRAM:** 16GB
- **OS:** WSL2 on Windows
- **Python:** 3.12
- **Working PyTorch:** 2.6.0+rocm6.4.2.git76481f7c
- **ROCm:** 6.4.43484-123eb5128
