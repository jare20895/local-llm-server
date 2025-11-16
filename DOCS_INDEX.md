# Documentation Index

Complete documentation for the Homelab LLM Server with ROCm GPU support on WSL2.

## Quick Navigation

### Getting Started
| Document | Purpose | When to Use |
|----------|---------|-------------|
| [README.md](README.md) | Project overview and features | First-time visitors |
| [QUICKSTART.md](QUICKSTART.md) | 3-step quick start guide | Want to get running fast |
| **[ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md)** | **Complete ROCm setup guide** | **Setting up AMD GPU in WSL2** |

### Setup Guides
| Document | Purpose | Environment |
|----------|---------|-------------|
| **[ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md)** | **Comprehensive guide covering Python venv AND Docker** | **Both** |
| [SETUP_LOCAL.md](SETUP_LOCAL.md) | Detailed local development setup | Python venv |
| [DOCKER.md](DOCKER.md) | Docker deployment guide | Docker |

### Troubleshooting
| Document | Purpose | When to Use |
|----------|---------|-------------|
| [FIX_WSL_GPU.md](FIX_WSL_GPU.md) | WSL GPU detection troubleshooting | GPU not detected |
| [WSL_GPU_FIX_SUMMARY.md](WSL_GPU_FIX_SUMMARY.md) | Summary of GPU fix (PyTorch wheels) | Quick reference |
| **[ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md)** | **Complete troubleshooting section** | **Comprehensive diagnostics** |

---

## Common Scenarios

### "I just want to get started quickly"
→ [QUICKSTART.md](QUICKSTART.md)

### "I need to set up AMD GPU on WSL2"
→ **[ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md)** - Start here!

### "GPU is not detected"
→ **[ROCM_SETUP_GUIDE.md - Troubleshooting](ROCM_SETUP_GUIDE.md#troubleshooting)** or [FIX_WSL_GPU.md](FIX_WSL_GPU.md)

### "I want to use Python virtual environment"
→ **[ROCM_SETUP_GUIDE.md - Python Setup](ROCM_SETUP_GUIDE.md#python-virtual-environment-setup)**

### "I want to use Docker"
→ **[ROCM_SETUP_GUIDE.md - Docker Setup](ROCM_SETUP_GUIDE.md#docker-setup)** or [DOCKER.md](DOCKER.md)

### "GPU works in one venv but not another"
→ **[ROCM_SETUP_GUIDE.md - Issue 1](ROCM_SETUP_GUIDE.md#issue-1-no-gpu-detected-in-python-venv)**

### "GPU works locally but not in Docker"
→ **[ROCM_SETUP_GUIDE.md - Issue 2](ROCM_SETUP_GUIDE.md#issue-2-no-gpu-detected-in-docker)**

### "Should I use venv or Docker?"
→ **[ROCM_SETUP_GUIDE.md - Environment Comparison](ROCM_SETUP_GUIDE.md#environment-comparison)**

---

## Documentation Overview

### Core Documentation

#### [README.md](README.md)
- Project overview and architecture
- Feature list
- API documentation
- Basic installation
- Performance metrics
- Troubleshooting summary

**Best for:** Understanding what the project does

---

#### [ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md) ⭐ **NEW & COMPREHENSIVE**
**Complete setup guide covering all ROCm environments**

**Contents:**
1. **Prerequisites** - Windows driver installation, WSL2 setup
2. **Python Virtual Environment Setup** - Step-by-step venv setup with AMD wheels
3. **Docker Setup** - Both pre-built AMD image and custom build options
4. **Troubleshooting** - 6 common issues with detailed solutions
5. **Environment Comparison** - Venv vs Docker pros/cons

**Key Topics:**
- Why AMD wheels matter (PyTorch source issue)
- WSL runtime library fix explained
- Docker GPU passthrough requirements
- Performance optimization
- Model recommendations for different VRAM sizes

**Best for:**
- Setting up AMD GPU in WSL2 (venv or Docker)
- Understanding ROCm environment differences
- Comprehensive troubleshooting

---

#### [QUICKSTART.md](QUICKSTART.md)
- 3-step quick start
- System specs
- Quick commands
- Model recommendations
- Performance tips

**Best for:** Getting running quickly if you know what you're doing

---

### Setup Guides

#### [SETUP_LOCAL.md](SETUP_LOCAL.md)
- Python virtual environment setup
- ROCm PyTorch installation (AMD wheels)
- WSL runtime library fix
- Verification steps
- Common issues
- Environment variables
- Performance tips

**Best for:** Local development with Python venv

---

#### [DOCKER.md](DOCKER.md)
- Docker prerequisites
- WSL GPU library requirements
- Pre-built AMD image setup
- Custom build setup
- GPU verification in Docker
- Performance optimization
- Docker-specific troubleshooting

**Best for:** Docker deployment

---

### Troubleshooting Guides

#### [FIX_WSL_GPU.md](FIX_WSL_GPU.md)
**WSL GPU detection troubleshooting**

**Updated to reflect:**
- ✅ PyTorch wheel source as PRIMARY issue
- ✅ WSL runtime library fix as secondary requirement
- ✅ De-emphasizes amdgpu module (not the real issue)

**Contents:**
- Root cause analysis (PyTorch wheels + WSL fix)
- Step-by-step fix instructions
- Alternative WSL ROCm setup
- Known limitations
- Workarounds

**Best for:** Diagnosing GPU detection issues

---

#### [WSL_GPU_FIX_SUMMARY.md](WSL_GPU_FIX_SUMMARY.md)
- Concise summary of GPU fix
- PyTorch wheel comparison table
- Docker requirements
- Updated files reference

**Best for:** Quick reference after initial setup

---

## Key Insights for WSL/ROCm

### The #1 Issue: PyTorch Wheel Source

**Problem:**
Many users install PyTorch with `pip install torch --index-url https://download.pytorch.org/whl/rocm6.4`

**Why this fails in WSL:**
- PyTorch.org wheels are NOT extensively tested by AMD
- Missing WSL-specific compatibility fixes
- Even with "ROCm" in version string, may not work in WSL

**Solution:**
Use AMD-recommended wheels from repo.radeon.com:
```bash
./setup_rocm.sh  # Automated
```

**How to verify:**
```bash
python -c "import torch; print(torch.__version__)"

# ❌ Wrong: 2.5.1+rocm6.2 (from pytorch.org)
# ❌ Wrong: 2.9.1+cu128 (CUDA version)
# ✅ Correct: 2.6.0+rocm6.4.2.git76481f7c (from repo.radeon.com)
```

### The #2 Issue: WSL Runtime Library

**Problem:**
PyTorch bundles `libhsa-runtime64.so` which conflicts with WSL's GPU passthrough

**Solution:**
Remove the bundled library:
```bash
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*
```

**Automated in:** `./setup_rocm.sh`

### The #3 Issue: Docker GPU Passthrough

**Problem:**
Docker needs host GPU libraries mounted, not just `/dev/dxg`

**Solution:**
Add to `docker-compose.yml`:
```yaml
volumes:
  - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
  - /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1
```

**Already configured in:** Both `docker-compose.yml` files

### Common Misconception: amdgpu Module

**Misconception:**
"GPU doesn't work because amdgpu module isn't loaded"

**Reality:**
- WSL2 uses DirectX GPU passthrough via `/dev/dxg`
- Does NOT require `amdgpu` kernel module
- `lsmod | grep amdgpu` being empty is NORMAL
- Real issue: Wrong PyTorch wheel source

---

## File Structure

```
/home/jare16/LLM/
├── README.md                    # Project overview
├── DOCS_INDEX.md               # This file (navigation)
├── ROCM_SETUP_GUIDE.md         # ⭐ Comprehensive ROCm setup
├── QUICKSTART.md               # Quick start guide
├── SETUP_LOCAL.md              # Local Python venv setup
├── DOCKER.md                   # Docker deployment
├── FIX_WSL_GPU.md             # WSL GPU troubleshooting
├── WSL_GPU_FIX_SUMMARY.md     # GPU fix summary
├── setup_rocm.sh              # Automated PyTorch setup
├── verify_gpu.py              # GPU verification script
├── start.sh                   # Start server script
├── docker-compose.yml         # Pre-built AMD image
├── docker-compose-build.yml   # Custom build
└── Dockerfile                 # Custom image definition
```

---

## Quick Reference

### Essential Commands

**Verify GPU:**
```bash
python verify_gpu.py
```

**Install PyTorch (AMD wheels):**
```bash
./setup_rocm.sh
```

**Start server (local):**
```bash
./start.sh
```

**Start server (Docker - pre-built):**
```bash
docker-compose up -d
```

**Start server (Docker - custom build):**
```bash
docker-compose -f docker-compose-build.yml up -d
```

### Diagnostic Commands

**Check PyTorch version:**
```bash
python -c "import torch; print(torch.__version__)"
```

**Check GPU available:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Check WSL GPU device:**
```bash
ls -la /dev/dxg
```

**Check ROCm:**
```bash
rocm-smi
rocminfo | grep "Marketing Name"
```

**Check GPU in Docker:**
```bash
docker exec -it homelab-llm-server python -c "import torch; print(torch.cuda.is_available())"
```

---

## External Resources

### Official Documentation
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD ROCm PyTorch Installation](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/install-pytorch.html)
- [Microsoft WSL GPU Documentation](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [PyTorch Official Site](https://pytorch.org/)

### AMD Resources
- [AMD Driver Downloads](https://www.amd.com/en/support)
- [AMD PyTorch Wheels Repository](https://repo.radeon.com/rocm/manylinux/)
- [ROCm Docker Hub](https://hub.docker.com/r/rocm/pytorch/tags)

### Community
- [PyTorch Discussion Forums](https://discuss.pytorch.org/)
- [AMD ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
- [WSL GitHub Issues](https://github.com/microsoft/WSL/issues)

---

## Version Information

**Tested Configuration:**
- **GPU:** AMD Radeon RX 7900 GRE (RDNA 3, gfx1100)
- **VRAM:** 16GB
- **OS:** Windows 11 + WSL2 (Ubuntu)
- **Python:** 3.12
- **PyTorch:** 2.6.0+rocm6.4.2.git76481f7c (from repo.radeon.com)
- **ROCm:** 6.4.43484-123eb5128
- **Docker Base:** rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

**Supported AMD GPUs:**
- RX 6000 series (RDNA 2)
- RX 7000 series (RDNA 3)
- Other ROCm-compatible AMD GPUs

---

## Contributing

When updating documentation:
1. Update this index if adding new docs
2. Ensure cross-references are updated
3. Test all commands and code samples
4. Keep troubleshooting sections current

---

**Last Updated:** 2025-11-16

**Need help?** Start with [ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md) for comprehensive setup and troubleshooting.
