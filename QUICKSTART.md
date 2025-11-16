# Quick Start Guide

## Your System
- **GPU:** AMD Radeon RX 7900 GRE (gfx1100)
- **VRAM:** 16GB
- **Architecture:** Navi 31 (RDNA 3)
- **ROCm Version:** 11.0.0

## Important: PyTorch Version AND Source Matter!
**Issue:** GPU works in some venvs but not others?
**Root Cause:** PyTorch source AND WSL compatibility fix

The problem has TWO parts:
1. **Wheel Source:** AMD-recommended wheels (repo.radeon.com) ≠ PyTorch.org wheels
2. **WSL Fix:** Must remove bundled `libhsa-runtime64.so` for WSL compatibility

Examples:
- ❌ PyTorch 2.5.1+rocm6.2 from pytorch.org → GPU not detected
- ⚠️  PyTorch 2.9.1+rocm6.4 from pytorch.org → GPU not detected (no WSL fix!)
- ✅ PyTorch 2.6.0+rocm6.4.2 from repo.radeon.com + WSL fix → GPU works!

**Solution:** Use AMD wheels + WSL fix (automated in setup_rocm.sh)

## Fix in 3 Steps

### 1. Install ROCm PyTorch

**The Problem:** `requirements.txt` previously had `torch>=2.0.0` which installed CUDA version by default. This has been fixed.

```bash
cd /home/jare16/LLM
source venv/bin/activate

# Option A: Automated (Recommended)
./setup_rocm.sh

# Option B: Manual (AMD wheels + WSL fix)
pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
pip install *.whl
# CRITICAL: WSL fix
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*
pip install -r requirements.txt
```

### 2. Verify GPU Works
```bash
python verify_gpu.py
```

Expected output:
```
✅ GPU is available and working!
GPU 0: AMD Radeon RX 7900 GRE
Total Memory: 16.00 GB
```

### 3. Start Server
```bash
./start.sh
```

Or manually:
```bash
source venv/bin/activate
python main.py
```

Access at: http://localhost:8000

---

## Alternative: Use Docker (Recommended for Production)

Docker already has ROCm PyTorch configured:

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Check GPU in container
docker exec -it homelab-llm-server rocm-smi
```

---

## Troubleshooting

### "No GPU detected" in local venv
→ Run: `./setup_rocm.sh`

### "Permission denied: /dev/dxg"
→ Run: `sudo chmod 666 /dev/dxg`

### Model loads but slow (CPU-only)
→ Wrong PyTorch version, reinstall: `./setup_rocm.sh`

### Docker: "GPU not found"
→ Check: `docker exec -it homelab-llm-server rocm-smi`

---

## File Reference

| File | Purpose |
|------|---------|
| `setup_rocm.sh` | Install ROCm PyTorch in venv |
| `verify_gpu.py` | Check GPU detection |
| `start.sh` | Start server with correct env vars |
| `docker-compose.yml` | Run in Docker with GPU |
| `.env.example` | Environment variables reference |
| `SETUP_LOCAL.md` | Detailed local setup guide |
| `DOCKER.md` | Detailed Docker guide |

---

## Model Recommendations for 16GB VRAM

| Model Size | Quantization | Works? |
|------------|--------------|--------|
| 3B params | None | ✅ Fast |
| 7B params | None | ✅ Good |
| 13B params | None | ✅ Works |
| 30B params | 4-bit | ✅ Slower |
| 70B params | 4-bit | ⚠️ Tight fit |

Examples:
- **Qwen/Qwen2.5-3B-Instruct** - Fast, great for testing
- **meta-llama/Llama-3.1-8B-Instruct** - Balanced
- **mistralai/Mistral-7B-Instruct-v0.3** - Popular choice
- **TheBloke/Llama-2-13B-chat-GPTQ** - Larger, quantized

---

## Quick Commands

```bash
# Local development
source venv/bin/activate
./start.sh

# Docker
docker-compose up -d
docker-compose logs -f
docker-compose down

# Verify GPU
python verify_gpu.py

# Fix PyTorch
./setup_rocm.sh
```

---

## Performance Tips

1. **Disable logging** in UI Settings → 5-10% speed boost
2. **Use Flash Attention** (automatic in newer transformers)
3. **Set optimal env vars** (already in `start.sh` and `docker-compose.yml`)
4. **Monitor VRAM:** Keep 1-2GB free for overhead

---

## Next Steps

- [ ] Fix PyTorch: `./setup_rocm.sh`
- [ ] Verify GPU: `python verify_gpu.py`
- [ ] Start server: `./start.sh`
- [ ] Register model in UI
- [ ] Test inference

🎯 **Goal:** See "GPU Available: True" and fast inference!
