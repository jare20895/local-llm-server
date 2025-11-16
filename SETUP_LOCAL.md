# Local Development Setup with ROCm

This guide covers setting up the LLM server for local development with AMD GPUs and ROCm in WSL2.

## Problem: Wrong PyTorch Version or Source

If you see "No GPU detected", you likely have one of these issues:
1. CUDA version of PyTorch instead of ROCm
2. PyTorch.org wheels instead of AMD-recommended repo.radeon.com wheels
3. Missing WSL runtime lib fix

```bash
# Check current version
python -c "import torch; print(torch.__version__)"
# ❌ Output like: 2.9.1+cu128  (CUDA version - wrong for AMD GPUs)
# ⚠️  Output like: 2.9.1+rocm6.4 (PyTorch.org - may not work in WSL)
# ✅ Output should be: 2.6.0+rocm6.4.2.git76481f7c (AMD wheels - works in WSL!)
```

**Key Insight:** AMD recommends using wheels from repo.radeon.com, NOT pytorch.org. The AMD wheels include critical WSL compatibility fixes.

## Quick Fix: Install ROCm PyTorch

### Option 1: Automated Script (Recommended)

```bash
# Activate your venv
source venv/bin/activate

# Run the setup script
./setup_rocm.sh

# Verify installation
python verify_gpu.py
```

### Option 2: Manual Installation

```bash
# Activate venv
source venv/bin/activate

# Uninstall CUDA/wrong PyTorch
pip uninstall torch torchvision torchaudio pytorch-triton-rocm

# Download AMD-recommended wheels (Python 3.12)
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
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*

# Install other dependencies
pip install -r requirements-rocm.txt

# Verify
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

## Verification Steps

### 1. Check GPU Device Access

```bash
# WSL2 uses /dev/dxg for GPU
ls -la /dev/dxg

# Should show:
# crw-rw-rw- 1 root root 10, 62 ...

# If permission denied:
sudo chmod 666 /dev/dxg
```

### 2. Check ROCm Installation

```bash
# Check ROCm version
rocm-smi

# Should display your AMD GPU info
```

### 3. Run Verification Script

```bash
python verify_gpu.py
```

Expected output:
```
==============================================================
PyTorch Information
==============================================================
  PyTorch Version: 2.x.x+rocmX.X
  ROCm Version: X.X.X
  GPU Available: True
  GPU Count: 1
  GPU 0: AMD Radeon ...
    - Total Memory: XX.XX GB
    - Compute Capability: X.X

==============================================================
GPU Computation Test
==============================================================
  Using device: cuda:0
  Creating tensor on GPU...
  Performing matrix multiplication...
  Synchronizing...
  ✅ Success! Result shape: torch.Size([1000, 1000])
```

## Common Issues

### Issue 1: "No GPU detected" in PyTorch

**Cause:** Wrong PyTorch version (CUDA instead of ROCm)

**Fix:**
```bash
./setup_rocm.sh
```

### Issue 2: "/dev/dxg: Permission denied"

**Cause:** Insufficient permissions on GPU device

**Fix:**
```bash
sudo chmod 666 /dev/dxg
```

### Issue 3: "rocm-smi: command not found"

**Cause:** ROCm not installed on host system

**Fix:** Install ROCm drivers on Windows (for WSL2)
- Download from: https://www.amd.com/en/support
- Restart WSL after installation: `wsl --shutdown`

### Issue 4: Model loads but runs slowly (CPU-only)

**Symptoms:**
- Model loads successfully
- Frontend shows "No GPU detected"
- Inference is very slow

**Cause:** Model loaded on CPU instead of GPU

**Fix:**
1. Verify GPU is available: `python verify_gpu.py`
2. Check PyTorch version: Must be ROCm, not CUDA
3. Reinstall correct PyTorch: `./setup_rocm.sh`

## Environment Variables (Optional)

For better performance, you can set these before running:

```bash
# GPU selection (if you have multiple GPUs)
export HIP_VISIBLE_DEVICES=0

# Memory management
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# ROCm path
export ROCM_PATH=/opt/rocm

# Then run
python main.py
```

Or create a `.env` file:
```bash
HIP_VISIBLE_DEVICES=0
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
ROCM_PATH=/opt/rocm
```

## Running the Server

Once GPU is detected:

```bash
# Activate venv
source venv/bin/activate

# Start server
python main.py

# Access at http://localhost:8000
```

## Performance Tips

1. **Disable performance logging** for 5-10% speed boost (Settings tab in UI)

2. **Use appropriate model sizes** for your VRAM:
   - 8GB VRAM: Up to 7B parameter models
   - 12GB VRAM: Up to 13B parameter models
   - 16GB+ VRAM: 30B+ parameter models (with quantization)

3. **Enable Flash Attention** if supported (automatic in transformers)

4. **Use quantization** for larger models:
   - 4-bit: 4x memory reduction
   - 8-bit: 2x memory reduction

## Switching Between Docker and Local

### Use Docker:
```bash
docker-compose up -d
```
- Data in `./data/models.db`
- Models in Docker volume `hf-cache`

### Use Local:
```bash
source venv/bin/activate
python main.py
```
- Data in `./models.db` (different location!)
- Models in `~/.cache/huggingface/`

**Note:** Database locations are different, so registered models won't sync between Docker and local unless you:
- Set `DATABASE_PATH=/home/jare16/LLM/data/models.db` for local
- Or mount the same database path in Docker

## Next Steps

- ✅ GPU detected → Start using: `python main.py`
- ❌ GPU not detected → Run: `python verify_gpu.py` and check output
- 📝 More help → See DOCKER.md for containerized setup

## Resources

- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [WSL GPU Support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
