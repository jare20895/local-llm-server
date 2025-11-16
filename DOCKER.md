# Docker Deployment Guide

This guide covers running the Homelab LLM Server in Docker with ROCm GPU support.

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **ROCm drivers** installed on host system (Windows side for WSL)
3. **AMD GPU** with ROCm support
4. Running in **WSL2** with GPU passthrough enabled

### WSL-Specific Setup

For WSL2, ensure GPU passthrough is working and required libraries exist:

```bash
# 1. Check GPU device is accessible
ls -la /dev/dxg
# Should show: crw-rw-rw- 1 root root 10, 125 ...

# If permission denied:
sudo chmod 666 /dev/dxg

# 2. Verify WSL GPU libraries exist (CRITICAL for Docker GPU detection)
ls -la /usr/lib/wsl/lib/libdxcore.so
ls -la /opt/rocm/lib/libhsa-runtime64.so.1

# Both should exist. If not, reinstall AMD drivers on Windows
```

**Why these libraries are critical:**
- `/dev/dxg` - WSL GPU device
- `libdxcore.so` - DirectX core library for WSL GPU passthrough
- `libhsa-runtime64.so.1` - HSA runtime from host (mounted into container)

**The docker-compose files automatically mount these libraries into the container.**
Without these mounts, PyTorch won't detect the GPU in Docker even with the correct PyTorch version!

## Configuration

### GPU Detection (No Manual Configuration Needed!)

When using the recommended WSL library mounts, **no manual GPU version setting is needed**:

```yaml
volumes:
  - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
  - /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1
```

**Important:** Do NOT set `HSA_OVERRIDE_GFX_VERSION` in Docker when using library mounts. The mounted `libhsa-runtime64.so.1` from your host system automatically detects the correct GPU version.

<details>
<summary>Legacy: Manual GPU Version Override (only if library mounts don't work)</summary>

If for some reason the library mounts don't work on your system, you can manually set:

```yaml
environment:
  - HSA_OVERRIDE_GFX_VERSION=11.0.0  # RX 7900 series
```

Common values:
- **RX 6800/6900 XT (Navi 21)**: `10.3.0`
- **RX 7900 XT/XTX (Navi 31)**: `11.0.0`
- **RX 6700 XT (Navi 22)**: `10.3.0`
- **RX 5700 XT (Navi 10)**: `10.1.0`

But this is rarely needed with proper library mounts!
</details>

### Hugging Face Token (Optional)

If you need to access private or gated models:

1. Get your token from https://huggingface.co/settings/tokens
2. Uncomment and set in `docker-compose.yml`:
   ```yaml
   - HF_TOKEN=your_token_here
   ```

## Docker Compose Options

This project provides two Docker Compose configurations:

### Option 1: Pre-built AMD Image (RECOMMENDED)
**File:** `docker-compose.yml`

Uses AMD's official pre-built image directly - fastest and most reliable.

```bash
# Start using AMD's official image (recommended)
docker-compose up -d

# View logs
docker-compose logs -f llm-server
```

**Advantages:**
- ✅ No build time (starts immediately)
- ✅ AMD-tested and verified
- ✅ Includes all WSL compatibility fixes
- ✅ Most reliable option

**How it works:**
- Uses `image: rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0`
- Mounts your code as a volume
- Installs Python dependencies on first start

### Option 2: Custom Build
**File:** `docker-compose-build.yml`

Builds a custom image from the Dockerfile.

```bash
# Build and start custom image
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

### Starting/Stopping

```bash
# Using pre-built image (default)
docker-compose up -d
docker-compose down
docker-compose restart

# Using custom build
docker-compose -f docker-compose-build.yml up -d
docker-compose -f docker-compose-build.yml down
docker-compose -f docker-compose-build.yml restart
```

### Accessing the Server

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Data Persistence

The following data is persisted between container restarts:

1. **Database**: `./data/models.db` - Model registry and performance logs
2. **HuggingFace Cache**: Docker volume `hf-cache` - Downloaded models

## Checking GPU Access

### Inside Docker Container:
```bash
docker exec -it homelab-llm-server bash
rocm-smi

# Or check PyTorch GPU access:
docker exec -it homelab-llm-server python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### On Host (WSL):
```bash
# Verify GPU device exists
ls -la /dev/dxg

# Check ROCm installation
rocm-smi
```

## Troubleshooting

### GPU Not Detected

1. Verify ROCm installation on host:
   ```bash
   rocm-smi
   ```

2. Check device permissions:
   ```bash
   ls -la /dev/kfd /dev/dri
   ```

3. Ensure you're in the `video` and `render` groups:
   ```bash
   groups
   ```

### Permission Issues

If you get permission errors accessing `/dev/kfd` or `/dev/dri`:

```bash
# Add your user to the groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Log out and back in, then restart Docker
```

### Out of Memory

If models fail to load due to VRAM:

1. Check available VRAM:
   ```bash
   docker exec -it homelab-llm-server rocm-smi
   ```

2. Try smaller models or enable quantization in the web UI

### Container Won't Start

Check logs for detailed error messages:
```bash
docker-compose logs
```

Common issues:
- ROCm version mismatch (check HSA_OVERRIDE_GFX_VERSION)
- Insufficient permissions for GPU devices
- Port 8000 already in use

## Performance Optimization

### Environment Variables

Adjust in `docker-compose.yml`:

```yaml
environment:
  # Increase for better memory management
  - PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256

  # Use specific GPU if you have multiple
  - HIP_VISIBLE_DEVICES=0,1  # Use first two GPUs
```

### Disable Performance Logging

For 5-10% speed improvement, disable detailed metrics via API or Settings tab in UI.

## Updating

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

## Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes downloaded models and database)
docker-compose down -v

# Remove images
docker rmi homelab-llm-server
```

## Docker Image Information

The Dockerfile now uses AMD's recommended image:
```dockerfile
FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

**Why this specific image?**
- AMD officially tests and recommends ROCm 6.4.2 for RX 7900 series
- Includes PyTorch 2.6.0 with WSL compatibility fixes
- Python 3.12 matches the project requirements
- Much more reliable than 'latest' tag

**For other Python versions:**
- Python 3.10: `rocm/pytorch:rocm6.4.2_ubuntu22.04_py3.10_pytorch_release_2.6.0`
- Python 3.11: `rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.11_pytorch_release_2.6.0`

Available tags: https://hub.docker.com/r/rocm/pytorch/tags

**Note:** For RX 7900 GRE and other RDNA 3 GPUs, ROCm 6.4.2 is the recommended minimum version.

## Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [Docker ROCm Support](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
