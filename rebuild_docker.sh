#!/bin/bash
# Rebuild Docker image with fixed PyTorch configuration

set -e

echo "=========================================="
echo "Rebuilding Docker Image"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Stop current container (if running)"
echo "  2. Remove old image"
echo "  3. Rebuild with ROCm PyTorch (no CUDA override)"
echo "  4. Start container"
echo "  5. Verify GPU detection"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Stopping and removing old container..."
docker-compose down

echo ""
echo "Step 2: Removing old image..."
docker rmi homelab-llm-server_llm-server 2>/dev/null || true
docker rmi $(docker images -q rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.3.0) 2>/dev/null || echo "Base image will be reused"

echo ""
echo "Step 3: Building new image..."
docker-compose build --no-cache

echo ""
echo "Step 4: Starting container..."
docker-compose up -d

echo ""
echo "Waiting for container to start..."
sleep 5

echo ""
echo "Step 5: Verifying GPU detection..."
echo "=========================================="
echo "PyTorch Version Check:"
docker exec homelab-llm-server python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.version.hip if hasattr(torch.version, \"hip\") else \"Not found\"}'); print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "ROCm SMI Output:"
docker exec homelab-llm-server rocm-smi

echo ""
echo "=========================================="
echo "✅ Rebuild complete!"
echo ""
echo "Expected results:"
echo "  - PyTorch: 2.3.0+rocm6.2 (or similar)"
echo "  - GPU Available: True"
echo "  - GPU: AMD Radeon RX 7900 GRE"
echo ""
echo "Access the server at: http://localhost:8000"
echo "View logs: docker-compose logs -f"
echo ""
