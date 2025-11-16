#!/bin/bash
# Quick start script for local development with ROCm

set -e

# Set ROCm environment variables for RX 7900 GRE
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

echo "=========================================="
echo "Starting LLM Server"
echo "=========================================="
echo ""
echo "GPU: AMD Radeon RX 7900 GRE (gfx1100)"
echo "ROCm Override: $HSA_OVERRIDE_GFX_VERSION"
echo ""

# Check if in venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Not in virtual environment"
    echo "Activating venv..."
    source venv/bin/activate
fi

# Quick GPU check
echo "Checking GPU..."
python -c "import torch; gpu_ok = torch.cuda.is_available(); print(f'GPU Available: {gpu_ok}'); exit(0 if gpu_ok else 1)" || {
    echo ""
    echo "❌ GPU not detected!"
    echo ""
    echo "Possible fixes:"
    echo "  1. Install ROCm PyTorch: ./setup_rocm.sh"
    echo "  2. Verify GPU access: python verify_gpu.py"
    echo ""
    exit 1
}

echo "✅ GPU detected!"
echo ""
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python main.py
