#!/bin/bash
# Setup script for ROCm PyTorch in local venv

set -e

echo "=========================================="
echo "ROCm PyTorch Installation Script"
echo "=========================================="

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: Not in a virtual environment!"
    echo "Please activate your venv first:"
    echo "  source venv/bin/activate"
    exit 1
fi

echo ""
echo "Current PyTorch installation:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  ROCm: {torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch not installed or import failed"

echo ""
echo "This will:"
echo "  1. Uninstall current PyTorch (CUDA version)"
echo "  2. Install ROCm-compatible PyTorch"
echo "  3. Install other dependencies from requirements.txt"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Uninstalling CUDA PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo "Step 2: Downloading AMD-recommended PyTorch wheels..."
# AMD recommends using wheels from repo.radeon.com instead of pytorch.org
# These are tested more extensively and include WSL compatibility fixes
wget -q https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
wget -q https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
wget -q https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
wget -q https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

echo ""
echo "Step 3: Installing AMD PyTorch wheels..."
pip install torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl \
            torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl \
            torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl \
            pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

echo ""
echo "Step 4: Applying WSL runtime lib fix..."
# Critical fix for WSL: Remove bundled libhsa-runtime64.so
# WSL uses the host system's HSA runtime library instead
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*
echo "  Removed bundled libhsa-runtime64.so (WSL compatibility)"

echo ""
echo "Step 5: Installing other dependencies..."
pip install fastapi uvicorn[standard] pydantic sqlmodel sqlalchemy transformers accelerate psutil python-multipart

echo ""
echo "Step 6: Cleaning up wheel files..."
rm -f torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
rm -f torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
rm -f torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
rm -f pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'ROCm Available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'hip'):
    print(f'ROCm Version: {torch.version.hip}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
else:
    print('WARNING: No GPU detected!')
    print('This could mean:')
    print('  - ROCm drivers not installed')
    print('  - GPU not accessible')
    print('  - Need to check /dev/dxg permissions in WSL')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify GPU access: python verify_gpu.py"
echo "  2. Start the server: python main.py"
echo ""
