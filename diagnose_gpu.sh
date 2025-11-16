#!/bin/bash
# GPU Diagnostic Script for WSL + ROCm

echo "============================================================"
echo "WSL GPU Diagnostic Tool"
echo "============================================================"
echo ""

# Test 1: Device Node
echo "Test 1: GPU Device Node"
echo "----------------------------"
if [ -e /dev/dxg ]; then
    ls -la /dev/dxg
    echo "✅ /dev/dxg exists"
else
    echo "❌ /dev/dxg NOT FOUND"
    echo "   → Windows AMD drivers may not be installed"
fi
echo ""

# Test 2: Kernel Modules
echo "Test 2: Kernel Modules"
echo "----------------------------"
if lsmod | grep -q amdgpu; then
    echo "✅ amdgpu module loaded:"
    lsmod | grep amdgpu
else
    echo "❌ amdgpu module NOT loaded"
    echo "   → This is the main issue!"
    echo "   → WSL kernel doesn't have AMD compute drivers"
fi
echo ""

# Test 3: ROCm HSA
echo "Test 3: ROCm HSA Layer"
echo "----------------------------"
if command -v rocminfo &> /dev/null; then
    if rocminfo 2>/dev/null | grep -q "gfx1100"; then
        echo "✅ rocminfo detects GPU:"
        rocminfo 2>/dev/null | grep "Marketing Name"
    else
        echo "❌ rocminfo doesn't detect GPU"
    fi
else
    echo "⚠️  rocminfo not installed"
fi
echo ""

# Test 4: PyTorch ROCm
echo "Test 4: PyTorch GPU Detection"
echo "----------------------------"
if [ -d "venv" ]; then
    source venv/bin/activate
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
if '+rocm' in torch.__version__:
    print('✅ ROCm PyTorch installed')
else:
    print('❌ CUDA PyTorch (wrong version)')
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print('✅ GPU detected by PyTorch!')
else:
    print('❌ GPU NOT detected by PyTorch')
"
else
    echo "⚠️  venv not found"
fi
echo ""

# Test 5: WSL Kernel Version
echo "Test 5: WSL Kernel"
echo "----------------------------"
uname -a
echo ""

# Test 6: Docker GPU Access
echo "Test 6: Docker GPU Access"
echo "----------------------------"
if command -v docker &> /dev/null; then
    if docker ps | grep -q homelab-llm-server; then
        echo "Testing GPU in Docker container..."
        docker exec homelab-llm-server python -c "import torch; print('GPU in Docker:', torch.cuda.is_available())" 2>/dev/null || echo "Container not running or Python error"
    else
        echo "⚠️  Container not running"
    fi
else
    echo "⚠️  Docker not installed"
fi
echo ""

# Summary
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""
echo "Issue Breakdown:"
echo ""
if [ -e /dev/dxg ]; then
    echo "✅ GPU device exists in WSL (/dev/dxg)"
else
    echo "❌ GPU device missing - Install Windows AMD drivers"
fi

if lsmod | grep -q amdgpu; then
    echo "✅ Kernel module loaded (amdgpu)"
else
    echo "❌ Kernel module NOT loaded (amdgpu)"
    echo "   → MAIN PROBLEM: WSL kernel can't access GPU compute"
    echo "   → Likely cause: Windows AMD driver doesn't support WSL"
    echo "   → or Docker Desktop broke the passthrough"
fi

if command -v rocminfo &> /dev/null && rocminfo 2>/dev/null | grep -q "gfx1100"; then
    echo "✅ HSA layer can see GPU (rocminfo works)"
else
    echo "❌ HSA layer can't see GPU"
fi

echo ""
echo "Recommended Actions:"
echo ""

if ! lsmod | grep -q amdgpu; then
    echo "1. ⚠️  CRITICAL: Fix Windows AMD driver installation"
    echo "   → See FIX_WSL_GPU.md for detailed steps"
    echo "   → Uninstall AMD drivers completely (use AMD Cleanup Utility)"
    echo "   → Reinstall latest AMD drivers with WSL support"
    echo "   → Run 'wsl --shutdown' and restart WSL"
    echo ""
    echo "2. If drivers are correct, this may be a WSL2 limitation:"
    echo "   → Consider using native Linux"
    echo "   → Or use CPU mode temporarily"
fi

echo ""
echo "Next Steps:"
echo "  - Read: FIX_WSL_GPU.md"
echo "  - Run this script again after fixing Windows drivers"
echo ""
