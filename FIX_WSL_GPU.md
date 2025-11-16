# Fix WSL GPU Access After Docker Desktop Installation

## Problem
GPU worked in WSL → Installed Docker Desktop → Display issues → Reinstalled AMD drivers → WSL GPU broken

## Root Cause
The issue has **TWO critical components**:

### 1. Wrong PyTorch Wheel Source (PRIMARY ISSUE)
- **Problem:** Using PyTorch.org wheels instead of AMD-recommended repo.radeon.com wheels
- **Impact:** PyTorch.org wheels don't include all WSL compatibility fixes
- **Solution:** Must use official AMD wheels from repo.radeon.com

AMD specifically states:
> "AMD recommends proceeding with ROCm WHLs available at repo.radeon.com. The ROCm WHLs available at PyTorch.org are not tested extensively by AMD as the WHLs change regularly when the nightly builds are updated."

### 2. WSL Runtime Library Conflict
- **Problem:** PyTorch's bundled `libhsa-runtime64.so` conflicts with WSL's GPU passthrough
- **Impact:** Even with correct wheels, GPU won't be detected without removing this library
- **Solution:** Remove bundled library to use host system's HSA runtime

**Note:** While the amdgpu module status can indicate GPU issues, the primary problem is using the wrong PyTorch wheels and not applying the WSL runtime library fix.

## Solution

### On Windows (Host)

#### 1. Completely Uninstall AMD Drivers
1. Download **AMD Cleanup Utility**: https://www.amd.com/en/support/kb/faq/gpu-601
2. Run it to completely remove AMD drivers
3. Restart Windows

#### 2. Install Latest AMD Drivers
1. Download latest **AMD Software: Adrenalin Edition** for RX 7900 series
   - https://www.amd.com/en/support/graphics/amd-radeon-7000-series/amd-radeon-7900-series/amd-radeon-rx-7900-gre
2. **Important:** During installation, ensure these are selected:
   - AMD Software
   - AMD Display Driver
   - **ROCm** (if available as option)
3. Restart Windows

#### 3. Verify Windows GPU Works
1. Open **Device Manager**
2. Check **Display adapters** → Should show "AMD Radeon RX 7900 GRE" with no errors
3. Open AMD Software → Should show GPU info

### On WSL

#### 4. Update WSL Kernel
```bash
# On Windows PowerShell (as Administrator)
wsl --update
wsl --shutdown
```

#### 5. Check GPU Access in WSL
```bash
# In WSL
ls -la /dev/dxg
# Should show: crw-rw-rw- 1 root root 10, 125 ...

# Check if HSA can see GPU
rocminfo | grep "Marketing Name"
# Should show: AMD Radeon RX 7900 GRE

# Optional: Check if amdgpu module loads (informational, not critical)
lsmod | grep amdgpu
# May be empty in WSL2 - this is normal and not the primary issue
# GPU can work via DirectX passthrough without amdgpu module
```

#### 6. Fix Permissions
```bash
sudo chmod 666 /dev/dxg
```

#### 7. Install Correct PyTorch Version (AMD Wheels)

**CRITICAL:** Use AMD-recommended wheels from repo.radeon.com, NOT pytorch.org!

```bash
cd /home/jare16/LLM
source venv/bin/activate

# Option A: Automated (Recommended)
./setup_rocm.sh

# Option B: Manual AMD Wheel Installation
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
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f ${location}/torch/lib/libhsa-runtime64.so*

# Test GPU
python verify_gpu.py
```

**Why AMD wheels matter:**
- ✅ Tested by AMD specifically for WSL environments
- ✅ Include WSL-specific compatibility fixes
- ✅ More stable and reliable for RX 7900 series
- ❌ PyTorch.org wheels are untested by AMD and may not work in WSL

## Alternative: WSL-Specific ROCm Setup

If above doesn't work, you may need WSL-specific ROCm configuration:

### Install ROCm in WSL (Inside WSL, not Windows)
```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update

# Install ROCm components
sudo apt install -y rocm-hip-runtime rocm-device-libs

# Add user to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Restart WSL
exit
# In Windows PowerShell:
wsl --shutdown
# Restart WSL
```

## Known Limitations

### WSL2 + ROCm Issues
1. **Limited Support**: ROCm on WSL2 is not officially fully supported by AMD
2. **Driver Dependencies**: Requires proper Windows-side driver with WSL passthrough
3. **Kernel Module**: WSL kernel may not include amdgpu module
4. **Docker Desktop Conflicts**: Docker Desktop can interfere with GPU passthrough

### Current Status (Before Fix)
- ✅ **rocminfo works** - HSA can detect GPU
- ❌ **PyTorch can't use GPU** - Wrong wheel source or missing WSL fix
- ⚠️  **amdgpu module may be missing** - Not the primary issue, symptom of driver passthrough limitations

## Workarounds

### Option 1: Use Native Linux (Recommended)
- Install Ubuntu/Fedora on bare metal
- Full ROCm support
- No WSL limitations

### Option 2: Use Windows ROCm (If Available)
- Install ROCm natively on Windows
- Use Windows Python environment
- Avoids WSL complications

### Option 3: CPU Mode (Temporary)
- Use CPU for development
- Deploy to Linux server with GPU for production

### Option 4: Wait for Better WSL Support
- Microsoft/AMD are improving WSL GPU support
- Future WSL kernels may have better amdgpu integration

## Testing Commands

### Verify Windows Driver
```powershell
# In PowerShell
Get-PnpDevice | Where-Object {$_.FriendlyName -like "*AMD*Radeon*"}
```

### Verify WSL GPU Access
```bash
# Device exists
ls -la /dev/dxg

# HSA layer works
rocminfo | grep -i "marketing name"

# Kernel module loaded (informational, not required)
lsmod | grep amdgpu

# PyTorch can use GPU (KEY TEST!)
python -c "import torch; print(torch.cuda.is_available())"
```

## Expected Results After Fix

### Before (Wrong PyTorch Wheels):
```bash
python -c "import torch; print(torch.__version__)"
# 2.5.1+rocm6.2 (from pytorch.org - doesn't work)
# OR 2.9.1+cu128 (CUDA version - completely wrong)

python -c "import torch; print(torch.cuda.is_available())"
# False
```

### After (AMD Wheels + WSL Fix):
```bash
python -c "import torch; print(torch.__version__)"
# 2.6.0+rocm6.4.2.git76481f7c (from repo.radeon.com - works!)

python -c "import torch; print(torch.cuda.is_available())"
# True

python verify_gpu.py
# ✅ GPU is available and working!
# GPU 0: AMD Radeon RX 7900 GRE
# Total Memory: 15.98 GB
```

**Note:** The amdgpu module may or may not load in WSL2 - this is normal. GPU access works via DirectX passthrough (`/dev/dxg`) even without the amdgpu kernel module.

## If All Else Fails

Contact support:
1. **AMD Support**: Report WSL2 ROCm issues
2. **Microsoft WSL**: Report GPU passthrough issues
3. **PyTorch Forums**: WSL + ROCm compatibility

Or consider switching to:
- Native Linux installation
- Remote Linux server with GPU
- Cloud GPU (vast.ai, runpod.io, etc.)

## Resources
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [WSL GPU Compute](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [Docker Desktop WSL Backend](https://docs.docker.com/desktop/wsl/)
