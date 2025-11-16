# Fix WSL GPU Access After Docker Desktop Installation

## Problem
GPU worked in WSL → Installed Docker Desktop → Display issues → Reinstalled AMD drivers → WSL GPU broken

## Root Cause
AMD driver reinstall didn't properly configure WSL GPU passthrough. The compute drivers (amdgpu module) aren't loading in WSL kernel.

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

# THIS IS THE KEY TEST - Check if amdgpu module loads
lsmod | grep amdgpu
# Should show amdgpu module (if working)
# If empty, drivers aren't passing through
```

#### 6. Fix Permissions
```bash
sudo chmod 666 /dev/dxg
```

#### 7. Install Correct PyTorch Version
```bash
cd /home/jare16/LLM
source venv/bin/activate

# Install ROCm 6.4 PyTorch (better compatibility with RX 7900 GRE)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Test GPU
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python verify_gpu.py
```

**Note:** ROCm 6.4 has better WSL compatibility than 6.2 for RX 7900 series.

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

### Current Status
- ✅ **rocminfo works** - HSA can detect GPU
- ❌ **PyTorch can't use GPU** - Compute drivers not initialized
- ❌ **amdgpu module missing** - Kernel driver not loaded

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

# Kernel module loaded (KEY!)
lsmod | grep amdgpu

# PyTorch can use GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Expected Results After Fix

### Before (Current State):
```bash
lsmod | grep amdgpu
# (empty - no module loaded)

python -c "import torch; print(torch.cuda.is_available())"
# False
```

### After (Fixed):
```bash
lsmod | grep amdgpu
# amdgpu    12345678  0
# (module loaded!)

python -c "import torch; print(torch.cuda.is_available())"
# True
```

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
