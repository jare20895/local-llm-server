#!/usr/bin/env python3
"""
GPU Verification Script for ROCm
Checks if PyTorch can detect and use AMD GPUs
"""

import sys
import os

def check_rocm_environment():
    """Check ROCm environment variables"""
    print("=" * 60)
    print("ROCm Environment Variables")
    print("=" * 60)

    env_vars = [
        'ROCM_PATH',
        'HSA_OVERRIDE_GFX_VERSION',
        'HIP_VISIBLE_DEVICES',
        'PYTORCH_HIP_ALLOC_CONF'
    ]

    for var in env_vars:
        value = os.getenv(var, 'Not set')
        print(f"  {var}: {value}")
    print()

def check_pytorch():
    """Check PyTorch installation and ROCm support"""
    print("=" * 60)
    print("PyTorch Information")
    print("=" * 60)

    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")

        # Check for ROCm
        if hasattr(torch.version, 'hip'):
            print(f"  ROCm Version: {torch.version.hip}")
        else:
            print("  ROCm Version: Not installed (CUDA or CPU build)")

        # Check CUDA/ROCm availability
        cuda_available = torch.cuda.is_available()
        print(f"  GPU Available: {cuda_available}")

        if cuda_available:
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

                # Get memory info
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)  # Convert to GB
                print(f"    - Total Memory: {total_mem:.2f} GB")
                print(f"    - Compute Capability: {props.major}.{props.minor}")
        else:
            print("\n  ⚠️  WARNING: No GPU detected!")
            print("\n  Possible causes:")
            print("    1. ROCm drivers not installed on host")
            print("    2. PyTorch installed with CUDA instead of ROCm")
            print("       → Run: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
            print("    3. GPU not accessible in WSL")
            print("       → Check: ls -la /dev/dxg")
            print("    4. Insufficient permissions")
            print("       → Try: sudo chmod 666 /dev/dxg")

        print()

    except ImportError:
        print("  ❌ PyTorch not installed!")
        print("  Run: pip install -r requirements.txt")
        return False

    return cuda_available

def test_gpu_computation():
    """Test actual GPU computation"""
    print("=" * 60)
    print("GPU Computation Test")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("  ⏭️  Skipped (no GPU available)")
            print()
            return False

        device = torch.device('cuda:0')
        print(f"  Using device: {device}")

        # Simple computation test
        print("  Creating tensor on GPU...")
        x = torch.randn(1000, 1000, device=device)

        print("  Performing matrix multiplication...")
        y = torch.matmul(x, x)

        print("  Synchronizing...")
        torch.cuda.synchronize()

        print(f"  ✅ Success! Result shape: {y.shape}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print()

        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print()
        return False

def check_transformers():
    """Check if transformers library can use GPU"""
    print("=" * 60)
    print("Transformers Library Test")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not torch.cuda.is_available():
            print("  ⏭️  Skipped (no GPU available)")
            print()
            return False

        print("  Loading tiny model for testing...")
        model_name = "gpt2"  # Small model for quick test

        # This will download if not cached
        print(f"  Downloading {model_name} (if needed)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print("  Moving model to GPU...")
        device = torch.device('cuda:0')
        model = model.to(device)

        print("  Running inference test...")
        inputs = tokenizer("Hello, GPU!", return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        result = tokenizer.decode(outputs[0])

        print(f"  ✅ Success! Generated: {result[:50]}...")
        print()

        return True

    except ImportError:
        print("  ⚠️  Transformers not installed")
        print("  Run: pip install transformers")
        print()
        return False
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print()
        return False

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("GPU Verification for ROCm")
    print("=" * 60)
    print()

    check_rocm_environment()
    gpu_available = check_pytorch()

    if gpu_available:
        test_gpu_computation()
        check_transformers()

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if gpu_available:
        print("  ✅ GPU is available and working!")
        print("  You can now run: python main.py")
    else:
        print("  ❌ GPU not detected")
        print("  Please fix the issues above before running the server")
        sys.exit(1)

    print()

if __name__ == "__main__":
    main()
