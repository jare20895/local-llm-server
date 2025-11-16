#!/usr/bin/env python3
"""
Quick setup verification script for Homelab LLM Server.
This script checks if all dependencies are installed and the server can start.
"""

import sys
import subprocess


def check_dependency(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def main():
    print("=" * 60)
    print("Homelab LLM Server - Setup Verification")
    print("=" * 60)
    print()

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        return False
    print("✓ Python version is compatible")
    print()

    # Check dependencies
    print("Checking dependencies...")
    dependencies = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("sqlmodel", "sqlmodel"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("psutil", "psutil"),
    ]

    all_installed = True
    for package, import_name in dependencies:
        if not check_dependency(package, import_name):
            all_installed = False

    print()

    if not all_installed:
        print("=" * 60)
        print("Some dependencies are missing. Install them with:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return False

    # Check if files exist
    print("Checking project files...")
    import os

    files_to_check = [
        "database.py",
        "main.py",
        "requirements.txt",
        "static/index.html",
        "static/style.css",
        "static/app.js",
    ]

    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} is MISSING")
            all_installed = False

    print()

    if all_installed:
        print("=" * 60)
        print("✓ All checks passed! You're ready to start the server.")
        print()
        print("To start the server, run:")
        print("  python3 main.py")
        print()
        print("Then navigate to: http://localhost:8000")
        print("API docs: http://localhost:8000/docs")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ Some checks failed. Please fix the issues above.")
        print("=" * 60)

    return all_installed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
