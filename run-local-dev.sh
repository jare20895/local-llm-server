#!/bin/bash
# Local development startup script
# This script activates the venv and sets environment variables for local dev

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting Local LLM Server (Development Mode)"
echo "=============================================="

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please create it first with: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Set environment variables for local dev (scoped to this process only)
export PRIMARY_CACHE_PATH="$SCRIPT_DIR/hf-cache"
export DATABASE_PATH="$SCRIPT_DIR/data/models.db"

echo "Environment configured:"
echo "  PRIMARY_CACHE_PATH=$PRIMARY_CACHE_PATH"
echo "  DATABASE_PATH=$DATABASE_PATH"
echo ""
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Run the application
cd "$SCRIPT_DIR"
python main.py
