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
export HF_HOME="$SCRIPT_DIR/hf-cache"
export DATABASE_PATH="$SCRIPT_DIR/data/models.db"

# Disable GPU for local dev (prevents conflict with Docker)
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""

echo "Environment configured:"
echo "  PRIMARY_CACHE_PATH=$PRIMARY_CACHE_PATH"
echo "  HF_HOME=$HF_HOME"
echo "  DATABASE_PATH=$DATABASE_PATH"
echo "  GPU Mode: DISABLED (CPU only - prevents conflict with Docker)"
echo ""
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Verify environment variables are set
echo "Verifying environment (as Python will see it):"
python -c "import os; print(f'  PRIMARY_CACHE_PATH from Python: {os.getenv(\"PRIMARY_CACHE_PATH\", \"NOT SET\")}')"
echo ""

# Update database cache paths for local dev
echo "Updating database cache paths for local development..."
sqlite3 "$SCRIPT_DIR/data/models.db" "UPDATE modelregistry SET cache_path = '$SCRIPT_DIR/hf-cache' WHERE cache_location = 'primary' AND cache_path = '/root/.cache/huggingface';"
echo "Database updated."
echo ""

# Trap to restore paths on exit
cleanup() {
    echo ""
    echo "Restoring database cache paths for Docker..."
    sqlite3 "$SCRIPT_DIR/data/models.db" "UPDATE modelregistry SET cache_path = '/root/.cache/huggingface' WHERE cache_location = 'primary' AND cache_path = '$SCRIPT_DIR/hf-cache';"
    echo "Database restored. Safe to run Docker now."
}
trap cleanup EXIT

# Run the application
cd "$SCRIPT_DIR"
python main.py
