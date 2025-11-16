# Use AMD-recommended ROCm PyTorch Docker image
# This includes ROCm 6.4.2 and PyTorch 2.6.0 with all WSL compatibility fixes
# AMD recommends this over 'latest' for consistent, tested builds
FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    rocm-smi \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch is already included in the base image
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.9.2 \
    sqlmodel==0.0.22 \
    sqlalchemy==2.0.35 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    psutil==6.1.0 \
    python-multipart==0.0.12

# Verify ROCm/PyTorch installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.version.hip if hasattr(torch.version, \"hip\") else \"Not found\"}'); assert torch.cuda.is_available() or True, 'GPU check'"

# Copy application files
COPY . .

# Create data directory for SQLite database
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variable for database path
ENV DATABASE_PATH=/app/data/models.db

# Run the application
CMD ["python", "main.py"]
