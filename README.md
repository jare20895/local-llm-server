# Homelab LLM Server

A robust, production-ready API for managing and serving local language models with comprehensive performance monitoring and a modern web UI.

## Features

- **Model Management**: Register, load, and unload models on-demand
- **Performance Monitoring**: Comprehensive scientific metrics collection including:
  - Token throughput (tokens/second)
  - GPU memory usage and utilization
  - CPU usage and system load
  - Time to first token (TTFT)
  - Generation parameters tracking
  - Quality metrics (repetition detection)
- **SQLite Database**: Persistent storage for model metadata and performance logs
- **Modern Web UI**: Clean, tabbed interface with sidebar navigation
- **OpenAPI Documentation**: Auto-generated Swagger and ReDoc documentation
- **Efficiency Mode**: Toggle performance logging for 5-10% speed improvement

## Architecture

The system consists of three main components:

1. **Database Layer** (`database.py`): SQLModel-based ORM with two tables:
   - `ModelRegistry`: Metadata about registered models
   - `PerformanceLog`: Detailed performance metrics for each inference

2. **API Layer** (`main.py`): FastAPI application with:
   - `ModelManager`: Singleton that manages the in-memory model
   - REST API endpoints for all operations
   - Comprehensive performance logging

3. **UI Layer** (`static/`): Lightweight single-page application
   - Vanilla JavaScript (no framework dependencies)
   - Modern CSS with clean design
   - Real-time status updates

## Installation

### Quick Start (AMD GPU with ROCm)

**For AMD GPUs in WSL2:** See [ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md) for the complete setup guide covering:
- Python virtual environment setup
- Docker environment setup
- Troubleshooting and diagnostics
- Environment comparison

**Quick 3-step setup:** See [QUICKSTART.md](QUICKSTART.md)

```bash
# Install ROCm PyTorch
./setup_rocm.sh

# Verify GPU
python verify_gpu.py

# Start server
./start.sh
```

### Standard Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start the server**:
```bash
python main.py
```

The server will start on `http://localhost:8000`

### Docker Installation (Recommended)

See [DOCKER.md](DOCKER.md) for complete Docker setup with ROCm GPU support.

```bash
docker-compose up -d
```

## Usage

### Web UI

Navigate to `http://localhost:8000` to access the web interface.

**Tabs:**
- **Inference**: Generate text with the loaded model
- **Models**: Register, load, and manage models
- **Analytics**: View performance statistics and logs
- **Settings**: Configure performance logging and validate models

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

**Model Registry:**
- `POST /api/models` - Register a new model
- `GET /api/models` - List all registered models
- `DELETE /api/models/{model_name}` - Delete a model

**Orchestration:**
- `GET /api/status` - Get current system status
- `POST /api/orchestrate/load` - Load a model into VRAM
- `POST /api/orchestrate/unload` - Unload current model
- `GET /api/orchestrate/validate` - Validate loaded model

**Configuration:**
- `GET /api/config/logging` - Check logging status
- `POST /api/config/logging` - Enable/disable performance logging

**Inference:**
- `POST /api/generate` - Generate text (hot path)

**Analytics:**
- `GET /api/analytics/performance/{model_name}` - Get aggregated stats
- `GET /api/analytics/logs/{model_name}` - Get recent performance logs

## Example: Register and Use a Model

### Via Web UI

1. Navigate to the **Models** tab
2. Click "Register New Model"
3. Fill in:
   - Model Name: `qwen-3b`
   - HF Path: `Qwen/Qwen2.5-3B-Instruct`
4. Click "Load" on the model card
5. Go to **Inference** tab and start generating

### Via API

```python
import requests

# 1. Register a model
requests.post("http://localhost:8000/api/models", json={
    "model_name": "qwen-3b",
    "hf_path": "Qwen/Qwen2.5-3B-Instruct",
    "trust_remote_code": False
})

# 2. Load the model
requests.post("http://localhost:8000/api/orchestrate/load", json={
    "model_name": "qwen-3b"
})

# 3. Generate text
response = requests.post("http://localhost:8000/api/generate", json={
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 256,
    "temperature": 0.7
})

print(response.json()["generated_text"])
```

## Performance Metrics

The system collects comprehensive metrics for scientific analysis:

### Token Metrics
- Input tokens
- Output tokens
- Total tokens

### Timing Metrics
- Total inference time (ms)
- Time to first token (ms)
- Tokens per second (throughput)

### Memory Metrics
- GPU peak memory allocation
- GPU reserved memory
- CPU memory usage (RSS)

### GPU Metrics (via nvidia-smi)
- GPU utilization percentage
- GPU temperature (°C)
- GPU power draw (watts)

### System Metrics
- CPU utilization percentage
- System load average (1-min)

### Generation Parameters
- Temperature
- Top-p
- Top-k
- Sampling mode
- Max tokens

### Quality Metrics
- Prompt hash (for duplicate detection)
- Repetition detection

### Model Configuration
- Data type (e.g., bfloat16)
- Device (e.g., cuda:0)
- Quantization method

## Database Schema

### ModelRegistry
```sql
- id (primary key)
- model_name (unique)
- hf_path
- trust_remote_code
- created_at
- parameter_count
- model_type
- architecture
- default_dtype
- context_length
- total_loads
- total_inferences
- last_loaded
```

### PerformanceLog
```sql
- id (primary key)
- timestamp
- model_registry_id (foreign key)
- [Token metrics: input_tokens, output_tokens, total_tokens]
- [Timing metrics: total_inference_ms, time_to_first_token_ms, tokens_per_second]
- [Memory metrics: gpu_mem_peak_alloc_mb, gpu_mem_reserved_mb, cpu_mem_rss_mb]
- [GPU metrics: gpu_utilization_percent, gpu_temperature_celsius, gpu_power_watts]
- [System metrics: cpu_utilization_percent, system_load_1min]
- [Generation params: temperature, top_p, top_k, do_sample, max_new_tokens]
- [Quality metrics: prompt_hash, repetition_detected]
- [Error tracking: error_occurred, error_message]
- [Model config: model_dtype, model_device, quantization]
```

## Configuration

### Efficiency Mode

Disable performance logging to gain 5-10% inference speed:

```python
requests.post("http://localhost:8000/api/config/logging", json={
    "enable": False
})
```

This disables collection of detailed metrics but still returns basic generation statistics.

## Production Considerations

1. **CORS**: Update `allow_origins` in `main.py` for production
2. **Database**: The SQLite file `models.db` is created automatically
3. **GPU**: Supports both CUDA (NVIDIA) and ROCm (AMD) GPUs
4. **Memory**: Ensure sufficient VRAM for your models (16GB recommended)
5. **Docker**: Use Docker for consistent deployment (see [DOCKER.md](DOCKER.md))

## Troubleshooting

**No GPU detected (AMD GPUs):**
- **PyTorch SOURCE matters!** Use AMD wheels from repo.radeon.com, NOT pytorch.org
- **WSL requires fix:** Must remove bundled libhsa-runtime64.so for GPU detection
- Quick fix: Run `./setup_rocm.sh` (handles AMD wheels + WSL fix automatically)
- See [SETUP_LOCAL.md](SETUP_LOCAL.md) for detailed AMD GPU setup
- **Key insight:** PyTorch.org wheels don't include WSL compatibility fixes. Always use AMD-recommended wheels for WSL:
  ```bash
  # ❌ Wrong (pytorch.org):
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.4

  # ✅ Correct (AMD repo.radeon.com):
  ./setup_rocm.sh  # Automated
  ```

**Model fails to load:**
- Check VRAM availability
- Verify Hugging Face path is correct
- Enable `trust_remote_code` if required

**GPU metrics not available:**
- NVIDIA: Ensure `nvidia-smi` is installed
- AMD: Ensure `rocm-smi` is installed and ROCm drivers are working
- Check GPU installation: `python verify_gpu.py`

**Performance logging not working:**
- Verify logging is enabled in Settings
- Check database write permissions

**For more help:**
- **Complete ROCm guide:** [ROCM_SETUP_GUIDE.md](ROCM_SETUP_GUIDE.md) - Comprehensive setup for Python venv and Docker
- Local setup: [SETUP_LOCAL.md](SETUP_LOCAL.md)
- Docker setup: [DOCKER.md](DOCKER.md)
- Quick reference: [QUICKSTART.md](QUICKSTART.md)
- GPU troubleshooting: [FIX_WSL_GPU.md](FIX_WSL_GPU.md)

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

Built with:
- FastAPI - Modern web framework
- Transformers - Hugging Face model library
- SQLModel - SQL database ORM
- PyTorch - Deep learning framework
