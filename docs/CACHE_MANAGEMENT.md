# Cache Management Guide

The Homelab LLM Server includes a dual-cache system to manage model storage across multiple drives, preventing your primary drive from running out of space.

## Overview

The cache management system provides:

- **Primary Cache**: Fast SSD for frequently used models (default: 100GB limit)
- **Secondary Cache**: Slower drive for infrequently used models (default: 2TB limit)
- **Custom Cache**: Specify any directory for one-off models
- **Background Monitoring**: Automatic space tracking every 5 minutes
- **UI Warnings**: Visual alerts when cache usage exceeds 90%

## Quick Start

### 1. Configure Cache Locations

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` to configure your cache locations:

```bash
# Primary cache (fast SSD)
PRIMARY_CACHE_PATH=/path/to/fast/ssd
PRIMARY_CACHE_LIMIT_GB=100

# Secondary cache (slower drive)
SECONDARY_CACHE_PATH=/mnt/z/llm-models-cache
SECONDARY_CACHE_LIMIT_GB=2000
```

### 2. Register a Model with Cache Location

When registering a new model in the UI:

1. Open the **Models** tab
2. Click **Register New Model**
3. Fill in model details
4. **Select Cache Location**:
   - `Primary (Fast SSD)`: Frequently used models
   - `Secondary (Slower Drive)`: Infrequently used models
   - `Custom Path`: Specify a custom directory
5. Enter estimated model size (MB) for space checking
6. Click **Register**

The system will warn you if the selected cache location doesn't have sufficient space.

### 3. Monitor Cache Usage

View cache statistics in the **Settings** tab:

- Current usage and limits
- Disk space available
- Color-coded progress bars:
  - **Green**: < 75% used
  - **Yellow**: 75-90% used
  - **Red**: > 90% used

Click **Refresh Cache Stats** to update the display.

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRIMARY_CACHE_PATH` | Path to primary cache directory | `~/.cache/huggingface/hub` |
| `PRIMARY_CACHE_LIMIT_GB` | Primary cache size limit in GB | `100` |
| `SECONDARY_CACHE_PATH` | Path to secondary cache directory | `/mnt/z/llm-models-cache` |
| `SECONDARY_CACHE_LIMIT_GB` | Secondary cache size limit in GB | `2000` |
| `CACHE_WARNING_THRESHOLD` | Warning threshold (0.9 = 90%) | `0.9` |

### Via API

#### Check Cache Space

```bash
curl -X POST "http://localhost:8000/api/cache/check?cache_location=primary&required_space_mb=5000"
```

#### Get Cache Statistics

```bash
curl "http://localhost:8000/api/cache/stats"
```

#### Get Recommended Cache Location

```bash
curl "http://localhost:8000/api/cache/recommend?estimated_size_mb=5000"
```

## How It Works

### Model Loading

When you load a model:

1. The system checks the model's `cache_location` from the database
2. Sets the HuggingFace `cache_dir` to the configured path
3. Downloads model files to that location (if not already cached)
4. Loads the model from the specified cache

### Space Monitoring

The background monitor:

1. Checks disk usage every 5 minutes
2. Calculates cache directory sizes
3. Updates usage statistics
4. Provides real-time data to the UI

### Space Validation

When registering a model:

1. System checks available space in selected cache
2. Compares against cache limit and estimated model size
3. Warns if usage would exceed 90% threshold
4. Prevents registration if insufficient space

## Best Practices

### 1. Estimate Model Sizes

Use these rough estimates:

| Model Size | Storage Needed |
|------------|----------------|
| 3B params  | ~6GB           |
| 7B params  | ~14GB          |
| 13B params | ~26GB          |
| 30B params | ~60GB          |
| 70B params | ~140GB         |

For quantized models, divide by the quantization ratio (e.g., 4-bit = /4).

### 2. Use Primary for Frequent Models

Store frequently used models in primary cache for faster loading:

- Development/testing models
- Production models
- Models you use daily

### 3. Use Secondary for Archival

Store infrequently used models in secondary cache:

- Experimental models
- Backup versions
- Models for occasional use

### 4. Monitor Regularly

Check cache usage in the Settings tab:

- Before downloading large models
- When experiencing performance issues
- When planning model additions

### 5. Set Realistic Limits

Configure limits based on actual disk space:

```bash
# Check available space
df -h /path/to/cache

# Set limit to 80% of available space
PRIMARY_CACHE_LIMIT_GB=80  # if you have 100GB available
```

## Troubleshooting

### Issue: "Insufficient space" error when registering

**Solution:**
1. Check cache usage in Settings tab
2. Select a different cache location (secondary or custom)
3. Increase cache limit if disk has more space
4. Delete unused models to free space

### Issue: Model downloads to wrong location

**Solution:**
1. Check model's `cache_location` in database
2. Verify environment variables are set correctly
3. Restart server after changing `.env` file

### Issue: Cache stats show 0% usage

**Solution:**
1. Check cache paths are correct and exist
2. Verify permissions on cache directories
3. Wait for background monitor to update (max 5 minutes)
4. Click "Refresh Cache Stats"

### Issue: Background monitoring not working

**Solution:**
1. Check server logs for errors
2. Verify cache paths are accessible
3. Restart server to reinitialize monitor

## API Reference

### Register Model with Cache Location

```python
import requests

response = requests.post("http://localhost:8000/api/models", json={
    "model_name": "llama-3-8b",
    "hf_path": "meta-llama/Meta-Llama-3-8B",
    "cache_location": "secondary",  # primary, secondary, or custom
    "cache_path": None,  # Required if cache_location is "custom"
    "estimated_size_mb": 14000  # ~14GB
})
```

### List Models with Cache Info

```python
response = requests.get("http://localhost:8000/api/models")
models = response.json()

for model in models:
    print(f"{model['model_name']}: {model['cache_location']} ({model['model_size_mb']}MB)")
```

## Migration Guide

### Moving Models Between Caches

Models are stored in HuggingFace cache directories. To move a model:

1. **Find the model files**:
   ```bash
   ls ~/.cache/huggingface/hub/models--<org>--<model>
   ```

2. **Copy to new location**:
   ```bash
   cp -r ~/.cache/huggingface/hub/models--<org>--<model> /mnt/z/llm-models-cache/
   ```

3. **Update database**:
   ```sql
   UPDATE modelregistry 
   SET cache_location = 'secondary',
       cache_path = '/mnt/z/llm-models-cache'
   WHERE model_name = 'your-model-name';
   ```

4. **Reload the model** to verify it works from new location

### Migrating from Single Cache

If you're upgrading from a version without cache management:

1. **Existing models** are automatically detected in the default cache
2. **New models** can be assigned to any cache location
3. **No migration required** - existing models continue to work

## Advanced Usage

### Custom Cache for Specific Use Cases

Use custom cache locations for:

- **Network drives**: Share models across machines
- **Temporary storage**: Test models without filling permanent cache
- **External drives**: Use USB/Thunderbolt drives for extra capacity

Example:

```python
response = requests.post("http://localhost:8000/api/models", json={
    "model_name": "test-model",
    "hf_path": "organization/model",
    "cache_location": "custom",
    "cache_path": "/mnt/external-drive/models"
})
```

### Programmatic Cache Management

```python
# Get cache recommendations
response = requests.get("http://localhost:8000/api/cache/recommend?estimated_size_mb=10000")
recommended = response.json()["recommended_cache"]

# Register model with recommended cache
requests.post("http://localhost:8000/api/models", json={
    "model_name": "auto-cached-model",
    "hf_path": "org/model",
    "cache_location": recommended
})
```

## Performance Considerations

### Primary vs Secondary Cache

| Aspect | Primary (SSD) | Secondary (HDD) |
|--------|---------------|-----------------|
| Speed | Fast (NVMe/SATA SSD) | Slower (7200 RPM HDD) |
| Model Load Time | ~5-10s | ~30-60s |
| Inference Speed | No difference (model in VRAM) | No difference |
| Cost per GB | Higher | Lower |
| Best for | Frequent use | Occasional use |

**Note:** Once a model is loaded into VRAM, inference speed is the same regardless of cache location.

## Support

For issues or questions:

1. Check server logs: `docker-compose logs` or `python main.py` output
2. Review this documentation
3. Open an issue on GitHub with:
   - Cache configuration (`.env` file)
   - Cache stats from Settings tab
   - Error messages from logs

## Future Enhancements

Planned features:

- Auto-migration of models based on usage frequency
- Cache cleanup tools (remove unused models)
- Smart caching (predict which models to keep in primary)
- Model deduplication (detect duplicate downloads)
- Cache compression support

---

**Last Updated:** 2025-11-16
