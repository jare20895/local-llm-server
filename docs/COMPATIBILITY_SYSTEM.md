# Model Compatibility System

## Overview

The compatibility system automatically tracks which models work with your hardware and prevents repeated crashes from incompatible models.

## Compatibility Statuses

| Status | Meaning | Badge Color | Action |
|--------|---------|-------------|--------|
| **unknown** | Model never tested | Gray | Load normally |
| **compatible** | Model loads successfully | Green | Load normally |
| **incompatible** | Model fails to load | Red | Requires force_load=true |
| **degraded** | Intermittent failures | Yellow | Load with caution |
| **testing** | Currently being tested | Blue | Server may be testing |

## How It Works

### 1. Pre-Load Database Update
```python
# BEFORE attempting to load the model
model.compatibility_status = "testing"
session.commit()  # ← Database is updated FIRST
```

This ensures that even if the server crashes, there's a record of the attempt.

### 2. Startup Cleanup
On server startup, any models stuck in "testing" status are automatically marked as "incompatible":

```
⚠️  Found 1 model(s) stuck in 'testing' status from previous crash
   - Marked 'Microsoft phi mini 4k' as incompatible (crashed during test)
✅ Compatibility status cleanup completed
```

### 3. Crash Detection
The system detects specific error patterns:
- `flash-attention` errors
- `CreateContext` failures (ROCm/WSL issues)
- `Assertion` failures
- `Aborted` signals (core dumps)

### 4. Degraded vs Incompatible

**Incompatible**: Model has **never or rarely** loaded successfully (< 3 times)
```python
if model.total_loads >= 3:
    model.compatibility_status = "degraded"  # Likely transient issue
else:
    model.compatibility_status = "incompatible"  # Hardware problem
```

**Degraded**: Model loaded successfully ≥3 times before, but now fails
- Likely cause: WSL GPU reset, driver glitch, memory issue
- Solution: Restart WSL/Docker or reboot

## UI Warnings

### Incompatible Model Warning
```
⚠️ WARNING: "model-name" is marked as INCOMPATIBLE with this hardware!

This model has failed to load previously and may crash the server.

Notes: [Hardware incompatibility detected: CreateContext fail...]

Do you want to retry loading anyway? (This will use force_load=true)
```

### Degraded Model Warning
```
⚠️ WARNING: "model-name" has DEGRADED compatibility!

This model has loaded successfully before but recently failed.
This might be a temporary issue (WSL/GPU reset needed).

Notes: [Intermittent issue detected (loaded successfully 5 times before)...]

Do you want to try loading anyway?
```

## Manual Override

### Force Loading Incompatible Models
Set `force_load=true` in the API request or confirm the UI warning:
```python
{
    "model_name": "phi-mini-4k",
    "force_load": true  # Override incompatibility check
}
```

### Manual Status Update
Use the enhanced model details modal (click model name):
1. Change **Compatibility Status** dropdown
2. Add **Compatibility Notes** explaining the issue
3. Add **Custom Load Configuration** (JSON) to fix the issue

Example custom config:
```json
{
    "attn_implementation": "eager",
    "torch_dtype": "bfloat16",
    "device_map": "auto"
}
```

## Database Schema

```sql
-- Compatibility fields in ModelRegistry
compatibility_status TEXT DEFAULT 'unknown',
compatibility_notes TEXT,
load_config TEXT  -- JSON configuration
```

## API Endpoints

### Check Compatibility
```bash
GET /api/models
```
Returns all models with their compatibility status.

### Update Compatibility
```bash
PATCH /api/models/{model_name}/config
Content-Type: application/json

{
    "compatibility_status": "degraded",
    "compatibility_notes": "Works after WSL restart",
    "load_config": "{\"attn_implementation\": \"eager\"}"
}
```

## Troubleshooting

### Model Stuck in "testing"
**Cause**: Server crashed before updating status

**Solution**: Restart server (automatic cleanup runs on startup)

### False Incompatibility
**Cause**: Transient issue marked model as incompatible

**Solution**:
1. Click model name → Edit details
2. Change status to "unknown"
3. Try loading again

### All Models Marked Incompatible
**Cause**: System-wide GPU/driver issue

**Solutions**:
1. Check GPU availability: `/api/status`
2. Restart WSL: `wsl --shutdown`
3. Check ROCm setup: `docs/ROCM_SETUP_GUIDE.md`
4. Check GPU fix: `docs/FIX_WSL_GPU.md`

## Best Practices

1. **Let the system learn**: Don't immediately override "incompatible" status
2. **Check compatibility notes**: They contain the actual error message
3. **Use custom load config**: Fix model-specific issues instead of force loading
4. **Monitor degraded models**: May indicate hardware/driver issues
5. **Keep notes updated**: Document what works and what doesn't

## Example Workflow

```bash
# 1. Register new model
POST /api/models
{
    "model_name": "new-model",
    "hf_path": "org/model-name",
    "cache_location": "primary"
}

# 2. Try loading (automatic compatibility test)
POST /api/orchestrate/load
{"model_name": "new-model"}

# 3. If it crashes:
# - Server marks as "testing" before crash
# - Restart server → auto-marks as "incompatible"

# 4. Check compatibility in UI:
# - Model card shows red "INCOMPATIBLE" badge
# - Click model name to see error details

# 5. Fix with custom config:
# - Click model name → Edit details
# - Add custom load_config JSON
# - Change status to "unknown"
# - Retry loading
```

## Related Documentation

- [Model Details Modal](MODEL_DETAILS_MODAL.md) - Edit compatibility settings
- [ROCm Setup Guide](ROCM_SETUP_GUIDE.md) - Hardware compatibility
- [WSL GPU Fix](FIX_WSL_GPU.md) - Common GPU issues
