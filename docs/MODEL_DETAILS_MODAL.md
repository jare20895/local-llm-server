# Model Details Modal

## Overview

The enhanced model details modal provides a comprehensive interface for editing all model settings, compatibility configuration, and custom loading parameters.

## Accessing the Modal

**Click on any model name** in the Models tab to open the modal.

The model name is displayed in blue with a hover effect to indicate it's clickable.

## Modal Sections

### 1. Model Information (Read-Only)

Basic model details that cannot be edited:
- **HuggingFace Path**: The model's repository path
- **Cache Location**: Where model files are stored (primary/secondary/custom)

### 2. Compatibility Configuration

#### Compatibility Status
Dropdown with options:
- `unknown` - Model never tested
- `compatible` - Model loads successfully
- `incompatible` - Model fails to load
- `degraded` - Intermittent failures
- `testing` - Currently being tested

**When to use each**:
- Set to `unknown` to retry a previously failed model
- Set to `degraded` if model works after WSL restart
- Set to `incompatible` if model permanently fails

#### Compatibility Notes
Free-text field to document:
- Specific error messages
- Hardware requirements
- Workarounds discovered
- When the issue occurs

**Example**:
```
Requires WSL restart every 3rd load. CreateContext fails with ROCm.
Works fine on second attempt after initial failure.
```

#### Custom Load Configuration (JSON)

Override default loading parameters with model-specific settings.

**Format**: Valid JSON object
```json
{
    "torch_dtype": "bfloat16",
    "attn_implementation": "eager",
    "device_map": "auto",
    "low_cpu_mem_usage": true
}
```

**Common Configurations**:

**For Phi Models (ROCm/WSL)**:
```json
{
    "attn_implementation": "eager",
    "torch_dtype": "bfloat16"
}
```

**For Large Models (Memory Issues)**:
```json
{
    "device_map": "auto",
    "low_cpu_mem_usage": true,
    "max_memory": {"0": "15GiB"}
}
```

**For Quantized Models**:
```json
{
    "load_in_8bit": true,
    "device_map": "auto"
}
```

**Leave empty** to use system defaults.

### 3. Benchmark Metrics (Collapsible)

Click "📊 Benchmark Metrics" to expand and edit quality/performance scores:

#### Quality Metrics (0-100 scale)
- **MMLU**: General knowledge
- **GPQA**: Graduate-level reasoning
- **HellaSwag**: Common sense reasoning
- **HumanEval**: Python coding ability
- **MBPP**: More Python coding
- **MATH**: Mathematical reasoning
- **TruthfulQA**: Honesty/hallucination resistance
- **Perplexity**: Fluency (lower is better, no max)

#### Operational Metrics
- **Max Throughput**: Tokens per second
- **Avg Latency TTFT**: Time to first token (milliseconds)
- **Quantization**: Type (e.g., "GGUF Q4_K_M", "None")

**All fields are optional** - only fill in what you know.

## Saving Changes

Click **"Save All Changes"** to update both:
1. Compatibility configuration (`PATCH /api/models/{name}/config`)
2. Benchmark metadata (`PATCH /api/models/{name}/metadata`)

Changes are saved atomically - both must succeed or the operation fails.

## Use Cases

### 1. Fix Incompatible Model

**Problem**: Model crashes with "CreateContext fail"

**Solution**:
1. Click model name
2. Set **Compatibility Status** → `degraded`
3. Add **Compatibility Notes**: "CreateContext fails on first load, works on retry"
4. Set **Custom Load Config**:
   ```json
   {
       "attn_implementation": "eager"
   }
   ```
5. Save and retry loading

### 2. Document Benchmark Scores

**Problem**: Need to track model performance for comparison

**Solution**:
1. Click model name
2. Expand **Benchmark Metrics**
3. Fill in known scores (from model card or your tests)
4. Save

Models with benchmark scores show them as badges on the model card.

### 3. Reset Failed Model

**Problem**: Model marked incompatible but you want to retry

**Solution**:
1. Click model name
2. Set **Compatibility Status** → `unknown`
3. Clear or update **Compatibility Notes**
4. Save
5. Try loading again (will re-test compatibility)

### 4. Configure WSL/ROCm Workaround

**Problem**: Model works but needs specific settings

**Solution**:
1. Click model name
2. Add **Custom Load Config**:
   ```json
   {
       "attn_implementation": "eager",
       "torch_dtype": "bfloat16",
       "device_map": "auto"
   }
   ```
3. Add **Compatibility Notes**: "ROCm WSL workaround - force eager attention"
4. Set **Compatibility Status** → `compatible`
5. Save

## Validation

### JSON Validation
The modal validates JSON before saving:
```javascript
if (loadConfigText) {
    try {
        JSON.parse(loadConfigText);
    } catch (e) {
        alert('Invalid JSON in Load Configuration field');
        return;
    }
}
```

**Invalid JSON will prevent saving** with an error message.

### Status Validation
Compatibility status must be one of:
- `unknown`
- `compatible`
- `incompatible`
- `degraded`
- `testing`

## API Reference

### Update Config Endpoint
```http
PATCH /api/models/{model_name}/config
Content-Type: application/json

{
    "compatibility_status": "degraded",
    "compatibility_notes": "Works after restart",
    "load_config": "{\"attn_implementation\": \"eager\"}"
}
```

**Response**: Updated model object

### Update Metadata Endpoint
```http
PATCH /api/models/{model_name}/metadata
Content-Type: application/json

{
    "mmlu_score": 72.5,
    "humaneval_score": 45.2,
    "max_throughput_tokens_sec": 38.7,
    "avg_latency_ms": 125.3
}
```

**Response**: Updated model object

## Visual Indicators

### Model Card Compatibility Badge

After setting compatibility status, the model card shows a color-coded badge:

| Status | Badge Color | Text Color |
|--------|-------------|------------|
| compatible | Light green | Dark green |
| incompatible | Light red | Dark red |
| degraded | Light yellow | Dark yellow |
| testing | Light blue | Dark blue |
| unknown | Light gray | Dark gray |

### Clickable Model Name

Model names have visual feedback:
- **Default**: Blue color (`#007bff`)
- **Hover**: Darker blue, underlined, slight shift right
- **Cursor**: Pointer (indicates clickable)

## Troubleshooting

### Modal Doesn't Appear

**Check**:
1. Open browser console (F12)
2. Look for JavaScript errors
3. Check for console logs:
   - `showEditDetailsModal called for: <model>`
   - `Found model: <data>`
   - `Modal should now be visible`

**Common Issues**:
- CSS not loaded → Check network tab
- JavaScript error → Check console for stack trace
- Modal element not found → Hard refresh (Ctrl+Shift+R)

### Changes Not Saving

**Check**:
1. Browser console for API errors
2. Server logs for validation errors
3. JSON syntax if using custom load config

**Common Issues**:
- Invalid JSON → Fix syntax
- Invalid status value → Use dropdown options
- Network error → Check server is running

### Custom Config Not Applied

**Verify**:
1. JSON is valid
2. Model is unloaded before testing
3. Server logs show "Using custom load config: {your config}"

**Debug**:
```bash
# Check database for saved config
sqlite3 models.db "SELECT model_name, load_config FROM modelregistry;"
```

## Browser Compatibility

Tested with:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

**Required Features**:
- CSS Grid
- Fetch API
- ES6 JavaScript
- HTML5 `<details>` element

## Related Documentation

- [Compatibility System](COMPATIBILITY_SYSTEM.md) - How status tracking works
- [Model Registry API](../README.md#api-endpoints) - API documentation
- [ROCm Setup](ROCM_SETUP_GUIDE.md) - Hardware configuration
