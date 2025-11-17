# Cache Directory Structure Reference

## Your Current Setup

### Project Cache (Shared between Local Dev & Docker)
```
/home/jare16/LLM/hf-cache/          ← PRIMARY_CACHE_PATH points HERE
├── hub/                             ← HuggingFace stores models here automatically
│   ├── models--Qwen--Qwen2.5-3B-Instruct/
│   ├── models--meta-llama--Llama-2-7b/
│   └── ...
├── transformers/                    ← Tokenizer configs
└── xet/                             ← XetHub cache
```

### Docker Mount (docker-compose.yml)
```yaml
volumes:
  - ./hf-cache:/root/.cache/huggingface
```

**What this means:**
- Docker container sees: `/root/.cache/huggingface`
- Actually writes to: `/home/jare16/LLM/hf-cache` on your host
- **Both local dev and Docker share the same files!**

### Secondary Cache (Slower Drive)
```
/mnt/z/llm-models-cache/            ← SECONDARY_CACHE_PATH points HERE
└── hub/                             ← Models go here automatically
    ├── models--large--model/
    └── ...
```

## How HuggingFace Cache Works

### When you specify `cache_dir`
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir="/home/jare16/LLM/hf-cache"  # ← Point to BASE directory
)
```

**HuggingFace automatically:**
1. Creates `hub/` subdirectory if it doesn't exist
2. Downloads model to: `/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/`
3. Stores tokenizer in: `/home/jare16/LLM/hf-cache/transformers/`

### ❌ WRONG
```python
cache_dir="/home/jare16/LLM/hf-cache/hub"  # Don't include /hub!
```

### ✅ CORRECT
```python
cache_dir="/home/jare16/LLM/hf-cache"  # Base directory only
```

## Environment Variables

### Current Configuration (.env)
```bash
# Primary cache - shared between local dev and Docker
PRIMARY_CACHE_PATH=/home/jare16/LLM/hf-cache
PRIMARY_CACHE_LIMIT_GB=100

# Secondary cache - larger, slower drive
SECONDARY_CACHE_PATH=/mnt/z/llm-models-cache
SECONDARY_CACHE_LIMIT_GB=2000
```

## Cache Location Behavior

### Primary Cache (Project's hf-cache)

**Local Development:**
```bash
source venv/bin/activate
python main.py
# Writes models to: /home/jare16/LLM/hf-cache/hub/
```

**Docker:**
```bash
docker-compose up -d
# Container writes to: /root/.cache/huggingface/hub/ (inside container)
# Which is actually: /home/jare16/LLM/hf-cache/hub/ (on host)
```

**Result:** ✅ Same files, no duplication!

### Secondary Cache

**Local Development:**
```bash
# Writes models to: /mnt/z/llm-models-cache/hub/
```

**Docker:**
```bash
# Also writes to: /mnt/z/llm-models-cache/hub/
# (if you register model with cache_location="secondary")
```

## File Paths in Database

When you register a model, the database stores:

```sql
model_name: "qwen-3b"
cache_location: "primary"
cache_path: "/home/jare16/LLM/hf-cache"  ← BASE directory
```

When loading, the system:
1. Reads `cache_path` from database
2. Passes it to HuggingFace: `from_pretrained(..., cache_dir="/home/jare16/LLM/hf-cache")`
3. HuggingFace loads from: `/home/jare16/LLM/hf-cache/hub/models--...`

## Checking Cache Contents

### View models in primary cache
```bash
ls -lh /home/jare16/LLM/hf-cache/hub/
```

### View models in secondary cache
```bash
ls -lh /mnt/z/llm-models-cache/hub/
```

### Check cache size
```bash
du -sh /home/jare16/LLM/hf-cache
du -sh /mnt/z/llm-models-cache
```

## Common Scenarios

### Scenario 1: Register model to primary cache
```json
{
  "model_name": "qwen-3b",
  "hf_path": "Qwen/Qwen2.5-3B-Instruct",
  "cache_location": "primary",
  "cache_path": null
}
```

**Result:** 
- System uses `PRIMARY_CACHE_PATH` = `/home/jare16/LLM/hf-cache`
- Model downloads to: `/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/`
- ✅ Available to both local dev and Docker

### Scenario 2: Register model to secondary cache
```json
{
  "model_name": "llama-70b",
  "hf_path": "meta-llama/Llama-2-70b",
  "cache_location": "secondary",
  "cache_path": null
}
```

**Result:**
- System uses `SECONDARY_CACHE_PATH` = `/mnt/z/llm-models-cache`
- Model downloads to: `/mnt/z/llm-models-cache/hub/models--meta-llama--Llama-2-70b/`

### Scenario 3: Custom cache location
```json
{
  "model_name": "temp-model",
  "hf_path": "org/model",
  "cache_location": "custom",
  "cache_path": "/mnt/usb-drive/models"
}
```

**Result:**
- Model downloads to: `/mnt/usb-drive/models/hub/models--org--model/`

## Legacy ~/.cache/huggingface

You mentioned you have `/home/jare16/.cache/huggingface`. This is HuggingFace's default location.

**Current situation:**
- ✅ New models: Go to `/home/jare16/LLM/hf-cache` (via PRIMARY_CACHE_PATH)
- ⚠️  Old models: May still be in `/home/jare16/.cache/huggingface`

**Options:**
1. **Leave them** - Old models stay there, new models use project cache
2. **Move them** - Copy to project cache for consolidation:
   ```bash
   cp -r ~/.cache/huggingface/hub/* /home/jare16/LLM/hf-cache/hub/
   ```
3. **Symlink** - Make ~/.cache/huggingface point to project cache:
   ```bash
   mv ~/.cache/huggingface ~/.cache/huggingface.backup
   ln -s /home/jare16/LLM/hf-cache ~/.cache/huggingface
   ```

## Summary

| Environment | Cache Location | Models Stored In |
|-------------|----------------|------------------|
| Local Dev (Primary) | `/home/jare16/LLM/hf-cache` | `hf-cache/hub/` |
| Docker (Primary) | `/root/.cache/huggingface` (container)<br>`/home/jare16/LLM/hf-cache` (host) | `hf-cache/hub/` |
| Both (Secondary) | `/mnt/z/llm-models-cache` | `llm-models-cache/hub/` |

**Key Point:** Always point `cache_dir` to the **BASE** directory. HuggingFace creates the `hub/` subdirectory automatically.
