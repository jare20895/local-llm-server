# Cache Behavior: Local Dev vs Docker

## Current Configuration (After Setup)

With the `.env` file configured:

```bash
PRIMARY_CACHE_PATH=/home/jare16/LLM/hf-cache
```

## How It Works

### Local Dev (Python venv)

**Step 1: Server reads configuration**
```python
# In database.py get_cache_config()
config = {
    "primary_path": os.getenv("PRIMARY_CACHE_PATH", "~/.cache/huggingface"),
    # With .env set: Returns "/home/jare16/LLM/hf-cache"
    # Without .env: Returns "~/.cache/huggingface" (fallback)
}
```

**Step 2: Model registration stores cache path**
```python
# User registers model via API
model = ModelRegistry(
    model_name="qwen-3b",
    cache_location="primary",
    cache_path="/home/jare16/LLM/hf-cache"  # ← From PRIMARY_CACHE_PATH
)
```

**Step 3: Model loading uses stored path**
```python
# User loads model via API
model_manager.load(
    hf_path="Qwen/Qwen2.5-3B-Instruct",
    cache_dir="/home/jare16/LLM/hf-cache"  # ← From database
)

# HuggingFace downloads to
AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir="/home/jare16/LLM/hf-cache"
)
# Result: Downloads to /home/jare16/LLM/hf-cache/hub/
```

**✅ Result for Local Dev:**
```
Downloads to: /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

### Docker

**Step 1: Container uses bind mount**
```yaml
# docker-compose.yml
volumes:
  - ./hf-cache:/root/.cache/huggingface
environment:
  - HF_HOME=/root/.cache/huggingface
  - PRIMARY_CACHE_PATH=/home/jare16/LLM/hf-cache  # ← From .env (via Docker)
```

**Step 2: Same process as local dev**
- Reads `PRIMARY_CACHE_PATH` from environment
- Stores in database: `cache_path="/home/jare16/LLM/hf-cache"`
- Passes to HuggingFace: `cache_dir="/home/jare16/LLM/hf-cache"`

**Step 3: Bind mount redirects writes**
```python
# Inside container, HuggingFace tries to write to:
/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/

# But Docker bind mount redirects to:
/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/

# Which is actually:
./hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/ (on host)
```

**✅ Result for Docker:**
```
Downloads to: /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

## Visual Comparison

### Scenario 1: Using Our API (Recommended)

**Local Dev:**
```
User registers model
    ↓
API reads PRIMARY_CACHE_PATH = "/home/jare16/LLM/hf-cache"
    ↓
Stores in DB: cache_path = "/home/jare16/LLM/hf-cache"
    ↓
User loads model
    ↓
API reads cache_path from DB
    ↓
Passes to HuggingFace: cache_dir = "/home/jare16/LLM/hf-cache"
    ↓
Downloads to: /home/jare16/LLM/hf-cache/hub/models--...
```

**Docker:**
```
User registers model
    ↓
API reads PRIMARY_CACHE_PATH = "/home/jare16/LLM/hf-cache"
    ↓
Stores in DB: cache_path = "/home/jare16/LLM/hf-cache"
    ↓
User loads model
    ↓
API reads cache_path from DB
    ↓
Passes to HuggingFace: cache_dir = "/home/jare16/LLM/hf-cache"
    ↓
Bind mount redirects to: ./hf-cache/hub/models--...
```

**✅ Same files on disk!**

### Scenario 2: Direct Python Script (Without API)

If you bypass the API and run Python directly:

**Without specifying cache_dir:**
```python
# Direct Python script (not using our API)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
# No cache_dir parameter!

# Downloads to DEFAULT location:
# ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

**With specifying cache_dir:**
```python
# Direct Python script with cache_dir
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir="/home/jare16/LLM/hf-cache"  # ← Explicit
)

# Downloads to project cache:
# /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

## Summary Table

| Scenario | Uses .env? | Cache Location |
|----------|-----------|----------------|
| **Local Dev via API** | ✅ Yes | `/home/jare16/LLM/hf-cache/hub/` |
| **Docker via API** | ✅ Yes | `/home/jare16/LLM/hf-cache/hub/` (via bind mount) |
| **Direct Python (no cache_dir)** | ❌ No | `~/.cache/huggingface/hub/` |
| **Direct Python (with cache_dir)** | ❌ No* | `/home/jare16/LLM/hf-cache/hub/` |

*If you manually specify the path

## Key Points

### ✅ With Current Setup

1. **Local dev uses project cache** (`/home/jare16/LLM/hf-cache`)
2. **Docker uses project cache** (via bind mount)
3. **Same files shared** between local dev and Docker
4. **No duplication** of model downloads

### How It Works

1. `.env` file sets: `PRIMARY_CACHE_PATH=/home/jare16/LLM/hf-cache`
2. Python reads environment variable on startup
3. Database stores cache paths from environment
4. API passes cache paths to HuggingFace
5. HuggingFace downloads to specified location

### When Would Default Be Used?

The default (`~/.cache/huggingface`) would only be used if:

1. **No .env file** and you don't set environment variables
2. **Direct Python scripts** that don't use our API
3. **Manual HuggingFace calls** without cache_dir parameter

Since you're using the API and have a `.env` file, the default is NOT used.

## Verification

Check which cache is being used:

```bash
# Start server locally
source venv/bin/activate
python main.py

# In another terminal, watch the cache grow
watch -n 1 'du -sh /home/jare16/LLM/hf-cache/hub/'

# Register and load a model via UI
# You should see the size increasing in hf-cache/hub/

# Check default cache is NOT growing
du -sh ~/.cache/huggingface/hub/ 2>/dev/null || echo "Not used"
```

## Environment Variable Loading

Python loads `.env` automatically if you use python-dotenv:

```python
# Add to main.py if not already present
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

# Now os.getenv("PRIMARY_CACHE_PATH") works
```

**Current Status:** Our code uses `os.getenv()` which reads environment variables set by:
- Shell environment (export VAR=value)
- `.env` file (if using python-dotenv or docker-compose)
- Docker environment section

Docker Compose automatically loads `.env` file, so no additional setup needed!

## Testing

To confirm behavior:

```bash
# Test 1: Check environment variable
echo $PRIMARY_CACHE_PATH
# Should show: /home/jare16/LLM/hf-cache

# Test 2: Start server and check logs
python main.py 2>&1 | grep -i cache

# Test 3: Register and load a small model
# Watch: ls -lh /home/jare16/LLM/hf-cache/hub/
# Should see new model directory appear
```
