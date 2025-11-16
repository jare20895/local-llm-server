# Model Download Pipeline Documentation

This document explains exactly how and where models are downloaded and stored when running locally in a Python venv.

## Pipeline Overview

**IMPORTANT:** Models are downloaded **DIRECTLY** to the target cache directory. There is NO intermediate download or copying step.

```
User Action → API → Database → ModelManager → HuggingFace → Direct Download to Cache
```

## Detailed Step-by-Step Flow

### Step 1: User Registers a Model

**UI Action:**
- User fills in model registration form
- Selects cache location: "Primary (Fast SSD)"
- Enters estimated size: 6000 MB
- Clicks "Register"

**API Call:**
```http
POST /api/models
{
  "model_name": "qwen-3b",
  "hf_path": "Qwen/Qwen2.5-3B-Instruct",
  "trust_remote_code": false,
  "cache_location": "primary",
  "estimated_size_mb": 6000
}
```

**Backend Processing (main.py:326-387):**
```python
def register_model(request: ModelCreateRequest):
    # 1. Validate cache location
    if request.cache_location not in ["primary", "secondary", "custom"]:
        raise HTTPException(...)

    # 2. Check available space
    space_check = cache_manager.check_space_for_model(
        "primary",  # cache_location
        6000        # estimated_size_mb
    )
    # Returns: {
    #   "cache_path": "/home/jare16/LLM/hf-cache",
    #   "sufficient": True,
    #   "warning": False,
    #   ...
    # }

    # 3. Get actual cache path
    actual_cache_path = cache_manager.get_cache_path(
        "primary",  # Returns: "/home/jare16/LLM/hf-cache"
        None
    )

    # 4. Save to database
    new_model = ModelRegistry(
        model_name="qwen-3b",
        hf_path="Qwen/Qwen2.5-3B-Instruct",
        trust_remote_code=False,
        cache_location="primary",
        cache_path="/home/jare16/LLM/hf-cache"  # ← Stored in DB
    )
    session.add(new_model)
    session.commit()
```

**Database State:**
```sql
-- modelregistry table
id: 1
model_name: "qwen-3b"
hf_path: "Qwen/Qwen2.5-3B-Instruct"
cache_location: "primary"
cache_path: "/home/jare16/LLM/hf-cache"  -- ← Will be used for download
```

### Step 2: User Loads the Model

**UI Action:**
- User clicks "Load" button on model card

**API Call:**
```http
POST /api/orchestrate/load
{
  "model_name": "qwen-3b"
}
```

**Backend Processing (main.py:479-501):**
```python
def load_model(request: ModelLoadRequest):
    # 1. Look up model in database
    model = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == "qwen-3b")
    ).first()
    # Result: model.cache_path = "/home/jare16/LLM/hf-cache"

    # 2. Call ModelManager with cache_dir parameter
    model_manager.load(
        hf_path="Qwen/Qwen2.5-3B-Instruct",
        model_name="qwen-3b",
        model_id=1,
        trust_remote_code=False,
        cache_dir="/home/jare16/LLM/hf-cache"  # ← From database!
    )
```

### Step 3: ModelManager Downloads Model

**ModelManager.load() (main.py:50-89):**
```python
def load(self, hf_path, model_name, model_id, trust_remote_code, cache_dir=None):
    # Prepare kwargs for HuggingFace
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": False,
    }

    # Add cache_dir if specified
    if cache_dir:
        load_kwargs["cache_dir"] = "/home/jare16/LLM/hf-cache"  # ← Passed to HF

    # Call HuggingFace Transformers
    self.model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        **load_kwargs  # ← Includes cache_dir parameter
    )
```

### Step 4: HuggingFace Downloads DIRECTLY to Cache

**HuggingFace Transformers Internal Process:**

```python
# Inside transformers.AutoModelForCausalLM.from_pretrained()

# 1. Determine cache directory
if cache_dir is not None:
    cache_location = cache_dir  # = "/home/jare16/LLM/hf-cache"
else:
    cache_location = os.path.expanduser("~/.cache/huggingface")  # Default

# 2. Create model-specific subdirectory
model_path = os.path.join(
    cache_location,
    "hub",
    f"models--{org}--{model}"
)
# Result: "/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct"

# 3. Download files DIRECTLY to this location (NO intermediate step!)
for file in ["config.json", "pytorch_model.bin", "tokenizer.json", ...]:
    download_url_to_file(
        url=f"https://huggingface.co/{hf_path}/resolve/main/{file}",
        destination=f"{model_path}/{file}"  # ← Downloads HERE directly!
    )

# 4. Load model from downloaded files
model = load_model_from_directory(model_path)
```

## Final File Structure

After download completes, the filesystem looks like:

```
/home/jare16/LLM/hf-cache/
├── hub/
│   └── models--Qwen--Qwen2.5-3B-Instruct/
│       ├── snapshots/
│       │   └── abc123def456/           ← Git commit hash
│       │       ├── config.json         ← Model configuration
│       │       ├── generation_config.json
│       │       ├── model.safetensors   ← Model weights (~6GB)
│       │       ├── tokenizer.json      ← Tokenizer
│       │       ├── tokenizer_config.json
│       │       └── ...
│       └── refs/
│           └── main                    → points to abc123def456
├── transformers/                       ← Tokenizer cache
└── xet/                                ← XetHub cache
```

## Key Points

### ✅ Direct Download (What Actually Happens)

```
HuggingFace API
    ↓
Download DIRECTLY to: /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
    ↓
Files stored at final destination
```

### ❌ NO Intermediate Copy (Common Misconception)

```
❌ This does NOT happen:
   Download to ~/.cache/huggingface/  (intermediate)
       ↓
   Copy to /home/jare16/LLM/hf-cache/  (final)
```

### Why No Intermediate Step?

When you specify `cache_dir` parameter:
- HuggingFace uses it as the **download destination**
- Files are written directly to disk at that location
- No temporary location involved
- No copying needed

## Comparison: With vs Without cache_dir

### Without cache_dir (Default Behavior)
```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
# Downloads to: ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

### With cache_dir (Our Implementation)
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir="/home/jare16/LLM/hf-cache"  # ← Overrides default
)
# Downloads to: /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

**Result:** Same model files, different location. No copying involved.

## Environment Variable Hierarchy

HuggingFace checks in this order:

1. **`cache_dir` parameter** (highest priority)
   - If set: Use this location ← **We use this**

2. **`HF_HOME` environment variable**
   - If set: Use `$HF_HOME/hub`
   - Docker uses: `HF_HOME=/root/.cache/huggingface`

3. **Default location** (lowest priority)
   - Use `~/.cache/huggingface/hub`

Our implementation uses **method 1** (cache_dir parameter) for explicit control.

## Verification

To verify where a model was downloaded:

```bash
# Check primary cache
ls -lh /home/jare16/LLM/hf-cache/hub/

# Should show:
# drwxr-xr-x models--Qwen--Qwen2.5-3B-Instruct/

# Check model size
du -sh /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
# Should show: ~6.0G

# Check default cache (should be empty if cache_dir worked)
ls -lh ~/.cache/huggingface/hub/ 2>/dev/null || echo "Not used"
```

## Performance Implications

### Single Download (Current Implementation)
- **Download time:** ~2-5 minutes (6GB model on fast connection)
- **Disk I/O:** Write-once to final location
- **Network:** Single download from HuggingFace CDN

### If There Was Copying (Hypothetical)
- **Download time:** ~2-5 minutes
- **Copy time:** +1-2 minutes (6GB copy operation)
- **Disk I/O:** Write twice (download + copy)
- **Temporary space:** Needs 2x model size

**Benefit of direct download:** Faster, less disk I/O, less space needed!

## What About Model Updates?

When a model is updated on HuggingFace:

```python
# Load with revision parameter
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir="/home/jare16/LLM/hf-cache",
    revision="v2.0"  # Specific version
)
```

**Result:**
```
/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
├── snapshots/
│   ├── abc123/  ← v1.0 (old version, kept)
│   └── def456/  ← v2.0 (new version, downloaded)
└── refs/
    └── main → def456
```

HuggingFace downloads new version to new snapshot, keeps old version. No re-download of unchanged files.

## Summary: The Complete Flow

```
1. User clicks "Register Model"
   ↓
2. API validates and stores cache_path in database
   cache_path = "/home/jare16/LLM/hf-cache"
   ↓
3. User clicks "Load Model"
   ↓
4. API reads cache_path from database
   ↓
5. ModelManager calls from_pretrained(cache_dir=cache_path)
   ↓
6. HuggingFace downloads DIRECTLY to:
   /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
   ↓
7. Model loaded into VRAM
   ↓
8. Ready for inference!
```

**Total steps:** 1 download operation, 0 copy operations ✅

## Monitoring Download Progress

Watch the download in real-time:

```bash
# In another terminal
watch -n 1 'du -sh /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/ 2>/dev/null || echo "Downloading..."'
```

You'll see:
```
Downloading...
Downloading...
256M    /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
512M    /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
1.2G    /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
...
6.0G    /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

Growing size = downloading directly to that location!
