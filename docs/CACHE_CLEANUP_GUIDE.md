# Cache Cleanup Guide

## Two Types of Cleanup

### 1. Delete Model from Registry (with optional file deletion)
Delete a model that's registered in the database.

### 2. Clean Up Orphaned Models
Delete model files that exist on disk but are NOT in the database.

---

## Method 1: Delete Registered Model

### Via API

**Delete database record only** (keeps files):
```bash
curl -X DELETE "http://localhost:8000/api/models/qwen-3b"
```

**Delete database record AND files**:
```bash
curl -X DELETE "http://localhost:8000/api/models/qwen-3b?delete_files=true"
```

**Response:**
```json
{
  "message": "Model 'qwen-3b' deleted from registry and files deleted from disk",
  "freed_space_mb": 6144.5,
  "deleted_path": "/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct"
}
```

### Via Python
```python
import requests

# Delete with files
response = requests.delete(
    "http://localhost:8000/api/models/qwen-3b",
    params={"delete_files": True}
)
print(response.json())
```

---

## Method 2: Clean Up Orphaned Models

Orphaned models occur when:
- You deleted the database manually
- You copied model files without registering them
- Registration failed but files were downloaded
- You're migrating from a previous setup

### Step 1: Find Orphaned Models

```bash
curl "http://localhost:8000/api/cache/orphaned"
```

**Response:**
```json
{
  "orphaned_models": [
    {
      "hf_path": "Qwen/Qwen2.5-3B-Instruct",
      "directory": "/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct",
      "cache_type": "primary",
      "size_mb": 6144.5,
      "size_gb": 6.0
    },
    {
      "hf_path": "meta-llama/Llama-3-8B",
      "directory": "/mnt/z/llm-models-cache/hub/models--meta-llama--Llama-3-8B",
      "cache_type": "secondary",
      "size_mb": 15360.0,
      "size_gb": 15.0
    }
  ],
  "count": 2,
  "total_size_mb": 21504.5,
  "total_size_gb": 21.0
}
```

### Step 2: Delete Orphaned Models

```bash
curl -X POST "http://localhost:8000/api/cache/cleanup" \
  -H "Content-Type: application/json" \
  -d '["Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3-8B"]'
```

**Response:**
```json
{
  "deleted": [
    {
      "hf_path": "Qwen/Qwen2.5-3B-Instruct",
      "directory": "/home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct",
      "cache_type": "primary",
      "freed_space_mb": 6144.5
    },
    {
      "hf_path": "meta-llama/Llama-3-8B",
      "directory": "/mnt/z/llm-models-cache/hub/models--meta-llama--Llama-3-8B",
      "cache_type": "secondary",
      "freed_space_mb": 15360.0
    }
  ],
  "deleted_count": 2,
  "total_freed_mb": 21504.5,
  "errors": [],
  "error_count": 0
}
```

---

## Safety Features

### 1. Prevents Deleting Loaded Models
```bash
curl -X DELETE "http://localhost:8000/api/models/qwen-3b?delete_files=true"
```

**If model is loaded:**
```json
{
  "detail": "Cannot delete 'qwen-3b' - it is currently loaded. Unload it first."
}
```

### 2. Prevents Deleting Registered Models via Cleanup
```bash
curl -X POST "http://localhost:8000/api/cache/cleanup" \
  -d '["Qwen/Qwen2.5-3B-Instruct"]'
```

**If model is in database:**
```json
{
  "errors": [
    {
      "hf_path": "Qwen/Qwen2.5-3B-Instruct",
      "error": "Model exists in database - use DELETE /api/models/{name} instead"
    }
  ]
}
```

---

## Manual Cleanup (Advanced)

### Delete Specific Model Directory
```bash
# Find the model
ls /home/jare16/LLM/hf-cache/hub/

# Delete manually
rm -rf /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

### Clear Entire Primary Cache
```bash
# ⚠️ WARNING: Deletes ALL models in primary cache
rm -rf /home/jare16/LLM/hf-cache/hub/models--*
```

### Clear Entire Secondary Cache
```bash
# ⚠️ WARNING: Deletes ALL models in secondary cache
rm -rf /mnt/z/llm-models-cache/hub/models--*
```

---

## Blob Deduplication Explained

### Why You See New Blobs Per Model

HuggingFace stores model files as "blobs" named by SHA256 hash:

**Structure:**
```
hf-cache/hub/
├── models--Qwen--Qwen2.5-3B-Instruct/
│   ├── blobs/
│   │   ├── sha256:abc123...  (model weights - 6GB)
│   │   └── sha256:def456...  (config.json - 2KB)
│   └── snapshots/
│       └── commit_hash/
│           ├── model.safetensors → ../../blobs/sha256:abc123...
│           └── config.json → ../../blobs/sha256:def456...
└── models--meta-llama--Llama-3-8B/
    ├── blobs/
    │   ├── sha256:ghi789...  (different model - 14GB)
    │   └── sha256:jkl012...  (different config - 3KB)
    └── snapshots/
```

**Key Points:**

1. **Blobs are per-model repository**
   - Each model (Qwen, Llama, etc.) has its own blobs directory
   - No sharing of blobs ACROSS different models
   - Even if two models have identical files, separate blobs are created

2. **Blobs ARE shared within same model**
   - If you download v1.0 and v2.0 of the SAME model
   - Unchanged files share the same blob
   - Only changed files get new blobs

3. **Blobs are named by content hash (SHA256)**
   - Same file content = Same hash = Same blob name
   - Different file content = Different hash = Different blob name

### Example: Why Blobs Aren't Reused

```bash
# Download Qwen-3B
# Creates: hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/blobs/

# Download Llama-3-8B
# Creates: hf-cache/hub/models--meta-llama--Llama-3-8B/blobs/
# ↑ NEW blobs, even if some files are identical!
```

**This is expected and normal!** It's how HuggingFace Hub works.

### When Blobs ARE Reused

```bash
# Download Qwen-3B v1.0
# blobs/sha256:abc... (6GB weights v1.0)
# blobs/sha256:def... (2KB config)

# Update to Qwen-3B v2.0 (config unchanged, weights changed)
# blobs/sha256:abc... (6GB weights v1.0 - kept)
# blobs/sha256:def... (2KB config - REUSED!)
# blobs/sha256:ghi... (6GB weights v2.0 - new)
```

Only 6GB downloaded for v2.0, not 12GB!

---

## Best Practices

### 1. Regular Cleanup Schedule

```bash
# Weekly: Find orphaned models
curl "http://localhost:8000/api/cache/orphaned"

# Review the list
# Delete orphaned models if safe
curl -X POST "http://localhost:8000/api/cache/cleanup" \
  -H "Content-Type: application/json" \
  -d '["model/to/delete"]'
```

### 2. Delete Files When Removing Models

Instead of:
```bash
# ❌ Leaves files on disk
curl -X DELETE "http://localhost:8000/api/models/old-model"
```

Do this:
```bash
# ✅ Cleans up files
curl -X DELETE "http://localhost:8000/api/models/old-model?delete_files=true"
```

### 3. Monitor Cache Usage

Check cache stats regularly:
```bash
curl "http://localhost:8000/api/cache/stats"
```

### 4. Use Single Cache Directory

Avoid creating multiple cache directories:
```bash
# ✅ Good: All models in one place
PRIMARY_CACHE_PATH=/home/jare16/LLM/hf-cache

# ❌ Bad: Scattered caches
model1.cache_path=/cache/a
model2.cache_path=/cache/b
model3.cache_path=/cache/c
```

---

## Troubleshooting

### Issue: "Files not deleted" error

**Cause:** Permission denied or files in use

**Solution:**
```bash
# Check permissions
ls -la /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/

# Fix permissions
sudo chown -R $USER:$USER /home/jare16/LLM/hf-cache/

# Ensure model is unloaded
curl -X POST "http://localhost:8000/api/orchestrate/unload"

# Try delete again
curl -X DELETE "http://localhost:8000/api/models/model-name?delete_files=true"
```

### Issue: Orphaned models not found

**Cause:** Models in database but not on disk

**Solution:**
This is not a problem - the database record points to non-existent files.
Just delete the database record:
```bash
curl -X DELETE "http://localhost:8000/api/models/model-name"
```

### Issue: Cache size doesn't decrease after deletion

**Cause:** Operating system hasn't released disk space yet

**Solution:**
```bash
# Sync filesystem
sync

# Check actual usage
du -sh /home/jare16/LLM/hf-cache/
```

---

## Summary

**To delete a registered model:**
```bash
DELETE /api/models/{model_name}?delete_files=true
```

**To clean up orphaned files:**
```bash
1. GET /api/cache/orphaned
2. POST /api/cache/cleanup with list of paths
```

**To manually delete:**
```bash
rm -rf /home/jare16/LLM/hf-cache/hub/models--org--model/
```

**Remember:** Deleting model files is permanent! Make sure you don't need them before deleting.
