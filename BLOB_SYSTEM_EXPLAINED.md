# HuggingFace Blob System Explained

## How Blobs Work

### Content-Addressable Storage

HuggingFace Hub uses SHA256 hashes as filenames to enable deduplication:

```
hf-cache/hub/
└── models--Qwen--Qwen2.5-3B-Instruct/
    ├── blobs/
    │   ├── sha256:abc123def456...  ← Actual model file (6GB)
    │   ├── sha256:789ghi012jkl...  ← Actual config file (2KB)
    │   └── sha256:mno345pqr678...  ← Actual tokenizer file (500KB)
    ├── snapshots/
    │   └── commit_abc123/
    │       ├── model.safetensors    → symlink to ../../blobs/sha256:abc123def456...
    │       ├── config.json          → symlink to ../../blobs/sha256:789ghi012jkl...
    │       └── tokenizer.json       → symlink to ../../blobs/sha256:mno345pqr678...
    └── refs/
        └── main → commit_abc123
```

### Key Points

1. **Blobs are named by content hash (SHA256)**
   - Same content = Same hash = Same blob
   - Different content = Different hash = New blob

2. **Snapshots use symlinks**
   - Snapshots are git commits
   - Files in snapshots are symlinks to blobs
   - Multiple snapshots can share the same blobs

3. **Deduplication works WITHIN a model repository**
   - If you download v1.0 and v2.0 of the same model
   - Unchanged files share the same blobs
   - Only changed files get new blobs

4. **NO deduplication ACROSS different models**
   - Each model repo has its own blobs directory
   - Even if two models have identical files, separate blobs are created

## Why You See New Blobs Being Downloaded

### Scenario 1: Different Models (Expected)
```bash
# Download Qwen-3B
hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/blobs/sha256:xxx...

# Download Llama-3-8B (DIFFERENT model)
hf-cache/hub/models--meta-llama--Llama-3-8B/blobs/sha256:yyy...
```
**Result:** New blobs downloaded ✅ (This is expected and correct)

### Scenario 2: Same Model, Different Versions (Partial Reuse)
```bash
# Download v1.0
hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
├── blobs/
│   ├── sha256:abc (config.json - unchanged)
│   └── sha256:def (model weights v1.0)
└── snapshots/v1.0/

# Download v2.0 (config unchanged, weights changed)
hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
├── blobs/
│   ├── sha256:abc (config.json - REUSED!)
│   ├── sha256:def (model weights v1.0)
│   └── sha256:ghi (model weights v2.0 - NEW)
└── snapshots/v2.0/
```
**Result:** Config blob reused, weight blob new ✅

### Scenario 3: Using Different Cache Paths (Problem!)
```python
# Register model A with primary cache
model_a = {
    "cache_path": "/home/jare16/LLM/hf-cache"
}

# Register model B with secondary cache
model_b = {
    "cache_path": "/mnt/z/llm-models-cache"
}
```
**Result:** Each cache downloads its own blobs ❌ (No sharing between caches)

## Checking Your Blob Usage

### View Blobs for a Model
```bash
# List blobs
ls -lh /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/blobs/

# Output example:
# -rw-r--r-- 1 user user 2.1K Nov 16 10:00 sha256:abc123...  (config.json)
# -rw-r--r-- 1 user user 6.0G Nov 16 10:05 sha256:def456...  (model weights)
# -rw-r--r-- 1 user user 500K Nov 16 10:00 sha256:ghi789...  (tokenizer)
```

### View Snapshots
```bash
# List snapshots
ls -lh /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/

# View files in a snapshot
ls -lh /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/abc123def/

# Check if they're symlinks
file /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/abc123def/model.safetensors
# Output: symbolic link to ../../blobs/sha256:...
```

### Check Total Cache Size
```bash
# Size of entire cache
du -sh /home/jare16/LLM/hf-cache/

# Size per model
du -sh /home/jare16/LLM/hf-cache/hub/models--*
```

## Download Behavior

### First Download
```
HuggingFace API
    ↓
Check if blob exists locally (by SHA256 hash)
    ↓
If exists: Create symlink (instant)
If not exists: Download blob → Create symlink
```

### Subsequent Downloads (Same Model)
```
Download new version
    ↓
Check each file's SHA256
    ↓
Unchanged files: Reuse existing blob (instant)
Changed files: Download new blob
    ↓
Create new snapshot with symlinks
```

## Why Blobs Are Efficient

### Without Blobs (Naive Approach)
```
v1.0: 6GB model + 500KB tokenizer = 6.0005GB
v1.1: 6GB model + 500KB tokenizer = 6.0005GB (duplicate!)
Total: 12.001GB
```

### With Blobs (HuggingFace Approach)
```
v1.0:
  blobs/sha256:abc (6GB model)
  blobs/sha256:def (500KB tokenizer)

v1.1: (only tokenizer changed)
  blobs/sha256:abc (6GB model - REUSED!)
  blobs/sha256:ghi (500KB tokenizer - NEW)
  
Total: 6GB + 500KB + 500KB = ~6.001GB
```
**Savings:** ~6GB (50% reduction!)

## Common Misconceptions

### ❌ Misconception 1: "Same file content across models should share blobs"
**Reality:** Each model repo has separate blobs directory. Even if Qwen-3B and Llama-3-8B both have identical `config.json`, they get separate blobs.

**Why:** Git-based storage keeps each repo isolated.

### ❌ Misconception 2: "Deleting a snapshot frees up space"
**Reality:** Blobs remain even if snapshots are deleted (orphaned blobs).

**Why:** HuggingFace doesn't automatically garbage collect blobs.

### ❌ Misconception 3: "Using different cache_dir should still share blobs"
**Reality:** Different cache directories = Completely separate storage.

**Why:** No cross-cache deduplication.

## Best Practices

### 1. Use Single Cache Directory
```bash
# ✅ Good: All models in one cache
PRIMARY_CACHE_PATH=/home/jare16/LLM/hf-cache

# All models share infrastructure (not blobs, but same location)
```

### 2. Avoid Unnecessary Re-downloads
```python
# ✅ Good: Reuse same cache for related models
model_a.cache_path = "/home/jare16/LLM/hf-cache"
model_b.cache_path = "/home/jare16/LLM/hf-cache"

# ❌ Bad: Different caches for each model
model_a.cache_path = "/cache/model-a"
model_b.cache_path = "/cache/model-b"
```

### 3. Use Specific Revisions
```python
# Download specific version
from_pretrained("Qwen/Qwen2.5-3B-Instruct", revision="v2.0")

# Prevents accidental re-downloads of "latest"
```

## Cleanup (Manual)

HuggingFace doesn't auto-cleanup orphaned blobs. You can manually remove:

### Remove Specific Model
```bash
rm -rf /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

### Remove Orphaned Blobs (Advanced)
```bash
# Find blobs not referenced by any snapshot
cd /home/jare16/LLM/hf-cache/hub/models--Qwen--Qwen2.5-3B-Instruct/
find blobs/ -type f | while read blob; do
    if ! find snapshots/ -type l -exec readlink {} \; | grep -q "$(basename $blob)"; then
        echo "Orphaned: $blob"
    fi
done
```

## Summary

**Blob Naming:**
- Named by SHA256 hash of content
- Same content = Same hash = Same filename

**Deduplication:**
- ✅ Within same model repo across versions
- ❌ Across different model repos
- ❌ Across different cache directories

**When New Blobs Are Downloaded:**
1. ✅ Different models (expected)
2. ✅ File content changed (expected)
3. ❌ Using different cache paths (avoidable)
4. ❌ Re-downloading already-cached model (check cache_dir)

**To Minimize Downloads:**
- Use single cache directory for all models
- Check cache before re-downloading
- Use specific revisions instead of "latest"
