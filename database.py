# database.py
from sqlmodel import Field, SQLModel, create_engine, Relationship
from datetime import datetime
from typing import Optional, List
from pydantic import ConfigDict
from enum import Enum
import os

# --- Cache Location Enum ---
class CacheLocation(str, Enum):
    """Enum for cache location types."""
    PRIMARY = "primary"      # Fast SSD, frequently used models
    SECONDARY = "secondary"  # Slower drive, infrequently used models
    CUSTOM = "custom"        # User-specified custom location


# --- Table 1: The Model Registry ---
# Stores *metadata* about models, not the models themselves.
class ModelRegistry(SQLModel, table=True):
    model_config = ConfigDict(protected_namespaces=())

    id: Optional[int] = Field(default=None, primary_key=True)
    model_name: str = Field(unique=True, index=True)  # Your friendly name
    hf_path: str                                       # Hugging Face path
    trust_remote_code: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.now)

    # Cache location tracking
    cache_location: str = Field(default="primary")    # primary, secondary, or custom
    cache_path: Optional[str] = None                  # Full path where model files are stored
    model_size_mb: Optional[float] = None             # Size of model files in MB

    # Model metadata (populated on first load)
    parameter_count: Optional[int] = None  # Total number of parameters
    model_type: Optional[str] = None       # e.g., "LlamaForCausalLM", "GPT2LMHeadModel"
    architecture: Optional[str] = None     # e.g., "llama", "gpt2", "qwen"
    default_dtype: Optional[str] = None    # e.g., "bfloat16", "float16"
    context_length: Optional[int] = None   # Maximum context window

    # Benchmark Metrics (Quality/Capability Scores)
    mmlu_score: Optional[float] = None           # General knowledge (0-100)
    gpqa_score: Optional[float] = None           # Graduate-level reasoning
    hellaswag_score: Optional[float] = None      # Commonsense reasoning
    humaneval_score: Optional[float] = None      # Coding ability (Python)
    mbpp_score: Optional[float] = None           # More Python coding
    math_score: Optional[float] = None           # Mathematical reasoning
    truthfulqa_score: Optional[float] = None     # Honesty/hallucination resistance
    perplexity: Optional[float] = None           # Fluency (lower is better)

    # Operational Metrics (Speed & Usability)
    max_throughput_tokens_sec: Optional[float] = None  # Max tokens/second
    avg_latency_ms: Optional[float] = None             # Average TTFT in milliseconds
    quantization: Optional[str] = None                 # e.g., "GGUF Q4_K_M", "GPTQ 4-bit", "None"

    # Usage statistics
    total_loads: int = Field(default=0)    # How many times this model has been loaded
    total_inferences: int = Field(default=0)  # Total number of inferences run
    last_loaded: Optional[datetime] = None

    # Version tracking (for performance comparison over time)
    current_commit: Optional[str] = None       # Git commit hash of current snapshot
    current_version: Optional[str] = None      # Human-readable version tag (e.g., "v2.0", "main")
    last_updated: Optional[datetime] = None    # When model was last updated/downloaded
    update_available: bool = Field(default=False)  # Cached: is update available?
    last_update_check: Optional[datetime] = None   # When we last checked for updates

    # This links a model to all its performance logs
    logs: List["PerformanceLog"] = Relationship(back_populates="model")


# --- Table 2: The Performance Log ---
# Stores the results of each inference with comprehensive scientific metrics.
class PerformanceLog(SQLModel, table=True):
    model_config = ConfigDict(protected_namespaces=())

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now)

    # The "link" back to the model that was run
    model_registry_id: Optional[int] = Field(default=None, foreign_key="modelregistry.id")
    model: Optional[ModelRegistry] = Relationship(back_populates="logs")

    # Model version at time of inference (for performance tracking over time)
    model_version: Optional[str] = None  # Commit hash or version tag

    # === Token Metrics ===
    input_tokens: int
    output_tokens: int
    total_tokens: int  # input + output for easier analysis

    # === Timing Metrics ===
    total_inference_ms: float  # Total time from request to response
    time_to_first_token_ms: Optional[float] = None  # TTFT - critical for UX/streaming
    tokens_per_second: Optional[float] = None  # Throughput metric

    # === Memory Metrics ===
    gpu_mem_peak_alloc_mb: Optional[float] = None  # Peak VRAM during inference
    gpu_mem_reserved_mb: Optional[float] = None     # Total VRAM reserved
    cpu_mem_rss_mb: Optional[float] = None          # CPU RAM usage

    # === GPU Metrics (if available via nvidia-smi or similar) ===
    gpu_utilization_percent: Optional[float] = None
    gpu_temperature_celsius: Optional[float] = None
    gpu_power_watts: Optional[float] = None

    # === System Metrics ===
    cpu_utilization_percent: Optional[float] = None  # System CPU load during inference
    system_load_1min: Optional[float] = None         # Linux load average

    # === Generation Parameters (for correlation analysis) ===
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: bool = Field(default=True)
    max_new_tokens: Optional[int] = None

    # === Quality Metrics ===
    prompt_hash: Optional[str] = None  # Hash of prompt for duplicate detection
    repetition_detected: Optional[bool] = None  # Did the model repeat itself?

    # === Error Tracking ===
    error_occurred: bool = Field(default=False)
    error_message: Optional[str] = None

    # === Model Configuration ===
    model_dtype: Optional[str] = None  # e.g., "torch.bfloat16"
    model_device: Optional[str] = None  # e.g., "cuda:0"
    quantization: Optional[str] = None  # e.g., "4bit", "8bit", "none"


# --- Database Engine Setup ---
# Use environment variable if set (for Docker), otherwise use current directory
sqlite_file_name = os.getenv("DATABASE_PATH", "models.db")
sqlite_url = f"sqlite:///{sqlite_file_name}"

# We'll use echo=False for production, echo=True for debugging
engine = create_engine(sqlite_url, echo=False)


def create_db_and_tables():
    """Initialize the database and create all tables."""
    SQLModel.metadata.create_all(engine)


# --- Utility Functions for Metrics Collection ---

def get_gpu_metrics() -> dict:
    """
    Collect GPU metrics using nvidia-smi if available.
    Returns dict with utilization, temperature, power, etc.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode == 0:
            values = result.stdout.strip().split(",")
            return {
                "utilization": float(values[0].strip()),
                "temperature": float(values[1].strip()),
                "power": float(values[2].strip()),
            }
    except Exception:
        pass

    return {"utilization": None, "temperature": None, "power": None}


def get_system_metrics() -> dict:
    """
    Collect system-level metrics (CPU load, etc.)
    """
    import psutil
    import os

    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "load_avg": None,
    }

    # Get load average on Unix systems
    try:
        load_avg = os.getloadavg()
        metrics["load_avg"] = load_avg[0]  # 1-minute load average
    except AttributeError:
        # Windows doesn't have getloadavg
        pass

    return metrics


def hash_prompt(prompt: str) -> str:
    """Create a hash of the prompt for duplicate detection."""
    import hashlib
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def detect_repetition(text: str, threshold: float = 0.3) -> bool:
    """
    Simple repetition detector - checks if generated text has excessive repetition.
    Returns True if repetition ratio exceeds threshold.
    """
    if len(text) < 50:
        return False

    # Split into words
    words = text.lower().split()
    if len(words) < 10:
        return False

    # Check for repeated sequences
    unique_words = set(words)
    repetition_ratio = 1.0 - (len(unique_words) / len(words))

    return repetition_ratio > threshold


# --- Cache Management Utilities ---

def get_disk_usage(path: str) -> dict:
    """
    Get disk usage statistics for a given path.
    Returns dict with total, used, free space in GB and usage percentage.
    """
    import shutil

    try:
        total, used, free = shutil.disk_usage(path)
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100 if total > 0 else 0,
        }
    except Exception as e:
        print(f"Error getting disk usage for {path}: {e}")
        return {
            "total_gb": 0,
            "used_gb": 0,
            "free_gb": 0,
            "usage_percent": 0,
            "error": str(e)
        }


def get_directory_size(path: str) -> float:
    """
    Calculate total size of a directory in MB.
    Returns size in megabytes.
    """
    import os

    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating directory size for {path}: {e}")

    return total_size / (1024**2)  # Convert to MB


def get_cache_config() -> dict:
    """
    Get cache configuration from environment variables or defaults.
    Returns dict with primary and secondary cache paths and limits.
    """
    import os

    # Default HuggingFace cache location (base directory, not /hub)
    default_hf_cache = os.path.expanduser("~/.cache/huggingface")

    return {
        "primary_path": os.getenv("PRIMARY_CACHE_PATH", default_hf_cache),
        "primary_limit_gb": float(os.getenv("PRIMARY_CACHE_LIMIT_GB", "100")),
        "secondary_path": os.getenv("SECONDARY_CACHE_PATH", "/mnt/z/llm-models-cache"),
        "secondary_limit_gb": float(os.getenv("SECONDARY_CACHE_LIMIT_GB", "2000")),  # 2TB
        "usage_warning_threshold": float(os.getenv("CACHE_WARNING_THRESHOLD", "0.9")),  # 90%
    }


def check_cache_space(cache_location: str, required_space_mb: float = 0) -> dict:
    """
    Check if cache location has sufficient space.
    Returns dict with available space and whether it's sufficient.
    """
    config = get_cache_config()

    # Determine path based on cache location
    if cache_location == "primary":
        cache_path = config["primary_path"]
        limit_gb = config["primary_limit_gb"]
    elif cache_location == "secondary":
        cache_path = config["secondary_path"]
        limit_gb = config["secondary_limit_gb"]
    else:
        # For custom locations, we'll check the disk but not enforce limits
        return {"sufficient": True, "warning": False}

    # Create cache directory if it doesn't exist
    os.makedirs(cache_path, exist_ok=True)

    # Get current usage
    disk_usage = get_disk_usage(cache_path)
    cache_size_mb = get_directory_size(cache_path)
    cache_size_gb = cache_size_mb / 1024

    # Calculate if space is sufficient
    required_space_gb = required_space_mb / 1024
    available_after_gb = disk_usage["free_gb"] - required_space_gb
    projected_usage_gb = cache_size_gb + required_space_gb

    sufficient = (available_after_gb > 0 and
                 projected_usage_gb <= limit_gb)

    warning = (projected_usage_gb / limit_gb) >= config["usage_warning_threshold"]

    return {
        "cache_path": cache_path,
        "cache_size_gb": cache_size_gb,
        "cache_limit_gb": limit_gb,
        "disk_free_gb": disk_usage["free_gb"],
        "disk_total_gb": disk_usage["total_gb"],
        "disk_usage_percent": disk_usage["usage_percent"],
        "projected_usage_gb": projected_usage_gb,
        "usage_percent": (projected_usage_gb / limit_gb) * 100 if limit_gb > 0 else 0,
        "sufficient": sufficient,
        "warning": warning,
        "required_space_mb": required_space_mb,
    }


# --- Model Version Management ---

def get_local_model_commit(hf_path: str, cache_path: str) -> Optional[str]:
    """
    Get the current commit hash of a locally cached model.
    Reads from the refs/main symlink in the HuggingFace cache.

    Returns:
        Commit hash string, or None if not found
    """
    import os

    # Convert HF path to cache directory name (e.g., "Qwen/Qwen2.5-3B" -> "models--Qwen--Qwen2.5-3B")
    org_model = hf_path.replace("/", "--")
    model_cache_dir = os.path.join(cache_path, "hub", f"models--{org_model}")
    refs_main = os.path.join(model_cache_dir, "refs", "main")

    try:
        if os.path.exists(refs_main):
            # refs/main is a file containing the commit hash
            with open(refs_main, 'r') as f:
                commit = f.read().strip()
                return commit
    except Exception as e:
        print(f"Error reading local commit for {hf_path}: {e}")

    return None


def get_remote_model_info(hf_path: str, token: Optional[str] = None) -> dict:
    """
    Get remote model information from HuggingFace Hub.

    Returns:
        Dict with 'commit', 'version', 'last_modified', or error info
    """
    try:
        from huggingface_hub import model_info
        from datetime import datetime

        info = model_info(hf_path, token=token)

        return {
            "commit": info.sha,
            "version": getattr(info, 'model_version', None) or "main",
            "last_modified": info.lastModified,
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def check_model_update_available(hf_path: str, cache_path: str, token: Optional[str] = None) -> dict:
    """
    Check if a model update is available.

    Returns:
        Dict with update_available (bool), local_commit, remote_commit, version info
    """
    local_commit = get_local_model_commit(hf_path, cache_path)
    remote_info = get_remote_model_info(hf_path, token)

    if not remote_info["success"]:
        return {
            "update_available": False,
            "error": remote_info["error"],
            "local_commit": local_commit
        }

    remote_commit = remote_info["commit"]
    update_available = (local_commit != remote_commit) if local_commit else True

    return {
        "update_available": update_available,
        "local_commit": local_commit,
        "remote_commit": remote_commit,
        "remote_version": remote_info["version"],
        "last_modified": remote_info["last_modified"],
        "success": True
    }


def garbage_collect_model_blobs(hf_path: str, cache_path: str) -> dict:
    """
    Remove orphaned blobs that are not referenced by any snapshot.
    This cleans up old model versions after an update.

    Returns:
        Dict with deleted blob count and freed space
    """
    import os

    org_model = hf_path.replace("/", "--")
    model_cache_dir = os.path.join(cache_path, "hub", f"models--{org_model}")
    blobs_dir = os.path.join(model_cache_dir, "blobs")
    snapshots_dir = os.path.join(model_cache_dir, "snapshots")

    if not os.path.exists(blobs_dir) or not os.path.exists(snapshots_dir):
        return {"deleted_count": 0, "freed_mb": 0, "error": "Model cache not found"}

    try:
        # Get all blob files
        all_blobs = set()
        for blob_file in os.listdir(blobs_dir):
            blob_path = os.path.join(blobs_dir, blob_file)
            if os.path.isfile(blob_path):
                all_blobs.add(blob_file)

        # Get all referenced blobs (from symlinks in snapshots)
        referenced_blobs = set()
        for snapshot in os.listdir(snapshots_dir):
            snapshot_path = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(snapshot_path):
                for file in os.listdir(snapshot_path):
                    file_path = os.path.join(snapshot_path, file)
                    if os.path.islink(file_path):
                        # Resolve symlink and extract blob filename
                        target = os.readlink(file_path)
                        blob_name = os.path.basename(target)
                        referenced_blobs.add(blob_name)

        # Find orphaned blobs
        orphaned_blobs = all_blobs - referenced_blobs

        # Delete orphaned blobs
        deleted_count = 0
        freed_bytes = 0
        for blob in orphaned_blobs:
            blob_path = os.path.join(blobs_dir, blob)
            try:
                blob_size = os.path.getsize(blob_path)
                os.remove(blob_path)
                deleted_count += 1
                freed_bytes += blob_size
            except Exception as e:
                print(f"Error deleting blob {blob}: {e}")

        return {
            "deleted_count": deleted_count,
            "freed_mb": freed_bytes / (1024**2),
            "total_blobs": len(all_blobs),
            "referenced_blobs": len(referenced_blobs),
            "success": True
        }

    except Exception as e:
        return {
            "deleted_count": 0,
            "freed_mb": 0,
            "error": str(e),
            "success": False
        }
