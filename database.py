# database.py
from sqlmodel import Field, SQLModel, create_engine, Relationship
from datetime import datetime
from typing import Optional, List
from pydantic import ConfigDict
import os

# --- Table 1: The Model Registry ---
# Stores *metadata* about models, not the models themselves.
class ModelRegistry(SQLModel, table=True):
    model_config = ConfigDict(protected_namespaces=())

    id: Optional[int] = Field(default=None, primary_key=True)
    model_name: str = Field(unique=True, index=True)  # Your friendly name
    hf_path: str                                       # Hugging Face path
    trust_remote_code: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.now)

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
