# main.py
import torch
import time
import os
import re
import psutil
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select
from sqlalchemy import func
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from pydantic import BaseModel, Field, ConfigDict, field_validator

# --- Database Setup ---
from database import (
    ModelRegistry,
    PerformanceLog,
    CacheLocation,
    engine,
    create_db_and_tables,
    get_gpu_metrics,
    get_system_metrics,
    hash_prompt,
    detect_repetition,
    get_cache_config,
    check_cache_space,
    get_directory_size,
)

# --- Cache Management ---
from cache_manager import CacheManager


def _normalize_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip()


def normalize_model_name(value: str) -> str:
    """Trim whitespace from model names."""
    normalized = _normalize_string(value)
    return normalized or ""


def normalize_hf_path(value: str) -> str:
    """Trim whitespace around Hugging Face path segments."""
    normalized = _normalize_string(value)
    if not normalized:
        return ""
    return re.sub(r"\s*/\s*", "/", normalized)


# --- Global State: The Model Manager ---
class ModelManager:
    """Manages the in-memory model state (singleton)."""

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.loaded_model_name: Optional[str] = None
        self.loaded_model_id: Optional[int] = None
        self.performance_logging: bool = True  # On by default

    def load(
        self, hf_path: str, model_name: str, model_id: int, trust_remote_code: bool,
        cache_dir: Optional[str] = None, load_config_json: Optional[str] = None
    ):
        """Load a model into VRAM from specified cache location."""
        # 1. Unload any previous model
        self.unload()
        model_name = normalize_model_name(model_name)

        cache_info = f" (cache: {cache_dir})" if cache_dir else ""
        print(f"🔄 Loading model: {model_name} from {hf_path}{cache_info}...")
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            # Default load kwargs (safe defaults for ROCm/WSL)
            load_kwargs = {
                "torch_dtype": dtype,  # Using torch_dtype for compatibility with older transformers
                "device_map": "auto",
                "trust_remote_code": trust_remote_code,
                "attn_implementation": "eager",  # Use eager attention for ROCm/WSL compatibility
            }

            # Override with model-specific config if provided
            if load_config_json:
                try:
                    custom_config = json.loads(load_config_json)
                    print(f"📝 Using custom load config: {custom_config}")
                    load_kwargs.update(custom_config)
                except Exception as e:
                    print(f"⚠️  Failed to parse load_config, using defaults: {e}")

            if cache_dir:
                # Set cache directory for HuggingFace transformers
                load_kwargs["cache_dir"] = cache_dir

            # Workaround for Phi-3 and other models that check flash-attention early
            # Set environment variable to disable flash-attention before model import
            import os
            os.environ['FLASH_ATTENTION_SKIP_CUDA_BUILD'] = '1'

            # For problematic models, try to load config first and force eager attention
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    hf_path,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir if cache_dir else None
                )
                # Force eager attention in config
                if hasattr(config, '_attn_implementation'):
                    config._attn_implementation = 'eager'
                if hasattr(config, 'attn_implementation'):
                    config.attn_implementation = 'eager'
                # For Phi models specifically
                if 'Phi' in config.model_type or 'phi' in str(config.architectures):
                    print(f"Detected Phi model - forcing eager attention for ROCm compatibility")
                load_kwargs["config"] = config
            except Exception as e:
                print(f"Note: Could not pre-configure attention implementation: {e}")

            self.model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                **load_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_path,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir if cache_dir else None
            )
            self.loaded_model_name = model_name
            self.loaded_model_id = model_id
            print(f"✅ Model '{model_name}' loaded successfully.")
        except Exception as e:
            self.unload()  # Clean up on failure
            print(f"🔥 Failed to load model: {e}")
            raise e

    def unload(self):
        """Unload the current model and free VRAM."""
        if self.model:
            print(f"Unloading model: {self.loaded_model_name}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.loaded_model_name = None
            self.loaded_model_id = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ Model unloaded and VRAM cleared.")

    def validate(self) -> Dict[str, Any]:
        """Validate that the loaded model is working properly."""
        if not self.model or not self.tokenizer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model currently loaded",
            )

        try:
            # Run a simple test inference
            test_prompt = "Hello"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=5, do_sample=False
                )

            # If we got here, the model is working
            return {
                "status": "healthy",
                "model_name": self.loaded_model_name,
                "test_passed": True,
                "device": str(next(self.model.parameters()).device),
                "dtype": str(next(self.model.parameters()).dtype),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_name": self.loaded_model_name,
                "test_passed": False,
                "error": str(e),
            }


# --- Pydantic API Models ---
class ModelCreateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., description="Friendly name for the model")
    hf_path: str = Field(..., description="Hugging Face model path or local path")
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code"
    )
    cache_location: str = Field(
        default="primary", description="Cache location: primary, secondary, or custom"
    )
    cache_path: Optional[str] = Field(
        default=None, description="Custom cache path (required if cache_location is 'custom')"
    )
    estimated_size_mb: Optional[float] = Field(
        default=5000, description="Estimated model size in MB for space checking"
    )

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if isinstance(value, str):
            value = normalize_model_name(value)
        if not value:
            raise ValueError("model_name cannot be blank")
        return value

    @field_validator("hf_path", mode="before")
    @classmethod
    def validate_hf_path(cls, value: str) -> str:
        if isinstance(value, str):
            value = normalize_hf_path(value)
        if not value:
            raise ValueError("hf_path cannot be blank")
        return value

    @field_validator("cache_path", mode="before")
    @classmethod
    def validate_cache_path(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return value


class ModelLoadRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., description="Name of the model to load")
    force_load: bool = Field(default=False, description="Force load even if marked incompatible (for retesting)")

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if isinstance(value, str):
            value = normalize_model_name(value)
        if not value:
            raise ValueError("model_name cannot be blank")
        return value


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt")
    max_tokens: int = Field(default=256, description="Maximum tokens to generate", ge=1, le=4096)
    temperature: float = Field(default=0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, description="Top-p sampling parameter", ge=0.0, le=1.0)
    do_sample: bool = Field(default=True, description="Whether to use sampling")


class LoggingConfigRequest(BaseModel):
    enable: bool = Field(..., description="Enable or disable performance logging")


class ModelMetadataUpdateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Benchmark metrics
    mmlu_score: Optional[float] = None
    gpqa_score: Optional[float] = None
    hellaswag_score: Optional[float] = None
    humaneval_score: Optional[float] = None
    mbpp_score: Optional[float] = None
    math_score: Optional[float] = None
    truthfulqa_score: Optional[float] = None
    perplexity: Optional[float] = None

    # Operational metrics
    max_throughput_tokens_sec: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    quantization: Optional[str] = None


class ModelConfigUpdateRequest(BaseModel):
    """Request model for updating model-specific configuration."""
    model_config = ConfigDict(protected_namespaces=())

    load_config: Optional[str] = Field(default=None, description="JSON string of model loading parameters")
    compatibility_status: Optional[str] = Field(default=None, description="Compatibility status: unknown, compatible, incompatible, degraded")
    compatibility_notes: Optional[str] = Field(default=None, description="Notes about compatibility issues")


class ModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    model_name: str
    hf_path: str
    trust_remote_code: bool
    created_at: datetime

    # Cache location
    cache_location: str = "primary"
    cache_path: Optional[str] = None
    model_size_mb: Optional[float] = None

    # Model metadata
    parameter_count: Optional[int] = None
    model_type: Optional[str] = None
    architecture: Optional[str] = None
    default_dtype: Optional[str] = None
    context_length: Optional[int] = None

    # Benchmark metrics
    mmlu_score: Optional[float] = None
    gpqa_score: Optional[float] = None
    hellaswag_score: Optional[float] = None
    humaneval_score: Optional[float] = None
    mbpp_score: Optional[float] = None
    math_score: Optional[float] = None
    truthfulqa_score: Optional[float] = None
    perplexity: Optional[float] = None

    # Operational metrics
    max_throughput_tokens_sec: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    quantization: Optional[str] = None

    # Usage statistics
    total_loads: int = 0
    total_inferences: int = 0
    last_loaded: Optional[datetime] = None

    # Model-specific configuration
    load_config: Optional[str] = None
    compatibility_status: Optional[str] = "unknown"
    compatibility_notes: Optional[str] = None


class StatusResponse(BaseModel):
    loaded_model: Optional[str]
    loaded_model_id: Optional[int]
    performance_logging: bool
    gpu_available: bool
    gpu_memory_allocated_mb: Optional[float] = None
    gpu_memory_reserved_mb: Optional[float] = None


class GenerateResponse(BaseModel):
    generated_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    inference_time_ms: float
    tokens_per_second: float


class PerformanceStatsResponse(BaseModel):
    total_inferences: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_inference_ms: float
    avg_tokens_per_second: float
    avg_gpu_mem_mb: Optional[float] = None
    avg_cpu_mem_mb: Optional[float] = None


# --- Global Cache Manager Instance ---
cache_manager = CacheManager(check_interval=300)  # Check every 5 minutes


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    create_db_and_tables()

    # Check for models stuck in "testing" status (from server crashes)
    with Session(engine) as session:
        testing_models = session.exec(
            select(ModelRegistry).where(ModelRegistry.compatibility_status == "testing")
        ).all()

        if testing_models:
            print(f"⚠️  Found {len(testing_models)} model(s) stuck in 'testing' status from previous crash")
            for model in testing_models:
                # Mark as incompatible since the test didn't complete successfully
                model.compatibility_status = "incompatible"
                old_notes = model.compatibility_notes or ""
                crash_note = f"[Server crashed during compatibility test on {datetime.now().strftime('%Y-%m-%d %H:%M')}]"
                model.compatibility_notes = f"{crash_note} {old_notes}".strip()
                session.add(model)
                print(f"   - Marked '{model.model_name}' as incompatible (crashed during test)")

            session.commit()
            print("✅ Compatibility status cleanup completed")

    cache_manager.start_monitoring()
    print("🚀 Homelab LLM Server started")
    yield
    # Shutdown
    cache_manager.stop_monitoring()


# --- FastAPI App Setup ---
app = FastAPI(
    title="Homelab LLM Server",
    description="A robust API for managing and serving local language models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()  # Create the single global instance

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Database Dependency ---
def get_db_session():
    with Session(engine) as session:
        yield session


def _normalize_model_record(
    session: Session, model: ModelRegistry, normalized_name: str
) -> ModelRegistry:
    """Ensure model name/hf_path lack stray whitespace, persisting if possible."""
    updated = False
    cleaned_hf_path: Optional[str] = None

    if normalized_name and model.model_name != normalized_name:
        model.model_name = normalized_name
        updated = True

    if model.hf_path:
        cleaned_hf_path = normalize_hf_path(model.hf_path)
        if cleaned_hf_path != model.hf_path:
            model.hf_path = cleaned_hf_path
            updated = True

    if updated:
        try:
            session.add(model)
            session.commit()
        except Exception as exc:
            session.rollback()
            # Keep normalized values for the current request even if persisting failed
            model.model_name = normalized_name or model.model_name
            if cleaned_hf_path is not None:
                model.hf_path = cleaned_hf_path
            print(f"Warning: failed to normalize model record {model.id}: {exc}")

    return model


def get_model_by_name(session: Session, model_name: str) -> Optional[ModelRegistry]:
    """Fetch a model by name, ignoring stray whitespace."""
    normalized = normalize_model_name(model_name)
    if not normalized:
        return None

    model = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == normalized)
    ).first()

    if model:
        return _normalize_model_record(session, model, normalized)

    # Fallback for legacy rows with trailing spaces
    legacy_model = session.exec(
        select(ModelRegistry).where(func.trim(ModelRegistry.model_name) == normalized)
    ).first()

    if legacy_model:
        return _normalize_model_record(session, legacy_model, normalized)

    return None


# --- API Endpoints ---

# ===== Group 1: Model Registry (Database Ops) =====


@app.post(
    "/api/models",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Model Registry"],
)
def register_model(
    request: ModelCreateRequest, session: Session = Depends(get_db_session)
):
    """Register a new model with the API."""
    model_name = normalize_model_name(request.model_name)
    hf_path = normalize_hf_path(request.hf_path)

    # Check if model already exists
    existing = get_model_by_name(session, model_name)

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' already registered",
        )

    # Validate cache location
    if request.cache_location not in ["primary", "secondary", "custom"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache_location: {request.cache_location}. Must be 'primary', 'secondary', or 'custom'"
        )

    if request.cache_location == "custom" and not request.cache_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="cache_path is required when cache_location is 'custom'"
        )

    # Check if cache has sufficient space
    space_check = cache_manager.check_space_for_model(
        request.cache_location,
        request.estimated_size_mb or 5000
    )

    if not space_check["sufficient"]:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=f"Insufficient space in {request.cache_location} cache. "
                   f"Available: {space_check.get('disk_free_gb', 0):.1f}GB, "
                   f"Required: ~{(request.estimated_size_mb or 5000)/1024:.1f}GB. "
                   f"Consider using a different cache location."
        )

    # Get actual cache path
    actual_cache_path = cache_manager.get_cache_path(
        request.cache_location,
        request.cache_path
    )

    # Create new registry entry
    new_model = ModelRegistry(
        model_name=model_name,
        hf_path=hf_path,
        trust_remote_code=request.trust_remote_code,
        cache_location=request.cache_location,
        cache_path=actual_cache_path,
    )
    session.add(new_model)
    session.commit()
    session.refresh(new_model)

    return new_model


@app.get("/api/models", response_model=List[ModelResponse], tags=["Model Registry"])
def list_models(session: Session = Depends(get_db_session)):
    """List all registered models."""
    models = session.exec(select(ModelRegistry)).all()
    return models


@app.patch(
    "/api/models/{model_name}/metadata",
    response_model=ModelResponse,
    tags=["Model Registry"],
)
def update_model_metadata(
    model_name: str,
    request: ModelMetadataUpdateRequest,
    session: Session = Depends(get_db_session),
):
    """Update benchmark and operational metrics for a model."""
    normalized_name = normalize_model_name(model_name)
    model = get_model_by_name(session, normalized_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{normalized_name}' not found",
        )

    # Update only the fields that are provided (not None)
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(model, key, value)

    session.add(model)
    session.commit()
    session.refresh(model)

    return model


@app.patch(
    "/api/models/{model_name}/config",
    response_model=ModelResponse,
    tags=["Model Registry"],
)
def update_model_config(
    model_name: str,
    request: ModelConfigUpdateRequest,
    session: Session = Depends(get_db_session),
):
    """Update model-specific configuration (load params, compatibility status)."""
    normalized_name = normalize_model_name(model_name)
    model = get_model_by_name(session, normalized_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{normalized_name}' not found",
        )

    # Validate load_config is valid JSON if provided
    if request.load_config is not None:
        try:
            json.loads(request.load_config)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="load_config must be valid JSON",
            )

    # Validate compatibility_status
    if request.compatibility_status is not None:
        valid_statuses = ["unknown", "compatible", "incompatible", "degraded"]
        if request.compatibility_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"compatibility_status must be one of: {valid_statuses}",
            )

    # Update only the fields that are provided (not None)
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(model, key, value)

    session.add(model)
    session.commit()
    session.refresh(model)

    return model


@app.delete("/api/models/{model_name}", tags=["Model Registry"])
def delete_model(
    model_name: str,
    delete_files: bool = False,
    session: Session = Depends(get_db_session)
):
    """
    Delete a model from the registry.

    Args:
        model_name: Name of the model to delete
        delete_files: If True, also delete model files from disk (default: False)
    """
    normalized_name = normalize_model_name(model_name)
    model = get_model_by_name(session, normalized_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{normalized_name}' not found",
        )

    # Check if this model is currently loaded
    if model_manager.loaded_model_name and normalize_model_name(model_manager.loaded_model_name) == normalized_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete '{normalized_name}' - it is currently loaded. Unload it first.",
        )

    # Delete from database
    hf_path = normalize_hf_path(model.hf_path or "")
    cache_path = model.cache_path
    session.delete(model)
    session.commit()

    result = {"message": f"Model '{normalized_name}' deleted from registry"}

    # Optionally delete files from disk
    if delete_files and cache_path:
        import shutil

        # Construct the model directory path
        # HuggingFace stores models as: cache_path/hub/models--org--model/
        org_model = hf_path.replace("/", "--")
        model_dir = os.path.join(cache_path, "hub", f"models--{org_model}")

        if os.path.exists(model_dir):
            try:
                # Get size before deletion for reporting
                size_mb = get_directory_size(model_dir)

                # Delete the directory
                shutil.rmtree(model_dir)

                result["message"] += f" and files deleted from disk"
                result["freed_space_mb"] = size_mb
                result["deleted_path"] = model_dir

                print(f"🗑️  Deleted model files: {model_dir} ({size_mb:.1f} MB freed)")
            except Exception as e:
                result["message"] += f" but failed to delete files: {str(e)}"
                result["error"] = str(e)
        else:
            result["message"] += " (no files found on disk)"
            result["warning"] = f"Model directory not found: {model_dir}"

    return result


@app.get("/api/models/{model_name}/updates", tags=["Model Registry"])
def check_for_updates(
    model_name: str,
    session: Session = Depends(get_db_session)
):
    """
    Check if a model has updates available on HuggingFace Hub.

    Returns:
        update_available: bool
        local_commit: current commit hash
        remote_commit: latest commit hash
        version info and last modified date
    """
    from database import check_model_update_available

    normalized_name = normalize_model_name(model_name)
    model = get_model_by_name(session, normalized_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{normalized_name}' not found",
        )

    if not model.cache_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{normalized_name}' has no cache path - cannot check for updates",
        )

    # Check for updates
    hf_path = normalize_hf_path(model.hf_path or "")
    update_info = check_model_update_available(
        hf_path=hf_path,
        cache_path=model.cache_path,
        token=os.getenv("HF_TOKEN")
    )

    if not update_info.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check for updates: {update_info.get('error')}",
        )

    # Update cache in database
    model.update_available = update_info["update_available"]
    model.last_update_check = datetime.now()
    if not model.current_commit and update_info.get("local_commit"):
        model.current_commit = update_info["local_commit"]

    session.add(model)
    session.commit()

    return {
        "model_name": normalized_name,
        "update_available": update_info["update_available"],
        "local_commit": update_info.get("local_commit"),
        "remote_commit": update_info.get("remote_commit"),
        "remote_version": update_info.get("remote_version"),
        "last_modified": update_info.get("last_modified"),
        "checked_at": datetime.now().isoformat()
    }


@app.post("/api/models/{model_name}/update", tags=["Model Registry"])
def update_model(
    model_name: str,
    garbage_collect: bool = True,
    session: Session = Depends(get_db_session)
):
    """
    Update a model to the latest version from HuggingFace Hub.

    Args:
        model_name: Name of the model to update
        garbage_collect: If True, clean up old blobs after update (default: True)

    Note:
        - HuggingFace automatically reuses unchanged blobs
        - Only new/changed files are downloaded
        - Old snapshots are kept unless garbage_collect=True
    """
    from database import get_local_model_commit, garbage_collect_model_blobs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    normalized_name = normalize_model_name(model_name)
    model = get_model_by_name(session, normalized_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{normalized_name}' not found",
        )

    # Check if model is currently loaded
    if model_manager.loaded_model_name and normalize_model_name(model_manager.loaded_model_name) == normalized_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update '{normalized_name}' - it is currently loaded. Unload it first.",
        )

    if not model.cache_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{normalized_name}' has no cache path",
        )

    hf_path = normalize_hf_path(model.hf_path or "")
    # Get current version before update
    old_commit = get_local_model_commit(hf_path, model.cache_path)

    try:
        # Download latest version (HuggingFace reuses unchanged blobs automatically)
        print(f"📥 Downloading latest version of {hf_path}...")

        # Download model and tokenizer
        AutoModelForCausalLM.from_pretrained(
            hf_path,
            cache_dir=model.cache_path,
            trust_remote_code=model.trust_remote_code,
            revision="main",  # Always get latest
            # Don't load into memory, just download
            low_cpu_mem_usage=True
        )

        AutoTokenizer.from_pretrained(
            hf_path,
            cache_dir=model.cache_path,
            trust_remote_code=model.trust_remote_code,
            revision="main"
        )

        # Get new version
        new_commit = get_local_model_commit(hf_path, model.cache_path)

        # Update model record
        model.current_commit = new_commit
        model.last_updated = datetime.now()
        model.update_available = False

        session.add(model)
        session.commit()

        result = {
            "message": f"Model '{normalized_name}' updated successfully",
            "old_commit": old_commit,
            "new_commit": new_commit,
            "updated_at": datetime.now().isoformat()
        }

        # Garbage collect old blobs if requested
        if garbage_collect:
            print(f"🧹 Cleaning up orphaned blobs...")
            gc_result = garbage_collect_model_blobs(hf_path, model.cache_path)

            if gc_result.get("success"):
                result["garbage_collection"] = {
                    "deleted_blobs": gc_result["deleted_count"],
                    "freed_mb": gc_result["freed_mb"],
                    "total_blobs": gc_result["total_blobs"],
                    "referenced_blobs": gc_result["referenced_blobs"]
                }
                print(f"🗑️  Deleted {gc_result['deleted_count']} orphaned blobs ({gc_result['freed_mb']:.1f} MB freed)")
            else:
                result["garbage_collection_error"] = gc_result.get("error")

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {str(e)}",
        )


# ===== Group 2: Orchestration (State Ops) =====


@app.get("/api/status", response_model=StatusResponse, tags=["Orchestration"])
def get_status():
    """Check which model is currently loaded in VRAM."""
    gpu_allocated = None
    gpu_reserved = None

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        gpu_reserved = torch.cuda.memory_reserved() / (1024**2)  # MB

    return StatusResponse(
        loaded_model=model_manager.loaded_model_name,
        loaded_model_id=model_manager.loaded_model_id,
        performance_logging=model_manager.performance_logging,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_allocated_mb=gpu_allocated,
        gpu_memory_reserved_mb=gpu_reserved,
    )


@app.post("/api/orchestrate/load", tags=["Orchestration"])
def load_model(request: ModelLoadRequest, session: Session = Depends(get_db_session)):
    """Load a model into VRAM. This will be a slow request."""
    # Look up the model in the registry
    requested_name = normalize_model_name(request.model_name)
    model = get_model_by_name(session, requested_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{requested_name}' not found in registry. Register it first.",
        )

    # Check compatibility status (unless force_load is True)
    if model.compatibility_status == "incompatible" and not request.force_load:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{requested_name}' is marked as incompatible with this hardware. "
                   f"Notes: {model.compatibility_notes or 'No details available'}. "
                   f"Use force_load=true to retry anyway.",
        )

    # Warn about degraded models (but allow loading)
    if model.compatibility_status == "degraded":
        print(f"⚠️  Loading degraded model '{requested_name}' - may have intermittent issues. "
              f"Notes: {model.compatibility_notes or 'Unknown issue'}")

    # Log force load attempt
    if request.force_load and model.compatibility_status == "incompatible":
        print(f"⚠️  Force loading incompatible model '{requested_name}' - testing for intermittent issues")

    # IMPORTANT: Update database BEFORE attempting to load to track compatibility testing
    # This ensures that even if the server crashes, we have a record of the attempt
    original_status = model.compatibility_status
    model.compatibility_status = "testing"
    session.add(model)
    session.commit()
    print(f"🔍 Testing compatibility for '{requested_name}'...")

    # Load the model with cache location and custom config
    hf_path = normalize_hf_path(model.hf_path or "")
    try:
        model_manager.load(
            hf_path=hf_path,
            model_name=model.model_name,
            model_id=model.id,
            trust_remote_code=model.trust_remote_code,
            cache_dir=model.cache_path,
            load_config_json=model.load_config,
        )

        # Mark as compatible on successful load
        if original_status == "unknown" or original_status == "testing":
            model.compatibility_status = "compatible"
        else:
            # Restore original status if it was already set (e.g., degraded)
            model.compatibility_status = original_status

        # Update model metadata in registry
        model.total_loads += 1
        model.last_loaded = datetime.now()

        # Try to get model metadata
        try:
            # Count parameters
            param_count = sum(
                p.numel() for p in model_manager.model.parameters()
            )
            model.parameter_count = param_count

            # Get model type
            model.model_type = model_manager.model.__class__.__name__

            # Try to get architecture from config
            if hasattr(model_manager.model, "config"):
                config = model_manager.model.config
                if hasattr(config, "model_type"):
                    model.architecture = config.model_type
                if hasattr(config, "max_position_embeddings"):
                    model.context_length = config.max_position_embeddings

            # Get dtype
            model.default_dtype = str(next(model_manager.model.parameters()).dtype)

            # Calculate actual model size on disk (if not already set)
            if not model.model_size_mb and model.cache_path:
                model_size_mb = get_directory_size(model.cache_path)
                model.model_size_mb = model_size_mb

            # Capture version information if not already set
            if not model.current_commit and model.cache_path:
                from database import get_local_model_commit
                commit = get_local_model_commit(hf_path, model.cache_path)
                if commit:
                    model.current_commit = commit
                    model.current_version = "main"  # Default to main branch
                    if not model.last_updated:
                        model.last_updated = datetime.now()

        except Exception as e:
            print(f"Warning: Could not collect full model metadata: {e}")

        session.add(model)
        session.commit()

        return {
            "message": f"Model '{requested_name}' loaded successfully",
            "model_id": model.id,
            "parameter_count": model.parameter_count,
            "model_type": model.model_type,
            "compatibility_status": model.compatibility_status,
        }
    except Exception as e:
        # Check if it's a flash-attention or ROCm compatibility issue
        error_str = str(e)

        # Determine compatibility status based on error type and history
        if "flash-attention" in error_str.lower() or "createcontext" in error_str.lower() or \
           "assertion" in error_str.lower() or "aborted" in error_str.lower():
            # Intermittent issue detection: if model loaded successfully 3+ times before,
            # it's likely a transient issue (WSL GPU reset, driver glitch, etc.)
            if model.total_loads >= 3:
                model.compatibility_status = "degraded"
                model.compatibility_notes = f"Intermittent issue detected (loaded successfully {model.total_loads} times before). " \
                                          f"May require WSL/GPU restart. Error: {error_str[:150]}"
                print(f"⚠️  Intermittent issue detected for '{model.model_name}' - marking as degraded (was loaded {model.total_loads} times successfully)")
            else:
                # Permanent incompatibility - model has never/rarely loaded successfully
                model.compatibility_status = "incompatible"
                model.compatibility_notes = f"Hardware incompatibility detected: {error_str[:200]}"
                print(f"❌ Marking '{model.model_name}' as incompatible - no successful load history")
        else:
            # Generic error - mark as incompatible but with generic message
            model.compatibility_status = "incompatible"
            model.compatibility_notes = f"Load failed with error: {error_str[:200]}"
            print(f"❌ Marking '{model.model_name}' as incompatible - generic load error")

        # CRITICAL: Save to database even if server crashes after this
        session.add(model)
        session.commit()
        print(f"💾 Compatibility status saved to database: {model.compatibility_status}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@app.post("/api/orchestrate/unload", tags=["Orchestration"])
def unload_model():
    """Unload the current model and free VRAM."""
    if not model_manager.model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No model currently loaded"
        )

    model_name = model_manager.loaded_model_name
    model_manager.unload()

    return {"message": f"Model '{model_name}' unloaded successfully"}


@app.get("/api/orchestrate/validate", tags=["Orchestration"])
def validate_model():
    """Validate that the loaded model is working properly."""
    return model_manager.validate()


# ===== Group 3: Configuration (State Ops) =====


@app.get("/api/config/logging", tags=["Configuration"])
def get_logging_config():
    """Check if performance logging is enabled."""
    return {"performance_logging": model_manager.performance_logging}


@app.post("/api/config/logging", tags=["Configuration"])
def set_logging_config(request: LoggingConfigRequest):
    """Enable or disable performance logging (efficiency mode)."""
    model_manager.performance_logging = request.enable
    return {
        "message": f"Performance logging {'enabled' if request.enable else 'disabled'}",
        "performance_logging": model_manager.performance_logging,
    }


# ===== Group 4: Inference (The "Work") =====


@app.post(
    "/api/generate", response_model=GenerateResponse, tags=["Inference"]
)
def generate_text(
    request: PromptRequest, session: Session = Depends(get_db_session)
):
    """Run inference - this is the hot path."""
    # Check if model is loaded
    if not model_manager.model or not model_manager.tokenizer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model currently loaded. Load a model first using /api/orchestrate/load",
        )

    # Prepare logging (if enabled)
    process = None
    start_time = None
    tokenization_end_time = None
    first_token_time = None
    gpu_mem_before = None
    gpu_metrics_before = None
    system_metrics = None

    if model_manager.performance_logging:
        process = psutil.Process()
        start_time = time.perf_counter()

        # Collect GPU metrics before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_mem_before = torch.cuda.memory_allocated()

        # Collect GPU utilization, temp, power
        gpu_metrics_before = get_gpu_metrics()

        # Collect system metrics
        system_metrics = get_system_metrics()

    # Tokenize input
    inputs = model_manager.tokenizer(request.prompt, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]

    if model_manager.performance_logging:
        tokenization_end_time = time.perf_counter()

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate
    error_occurred = False
    error_message = None

    try:
        with torch.no_grad():
            # For TTFT measurement, we'd need streaming support
            # For now, we'll approximate it
            outputs = model_manager.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
            )

            if model_manager.performance_logging:
                first_token_time = time.perf_counter()

    except Exception as e:
        error_occurred = True
        error_message = str(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )

    # Decode output
    generated_text = model_manager.tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )
    output_length = outputs.shape[1] - input_length

    # Capture metrics (if enabled)
    inference_time_ms = 0
    ttft_ms = None
    gpu_mem_peak_mb = None
    gpu_mem_reserved_mb = None
    cpu_mem_mb = None
    tokens_per_second = 0

    if model_manager.performance_logging:
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        # Approximate TTFT (in real implementation, would need streaming)
        if first_token_time:
            ttft_ms = (first_token_time - tokenization_end_time) * 1000

        tokens_per_second = (
            output_length / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
        )

        # GPU memory metrics
        if torch.cuda.is_available():
            gpu_mem_peak = torch.cuda.max_memory_allocated()
            gpu_mem_peak_mb = gpu_mem_peak / (1024**2)
            gpu_mem_reserved_mb = torch.cuda.memory_reserved() / (1024**2)

        # CPU memory
        cpu_mem_mb = process.memory_info().rss / (1024**2)

        # Detect repetition in output
        repetition_flag = detect_repetition(generated_text)

        # Hash the prompt for duplicate detection
        prompt_hash_val = hash_prompt(request.prompt)

        # Get model configuration
        model_device = str(next(model_manager.model.parameters()).device)
        model_dtype = str(next(model_manager.model.parameters()).dtype)

        # Update model inference count in registry
        model_record = session.exec(
            select(ModelRegistry).where(
                ModelRegistry.id == model_manager.loaded_model_id
            )
        ).first()
        if model_record:
            model_record.total_inferences += 1
            session.add(model_record)

        # Save comprehensive log to database
        log_entry = PerformanceLog(
            model_registry_id=model_manager.loaded_model_id,
            model_version=model_record.current_commit if model_record else None,
            # Token metrics
            input_tokens=input_length,
            output_tokens=output_length,
            total_tokens=input_length + output_length,
            # Timing metrics
            total_inference_ms=inference_time_ms,
            time_to_first_token_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            # Memory metrics
            gpu_mem_peak_alloc_mb=gpu_mem_peak_mb,
            gpu_mem_reserved_mb=gpu_mem_reserved_mb,
            cpu_mem_rss_mb=cpu_mem_mb,
            # GPU metrics
            gpu_utilization_percent=gpu_metrics_before.get("utilization"),
            gpu_temperature_celsius=gpu_metrics_before.get("temperature"),
            gpu_power_watts=gpu_metrics_before.get("power"),
            # System metrics
            cpu_utilization_percent=system_metrics.get("cpu_percent"),
            system_load_1min=system_metrics.get("load_avg"),
            # Generation parameters
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
            max_new_tokens=request.max_tokens,
            # Quality metrics
            prompt_hash=prompt_hash_val,
            repetition_detected=repetition_flag,
            # Error tracking
            error_occurred=error_occurred,
            error_message=error_message,
            # Model configuration
            model_dtype=model_dtype,
            model_device=model_device,
        )
        session.add(log_entry)
        session.commit()
    else:
        # Still calculate basic metrics for response
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        tokens_per_second = (
            output_length / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
        )

    return GenerateResponse(
        generated_text=generated_text,
        input_tokens=input_length,
        output_tokens=output_length,
        total_tokens=input_length + output_length,
        inference_time_ms=inference_time_ms,
        tokens_per_second=tokens_per_second,
    )


@app.post("/api/generate/stream", tags=["Inference"])
async def generate_text_stream(
    request: PromptRequest, session: Session = Depends(get_db_session)
):
    """Run inference with token streaming - provides real-time token generation."""
    # Check if model is loaded
    if not model_manager.model or not model_manager.tokenizer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model currently loaded. Load a model first using /api/orchestrate/load",
        )

    async def generate_stream():
        start_time = time.perf_counter()
        ttft = None
        input_length = 0
        output_length = 0
        error_occurred = False
        error_message = None

        try:
            # Tokenize input
            inputs = model_manager.tokenizer(request.prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'input_tokens': input_length})}\n\n"

            # Create streamer with timeout to prevent buffering
            streamer = TextIteratorStreamer(
                model_manager.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=0.1  # Small timeout to ensure immediate yielding
            )

            # Generation kwargs
            generation_kwargs = {
                **inputs,
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": request.do_sample,
                "streamer": streamer,
            }

            # Run generation in a thread to avoid blocking
            generation_thread = threading.Thread(
                target=lambda: model_manager.model.generate(**generation_kwargs)
            )
            generation_thread.start()

            # Stream tokens
            token_count = 0
            import sys
            for token_text in streamer:
                if ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000
                    print(f"TTFT: {ttft}ms", flush=True)  # Debug

                token_count += 1
                chunk = f"data: {json.dumps({'type': 'token', 'text': token_text})}\n\n"
                print(f"Yielding token {token_count}: {repr(token_text)}", flush=True)  # Debug
                yield chunk

                # Ensure immediate flush by yielding empty string
                # This forces the response to be sent immediately
                sys.stdout.flush()

            generation_thread.join()
            output_length = token_count
            print(f"Total tokens generated: {output_length}", flush=True)  # Debug

        except Exception as e:
            error_occurred = True
            error_message = str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return

        # Calculate final stats
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        tokens_per_second = (
            output_length / (total_time_ms / 1000) if total_time_ms > 0 else 0
        )

        # Log to database if enabled
        if model_manager.performance_logging and not error_occurred:
            try:
                # Get metrics
                gpu_mem_peak_mb = None
                if torch.cuda.is_available():
                    gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

                model_device = str(next(model_manager.model.parameters()).device)
                model_dtype = str(next(model_manager.model.parameters()).dtype)

                # Update model inference count
                model_record = session.exec(
                    select(ModelRegistry).where(
                        ModelRegistry.id == model_manager.loaded_model_id
                    )
                ).first()
                if model_record:
                    model_record.total_inferences += 1
                    session.add(model_record)

                # Save log
                log_entry = PerformanceLog(
                    model_registry_id=model_manager.loaded_model_id,
                    input_tokens=input_length,
                    output_tokens=output_length,
                    total_tokens=input_length + output_length,
                    total_inference_ms=total_time_ms,
                    time_to_first_token_ms=ttft,
                    tokens_per_second=tokens_per_second,
                    gpu_mem_peak_alloc_mb=gpu_mem_peak_mb,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    max_new_tokens=request.max_tokens,
                    model_dtype=model_dtype,
                    model_device=model_device,
                    error_occurred=error_occurred,
                    error_message=error_message,
                )
                session.add(log_entry)
                session.commit()
            except Exception as e:
                print(f"Warning: Failed to log performance: {e}")

        # Send final stats
        final_stats = {
            'type': 'done',
            'input_tokens': input_length,
            'output_tokens': output_length,
            'total_tokens': input_length + output_length,
            'inference_time_ms': total_time_ms,
            'ttft_ms': ttft,
            'tokens_per_second': tokens_per_second,
        }
        yield f"data: {json.dumps(final_stats)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ===== Group 5: Performance Analytics =====


@app.get(
    "/api/analytics/performance/{model_name}",
    response_model=PerformanceStatsResponse,
    tags=["Analytics"],
)
def get_performance_stats(
    model_name: str, session: Session = Depends(get_db_session)
):
    """Get aggregated performance statistics for a specific model."""
    # Find the model
    normalized_name = normalize_model_name(model_name)
    model = get_model_by_name(session, normalized_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{normalized_name}' not found",
        )

    # Get all logs for this model
    logs = session.exec(
        select(PerformanceLog).where(PerformanceLog.model_registry_id == model.id)
    ).all()

    if not logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No performance logs found for model '{normalized_name}'",
        )

    # Calculate statistics
    total = len(logs)
    avg_input = sum(log.input_tokens for log in logs) / total
    avg_output = sum(log.output_tokens for log in logs) / total
    avg_inference = sum(log.total_inference_ms for log in logs) / total
    avg_tps = sum(log.tokens_per_second for log in logs if log.tokens_per_second) / total

    gpu_logs = [log.gpu_mem_peak_alloc_mb for log in logs if log.gpu_mem_peak_alloc_mb]
    avg_gpu = sum(gpu_logs) / len(gpu_logs) if gpu_logs else None

    cpu_logs = [log.cpu_mem_rss_mb for log in logs if log.cpu_mem_rss_mb]
    avg_cpu = sum(cpu_logs) / len(cpu_logs) if cpu_logs else None

    return PerformanceStatsResponse(
        total_inferences=total,
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        avg_inference_ms=avg_inference,
        avg_tokens_per_second=avg_tps,
        avg_gpu_mem_mb=avg_gpu,
        avg_cpu_mem_mb=avg_cpu,
    )


@app.get("/api/analytics/logs/{model_name}", tags=["Analytics"])
def get_performance_logs(
    model_name: str,
    limit: int = 100,
    session: Session = Depends(get_db_session),
):
    """Get recent performance logs for a specific model."""
    # Find the model
    normalized_name = normalize_model_name(model_name)
    model = get_model_by_name(session, normalized_name)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{normalized_name}' not found",
        )

    # Get logs
    statement = (
        select(PerformanceLog)
        .where(PerformanceLog.model_registry_id == model.id)
        .order_by(PerformanceLog.timestamp.desc())
        .limit(limit)
    )
    logs = session.exec(statement).all()

    return {"model_name": normalized_name, "logs": logs, "count": len(logs)}


# ===== Group 6: Cache Management =====


@app.get("/api/cache/orphaned", tags=["Cache Management"])
def find_orphaned_models(session: Session = Depends(get_db_session)):
    """
    Find model files on disk that are not in the database (orphaned models).

    Returns:
        List of orphaned model directories that can be safely deleted
    """
    config = get_cache_config()
    orphaned = []

    for cache_type in ["primary", "secondary"]:
        cache_path = config["primary_path"] if cache_type == "primary" else config["secondary_path"]
        hub_path = os.path.join(cache_path, "hub")

        if not os.path.exists(hub_path):
            continue

        # List all model directories
        try:
            for item in os.listdir(hub_path):
                if item.startswith("models--"):
                    model_dir = os.path.join(hub_path, item)

                    # Extract org and model name
                    # Format: models--org--model
                    parts = item.replace("models--", "").split("--")
                    if len(parts) >= 2:
                        hf_path = "/".join(parts)  # org/model

                        # Check if this model exists in database
                        db_model = session.exec(
                            select(ModelRegistry).where(ModelRegistry.hf_path == hf_path)
                        ).first()

                        if not db_model:
                            # Orphaned - not in database
                            size_mb = get_directory_size(model_dir)
                            orphaned.append({
                                "hf_path": hf_path,
                                "directory": model_dir,
                                "cache_type": cache_type,
                                "size_mb": size_mb,
                                "size_gb": size_mb / 1024
                            })
        except Exception as e:
            print(f"Error scanning {hub_path}: {e}")

    return {
        "orphaned_models": orphaned,
        "count": len(orphaned),
        "total_size_mb": sum(m["size_mb"] for m in orphaned),
        "total_size_gb": sum(m["size_gb"] for m in orphaned)
    }


@app.post("/api/cache/cleanup", tags=["Cache Management"])
def cleanup_orphaned_models(
    hf_paths: List[str],
    session: Session = Depends(get_db_session)
):
    """
    Delete orphaned model files from disk.

    Args:
        hf_paths: List of HuggingFace paths to delete (e.g., ["Qwen/Qwen2.5-3B-Instruct"])

    Returns:
        Summary of deleted files
    """
    import shutil

    config = get_cache_config()
    deleted = []
    errors = []

    for hf_path in hf_paths:
        # Check database to ensure it's actually orphaned
        db_model = session.exec(
            select(ModelRegistry).where(ModelRegistry.hf_path == hf_path)
        ).first()

        if db_model:
            errors.append({
                "hf_path": hf_path,
                "error": "Model exists in database - use DELETE /api/models/{name} instead"
            })
            continue

        # Try both caches
        org_model = hf_path.replace("/", "--")
        deleted_from_cache = False

        for cache_type in ["primary", "secondary"]:
            cache_path = config["primary_path"] if cache_type == "primary" else config["secondary_path"]
            model_dir = os.path.join(cache_path, "hub", f"models--{org_model}")

            if os.path.exists(model_dir):
                try:
                    size_mb = get_directory_size(model_dir)
                    shutil.rmtree(model_dir)
                    deleted.append({
                        "hf_path": hf_path,
                        "directory": model_dir,
                        "cache_type": cache_type,
                        "freed_space_mb": size_mb
                    })
                    deleted_from_cache = True
                    print(f"🗑️  Cleaned up orphaned model: {model_dir} ({size_mb:.1f} MB)")
                except Exception as e:
                    errors.append({
                        "hf_path": hf_path,
                        "directory": model_dir,
                        "error": str(e)
                    })

        if not deleted_from_cache:
            errors.append({
                "hf_path": hf_path,
                "error": "Model directory not found in any cache"
            })

    return {
        "deleted": deleted,
        "deleted_count": len(deleted),
        "total_freed_mb": sum(d["freed_space_mb"] for d in deleted),
        "errors": errors,
        "error_count": len(errors)
    }


@app.get("/api/cache/stats", tags=["Cache Management"])
def get_cache_stats():
    """Get current cache statistics for all cache locations."""
    summary = cache_manager.get_cache_summary()
    return summary


@app.get("/api/cache/config", tags=["Cache Management"])
def get_cache_config_endpoint():
    """Get cache configuration settings."""
    config = get_cache_config()
    return config


@app.post("/api/cache/check", tags=["Cache Management"])
def check_cache_space_endpoint(
    cache_location: str = "primary",
    required_space_mb: float = 0
):
    """
    Check if cache location has sufficient space for a model.

    Args:
        cache_location: "primary", "secondary", or "custom"
        required_space_mb: Required space in MB (e.g., estimated model size)

    Returns:
        Space availability information
    """
    space_info = cache_manager.check_space_for_model(cache_location, required_space_mb)
    return space_info


@app.get("/api/cache/recommend", tags=["Cache Management"])
def get_recommended_cache(estimated_size_mb: float = 5000):
    """
    Get recommended cache location based on available space.

    Args:
        estimated_size_mb: Estimated model size in MB

    Returns:
        Recommended cache location ("primary" or "secondary")
    """
    recommended = cache_manager.get_recommended_cache(estimated_size_mb)
    return {
        "recommended_cache": recommended,
        "estimated_size_mb": estimated_size_mb,
        "estimated_size_gb": estimated_size_mb / 1024,
    }


# ===== Health Check =====


@app.get("/health", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
    }


# ===== Serve UI =====


@app.get("/")
def serve_ui():
    """Serve the web UI."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
