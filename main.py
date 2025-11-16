# main.py
import torch
import time
import os
import psutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field

# --- Database Setup ---
from database import (
    ModelRegistry,
    PerformanceLog,
    engine,
    create_db_and_tables,
    get_gpu_metrics,
    get_system_metrics,
    hash_prompt,
    detect_repetition,
)


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
        self, hf_path: str, model_name: str, model_id: int, trust_remote_code: bool
    ):
        """Load a model into VRAM."""
        # 1. Unload any previous model
        self.unload()

        print(f"🔄 Loading model: {model_name} from {hf_path}...")
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_path, trust_remote_code=trust_remote_code
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
    model_name: str = Field(..., description="Friendly name for the model")
    hf_path: str = Field(..., description="Hugging Face model path or local path")
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code"
    )


class ModelLoadRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to load")


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt")
    max_tokens: int = Field(default=256, description="Maximum tokens to generate", ge=1, le=4096)
    temperature: float = Field(default=0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, description="Top-p sampling parameter", ge=0.0, le=1.0)
    do_sample: bool = Field(default=True, description="Whether to use sampling")


class LoggingConfigRequest(BaseModel):
    enable: bool = Field(..., description="Enable or disable performance logging")


class ModelResponse(BaseModel):
    id: int
    model_name: str
    hf_path: str
    trust_remote_code: bool
    created_at: datetime


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


# --- FastAPI App Setup ---
app = FastAPI(
    title="Homelab LLM Server",
    description="A robust API for managing and serving local language models",
    version="1.0.0",
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


# --- On Startup ---
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    print("🚀 Homelab LLM Server started")


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
    # Check if model already exists
    existing = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == request.model_name)
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{request.model_name}' already registered",
        )

    # Create new registry entry
    new_model = ModelRegistry(
        model_name=request.model_name,
        hf_path=request.hf_path,
        trust_remote_code=request.trust_remote_code,
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


@app.delete("/api/models/{model_name}", tags=["Model Registry"])
def delete_model(model_name: str, session: Session = Depends(get_db_session)):
    """Delete a model from the registry."""
    model = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == model_name)
    ).first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    # Check if this model is currently loaded
    if model_manager.loaded_model_name == model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete '{model_name}' - it is currently loaded. Unload it first.",
        )

    session.delete(model)
    session.commit()

    return {"message": f"Model '{model_name}' deleted successfully"}


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
    model = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == request.model_name)
    ).first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_name}' not found in registry. Register it first.",
        )

    # Load the model
    try:
        model_manager.load(
            hf_path=model.hf_path,
            model_name=model.model_name,
            model_id=model.id,
            trust_remote_code=model.trust_remote_code,
        )

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

        except Exception as e:
            print(f"Warning: Could not collect full model metadata: {e}")

        session.add(model)
        session.commit()

        return {
            "message": f"Model '{request.model_name}' loaded successfully",
            "model_id": model.id,
            "parameter_count": model.parameter_count,
            "model_type": model.model_type,
        }
    except Exception as e:
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
    model = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == model_name)
    ).first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    # Get all logs for this model
    logs = session.exec(
        select(PerformanceLog).where(PerformanceLog.model_registry_id == model.id)
    ).all()

    if not logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No performance logs found for model '{model_name}'",
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
    model = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == model_name)
    ).first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    # Get logs
    statement = (
        select(PerformanceLog)
        .where(PerformanceLog.model_registry_id == model.id)
        .order_by(PerformanceLog.timestamp.desc())
        .limit(limit)
    )
    logs = session.exec(statement).all()

    return {"model_name": model_name, "logs": logs, "count": len(logs)}


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
