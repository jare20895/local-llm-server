#!/usr/bin/env python3
"""
Seed the SQLite database with repeatable ModelRegistry and PerformanceLog data.

The script is idempotent: it inserts new records only when they are missing and
never mutates existing rows (e.g., an existing model with the same name is left
untouched). Run with the same DATABASE_PATH that the app uses, for example:

    DATABASE_PATH=data/models.db python seed_database.py
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Default to the repo-local database unless the caller already exported a path.
os.environ.setdefault("DATABASE_PATH", "data/models.db")

from sqlmodel import Session, select

from database import ModelRegistry, PerformanceLog, create_db_and_tables, engine


NOW = datetime.utcnow()


def build_model_seeds() -> List[Dict]:
    """Return dictionaries describing the seed ModelRegistry rows."""
    return [
        {
            "model_name": "Local Llama 3B Instruct",
            "hf_path": "meta-llama/Llama-3.1-3B-Instruct",
            "cache_location": "primary",
            "cache_path": "/home/jare16/.cache/huggingface/llama-3.1-3b",
            "model_size_mb": 4210.8,
            "parameter_count": 3_200_000_000,
            "model_type": "LlamaForCausalLM",
            "architecture": "llama",
            "default_dtype": "bfloat16",
            "context_length": 8192,
            "mmlu_score": 74.6,
            "gpqa_score": 43.1,
            "hellaswag_score": 84.2,
            "humaneval_score": 62.4,
            "mbpp_score": 71.5,
            "math_score": 44.0,
            "truthfulqa_score": 67.3,
            "perplexity": 5.8,
            "max_throughput_tokens_sec": 62.4,
            "avg_latency_ms": 210.3,
            "quantization": "none",
            "total_loads": 7,
            "total_inferences": 42,
            "last_loaded": NOW - timedelta(days=2, hours=5),
            "current_commit": "9d2b1c7",
            "current_version": "2024-11-qa",
            "last_updated": NOW - timedelta(days=5),
            "update_available": False,
            "last_update_check": NOW - timedelta(days=1),
            "load_config": '{"dtype":"bfloat16","device":"cuda:0"}',
            "compatibility_status": "compatible",
            "compatibility_notes": "Validated on single GPU workstation.",
        },
        {
            "model_name": "Edge TinyMix 1B",
            "hf_path": "local/edge-tinymix-1b",
            "trust_remote_code": True,
            "cache_location": "secondary",
            "cache_path": "/mnt/models/edge-tinymix-1b",
            "model_size_mb": 980.2,
            "parameter_count": 1_050_000_000,
            "model_type": "MambaForCausalLM",
            "architecture": "mamba",
            "default_dtype": "float16",
            "context_length": 4096,
            "mmlu_score": 54.8,
            "gpqa_score": 33.0,
            "hellaswag_score": 71.1,
            "humaneval_score": 38.5,
            "mbpp_score": 45.0,
            "math_score": 24.7,
            "truthfulqa_score": 58.9,
            "perplexity": 9.7,
            "max_throughput_tokens_sec": 118.5,
            "avg_latency_ms": 95.2,
            "quantization": "GPTQ 4-bit",
            "total_loads": 15,
            "total_inferences": 220,
            "last_loaded": NOW - timedelta(days=1, hours=3),
            "current_commit": "d41a11b",
            "current_version": "v1.2.0-beta",
            "last_updated": NOW - timedelta(days=12),
            "update_available": True,
            "last_update_check": NOW - timedelta(hours=6),
            "load_config": '{"quant":"gptq","group_size":128}',
            "compatibility_status": "degraded",
            "compatibility_notes": "Runs on consumer GPU when max_new_tokens <= 2048.",
        },
    ]


def build_log_seeds() -> List[Dict]:
    """Return dictionaries describing the seed PerformanceLog rows."""
    return [
        {
            "model_name": "Local Llama 3B Instruct",
            "timestamp": NOW - timedelta(hours=2),
            "model_version": "llama-3.1-3b-vqa-2024-11",
            "input_tokens": 1536,
            "output_tokens": 384,
            "total_tokens": 1920,
            "total_inference_ms": 940.0,
            "time_to_first_token_ms": 220.0,
            "tokens_per_second": 2042.5,
            "gpu_mem_peak_alloc_mb": 13200.0,
            "gpu_mem_reserved_mb": 15360.0,
            "cpu_mem_rss_mb": 4200.0,
            "gpu_utilization_percent": 86.0,
            "gpu_temperature_celsius": 67.0,
            "gpu_power_watts": 295.0,
            "cpu_utilization_percent": 42.5,
            "system_load_1min": 4.2,
            "temperature": 0.15,
            "top_p": 0.9,
            "top_k": 40,
            "do_sample": True,
            "max_new_tokens": 512,
            "prompt_hash": "seed-local-llama-gen-202501",
            "repetition_detected": False,
            "error_occurred": False,
            "error_message": None,
            "model_dtype": "torch.bfloat16",
            "model_device": "cuda:0",
            "quantization": "none",
        },
        {
            "model_name": "Edge TinyMix 1B",
            "timestamp": NOW - timedelta(days=1, hours=1),
            "model_version": "edge-tinymix-1b-q4",
            "input_tokens": 768,
            "output_tokens": 256,
            "total_tokens": 1024,
            "total_inference_ms": 610.0,
            "time_to_first_token_ms": 140.0,
            "tokens_per_second": 1678.7,
            "gpu_mem_peak_alloc_mb": 5100.0,
            "gpu_mem_reserved_mb": 5632.0,
            "cpu_mem_rss_mb": 2500.0,
            "gpu_utilization_percent": 63.0,
            "gpu_temperature_celsius": 59.0,
            "gpu_power_watts": 180.0,
            "cpu_utilization_percent": 35.0,
            "system_load_1min": 2.1,
            "temperature": 0.35,
            "top_p": 0.92,
            "top_k": 80,
            "do_sample": True,
            "max_new_tokens": 256,
            "prompt_hash": "seed-edge-tinymix-benchmark-202501",
            "repetition_detected": False,
            "error_occurred": False,
            "error_message": None,
            "model_dtype": "torch.float16",
            "model_device": "cuda:0",
            "quantization": "gptq-4bit",
        },
    ]


def get_or_create_model(session: Session, payload: Dict) -> Tuple[ModelRegistry, bool]:
    """Fetch an existing model by name or create it."""
    model = session.exec(
        select(ModelRegistry).where(ModelRegistry.model_name == payload["model_name"])
    ).first()
    if model:
        return model, False

    model = ModelRegistry(**payload)
    session.add(model)
    session.commit()
    session.refresh(model)
    return model, True


def log_exists(session: Session, prompt_hash: str) -> bool:
    """Return True if a performance log with the prompt hash already exists."""
    result = session.exec(
        select(PerformanceLog).where(PerformanceLog.prompt_hash == prompt_hash)
    ).first()
    return result is not None


def seed_database() -> None:
    """Create tables (if missing) and insert the seed data."""
    create_db_and_tables()

    model_seeds = build_model_seeds()
    log_seeds = build_log_seeds()

    with Session(engine) as session:
        model_cache: Dict[str, ModelRegistry] = {}
        new_models = skipped_models = 0

        for entry in model_seeds:
            model, created = get_or_create_model(session, entry)
            model_cache[model.model_name] = model
            if created:
                new_models += 1
                print(f"✓ Created model '{model.model_name}' (id={model.id})")
            else:
                skipped_models += 1
                print(f"- Model '{model.model_name}' already exists, skipping")

        new_logs = skipped_logs = 0

        for entry in log_seeds:
            prompt_hash = entry["prompt_hash"]
            if log_exists(session, prompt_hash):
                skipped_logs += 1
                print(f"- Performance log '{prompt_hash}' already exists, skipping")
                continue

            entry_copy = deepcopy(entry)
            model_name = entry_copy.pop("model_name")
            model = model_cache.get(model_name)
            if model is None:
                model = session.exec(
                    select(ModelRegistry).where(ModelRegistry.model_name == model_name)
                ).first()

            if model is None:
                print(f"! Unable to find model '{model_name}' for log '{prompt_hash}'")
                skipped_logs += 1
                continue

            entry_copy["model_registry_id"] = model.id
            log = PerformanceLog(**entry_copy)
            session.add(log)
            session.commit()
            new_logs += 1
            print(f"✓ Created performance log '{prompt_hash}' for '{model_name}'")

    print(
        f"\nSeeding complete: {new_models} model(s) created, "
        f"{skipped_models} already present. "
        f"{new_logs} performance log(s) created, {skipped_logs} already present."
    )


if __name__ == "__main__":
    seed_database()
