import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------
# Configuration
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional: Use bfloat16/FP16 for lower memory usage on ROCm
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Global model and tokenizer (loaded on startup)
tokenizer = None
model = None

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Qwen2.5-Coder-7B API")

# -------------------------
# Startup event: Load model only once
# -------------------------
@app.on_event("startup")
async def load_model():
    global tokenizer, model
    
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print("🔄 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=TORCH_DTYPE,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        offload_folder="offload",
        trust_remote_code=True
    )
    model.eval()

    print(f"✅ Model loaded on device(s) {DEVICE}")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/generate")
async def generate(req: PromptRequest):
    if model is None:
        return {"error": "Model not loaded yet. Try again in a moment."}
    
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}

@app.get("/")
async def root():
    # FIX 2: Removed the stray "(pytorch)" from this line
    return {"message": "Qwen2.5-Coder-7B-Instruct API running"}