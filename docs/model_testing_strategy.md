# Model Testing & Load Strategy

This guide captures the current loading behavior in `main.py`, outlines a repeatable validation workflow, and documents the knobs we can safely tune (with Phi‑3 Mini as the running example).

## 1. How the API loads models today

`ModelManager.load(...)` in `main.py` performs these steps:

1. **Unload & sanitize** – frees the existing model, normalizes the user‑provided name/path.
2. **Compose default kwargs** – always sets:
   - `torch_dtype=torch.bfloat16` when a GPU is visible, otherwise `float32`.
   - `device_map="auto"` so Hugging Face partitions weights across devices (single ROCm GPU today).
   - `attn_implementation="eager"` to avoid Flash‑Attention code paths that fail on ROCm/WSL.
   - `trust_remote_code` from the registry entry (required for custom architectures).
3. **Apply overrides** – if the DB record contains `load_config_json` we merge it into `load_kwargs`.
4. **Cache options** – when the registry points at a custom cache directory we inject `cache_dir`.
5. **Disable flash attention globally** – sets `FLASH_ATTENTION_SKIP_CUDA_BUILD=1` before imports.
6. **Preload config** – fetches `AutoConfig`, forces eager attention flags, and passes the config object into `from_pretrained`.
7. **Instantiate model & tokenizer** – both are cached in the singleton and exposed to inference routes.

Any tuning therefore happens either by editing `load_config_json` for a model row or by adding logic before `from_pretrained`.

## 2. Pre‑test workflow

Follow this checklist for every new model revision:

1. **Inspect the Hugging Face card** for:
   - Required `trust_remote_code`, tokenizer caveats (`use_fast=False`, special BOS/EOS handling).
   - Suggested inference dtype (bfloat16 vs float16 vs int8/4bit).
   - Extra generation configuration (e.g., `sliding_window`, `rope_scaling` hints).
2. **Download the config JSON** (`config.json`) to capture architectural limits.
3. **Estimate memory footprint**  
   Parameters ≈ `hidden_size × intermediate_size × layers × 2` (rough rule). For Phi‑3 Mini:
   `3,072 × 8,192 × 32 × 2 ≈ 1.6e9` multiplications -> ~3.8B parameters → ~7.5 GB in BF16 before optimizer states.
   Ensure GPU free memory > 1.2× parameter bytes for activations + KV cache.
4. **Capture tester metadata** – stage the model (`/api/models/offline`) so Vector‑Tester records load attempts with the correct `model_id`.
5. **Define a test profile** that encodes load/unload/validate/inference steps so that failures are reproducible.

## 3. Reading the Phi‑3 config

Key fields from the provided JSON:

| Field | Implication / Lever |
|-------|---------------------|
| `model_type: "phi3"` + `auto_map` referencing `configuration_phi3` | Requires `trust_remote_code=True` so our loader imports the Microsoft custom modules. |
| `torch_dtype: "bfloat16"` | Matches the API default; if VRAM pressure is high we can override to `float16` via `{"torch_dtype":"float16"}` in `load_config_json`. |
| `num_hidden_layers: 32`, `hidden_size: 3072`, `intermediate_size: 8192` | Confirms ~3.8B parameters. On 16 GB GPUs this leaves limited headroom; enabling `low_cpu_mem_usage=True` can help the initial load. |
| `num_attention_heads = num_key_value_heads = 32` | No grouped-query attention; KV cache scales with full head count. For long prompts lower `max_tokens` and `sliding_window`. |
| `sliding_window: 2047`, `max_position_embeddings: 4096` | The model expects MQA sliding attention. Hugging Face often recommends setting `model.config.sliding_window` before generation; ensure validation prompts stay within 2k tokens unless we enable RoPE scaling. |
| `attention_bias: false`, `attn_implementation` unspecified | Confirms we must keep `attn_implementation="eager"` as in `main.py`. Flash attention kernels would be unsupported on ROCm. |
| `use_cache: true` | Streaming is safe, but ensure we don’t disable cache when experimenting. |

## 4. Tunable load settings

Use the `load_config_json` column (or extend `ModelManager.load`) to inject the following knobs when Phi crashes:

| Setting | When to try | Example JSON override |
|---------|-------------|-----------------------|
| `low_cpu_mem_usage` | OOM during weight load | `{"low_cpu_mem_usage": true}` |
| `max_memory` | Force smaller GPU allocation map | `{"max_memory": {"0": "12GiB", "cpu": "32GiB"}}` |
| `torch_dtype` | BF16 unsupported / OOM | `{"torch_dtype": "float16"}` (matches config field names) |
| `device_map` | Auto placement fails | `{"device_map": {"": "cuda:0"}}` or `"sequential"` for CPU offload tests |
| `rope_scaling` | Need >4k tokens | `{"rope_scaling": {"type": "linear", "factor": 2.0}}` (only if model card allows) |
| `attn_implementation` | Guard rails | Already forced to `"eager"`; keep it explicit in overrides |
| Tokenizer hints | HF cards often require `use_fast=False` | currently we rely on AutoTokenizer defaults; override by editing `main.py` or storing flag in registry |

Combine overrides carefully—start with one change per attempt and log it via Vector‑Tester’s “Alternate config notes” so we can correlate results.

## 5. Suggested testing sequence

1. **Dry run on CPU** (`device_map="cpu"`, `torch_dtype="float32"`) just to ensure weights deserialize.
2. **GPU load with eager attention & BF16** (current defaults). Monitor Docker logs for ROCm kernel panics.
3. **Fallback to float16** if kernels complain about unsupported BF16 ops.
4. **Constrain memory map** if ROCm reports OOM even after float16. Set `max_memory` or run with `low_cpu_mem_usage`.
5. **Validate    Model Card Section	NULL	NULL	NULL	Note that by default, the Phi-3 Mini-4K-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:
* NVIDIA A100
* NVIDIA A6000
* NVIDIA H100inference** with a short 10-token prompt (Vector‑Tester’s “Validate” button). If `sliding_window` errors appear, reduce prompt length.
6. **Document** the exact `load_config_json`, commit it to the registry, and snapshot logs via Vector‑Tester so future tests start from a known-good configuration.

## 6. What to capture from Hugging Face

Before each attempt, add these to the model’s notes or tester log:

- **Model card URL & revision hash** – ensures we can reproduce issues if upstream updates.
- **Required extra files** (tokenizer merges, generation configs).
- **Recommended inference parameters** (temperature, top_p, stop sequences).
- **Hardware assumptions** (some repos explicitly mention CUDA kernels only; list them so we know to avoid unsupported models on ROCm).

This documentation + the Vector‑Tester workflow should make future Phi attempts structured: we inspect config, decide which override to try, record it, and iterate without guessing.
