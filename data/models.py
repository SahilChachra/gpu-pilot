# ── Model Database ─────────────────────────────────────────────────────────────
# params_b        = total parameter count (billions)
# active_params_b = parameters active per forward pass (= params_b for dense, < for MoE)
# family          = model family for grouping
# arch            = "decoder" | "moe"
# layers          = transformer depth
# heads           = number of attention heads
# kv_heads        = GQA/MQA key-value heads (= heads for MHA, < heads for GQA/MQA)
# head_dim        = attention head dimension
# hidden          = hidden state size
# context         = max context window (tokens)
# use_cases       = subset of ["chat","code","reasoning","math","vision","multilingual"]
# license         = "llama3" | "apache2" | "mistral" | "gemma" | "commercial" | "mit"
# notes           = key deployment facts

MODELS = {
    # ── Llama 3.x ────────────────────────────────────────────────────────────
    "meta-llama/Llama-3.2-1B-Instruct": {
        "params_b": 1,   "active_params_b": 1,   "family": "llama3",  "arch": "decoder",
        "layers": 16, "heads": 32, "kv_heads": 8,  "head_dim": 64,  "hidden": 2048,
        "context": 131072, "use_cases": ["chat"], "license": "llama3",
        "notes": "Ultra-small. Fits on any GPU. Good for edge / high-throughput triage.",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "params_b": 3,   "active_params_b": 3,   "family": "llama3",  "arch": "decoder",
        "layers": 28, "heads": 24, "kv_heads": 8,  "head_dim": 128, "hidden": 3072,
        "context": 131072, "use_cases": ["chat", "code"], "license": "llama3",
        "notes": "Strong 3B. Punches above its weight. Good quality/cost on T4 or L4.",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "params_b": 8,   "active_params_b": 8,   "family": "llama3",  "arch": "decoder",
        "layers": 32, "heads": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 4096,
        "context": 131072, "use_cases": ["chat", "code", "reasoning"], "license": "llama3",
        "notes": "Best-in-class 8B. 128K context. Fits comfortably on a single A10G/L4.",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "params_b": 70,  "active_params_b": 70,  "family": "llama3",  "arch": "decoder",
        "layers": 80, "heads": 64, "kv_heads": 8,  "head_dim": 128, "hidden": 8192,
        "context": 131072, "use_cases": ["chat", "code", "reasoning"], "license": "llama3",
        "notes": "Industry-standard 70B. Needs 1×80GB or 2×40GB GPU. Use AWQ to halve VRAM.",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "params_b": 70,  "active_params_b": 70,  "family": "llama3",  "arch": "decoder",
        "layers": 80, "heads": 64, "kv_heads": 8,  "head_dim": 128, "hidden": 8192,
        "context": 131072, "use_cases": ["chat", "code", "reasoning"], "license": "llama3",
        "notes": "Improved 3.3 variant — outperforms 3.1-70B on most benchmarks. Same VRAM requirement.",
    },
    "meta-llama/Llama-3.1-405B-Instruct": {
        "params_b": 405, "active_params_b": 405, "family": "llama3",  "arch": "decoder",
        "layers": 126, "heads": 128, "kv_heads": 8, "head_dim": 128, "hidden": 16384,
        "context": 131072, "use_cases": ["chat", "code", "reasoning"], "license": "llama3",
        "notes": "Frontier open model. Requires 8×80GB at fp16 or 4×80GB at AWQ. Very expensive to serve.",
    },

    # ── Mistral / Mixtral ────────────────────────────────────────────────────
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "params_b": 7,   "active_params_b": 7,   "family": "mistral", "arch": "decoder",
        "layers": 32, "heads": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 4096,
        "context": 32768,  "use_cases": ["chat", "code"], "license": "apache2",
        "notes": "Proven 7B. Apache 2.0 — fully commercial. Solid baseline for comparison.",
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "params_b": 12,  "active_params_b": 12,  "family": "mistral", "arch": "decoder",
        "layers": 40, "heads": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["chat", "code", "multilingual"], "license": "apache2",
        "notes": "12B with 128K context. Mistral x NVIDIA collaboration. Apache 2.0.",
    },
    "mistralai/Mistral-Small-3.1-24B-Instruct": {
        "params_b": 24,  "active_params_b": 24,  "family": "mistral", "arch": "decoder",
        "layers": 40, "heads": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["chat", "code", "vision"], "license": "mistral",
        "notes": "Strong 24B with vision capability. Fits on 40GB GPU at fp16.",
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "params_b": 46,  "active_params_b": 13,  "family": "mixtral", "arch": "moe",
        "layers": 32, "heads": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 4096,
        "context": 32768,  "use_cases": ["chat", "code", "multilingual"], "license": "apache2",
        "notes": "MoE: 46B total, ~13B active. All 8 experts loaded in VRAM but only 2 active per token. Throughput like 13B, VRAM like 46B.",
    },
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "params_b": 140, "active_params_b": 39,  "family": "mixtral", "arch": "moe",
        "layers": 56, "heads": 48, "kv_heads": 8,  "head_dim": 128, "hidden": 6144,
        "context": 65536,  "use_cases": ["chat", "code", "reasoning", "multilingual"], "license": "apache2",
        "notes": "MoE: 140B total, ~39B active. High quality. Needs 4×80GB at fp16, 2×80GB at AWQ.",
    },
    "mistralai/Codestral-22B-v0.1": {
        "params_b": 22,  "active_params_b": 22,  "family": "mistral", "arch": "decoder",
        "layers": 32, "heads": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 6144,
        "context": 32768,  "use_cases": ["code"], "license": "mistral",
        "notes": "Best open code model. Fill-in-the-middle. Apache 2.0 for personal use.",
    },

    # ── Qwen 2.5 ─────────────────────────────────────────────────────────────
    "Qwen/Qwen2.5-7B-Instruct": {
        "params_b": 7,   "active_params_b": 7,   "family": "qwen2.5", "arch": "decoder",
        "layers": 28, "heads": 28, "kv_heads": 4,  "head_dim": 128, "hidden": 3584,
        "context": 131072, "use_cases": ["chat", "code", "multilingual"], "license": "apache2",
        "notes": "Top 7B. Very strong multilingual. MQA (4 KV heads) = small KV cache.",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "params_b": 14,  "active_params_b": 14,  "family": "qwen2.5", "arch": "decoder",
        "layers": 48, "heads": 40, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["chat", "code", "math", "multilingual"], "license": "apache2",
        "notes": "Strong 14B. Excellent math & code. Fits on single A10G/L4 at AWQ.",
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "params_b": 32,  "active_params_b": 32,  "family": "qwen2.5", "arch": "decoder",
        "layers": 64, "heads": 40, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["chat", "code", "math", "reasoning"], "license": "apache2",
        "notes": "Exceptional quality for its size. Competes with 70B models on code/math.",
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "params_b": 72,  "active_params_b": 72,  "family": "qwen2.5", "arch": "decoder",
        "layers": 80, "heads": 64, "kv_heads": 8,  "head_dim": 128, "hidden": 8192,
        "context": 131072, "use_cases": ["chat", "code", "math", "reasoning", "multilingual"], "license": "apache2",
        "notes": "Top open 72B. Apache 2.0. Beats many proprietary models. Same VRAM as Llama 70B.",
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "params_b": 7,   "active_params_b": 7,   "family": "qwen2.5", "arch": "decoder",
        "layers": 28, "heads": 28, "kv_heads": 4,  "head_dim": 128, "hidden": 3584,
        "context": 131072, "use_cases": ["code"], "license": "apache2",
        "notes": "7B specialized coder. Strong FIM support. Very low KV cache thanks to MQA.",
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "params_b": 32,  "active_params_b": 32,  "family": "qwen2.5", "arch": "decoder",
        "layers": 64, "heads": 40, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["code"], "license": "apache2",
        "notes": "Best open code model overall. Matches GPT-4o on many coding benchmarks.",
    },
    "Qwen/QwQ-32B": {
        "params_b": 32,  "active_params_b": 32,  "family": "qwen2.5", "arch": "decoder",
        "layers": 64, "heads": 40, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["reasoning", "math", "code"], "license": "apache2",
        "notes": "32B reasoning / thinking model. Long chain-of-thought. Budget DeepSeek-R1 alternative.",
    },

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "deepseek-ai/DeepSeek-V3": {
        "params_b": 671, "active_params_b": 37,  "family": "deepseek", "arch": "moe",
        "layers": 61, "heads": 128, "kv_heads": 128, "head_dim": 128, "hidden": 7168,
        "context": 131072, "use_cases": ["chat", "code", "reasoning", "math"], "license": "mit",
        "notes": "MoE: 671B total, ~37B active per token. MLA attention = very small KV cache. Needs 8×80GB fp16 / 4×80GB AWQ.",
    },
    "deepseek-ai/DeepSeek-R1": {
        "params_b": 671, "active_params_b": 37,  "family": "deepseek", "arch": "moe",
        "layers": 61, "heads": 128, "kv_heads": 128, "head_dim": 128, "hidden": 7168,
        "context": 131072, "use_cases": ["reasoning", "math", "code"], "license": "mit",
        "notes": "MoE reasoning model. Same arch as V3. Long CoT — expect 4-10K output tokens. High VRAM, low active compute.",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "params_b": 8,   "active_params_b": 8,   "family": "deepseek", "arch": "decoder",
        "layers": 32, "heads": 32, "kv_heads": 8,  "head_dim": 128, "hidden": 4096,
        "context": 131072, "use_cases": ["reasoning", "math", "code"], "license": "mit",
        "notes": "R1 distilled into Llama 3.1 8B backbone. Excellent reasoning at tiny cost.",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "params_b": 7,   "active_params_b": 7,   "family": "deepseek", "arch": "decoder",
        "layers": 28, "heads": 28, "kv_heads": 4,  "head_dim": 128, "hidden": 3584,
        "context": 131072, "use_cases": ["reasoning", "math", "code"], "license": "mit",
        "notes": "R1 distilled into Qwen2.5-7B. Top 7B reasoning model.",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "params_b": 14,  "active_params_b": 14,  "family": "deepseek", "arch": "decoder",
        "layers": 48, "heads": 40, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["reasoning", "math", "code"], "license": "mit",
        "notes": "R1 distilled into Qwen2.5-14B. Best performance in the 14B class.",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "params_b": 32,  "active_params_b": 32,  "family": "deepseek", "arch": "decoder",
        "layers": 64, "heads": 40, "kv_heads": 8,  "head_dim": 128, "hidden": 5120,
        "context": 131072, "use_cases": ["reasoning", "math", "code"], "license": "mit",
        "notes": "R1 distilled into Qwen2.5-32B. Near full R1 quality at 1/20th the VRAM.",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        "params_b": 70,  "active_params_b": 70,  "family": "deepseek", "arch": "decoder",
        "layers": 80, "heads": 64, "kv_heads": 8,  "head_dim": 128, "hidden": 8192,
        "context": 131072, "use_cases": ["reasoning", "math", "code"], "license": "mit",
        "notes": "R1 distilled into Llama 3.3-70B. Near full R1 quality at far lower cost.",
    },

    # ── Google Gemma ─────────────────────────────────────────────────────────
    "google/gemma-2-2b-it": {
        "params_b": 2.6, "active_params_b": 2.6, "family": "gemma2",  "arch": "decoder",
        "layers": 26, "heads": 8,  "kv_heads": 4,  "head_dim": 256, "hidden": 2304,
        "context": 8192,   "use_cases": ["chat"], "license": "gemma",
        "notes": "Tiny but surprisingly capable. Sliding window attention limits context to 8K.",
    },
    "google/gemma-2-9b-it": {
        "params_b": 9,   "active_params_b": 9,   "family": "gemma2",  "arch": "decoder",
        "layers": 42, "heads": 16, "kv_heads": 8,  "head_dim": 256, "hidden": 3584,
        "context": 8192,   "use_cases": ["chat", "code"], "license": "gemma",
        "notes": "Strong 9B. Note: large head_dim=256 = bigger KV cache than Llama 8B.",
    },
    "google/gemma-2-27b-it": {
        "params_b": 27,  "active_params_b": 27,  "family": "gemma2",  "arch": "decoder",
        "layers": 46, "heads": 32, "kv_heads": 16, "head_dim": 128, "hidden": 4608,
        "context": 8192,   "use_cases": ["chat", "code"], "license": "gemma",
        "notes": "27B with good reasoning. 8K context limit. Fits on A100 40GB with fp16.",
    },

    # ── Microsoft Phi ─────────────────────────────────────────────────────────
    "microsoft/Phi-3.5-mini-instruct": {
        "params_b": 3.8, "active_params_b": 3.8, "family": "phi",     "arch": "decoder",
        "layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 96,  "hidden": 3072,
        "context": 131072, "use_cases": ["chat", "code", "reasoning"], "license": "mit",
        "notes": "Best 4B-class model for on-device / edge. 128K context. MIT license.",
    },
    "microsoft/phi-4": {
        "params_b": 14,  "active_params_b": 14,  "family": "phi",     "arch": "decoder",
        "layers": 40, "heads": 40, "kv_heads": 10, "head_dim": 128, "hidden": 5120,
        "context": 16384,  "use_cases": ["chat", "code", "reasoning", "math"], "license": "mit",
        "notes": "Phi-4 punches above 14B class on STEM. 16K context. MIT license.",
    },
}

# ── Quantization tables ────────────────────────────────────────────────────────

QUANT_BITS = {
    "fp16":     16,
    "bfloat16": 16,
    "int8":      8,
    "fp8":       8,
    "awq":       4,
    "gptq":      4,
}

# Bytes-per-parameter multiplier relative to fp16 baseline (2 bytes/param)
QUANT_OVERHEAD = {
    "fp16":     1.00,
    "bfloat16": 1.00,
    "int8":     0.52,   # slight overhead for scale factors
    "fp8":      0.52,
    "awq":      0.27,   # 4-bit + zero-points + scales
    "gptq":     0.27,
}

# Quantization methods that benefit from FP8 Tensor Engine hardware
QUANT_USES_FP8_ENGINE = {"fp8", "awq"}
