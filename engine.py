"""
Calculation and recommendation engine for vLLM GPU Advisor.
All pure functions — no Flask, no I/O.
"""
import math

from data.gpus   import GPUS
from data.models import MODELS, QUANT_BITS, QUANT_OVERHEAD, QUANT_USES_FP8_ENGINE


# ── Memory estimators ──────────────────────────────────────────────────────────

def calc_model_vram_gb(params_b: float, quant: str) -> float:
    """Estimate weight VRAM in GB (fp16 baseline scaled by quant overhead)."""
    overhead = QUANT_OVERHEAD.get(quant, 1.0)
    # fp16 baseline: 2 bytes per parameter
    # params_b × 10^9 × 2 bytes / 10^9 = params_b × 2
    return round(params_b * 2.0 * overhead, 1)


def calc_kv_cache_gb_per_seq(model_info: dict, context_len: int, kv_dtype: str = "fp16") -> float:
    """KV cache per sequence in GB.

    Formula: 2 (K+V) × layers × kv_heads × head_dim × context_len × bytes_per_elem
    """
    bytes_per_elem = 1 if kv_dtype == "fp8" else 2
    kv_bytes = (
        2
        * model_info["layers"]
        * model_info["kv_heads"]
        * model_info["head_dim"]
        * context_len
        * bytes_per_elem
    )
    return kv_bytes / 1e9


def calc_max_batch(total_vram_gb: float, model_vram_gb: float, kv_per_seq_gb: float, gpu_util: float = 0.90) -> int:
    """Max concurrent sequences given memory constraints."""
    kv_budget = total_vram_gb * gpu_util - model_vram_gb
    kv_budget *= 0.95   # reserve ~5% for activations / framework overhead
    if kv_budget <= 0 or kv_per_seq_gb <= 0:
        return 0
    return max(0, int(kv_budget / kv_per_seq_gb))


def min_gpus_for_model(model_vram_gb: float, gpu_vram_gb: float, gpu_util: float = 0.88) -> int:
    """Minimum number of GPUs needed for the model weights to fit."""
    per_gpu_usable = gpu_vram_gb * gpu_util
    return max(1, math.ceil(model_vram_gb / per_gpu_usable))


# ── Throughput estimator ───────────────────────────────────────────────────────

def estimate_throughput(gpu_info: dict, model_info: dict, quant: str, batch_size: int) -> int:
    """Roofline throughput model: min(BW-bound, compute-bound) tokens/sec.

    Decode phase:
    - Memory-BW bound (small batch): each token reads all active weights once.
      TPS ≈ BW / (active_params × bytes_per_param) × batch_size
    - Compute bound (large batch):
      TPS ≈ TFLOPS × η / (2 × active_params)
    """
    active_params_b = model_info.get("active_params_b", model_info["params_b"])
    bytes_pp = QUANT_BITS.get(quant, 16) / 8.0

    use_fp8_engine = quant in QUANT_USES_FP8_ENGINE and gpu_info["fp8"]
    tflops = gpu_info["tflops_fp8"] if use_fp8_engine else gpu_info["tflops_bf16"]
    bw_gb  = gpu_info["bw"]

    weight_bytes_gb = active_params_b * bytes_pp
    if weight_bytes_gb <= 0:
        return 0

    bw_tps = (bw_gb / weight_bytes_gb) * batch_size

    # Tensor-core efficiency improves with batch size
    if   batch_size >= 256: eta = 0.60
    elif batch_size >= 64:  eta = 0.50
    elif batch_size >= 16:  eta = 0.40
    elif batch_size >= 4:   eta = 0.30
    else:                   eta = 0.20

    compute_tps = (tflops * 1e12 * eta) / (2.0 * active_params_b * 1e9)

    return round(min(bw_tps, compute_tps))


# ── Cost helpers ───────────────────────────────────────────────────────────────

def cost_per_1m_tokens(hourly_rate, tps: int):
    """$/1M tokens given hourly rate and tokens/sec. Returns None if inputs invalid."""
    if tps <= 0 or hourly_rate is None:
        return None
    return round(hourly_rate / tps * 1e6 / 3600, 4)


# ── GPU recommender ────────────────────────────────────────────────────────────

def recommend_gpus(
    model_id: str,
    quant: str,
    context_len: int,
    target_batch: int,
    target_tps: int,
    num_gpus: int = 1,
    model_override: dict = None,
) -> list:
    """Return up to 10 ranked GPU recommendations with rich detail."""
    model = model_override or MODELS.get(model_id)
    if not model:
        return []

    model_vram = calc_model_vram_gb(model["params_b"], quant)
    kv_per_seq = calc_kv_cache_gb_per_seq(model, context_len)

    results = []
    for gpu_name, gpu in GPUS.items():
        total_vram    = gpu["vram"] * num_gpus
        max_b         = calc_max_batch(total_vram, model_vram, kv_per_seq)
        batch_for_tps = min(max_b, target_batch or 64)
        est_tps       = estimate_throughput(gpu, model, quant, batch_for_tps)

        fits        = model_vram <= total_vram * 0.90
        meets_batch = max_b >= (target_batch or 1)
        meets_tps   = est_tps >= (target_tps or 0)
        min_gpus_n  = min_gpus_for_model(model_vram, gpu["vram"])

        score = 0
        if fits:        score += 40
        if meets_batch: score += 25
        if meets_tps:   score += 20
        if gpu["runpod_hr"] and est_tps > 0:
            score += min(15, int(est_tps / gpu["runpod_hr"] / 50))

        results.append({
            "gpu":           gpu_name,
            "arch":          gpu["arch"],
            "tier":          gpu["tier"],
            "vram_gb":       total_vram,
            "model_vram_gb": model_vram,
            "kv_budget_gb":  round(max(0, total_vram * 0.90 - model_vram), 1),
            "max_batch":     max_b,
            "est_tps":       est_tps,
            "fits":          fits,
            "meets_batch":   meets_batch,
            "meets_tps":     meets_tps,
            "min_gpus":      min_gpus_n,
            "fp8_support":   gpu["fp8"],
            "nvlink":        gpu["nvlink"],
            "mig_slices":    gpu["mig"],
            "pricing": {
                "runpod": gpu["runpod_hr"] * num_gpus if gpu["runpod_hr"] else None,
                "lambda": gpu["lambda_hr"] * num_gpus if gpu["lambda_hr"] else None,
                "vastai": gpu["vastai_hr"] * num_gpus if gpu["vastai_hr"] else None,
            },
            "cost_per_1m": {
                "runpod": cost_per_1m_tokens(gpu["runpod_hr"] * num_gpus if gpu["runpod_hr"] else None, est_tps),
                "lambda": cost_per_1m_tokens(gpu["lambda_hr"] * num_gpus if gpu["lambda_hr"] else None, est_tps),
                "vastai": cost_per_1m_tokens(gpu["vastai_hr"] * num_gpus if gpu["vastai_hr"] else None, est_tps),
            },
            "notes": gpu["notes"],
            "score": score,
        })

    results.sort(key=lambda x: (-x["score"], x["pricing"]["runpod"] or 9999))
    return results[:10]


# ── Config recommender ─────────────────────────────────────────────────────────

def recommend_config(
    gpu_name: str,
    model_id: str,
    quant: str,
    context_len: int,
    priority: str,
    model_override: dict = None,
) -> dict:
    """Given GPU + model, return the best vLLM serve config."""
    if gpu_name not in GPUS:
        return {}
    model = model_override or MODELS.get(model_id)
    if not model:
        return {}

    gpu        = GPUS[gpu_name]
    model_vram = calc_model_vram_gb(model["params_b"], quant)
    kv_per_seq = calc_kv_cache_gb_per_seq(model, context_len)
    tp_size    = min_gpus_for_model(model_vram, gpu["vram"])
    max_seqs   = calc_max_batch(gpu["vram"] * tp_size, model_vram, kv_per_seq)

    if priority == "throughput":
        cfg_max_seqs = min(512, max(64, max_seqs))
        cfg_batched  = 8192
        cfg_delay    = 0.3
        gpu_util     = 0.93
    elif priority == "latency":
        cfg_max_seqs = min(32, max(1, max_seqs))
        cfg_batched  = 1024
        cfg_delay    = 0.0
        gpu_util     = 0.90
    elif priority == "cost":
        cfg_max_seqs = min(512, max_seqs)
        cfg_batched  = 16384
        cfg_delay    = 0.5
        gpu_util     = 0.93
    else:  # balanced
        cfg_max_seqs = min(128, max(16, max_seqs))
        cfg_batched  = 4096
        cfg_delay    = 0.0
        gpu_util     = 0.90

    use_fp8_kv = gpu["fp8"] and quant in ("fp8", "awq", "fp16", "bfloat16") and priority in ("throughput", "cost")
    use_quant  = quant if quant not in ("fp16", "bfloat16") else None

    if gpu["fp8"] and quant in ("fp16", "bfloat16"):
        suggested_quant = "fp8"
    elif not gpu["fp8"] and quant in ("fp16", "bfloat16"):
        suggested_quant = "awq"
    else:
        suggested_quant = quant

    cmd = [f"vllm serve {model_id}"]
    if tp_size > 1:
        cmd.append(f"  --tensor-parallel-size {tp_size}")
    if use_quant:
        cmd.append(f"  --quantization {use_quant}")
    cmd.append(f"  --max-num-seqs {cfg_max_seqs}")
    cmd.append(f"  --max-num-batched-tokens {cfg_batched}")
    cmd.append(f"  --gpu-memory-utilization {gpu_util}")
    cmd.append("  --enable-prefix-caching")
    if use_fp8_kv:
        cmd.append("  --kv-cache-dtype fp8_e4m3")
    if cfg_delay > 0:
        cmd.append(f"  --scheduler-delay-factor {cfg_delay}")
    cmd.append("  --dtype bfloat16")

    est_tps = estimate_throughput(gpu, model, quant, cfg_max_seqs)

    return {
        "tp_size":                tp_size,
        "max_num_seqs":           cfg_max_seqs,
        "max_num_batched_tokens": cfg_batched,
        "gpu_memory_utilization": gpu_util,
        "enable_prefix_caching":  True,
        "kv_cache_dtype":         "fp8_e4m3" if use_fp8_kv else "auto",
        "scheduler_delay_factor": cfg_delay,
        "quantization":           use_quant,
        "suggested_quant":        suggested_quant,
        "model_vram_gb":          model_vram,
        "max_batch_possible":     max_seqs,
        "command":                " \\\n".join(cmd),
        "est_tps":                est_tps,
        "cost_per_day": {
            "runpod": round(gpu["runpod_hr"] * tp_size * 24, 2) if gpu["runpod_hr"] else None,
            "lambda": round(gpu["lambda_hr"] * tp_size * 24, 2) if gpu["lambda_hr"] else None,
            "vastai": round(gpu["vastai_hr"] * tp_size * 24, 2) if gpu["vastai_hr"] else None,
        },
        "cost_per_1m": {
            "runpod": cost_per_1m_tokens(gpu["runpod_hr"] * tp_size if gpu["runpod_hr"] else None, est_tps),
            "lambda": cost_per_1m_tokens(gpu["lambda_hr"] * tp_size if gpu["lambda_hr"] else None, est_tps),
            "vastai": cost_per_1m_tokens(gpu["vastai_hr"] * tp_size if gpu["vastai_hr"] else None, est_tps),
        },
        "model_arch":  model["arch"],
        "model_notes": model.get("notes", ""),
    }
