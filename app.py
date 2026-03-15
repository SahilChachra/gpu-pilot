"""
vLLM GPU Advisor — Flask backend
Run:  python app.py
Open: http://localhost:5050
"""
import os

import requests as http
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from data.gpus   import GPUS
from data.models import MODELS, QUANT_BITS
from engine      import (
    calc_kv_cache_gb_per_seq,
    calc_max_batch,
    calc_model_vram_gb,
    recommend_config,
    recommend_gpus,
)
from pricing     import PRICE_CACHE, refresh_prices, start_price_refresh_thread
from advisor     import stream_advisor

app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_custom_model(params_b: float, context_len: int) -> dict:
    """Construct a synthetic model dict from a bare parameter count."""
    return {
        "params_b": params_b, "active_params_b": params_b,
        "family": "custom", "arch": "decoder",
        "layers": max(1, int(params_b * 4.57)),
        "heads": 32, "kv_heads": 8, "head_dim": 128, "hidden": 4096,
        "context": context_len, "use_cases": [], "license": "unknown",
        "notes": "Custom model — architecture estimated from parameter count.",
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        gpus=list(GPUS.keys()),
        models=list(MODELS.keys()),
        quant_types=list(QUANT_BITS.keys()),
    )


@app.route("/api/gpus")
def api_gpus():
    return jsonify(GPUS)


@app.route("/api/models")
def api_models():
    return jsonify(MODELS)


@app.route("/api/hf-model-info")
def api_hf_model_info():
    """Fetch architecture details from HuggingFace config.json and estimate param count."""
    model_id = request.args.get("model_id", "").strip()
    if not model_id or "/" not in model_id:
        return jsonify({"error": "Expected 'org/model-name' format"}), 400

    try:
        r = http.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            timeout=10,
            headers={"Accept": "application/json"},
            allow_redirects=True,
        )
        if r.status_code == 401:
            return jsonify({"error": "Gated model — requires HuggingFace token"}), 403
        if r.status_code == 404:
            return jsonify({"error": "Model not found on HuggingFace"}), 404
        r.raise_for_status()
        cfg = r.json()
    except Exception as e:
        return jsonify({"error": f"HuggingFace fetch failed: {e}"}), 502

    num_layers   = cfg.get("num_hidden_layers", 32)
    num_heads    = cfg.get("num_attention_heads", 32)
    kv_heads     = cfg.get("num_key_value_heads", num_heads)
    hidden       = cfg.get("hidden_size", 4096)
    head_dim     = cfg.get("head_dim", hidden // max(1, num_heads))
    intermediate = cfg.get("intermediate_size", hidden * 4)
    vocab_size   = cfg.get("vocab_size", 32000)
    model_type   = cfg.get("model_type", "unknown")

    # Parameter count from architecture
    attn = (num_heads * head_dim + 2 * kv_heads * head_dim + num_heads * head_dim) * hidden
    swiglu_acts = {"silu", "gelu_new", "swiglu", "geglu", "gelu_fast", "gelu_pytorch_tanh"}
    ffn_mult    = 3 if cfg.get("hidden_act", "silu") in swiglu_acts else 2
    per_layer   = attn + ffn_mult * hidden * intermediate + 2 * hidden
    emb         = vocab_size * hidden
    lm_head     = 0 if cfg.get("tie_word_embeddings", True) else vocab_size * hidden
    params_b    = round((emb + num_layers * per_layer + hidden + lm_head) / 1e9, 2)

    num_experts = cfg.get("num_experts") or cfg.get("num_local_experts") or 0
    is_moe = bool(num_experts) or "moe" in model_type.lower() or "mixture" in model_type.lower()

    return jsonify({
        "params_b":    params_b,
        "layers":      num_layers,
        "heads":       num_heads,
        "kv_heads":    kv_heads,
        "head_dim":    head_dim,
        "hidden":      hidden,
        "model_type":  model_type,
        "is_moe":      is_moe,
        "num_experts": num_experts,
    })


@app.route("/api/recommend-gpu", methods=["POST"])
def api_recommend_gpu():
    data         = request.json
    model_id     = data.get("model_id", "")
    quant        = data.get("quant", "fp16")
    context_len  = int(data.get("context_len", 4096))
    target_batch = int(data.get("target_batch", 32))
    target_tps   = int(data.get("target_tps", 0))
    num_gpus     = int(data.get("num_gpus", 1))

    custom_model = None
    if model_id not in MODELS and model_id:
        custom_model = _build_custom_model(float(data.get("custom_params_b", 7)), context_len)

    recs       = recommend_gpus(model_id, quant, context_len, target_batch, target_tps, num_gpus, custom_model)
    model_info = custom_model or MODELS.get(model_id, {})
    kv_per_seq = round(calc_kv_cache_gb_per_seq(model_info, context_len), 4) if model_info else 0

    return jsonify({
        "recommendations": recs,
        "model_vram_gb":   calc_model_vram_gb(model_info.get("params_b", 7), quant) if model_id in MODELS else None,
        "kv_per_seq_gb":   kv_per_seq,
        "model_arch":      model_info.get("arch", "decoder"),
        "model_notes":     model_info.get("notes", ""),
        "use_cases":       model_info.get("use_cases", []),
        "is_moe":          model_info.get("arch") == "moe",
    })


@app.route("/api/recommend-config", methods=["POST"])
def api_recommend_config():
    data        = request.json
    gpu_name    = data.get("gpu", "")
    model_id    = data.get("model_id", "")
    quant       = data.get("quant", "fp16")
    context_len = int(data.get("context_len", 4096))
    priority    = data.get("priority", "balanced")

    if gpu_name not in GPUS:
        return jsonify({"error": f"Unknown GPU: {gpu_name}"}), 400

    custom_model = None
    if model_id not in MODELS:
        custom_model = _build_custom_model(float(data.get("custom_params_b", 7)), context_len)

    return jsonify(recommend_config(gpu_name, model_id, quant, context_len, priority, custom_model))


@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    data        = request.json
    params_b    = float(data.get("params_b", 7))
    quant       = data.get("quant", "fp16")
    context_len = int(data.get("context_len", 4096))
    num_layers  = int(data.get("num_layers", 32))
    kv_heads    = int(data.get("kv_heads", 8))
    head_dim    = int(data.get("head_dim", 128))
    gpu_vram    = float(data.get("gpu_vram", 80))
    num_gpus    = int(data.get("num_gpus", 1))
    gpu_util    = float(data.get("gpu_util", 0.90))
    kv_dtype    = data.get("kv_dtype", "fp16")

    model_vram = calc_model_vram_gb(params_b, quant)
    kv_per_seq = calc_kv_cache_gb_per_seq(
        {"layers": num_layers, "kv_heads": kv_heads, "head_dim": head_dim},
        context_len, kv_dtype,
    )
    total_vram = gpu_vram * num_gpus
    max_b      = calc_max_batch(total_vram, model_vram, kv_per_seq, gpu_util)

    return jsonify({
        "model_vram_gb":       model_vram,
        "kv_per_seq_gb":       round(kv_per_seq, 4),
        "total_vram_gb":       total_vram,
        "available_for_kv_gb": round(total_vram * gpu_util - model_vram, 2),
        "max_concurrent_seqs": max_b,
        "fits_on_gpu":         model_vram <= total_vram * gpu_util,
    })


@app.route("/api/advisor", methods=["POST"])
def api_advisor():
    messages = request.json.get("messages", [])
    return Response(
        stream_with_context(stream_advisor(messages)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/prices")
def api_prices():
    """Return live price cache status and per-GPU prices."""
    live = {
        name: {
            "runpod_hr": g.get("runpod_hr"),
            "lambda_hr": g.get("lambda_hr"),
            "vastai_hr": g.get("vastai_hr"),
        }
        for name, g in GPUS.items()
    }
    return jsonify({
        "prices":       live,
        "last_updated": PRICE_CACHE.get("last_updated"),
        "runpod_hits":  PRICE_CACHE.get("runpod_hits", 0),
        "vastai_hits":  PRICE_CACHE.get("vastai_hits", 0),
        "errors":       PRICE_CACHE.get("errors", []),
        "is_live":      PRICE_CACHE.get("last_updated") is not None,
    })


@app.route("/api/refresh-prices", methods=["POST"])
def api_refresh_prices():
    """Manually trigger a synchronous price refresh (~2-3 s)."""
    refresh_prices()
    return jsonify({
        "ok":           True,
        "last_updated": PRICE_CACHE.get("last_updated"),
        "runpod_hits":  PRICE_CACHE.get("runpod_hits", 0),
        "vastai_hits":  PRICE_CACHE.get("vastai_hits", 0),
        "errors":       PRICE_CACHE.get("errors", []),
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))

    # With Flask debug=True the werkzeug reloader forks a child process;
    # WERKZEUG_RUN_MAIN=true is set only in the child — start the thread there.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get("WERKZEUG_RUN_MAIN"):
        start_price_refresh_thread()

    print(f"\n vLLM GPU Advisor  http://localhost:{port}")
    print(f"   GPUs in database:   {len(GPUS)}")
    print(f"   Models in database: {len(MODELS)}")
    print(f"   Set ANTHROPIC_API_KEY to enable AI chat")
    print(f"   Live pricing: fetching in background...\n")
    app.run(debug=True, port=port, threaded=True)
