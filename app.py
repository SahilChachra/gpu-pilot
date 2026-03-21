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
    calc_image_tokens,
    calc_kv_cache_gb_per_seq,
    calc_max_batch,
    calc_model_vram_gb,
    calc_vlm_overhead,
    recommend_config,
    recommend_gpus,
)
from pricing     import PRICE_CACHE, refresh_prices, start_price_refresh_thread
from advisor     import stream_advisor

app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_custom_model(params_b: float, context_len: int, vlm_params: dict | None = None,
                        total_params_b: float | None = None) -> dict:
    """Construct a synthetic model dict from a bare parameter count, optionally with VLM fields.

    params_b       — active/compute parameter count (used for throughput)
    total_params_b — total params including all MoE experts (used for VRAM sizing)
    """
    vram_params = total_params_b if total_params_b and total_params_b > params_b else params_b
    base = {
        "params_b": vram_params, "active_params_b": params_b,
        "family": "custom", "arch": "decoder",
        "layers": max(1, int(vram_params * 4.57)),
        "heads": 32, "kv_heads": 8, "head_dim": 128, "hidden": 4096,
        "context": context_len, "use_cases": [], "license": "unknown",
        "notes": "Custom model — architecture estimated from parameter count.",
    }
    if vlm_params:
        base.update({
            "is_vlm":               True,
            "vision_encoder":       vlm_params.get("vision_encoder", "Unknown encoder"),
            "vision_encoder_gb":    float(vlm_params.get("vision_encoder_gb", 0.6)),
            "patch_size":           int(vlm_params.get("patch_size", 14)),
            "img_size":             int(vlm_params.get("img_size", 336)),
            "dynamic_res":          bool(vlm_params.get("dynamic_res", False)),
            "tile_based":           bool(vlm_params.get("tile_based", False)),
            "img_token_merge":      int(vlm_params.get("img_token_merge", 1)),
            "img_tokens_per_image": int(vlm_params.get("img_tokens_per_image") or 576),
            "max_tiles":            int(vlm_params.get("max_tiles") or 4),
            "cross_attention_vision": bool(vlm_params.get("cross_attention_vision", False)),
        })
    return base


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


_VLM_MODEL_TYPES = {
    "qwen2_vl", "llava", "llava_next", "llava_onevision", "mllama",
    "pixtral", "idefics2", "idefics3", "internvl_chat", "paligemma",
    "phi3_v", "blip", "blip-2", "instructblip", "kosmos-2",
    "qwen2_5_vl", "molmo", "emu3", "aria",
}


def _extract_vlm_info(cfg: dict) -> dict:
    """Return VLM-specific fields from a HuggingFace config, or is_vlm=False if not a VLM."""
    model_type   = cfg.get("model_type", "unknown").lower()
    vision_cfg   = cfg.get("vision_config", {})
    is_vlm       = model_type in _VLM_MODEL_TYPES or bool(vision_cfg)

    if not is_vlm:
        return {"is_vlm": False, "vision_encoder": None, "vision_encoder_gb": 0.0,
                "patch_size": 14, "img_size": 336, "dynamic_res": False, "tile_based": False,
                "img_token_merge": 1, "img_tokens_per_image": None, "max_tiles": None,
                "cross_attention_vision": False}

    # ── Cross-attention VLMs (LLaMA-3.2 Vision / mllama) ──────────────────
    cross_attention = model_type == "mllama"

    # ── Tile-based (LLaVA-NeXT / llava_next / llava_onevision) ────────────
    tile_based = model_type in {"llava_next", "llava_onevision"} or bool(cfg.get("image_grid_pinpoints"))

    # ── Dynamic resolution (Qwen2-VL, Pixtral, etc.) ──────────────────────
    dynamic_res = model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_5_moe", "pixtral", "emu3", "aria", "molmo"}

    # ── Patch size ────────────────────────────────────────────────────────
    if model_type == "pixtral":
        patch_size = vision_cfg.get("image_patch_size", 16)
    else:
        patch_size = vision_cfg.get("patch_size", 14)

    # ── Image size (for tile/fixed models) ────────────────────────────────
    img_size = vision_cfg.get("image_size", 336)

    # ── Merge stride for dynamic models ───────────────────────────────────
    # spatial_merge_size lives inside vision_config for Qwen2-VL
    img_token_merge = vision_cfg.get("spatial_merge_size") or cfg.get("spatial_merge_size") or 1

    # ── Max tiles for tile-based models ───────────────────────────────────
    max_tiles = None
    if tile_based:
        pinpoints = cfg.get("image_grid_pinpoints", [])
        max_tiles = max(len(pinpoints), 4) if pinpoints else 4

    # ── Fixed token count for classic LLaVA ──────────────────────────────
    img_tokens_per_image = None
    if not dynamic_res and not tile_based and patch_size and img_size:
        img_tokens_per_image = (img_size // patch_size) ** 2

    # ── Vision encoder name + VRAM estimate ──────────────────────────────
    embed_dim = vision_cfg.get("embed_dim") or vision_cfg.get("hidden_size", 0)
    depth     = vision_cfg.get("depth") or vision_cfg.get("num_hidden_layers", 0)

    if embed_dim and depth:
        # Rough ViT param count: depth × (4 × embed_dim^2 + 4 × embed_dim × mlp)
        mlp_dim = vision_cfg.get("intermediate_size", embed_dim * 4)
        enc_params = depth * (4 * embed_dim ** 2 + 2 * embed_dim * mlp_dim) + embed_dim
        vision_encoder_gb = round(enc_params * 2 / 1e9, 2)  # fp16
        vision_encoder = f"ViT ({round(enc_params/1e6)}M params)"
    elif model_type == "mllama":
        vision_encoder_gb = 1.7
        vision_encoder = "CLIP ViT-H (1.1B)"
    elif model_type in {"llava", "llava_next", "llava_onevision"}:
        vision_encoder_gb = 0.6
        vision_encoder = "CLIP ViT-L/14@336"
    elif model_type == "pixtral":
        vision_encoder_gb = 0.8
        vision_encoder = "Pixtral ViT (400M)"
    else:
        vision_encoder_gb = round(embed_dim * 2 / 1e6, 2) if embed_dim else 0.6
        vision_encoder = f"{model_type} ViT"

    return {
        "is_vlm":               True,
        "vision_encoder":       vision_encoder,
        "vision_encoder_gb":    vision_encoder_gb,
        "patch_size":           patch_size,
        "img_size":             img_size,
        "dynamic_res":          dynamic_res,
        "tile_based":           tile_based,
        "img_token_merge":      img_token_merge,
        "img_tokens_per_image": img_tokens_per_image,
        "max_tiles":            max_tiles,
        "cross_attention_vision": cross_attention,
    }


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

    model_type = cfg.get("model_type", "unknown")

    # Many newer models (Qwen3.5-MoE, multimodal hybrids) nest LLM fields inside
    # text_config / language_config. Merge so top-level keys always win.
    nested = cfg.get("text_config") or cfg.get("language_config") or {}
    resolved = {**nested, **cfg}  # top-level overrides nested

    num_layers   = resolved.get("num_hidden_layers", 32)
    num_heads    = resolved.get("num_attention_heads", 32)
    kv_heads     = resolved.get("num_key_value_heads", num_heads)
    hidden       = resolved.get("hidden_size", 4096)
    head_dim     = resolved.get("head_dim", hidden // max(1, num_heads))
    intermediate = resolved.get("intermediate_size", hidden * 4)
    vocab_size   = resolved.get("vocab_size", 32000)

    # MoE architecture fields
    num_experts         = resolved.get("num_experts") or resolved.get("num_local_experts") or 0
    num_experts_per_tok = resolved.get("num_experts_per_tok") or resolved.get("num_activated_experts") or 0
    moe_intermediate    = resolved.get("moe_intermediate_size") or 0
    shared_intermediate = resolved.get("shared_expert_intermediate_size") or 0
    is_moe = bool(num_experts) or "moe" in model_type.lower() or "mixture" in model_type.lower()

    # Parameter count from architecture
    attn = (num_heads * head_dim + 2 * kv_heads * head_dim + num_heads * head_dim) * hidden
    swiglu_acts = {"silu", "gelu_new", "swiglu", "geglu", "gelu_fast", "gelu_pytorch_tanh"}
    ffn_mult    = 3 if resolved.get("hidden_act", "silu") in swiglu_acts else 2

    if is_moe and moe_intermediate and num_experts:
        # Total: all N expert FFNs + shared expert (always active)
        ffn_total  = ffn_mult * hidden * (moe_intermediate * num_experts + shared_intermediate)
        # Active: only experts_per_tok FFNs fire per token + shared expert
        active_k   = num_experts_per_tok or 1
        ffn_active = ffn_mult * hidden * (moe_intermediate * active_k + shared_intermediate)
    else:
        ffn_total = ffn_active = ffn_mult * hidden * intermediate

    emb     = vocab_size * hidden
    lm_head = 0 if resolved.get("tie_word_embeddings", True) else vocab_size * hidden

    params_b        = round((emb + num_layers * (attn + ffn_total  + 2 * hidden) + hidden + lm_head) / 1e9, 2)
    active_params_b = round((emb + num_layers * (attn + ffn_active + 2 * hidden) + hidden + lm_head) / 1e9, 2)

    vlm_info = _extract_vlm_info(cfg)

    return jsonify({
        "params_b":           params_b,
        "active_params_b":    active_params_b,
        "layers":             num_layers,
        "heads":              num_heads,
        "kv_heads":           kv_heads,
        "head_dim":           head_dim,
        "hidden":             hidden,
        "model_type":         model_type,
        "is_moe":             is_moe,
        "num_experts":        num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        **vlm_info,
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
    num_images   = int(data.get("num_images", 1))
    img_h        = int(data.get("img_h", 336))
    img_w        = int(data.get("img_w", 336))

    custom_model = None
    if model_id not in MODELS and model_id:
        vlm_params = None
        if data.get("custom_is_vlm"):
            vlm_params = {k[len("custom_"):]: v for k, v in data.items() if k.startswith("custom_") and k != "custom_params_b" and k != "custom_is_vlm" and k != "custom_total_params_b"}
        total_params_b = float(data["custom_total_params_b"]) if data.get("custom_total_params_b") else None
        custom_model = _build_custom_model(float(data.get("custom_params_b", 7)), context_len, vlm_params, total_params_b)

    recs       = recommend_gpus(model_id, quant, context_len, target_batch, target_tps,
                                num_gpus, custom_model, num_images, img_h, img_w)
    model_info = custom_model or MODELS.get(model_id, {})
    is_vlm     = model_info.get("is_vlm", False)
    img_tokens = calc_image_tokens(model_info, img_h, img_w) if is_vlm else 0
    encoder_gb, _ = calc_vlm_overhead(model_info, num_images, img_h, img_w)
    kv_per_seq = round(calc_kv_cache_gb_per_seq(
        model_info, context_len + img_tokens * num_images
    ), 4) if model_info else 0

    return jsonify({
        "recommendations":       recs,
        "model_vram_gb":         round(calc_model_vram_gb(model_info.get("params_b", 7), quant) + encoder_gb, 1)
                                 if model_info else None,
        "kv_per_seq_gb":         kv_per_seq,
        "model_arch":            model_info.get("arch", "decoder"),
        "model_notes":           model_info.get("notes", ""),
        "use_cases":             model_info.get("use_cases", []),
        "is_moe":                model_info.get("arch") == "moe",
        "is_vlm":                is_vlm,
        "img_tokens_per_image":  img_tokens if is_vlm else None,
        "vision_encoder":        model_info.get("vision_encoder") if is_vlm else None,
        "encoder_vram_gb":       round(encoder_gb, 2) if is_vlm else None,
        "cross_attention_vision": model_info.get("cross_attention_vision", False),
    })


@app.route("/api/recommend-config", methods=["POST"])
def api_recommend_config():
    data        = request.json
    gpu_name    = data.get("gpu", "")
    model_id    = data.get("model_id", "")
    quant       = data.get("quant", "fp16")
    context_len = int(data.get("context_len", 4096))
    priority    = data.get("priority", "balanced")
    num_images  = int(data.get("num_images", 1))
    img_h       = int(data.get("img_h", 336))
    img_w       = int(data.get("img_w", 336))

    if gpu_name not in GPUS:
        return jsonify({"error": f"Unknown GPU: {gpu_name}"}), 400

    custom_model = None
    if model_id not in MODELS:
        vlm_params = None
        if data.get("custom_is_vlm"):
            vlm_params = {k[len("custom_"):]: v for k, v in data.items() if k.startswith("custom_") and k != "custom_params_b" and k != "custom_is_vlm" and k != "custom_total_params_b"}
        total_params_b = float(data["custom_total_params_b"]) if data.get("custom_total_params_b") else None
        custom_model = _build_custom_model(float(data.get("custom_params_b", 7)), context_len, vlm_params, total_params_b)

    return jsonify(recommend_config(
        gpu_name, model_id, quant, context_len, priority, custom_model,
        num_images, img_h, img_w
    ))


@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    data               = request.json
    params_b           = float(data.get("params_b", 7))
    quant              = data.get("quant", "fp16")
    context_len        = int(data.get("context_len", 4096))
    num_layers         = int(data.get("num_layers", 32))
    kv_heads           = int(data.get("kv_heads", 8))
    head_dim           = int(data.get("head_dim", 128))
    gpu_vram           = float(data.get("gpu_vram", 80))
    num_gpus           = int(data.get("num_gpus", 1))
    gpu_util           = float(data.get("gpu_util", 0.90))
    kv_dtype           = data.get("kv_dtype", "fp16")
    vision_encoder_gb  = float(data.get("vision_encoder_gb", 0))
    num_images         = int(data.get("num_images", 1))
    img_tokens_each    = int(data.get("img_tokens", 0))

    lm_vram    = calc_model_vram_gb(params_b, quant)
    model_vram = lm_vram + vision_encoder_gb
    eff_context = context_len + img_tokens_each * num_images
    kv_per_seq = calc_kv_cache_gb_per_seq(
        {"layers": num_layers, "kv_heads": kv_heads, "head_dim": head_dim},
        eff_context, kv_dtype,
    )
    total_vram = gpu_vram * num_gpus
    max_b      = calc_max_batch(total_vram, model_vram, kv_per_seq, gpu_util)

    return jsonify({
        "model_vram_gb":       model_vram,
        "lm_vram_gb":          lm_vram,
        "vision_encoder_gb":   vision_encoder_gb,
        "kv_per_seq_gb":       round(kv_per_seq, 4),
        "total_vram_gb":       total_vram,
        "available_for_kv_gb": round(total_vram * gpu_util - model_vram, 2),
        "max_concurrent_seqs": max_b,
        "fits_on_gpu":         model_vram <= total_vram * gpu_util,
        "img_tokens_each":     img_tokens_each,
        "total_img_tokens":    img_tokens_each * num_images,
        "eff_context":         eff_context,
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
