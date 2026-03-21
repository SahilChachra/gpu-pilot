"""
AI advisor: Claude-powered streaming chat about vLLM configuration.
"""
import json
import os

from data.gpus   import GPUS
from data.models import MODELS

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


_vlm_models = [k for k, v in MODELS.items() if v.get("is_vlm")]

SYSTEM_PROMPT = f"""You are an expert vLLM infrastructure advisor with deep knowledge of:
- GPU hardware (H200, H100, B200, RTX 5090, A100, L40S, T4, etc.) — VRAM, bandwidth, TFLOP/s, NVLink
- vLLM configuration parameters and their exact effects on throughput, latency, memory
- LLM architectures — KV cache math, GQA/MQA, MoE active-params vs total-params, MLA attention
- Quantization: AWQ, GPTQ, FP8, INT8 — quality tradeoffs and GPU requirements
- Cloud GPU pricing on RunPod, Lambda Labs, and Vast.ai
- Vision-Language Models (VLMs) — image tokens, vision encoders, multi-image batching

VLM-specific expertise:
- Image token math: fixed-res (LLaVA-1.5: 576 tokens), tile-based (LLaVA-NeXT: up to 2880), dynamic (Qwen2-VL: ⌈H/28⌉×⌈W/28⌉, Pixtral: ⌈H/16⌉×⌈W/16⌉)
- Vision encoder VRAM: 0.6–1.7 GB extra (CLIP ViT-L=0.6GB, ViT-H=1.7GB, Qwen2-VL ViT=1.4GB, SigLIP=0.8GB)
- Cross-attention VLMs (LLaMA-3.2-Vision): image tokens bypass main KV cache — lower KV pressure
- Key vLLM VLM flags: --limit-mm-per-prompt image=N (max images per request), --mm-processor-kwargs (resolution control)
- High-res batching: 8× 1080p images ≈ 8×(⌈1080/28⌉×⌈1920/28⌉)=8×2470=19760 extra KV tokens — size accordingly

When recommending GPU/config, always:
1. Show the math (model VRAM + encoder VRAM, KV cache per sequence incl. image tokens, max concurrent seqs)
2. Give a concrete vLLM serve command with all key flags
3. For VLMs: include --limit-mm-per-prompt and note effective context = text_ctx + img_tokens×N
4. Quote cost per 1M tokens on RunPod (and Lambda/Vast if available)
5. Note MoE implications if relevant (total VRAM ≠ active compute)
6. Warn about PCIe vs NVLink at multi-GPU scale, and OOM risks
7. Suggest quantization when it significantly changes the economics

Available GPUs: {', '.join(GPUS.keys())}
Available models (LLMs): {', '.join(k for k in MODELS if not MODELS[k].get('is_vlm'))}
Available models (VLMs): {', '.join(_vlm_models)}
Use markdown. Be concise but precise. Show numbers."""


def stream_advisor(messages: list):
    """Yield SSE-formatted chunks from Claude. Caller wraps in Flask Response."""
    if not _ANTHROPIC_AVAILABLE:
        yield "data: " + json.dumps({"text": "❌ Anthropic SDK not installed. Run: pip install anthropic"}) + "\n\n"
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        yield "data: " + json.dumps({"text": "❌ Set ANTHROPIC_API_KEY environment variable to use the AI advisor."}) + "\n\n"
        return

    client = anthropic.Anthropic(api_key=api_key)

    db_context = (
        f"\nGPU database: {json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'notes'} for k, v in GPUS.items()})}\n"
        f"Model database: {json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'notes'} for k, v in MODELS.items()})}\n"
    )
    enriched = messages.copy()
    if enriched and enriched[0]["role"] == "user":
        enriched[0] = {"role": "user", "content": enriched[0]["content"] + f"\n\n[DB Context: {db_context}]"}

    try:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=enriched,
        ) as stream:
            for text in stream.text_stream:
                yield "data: " + json.dumps({"text": text}) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"text": f"\n\n❌ Error: {str(e)}"}) + "\n\n"

    yield "data: [DONE]\n\n"
