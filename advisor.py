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


SYSTEM_PROMPT = f"""You are an expert vLLM infrastructure advisor with deep knowledge of:
- GPU hardware (H200, H100, B200, RTX 5090, A100, L40S, T4, etc.) — VRAM, bandwidth, TFLOP/s, NVLink
- vLLM configuration parameters and their exact effects on throughput, latency, memory
- LLM architectures — KV cache math, GQA/MQA, MoE active-params vs total-params, MLA attention
- Quantization: AWQ, GPTQ, FP8, INT8 — quality tradeoffs and GPU requirements
- Cloud GPU pricing on RunPod, Lambda Labs, and Vast.ai

When recommending GPU/config, always:
1. Show the math (model VRAM, KV cache per sequence, max concurrent sequences)
2. Give a concrete vLLM serve command with all key flags
3. Quote cost per 1M tokens on RunPod (and Lambda/Vast if available)
4. Note MoE implications if relevant (total VRAM ≠ active compute)
5. Warn about PCIe vs NVLink at multi-GPU scale, and OOM risks
6. Suggest quantization when it significantly changes the economics

Available GPUs: {', '.join(GPUS.keys())}
Available models: {', '.join(MODELS.keys())}
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
