# gpupilot

Your co-pilot for LLM inference — GPU recommendations, vLLM config generation, VRAM calculator, and live cloud pricing in one dashboard.

## Screenshots

| | |
|---|---|
| ![GPU Finder](screenshots/1%20-%20GPU%20Finder.png) | ![Config Generator](screenshots/2%20-%20Config%20Generator.png) |
| **GPU Finder** — ranked GPU recommendations with live cloud pricing | **Config Generator** — optimal `vllm serve` command for your GPU + model |
| ![VRAM Calculator](screenshots/3%20-%20VRAM%20Calculator.png) | ![Parameter Reference](screenshots/4%20-%20Parameter%20Reference.png) |
| **VRAM Calculator** — full memory breakdown and max concurrent sequences | **Parameter Reference** — every vLLM knob with focus-aware guidance |
| ![AI Advisor](screenshots/5%20-%20Chat%20interface.png) | |
| **AI Advisor** — chat with Claude about GPU selection and config tuning | |

## Features

| Tab | What it does |
|-----|-------------|
| **GPU Finder** | Select model + requirements → ranked GPU recommendations with cost/throughput estimates. Saves search history locally. |
| **Config Generator** | Select GPU + model + priority → optimal `vllm serve` command ready to copy. |
| **VRAM Calculator** | Enter any model architecture → VRAM breakdown + max concurrent sequences. |
| **Param Reference** | All vLLM parameters with focus-aware guidance (throughput / latency / memory / cost). |
| **AI Advisor** | Chat with Claude about GPU selection, config tuning, and inference math. Full conversation history persisted across sessions. |

## Quick Start

```bash
# 1. Install dependencies
pip install flask requests anthropic

# 2. (Optional) Set API key for AI Advisor tab
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
python app.py

# 4. Open browser
open http://localhost:5050
```

## Project Structure

```
gpupilot/
├── app.py              # Flask routes + entry point
├── engine.py           # VRAM / KV cache / throughput calculations + GPU recommender
├── pricing.py          # Live price fetching (RunPod GraphQL, Vast.ai REST) + background thread
├── advisor.py          # Claude AI advisor — system prompt + SSE streaming
├── data/
│   ├── gpus.py         # GPU database (27 GPUs) + RunPod/Vast.ai name maps
│   └── models.py       # Model database (31 models) + quantization tables
├── requirements.txt
├── templates/
│   └── index.html      # Dashboard HTML
└── static/
    ├── style.css        # Dark-theme UI
    ├── app.js           # All frontend logic
    └── params.js        # vLLM parameter definitions + focus metadata
```

## GPU Database

27 GPUs across 5 generations with VRAM, memory bandwidth, BF16/FP8 TFLOPS (dense), NVLink, MIG, and cloud pricing:

| Generation | GPUs |
|---|---|
| Blackwell datacenter | B200 SXM 192GB, B100 SXM 192GB |
| Hopper | H200 SXM 141GB, H100 SXM/NVL/PCIe 80-94GB |
| Ampere datacenter | A100 SXM/PCIe 80/40GB, A40, A30, A10G, A6000 |
| Ada Lovelace | L40S, L40, A6000 Ada, L4, RTX 4090, RTX 4080 Super |
| Blackwell consumer | RTX 5090, 5080, 5070 Ti, 5070 |
| Legacy | RTX 3090, T4, V100 SXM2 |

## Model Database

31 pre-configured models with exact architecture specs (layers, KV heads, head_dim, hidden size):

- **Llama 3.x** — 1B, 3B, 8B, 70B, 405B
- **Mistral / Mixtral** — 7B, Nemo 12B, Small 24B, 8×7B MoE, 8×22B MoE, Codestral 22B
- **Qwen 2.5** — 7B, 14B, 32B, 72B, Coder 7B/32B, QwQ 32B
- **DeepSeek** — V3, R1, R1 distills (8B / 7B / 14B / 32B / 70B)
- **Gemma 2** — 2B, 9B, 27B
- **Phi** — 3.5 Mini, Phi-4

For custom or unlisted models, enter the HuggingFace model ID — parameter count and architecture are fetched automatically from `config.json`.

## Live Pricing

Prices are fetched automatically on startup from **RunPod** (GraphQL) and **Vast.ai** (REST) and refreshed every 30 minutes. No API keys required. The sidebar shows a live indicator with last-updated time; a manual refresh button is also available.

## Calculation Engine

| Formula | Description |
|---|---|
| `params_b × 2 × quant_overhead` | Model weight VRAM (GB) |
| `2 × layers × kv_heads × head_dim × ctx_len × bytes` | KV cache per sequence |
| `(total_vram × util − model_vram) × 0.95 / kv_per_seq` | Max concurrent sequences |
| `min(BW / weight_bytes × batch, TFLOPS × η / 2 / params)` | Estimated tokens/sec (roofline) |

## Extending

- **Add a GPU** → edit `data/gpus.py`, add an entry to `GPUS` and both name maps
- **Add a model** → edit `data/models.py`, add an entry to `MODELS`
- **Tune calculations** → `engine.py` is pure Python, no Flask dependency
