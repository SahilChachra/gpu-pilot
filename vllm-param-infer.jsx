import { useState, useMemo } from "react";

// ─── DATA ─────────────────────────────────────────────────────────────────────

const FOCUS_META = {
  throughput:   { label: "Throughput",     icon: "⚡", color: "#22d3ee", bg: "#083344", desc: "Maximize tokens/sec & requests/sec" },
  latency:      { label: "Low Latency",    icon: "⏱",  color: "#fb923c", bg: "#431407", desc: "Minimize TTFT & inter-token delay" },
  memory:       { label: "Memory",         icon: "🗄",  color: "#a78bfa", bg: "#2e1065", desc: "Fit larger models, reduce OOM risk" },
  cost:         { label: "Cost Efficiency",icon: "💰", color: "#4ade80", bg: "#052e16", desc: "Best perf per GPU dollar" },
  comparison:   { label: "Model Comparison",icon: "⚖", color: "#f472b6", bg: "#500724", desc: "Fair apples-to-apples benchmarks" },
};

const CATEGORIES = ["All", "Parallelism", "Quantization", "KV Cache", "Batching", "Memory", "Model Loading", "Attention", "Speculative"];

const PARAMS = [
  // ── PARALLELISM ──────────────────────────────────────────────────────────
  {
    name: "--tensor-parallel-size",
    alias: "-tp",
    category: "Parallelism",
    type: "int",
    default: "1",
    focus: ["throughput", "memory", "cost"],
    impact: "critical",
    summary: "Split model weights across N GPUs using Megatron column/row parallelism.",
    detail: "Each GPU holds 1/N of every attention and FFN weight matrix. Forward pass requires all-reduce collectives to synchronize activations. vLLM uses Megatron-LM style TP — column-parallel for Q/K/V/up projections, row-parallel for output/down projections. Each GPU processes a distinct subset of attention heads in parallel, so effective FLOP/s scales linearly with GPU count (minus comms overhead).",
    tradeoffs: [
      { label: "↑ Throughput",    val: "Linear scaling on NVLink, ~70% on PCIe" },
      { label: "↑ Memory",        val: "Enables models larger than single GPU VRAM" },
      { label: "↓ Latency",       val: "All-reduce adds 1–5ms per layer on PCIe" },
      { label: "↓ Comms overhead",val: "Degrades above TP=2 on PCIe buses" },
    ],
    when: {
      throughput:  "Set to number of GPUs. Best on NVLink (A100/H100 SXM). Cap at 2 on PCIe.",
      latency:     "Avoid if latency is critical and model fits on 1 GPU — all-reduce adds overhead.",
      memory:      "Required when model > single GPU VRAM. 70B models need TP≥4 on 80GB GPUs.",
      cost:        "TP=4 on 4× A10G often cheaper than 1× A100 for same throughput.",
      comparison:  "Fix TP=1 across all models being compared to isolate model differences.",
    },
    example: "vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4",
    recipe: { throughput: "Match GPU count", latency: "1 if model fits", memory: "ceil(model_gb / gpu_vram)" },
  },
  {
    name: "--pipeline-parallel-size",
    alias: "-pp",
    category: "Parallelism",
    type: "int",
    default: "1",
    focus: ["memory", "throughput"],
    impact: "high",
    summary: "Split transformer layers across GPUs in pipeline stages.",
    detail: "Each GPU hosts a contiguous slice of layers. Micro-batches fill the pipeline to reduce bubble time. Less inter-GPU communication than TP (only activations at stage boundaries, not all-reduce). Combine with TP for hybrid parallelism: tp=4 pp=2 = 8 GPUs. The pipeline bubble ratio is (pp-1)/(m+pp-1) where m = micro-batches.",
    tradeoffs: [
      { label: "↑ Memory scale",  val: "Enables 100B+ models across nodes" },
      { label: "↓ Latency",       val: "Pipeline bubble adds idle time per batch" },
      { label: "↓ Complexity",    val: "Harder to tune, more failure modes" },
    ],
    when: {
      throughput:  "Use only when model doesn't fit on TP alone. Minimize pp for throughput.",
      memory:      "Combine with TP: total GPUs = tp × pp. Best for multi-node deployments.",
      latency:     "Avoid — pipeline bubble directly hurts latency.",
      cost:        "Can reduce cost on multi-node setups with slow interconnects.",
      comparison:  "Keep fixed across runs.",
    },
    example: "vllm serve <model> --tensor-parallel-size 4 --pipeline-parallel-size 2",
    recipe: { memory: "tp=node_gpus, pp=num_nodes" },
  },

  // ── QUANTIZATION ─────────────────────────────────────────────────────────
  {
    name: "--quantization",
    alias: "-q",
    category: "Quantization",
    type: "enum",
    options: "awq | gptq | fp8 | int8 | gguf | bitsandbytes | squeezellm",
    default: "None (fp16/bf16)",
    focus: ["throughput", "memory", "cost", "comparison"],
    impact: "critical",
    summary: "Reduce weight precision to shrink model size and accelerate matrix multiplications.",
    detail: "AWQ (4-bit): Activation-aware weight quantization. Identifies salient weights and protects them. Best quality at 4-bit. Fastest inference via fused CUDA kernels (uses GEMM with int4 weights, fp16 activations). GPTQ (4-bit): Layer-wise quantization using second-order error compensation. Slightly lower quality than AWQ at same bit-width. FP8 (8-bit floats): Weights AND activations in fp8 — requires H100/Ada Lovelace. Near fp16 quality with ~2× throughput boost. INT8: SmoothQuant scales outliers; broader GPU support (A10G, RTX series).",
    tradeoffs: [
      { label: "AWQ 4-bit",   val: "2× VRAM reduction, 1.5–2× faster decode, ~0.3 perplexity increase" },
      { label: "GPTQ 4-bit",  val: "Similar VRAM to AWQ, slightly slower kernels, easy model availability" },
      { label: "FP8 (H100)",  val: "1.5–2× faster with near fp16 quality — best option on H100" },
      { label: "INT8",        val: "~1.3× speedup, widely supported, more quality loss than fp8" },
    ],
    when: {
      throughput:  "FP8 on H100 > AWQ > GPTQ > INT8 for throughput. Non-quantized only if quality is paramount.",
      memory:      "AWQ/GPTQ cut memory ~50%. Essential for running 70B on 2× 40GB vs 4× 40GB.",
      cost:        "AWQ on smaller GPU often beats fp16 on larger GPU: same quality, lower $/hr.",
      latency:     "FP8 and AWQ have good latency. GPTQ slower due to dequantization overhead.",
      comparison:  "Benchmark same model at fp16 vs awq vs fp8 to quantify quality-speed tradeoff.",
    },
    example: "vllm serve TheBloke/Llama-2-13B-AWQ --quantization awq",
    recipe: { throughput: "fp8 on H100, awq elsewhere", memory: "awq or gptq for 4-bit", cost: "awq on smaller/cheaper GPU" },
  },
  {
    name: "--dtype",
    alias: "",
    category: "Quantization",
    type: "enum",
    options: "auto | half (fp16) | bfloat16 | float32",
    default: "auto (model's native dtype)",
    focus: ["throughput", "memory", "comparison"],
    impact: "high",
    summary: "Base floating point dtype for weights and activations.",
    detail: "bfloat16: 8-bit exponent (same as fp32), 7-bit mantissa. Better dynamic range, fewer NaN/Inf issues. Preferred for modern LLMs (Llama 3, Mistral, Gemma). fp16: 5-bit exponent, 10-bit mantissa. Better precision but can overflow on large activations. Slightly faster on Volta/Turing. float32: Full precision — doubles VRAM, halves throughput. Never use in production.",
    tradeoffs: [
      { label: "bf16 vs fp16",  val: "bf16 more stable, fp16 marginally faster on old GPUs" },
      { label: "fp32",          val: "2× memory cost, near-zero benefit for inference" },
    ],
    when: {
      throughput:  "Use bf16 for modern models, fp16 for Volta/Turing GPUs.",
      memory:      "Both fp16/bf16 are identical memory footprint. Avoid fp32.",
      comparison:  "Fix dtype across all model comparisons — different dtypes bias benchmarks.",
      cost:        "bf16/fp16 equivalent cost. Never fp32.",
      latency:     "fp16 marginally faster on older hardware. bf16 on Ampere+.",
    },
    example: "vllm serve <model> --dtype bfloat16",
    recipe: { comparison: "Fix to bfloat16 for all runs" },
  },

  // ── KV CACHE ─────────────────────────────────────────────────────────────
  {
    name: "--enable-prefix-caching",
    alias: "",
    category: "KV Cache",
    type: "bool",
    default: "false",
    focus: ["throughput", "latency", "cost"],
    impact: "critical",
    summary: "Cache and reuse KV tensors for shared prompt prefixes across requests.",
    detail: "vLLM hashes token sequences using a radix tree. When a new request shares a prefix with a cached sequence (e.g., same system prompt), the cached KV blocks are reused — no recomputation. The cache is copy-on-write, so different completions share the same prefix blocks. Prefix caching is most effective when: (1) system prompts are long, (2) few-shot examples are repeated, (3) multi-turn chat accumulates context.",
    tradeoffs: [
      { label: "↑ Throughput",   val: "Up to 5× speedup on workloads with 80%+ shared prefix" },
      { label: "↑ TTFT",         val: "Prefill of shared tokens is skipped — major TTFT reduction" },
      { label: "↓ Memory",       val: "Slightly more KV cache fragmentation in rare edge cases" },
    ],
    when: {
      throughput:  "Always enable for chat APIs. Dramatic gains when system prompts > 200 tokens.",
      latency:     "Enable — reduces TTFT by skipping prefill on shared tokens.",
      cost:        "Enables serving more requests/GPU-hour by reducing compute per request.",
      memory:      "Neutral to slightly negative. Blocks are reused, not duplicated.",
      comparison:  "Disable for apples-to-apples model comparison (isolates model speed from caching artifacts).",
    },
    example: "vllm serve <model> --enable-prefix-caching",
    recipe: { throughput: "Always on", latency: "Always on", cost: "Always on", comparison: "Disable" },
  },
  {
    name: "--kv-cache-dtype",
    alias: "",
    category: "KV Cache",
    type: "enum",
    options: "auto | fp8 | fp8_e5m2 | fp8_e4m3",
    default: "auto (matches model dtype)",
    focus: ["throughput", "memory", "cost"],
    impact: "high",
    summary: "Quantize KV cache tensors to fp8 — halves KV memory usage.",
    detail: "KV cache typically consumes 30–60% of GPU memory during serving. fp8_e4m3 quantizes every K and V tensor as it's written into the cache, freeing ~50% of KV memory. This allows either: more concurrent sequences in the same VRAM, or longer context windows. Requires H100 or Ada Lovelace (RTX 4090, L40S). fp8_e4m3 (4 exponent, 3 mantissa) is better suited for KV values than fp8_e5m2.",
    tradeoffs: [
      { label: "↑ 2× KV capacity",    val: "Double max_num_seqs or max_model_len in same VRAM" },
      { label: "↑ Throughput",         val: "Higher batch size → more tok/s" },
      { label: "↓ Quality (minor)",    val: "~0.1–0.3 perplexity increase on long sequences" },
      { label: "H100/Ada only",        val: "Falls back to fp16 on unsupported hardware" },
    ],
    when: {
      throughput:  "Enable on H100 — more KV space = bigger batches = more tok/s.",
      memory:      "Essential when hitting KV OOM at large batch sizes.",
      cost:        "More seq/GPU = higher utilization = lower cost per request.",
      latency:     "Neutral — doesn't affect per-request latency directly.",
      comparison:  "Disable — only available on H100+, biases hardware comparison.",
    },
    example: "vllm serve <model> --kv-cache-dtype fp8_e4m3",
    recipe: { throughput: "fp8_e4m3 on H100", memory: "fp8_e4m3 to double capacity" },
  },
  {
    name: "--block-size",
    alias: "",
    category: "KV Cache",
    type: "int",
    options: "8 | 16 | 32",
    default: "16",
    focus: ["memory", "throughput"],
    impact: "medium",
    summary: "Token granularity of PagedAttention KV blocks.",
    detail: "PagedAttention divides KV memory into fixed-size logical blocks (like OS virtual memory pages). block_size=16 means each block holds KV tensors for 16 tokens. Smaller blocks: less internal fragmentation (wasted space at end of sequences shorter than a block), but higher block table management overhead and worse cache line alignment. Larger blocks: better GPU memory access patterns, fewer metadata ops, but more waste for short sequences.",
    tradeoffs: [
      { label: "block=8",   val: "Best for very short sequences (<64 tokens). Less waste." },
      { label: "block=16",  val: "Optimal for mixed workloads. Default for a reason." },
      { label: "block=32",  val: "Best for long-context workloads (>4K tokens). Better prefetch patterns." },
    ],
    when: {
      memory:      "Use 8 for short-seq workloads to reduce fragmentation. Use 32 for RAG/long-context.",
      throughput:  "32 improves GPU memory access patterns for long sequences.",
      latency:     "Minimal impact. Slightly faster with 32 for long sequences.",
      comparison:  "Keep at default 16.",
      cost:        "Reducing fragmentation with appropriate block size can increase effective batch size.",
    },
    example: "vllm serve <model> --block-size 32  # for long-context RAG",
    recipe: { memory: "8 for short, 32 for long context" },
  },
  {
    name: "--swap-space",
    alias: "",
    category: "KV Cache",
    type: "float (GiB)",
    default: "4",
    focus: ["memory", "throughput"],
    impact: "low",
    summary: "CPU RAM reserved for swapping KV blocks when GPU cache is exhausted.",
    detail: "When GPU KV cache is full, vLLM can offload lower-priority sequence KV blocks to CPU RAM (via PCIe DMA) instead of aborting them. The scheduler prioritizes decode-phase sequences over prefill; preempted prefills are swapped out. High swap space prevents request aborts under burst traffic but introduces latency when swap actually occurs (PCIe bandwidth = 32 GB/s vs HBM 2+ TB/s).",
    tradeoffs: [
      { label: "↑ Fewer aborts",   val: "Handles traffic spikes without dropping requests" },
      { label: "↓ Latency spike",  val: "PCIe swap adds 10–100ms when triggered" },
      { label: "↓ CPU RAM",        val: "Consumes system memory" },
    ],
    when: {
      memory:      "Increase to 16–32 GiB if you see high request abort rates.",
      throughput:  "Neutral in steady state. Prevents throughput collapse under bursts.",
      latency:     "Keep low or disable if P99 latency is critical — swaps cause spikes.",
      cost:        "Higher swap space → fewer dropped requests → higher effective utilization.",
      comparison:  "Disable (set to 0) for clean benchmarks — swap distorts latency distributions.",
    },
    example: "vllm serve <model> --swap-space 16",
    recipe: { memory: "16-32 for production", comparison: "0 for benchmarks" },
  },

  // ── BATCHING ─────────────────────────────────────────────────────────────
  {
    name: "--max-num-seqs",
    alias: "",
    category: "Batching",
    type: "int",
    default: "256",
    focus: ["throughput", "latency", "cost"],
    impact: "critical",
    summary: "Maximum number of sequences in a single decode iteration (effective batch size).",
    detail: "Controls how many requests are processed simultaneously during the autoregressive decode phase. vLLM uses continuous batching — new requests are slotted in as others finish, without waiting for a fixed batch to complete. Higher max_num_seqs improves GPU utilization by keeping more work in flight. Limited by: KV cache capacity (each sequence needs blocks), and gpu_memory_utilization.",
    tradeoffs: [
      { label: "↑ Throughput",   val: "Higher GPU utilization → more tok/s" },
      { label: "↑ Cost eff.",    val: "More requests per GPU-hour" },
      { label: "↓ Latency",      val: "More queuing → higher average response time" },
      { label: "↓ Memory",       val: "Each sequence consumes KV blocks" },
    ],
    when: {
      throughput:  "Set as high as KV memory allows. Start at 256, increase until OOM.",
      latency:     "Lower (32–64) for tight latency SLAs — reduces queuing time.",
      cost:        "High (256–512) for batch/offline workloads to maximize GPU utilization.",
      memory:      "Lower if hitting KV cache OOM. Each seq uses block_size × max_seq_len × 2 × num_layers × head_dim bytes.",
      comparison:  "Fix to same value across model comparisons.",
    },
    example: "vllm serve <model> --max-num-seqs 512",
    recipe: { throughput: "256–512", latency: "32–64", cost: "512+", comparison: "fix at 128" },
  },
  {
    name: "--max-num-batched-tokens",
    alias: "",
    category: "Batching",
    type: "int",
    default: "max_model_len",
    focus: ["throughput", "latency"],
    impact: "critical",
    summary: "Max total tokens (prompt + decode) processed in one forward pass.",
    detail: "The primary lever for GPU compute utilization. A forward pass processes all prefill tokens for new requests plus one decode token per active sequence. Higher values pack more prefill tokens into each step, improving matmul efficiency (GPU likes large dense matrices). Bounded by GPU SRAM for attention computation and activation memory. Setting too high causes OOM. The scheduler respects this budget — it will batch as many prefill tokens as possible up to this limit per step.",
    tradeoffs: [
      { label: "↑ Throughput",    val: "Larger matrices = better GPU FLOP utilization" },
      { label: "↑ Prefill speed", val: "More tokens per step = faster time-to-decode-phase" },
      { label: "↓ Latency",       val: "Large prefill batches can delay decode steps (head-of-line blocking)" },
      { label: "↓ OOM risk",      val: "Scales with O(n²) attention for long contexts" },
    ],
    when: {
      throughput:  "Start at 4096, double to 8192 then 16384. Benchmark each step.",
      latency:     "Lower (1024–2048) to reduce prefill HOL blocking on decode latency.",
      cost:        "High (8192+) for batch/offline inference — pure throughput mode.",
      comparison:  "Fix to same value. Different models may have different optimal points.",
      memory:      "Reduce if hitting OOM during prefill.",
    },
    example: "vllm serve <model> --max-num-batched-tokens 8192",
    recipe: { throughput: "8192–16384", latency: "1024–2048", cost: "16384+", comparison: "fix at 4096" },
  },
  {
    name: "--max-model-len",
    alias: "",
    category: "Batching",
    type: "int",
    default: "model's max context (often 8K–128K)",
    focus: ["memory", "throughput", "comparison"],
    impact: "high",
    summary: "Truncate max context length to free KV cache memory for larger batches.",
    detail: "KV cache size = max_model_len × num_layers × 2 × num_heads × head_dim × bytes_per_element. Halving max_model_len roughly halves KV memory, allowing twice as many concurrent sequences. Use when: actual workload context is much shorter than model's maximum, you want more batch parallelism over long context. A 70B model with 128K context uses ~480GB of KV per sequence — impractical at scale.",
    tradeoffs: [
      { label: "↑ More sequences",   val: "Freed KV memory allows higher max_num_seqs" },
      { label: "↑ Throughput",       val: "Bigger batches → better GPU utilization" },
      { label: "↓ Context length",   val: "Long documents / conversations get truncated" },
    ],
    when: {
      memory:      "Set to 2–4× your p95 input length. Massive KV memory savings.",
      throughput:  "Reduce if your workload is short-context to allow bigger batches.",
      comparison:  "Fix to same value across models for fair KV memory comparison.",
      cost:        "Shorter context = more requests per GPU = lower cost per request.",
      latency:     "Minimal direct impact on per-token latency.",
    },
    example: "vllm serve <model> --max-model-len 4096",
    recipe: { memory: "2× your p95 prompt length", comparison: "fix to 4096 or 8192" },
  },
  {
    name: "--scheduler-delay-factor",
    alias: "",
    category: "Batching",
    type: "float",
    default: "0.0",
    focus: ["throughput", "cost"],
    impact: "medium",
    summary: "Delay scheduling to allow larger prefill batches to accumulate.",
    detail: "A delay factor of D means the scheduler waits up to D × last_prefill_latency before dispatching a new iteration. This allows more requests to arrive and be batched together into a single large prefill step, improving GPU utilization. Think of it as an admission control delay. The tradeoff is higher TTFT for all requests.",
    tradeoffs: [
      { label: "↑ Throughput",   val: "Larger prefill batches → better GPU efficiency" },
      { label: "↑ Cost",         val: "Higher utilization = more requests per GPU-hour" },
      { label: "↓ TTFT",         val: "Intentional delay before first token" },
    ],
    when: {
      throughput:  "Set 0.3–0.7 for batch/offline workloads where TTFT doesn't matter.",
      cost:        "Higher delay → denser batches → lower cost per token.",
      latency:     "Keep at 0.0 — directly increases TTFT.",
      comparison:  "Keep at 0.0 for fair comparisons.",
      memory:      "No impact.",
    },
    example: "vllm serve <model> --scheduler-delay-factor 0.5",
    recipe: { throughput: "0.3–0.7 for offline", latency: "0.0", cost: "0.5+" },
  },

  // ── MEMORY ───────────────────────────────────────────────────────────────
  {
    name: "--gpu-memory-utilization",
    alias: "",
    category: "Memory",
    type: "float (0–1)",
    default: "0.90",
    focus: ["throughput", "memory", "cost"],
    impact: "high",
    summary: "Fraction of GPU VRAM allocated to vLLM (weights + KV cache).",
    detail: "vLLM runs a profiling pass at startup to measure peak activation memory, then allocates remaining_memory × utilization for KV cache blocks. Higher utilization = more KV blocks = more concurrent sequences. At 0.90 with a 40GB GPU and a model using 25GB of weights: KV cache gets ~13.5GB. Bumping to 0.95 gives ~16.5GB — ~22% more KV capacity. Avoid 0.99+ as CUDA context and activation peaks can cause OOM.",
    tradeoffs: [
      { label: "↑ More KV cache",   val: "Higher max_num_seqs and longer sequences" },
      { label: "↑ Throughput",      val: "More batching → better GPU utilization" },
      { label: "↓ OOM risk",        val: "Activation peaks + fragmentation can exceed budget" },
    ],
    when: {
      throughput:  "Start at 0.90, push to 0.93 if stable, then 0.95. Stop before OOM.",
      memory:      "0.90–0.95 is the safe range. Measure actual KV block count at startup.",
      cost:        "Higher utilization = more concurrent work = lower cost per request.",
      latency:     "Minimal direct impact.",
      comparison:  "Fix to 0.90 for fair comparisons across GPUs with same VRAM.",
    },
    example: "vllm serve <model> --gpu-memory-utilization 0.93",
    recipe: { throughput: "0.93–0.95", memory: "0.90 safe default", comparison: "fix at 0.90" },
  },
  {
    name: "--enforce-eager",
    alias: "",
    category: "Memory",
    type: "bool",
    default: "false",
    focus: ["memory", "latency"],
    impact: "high",
    summary: "Disable CUDA graph capture — eager PyTorch mode.",
    detail: "CUDA graphs pre-compile and cache fixed GPU kernel sequences for common decode batch sizes. This eliminates CPU-side kernel launch overhead (typically 20–100µs per batch), giving 20–40% throughput improvement during decode. Graph capture happens at startup for batch sizes 1, 2, 4, 8 … max_num_seqs, consuming extra startup time and memory per graph. enforce_eager=true disables this entirely.",
    tradeoffs: [
      { label: "true: ↑ Startup speed",  val: "No graph capture = instant start" },
      { label: "true: ↑ Less VRAM",      val: "Saves memory used by CUDA graphs" },
      { label: "true: ↓ Decode throughput", val: "20–40% lower throughput — CPU kernel launch overhead" },
    ],
    when: {
      memory:      "Enable if running out of VRAM due to graph capture memory overhead.",
      latency:     "Disable (default) — CUDA graphs reduce decode latency.",
      throughput:  "Keep false (default). Never enable for production throughput.",
      cost:        "Keep false — lower throughput = higher cost per token.",
      comparison:  "Fix to false for all comparisons.",
    },
    example: "vllm serve <model> --enforce-eager  # debugging only",
    recipe: { memory: "true only if OOM at startup", latency: "false (default)" },
  },

  // ── ATTENTION BACKENDS ───────────────────────────────────────────────────
  {
    name: "--attention-backend",
    alias: "",
    category: "Attention",
    type: "enum",
    options: "FLASH_ATTN | FLASHINFER | TORCH_SDPA | TRITON | XFORMERS",
    default: "auto (best for hardware)",
    focus: ["throughput", "latency", "comparison"],
    impact: "high",
    summary: "Attention kernel implementation — major impact on both prefill and decode speed.",
    detail: "FLASH_ATTN (FlashAttention v2/v3): Best for most NVIDIA GPUs. Fused attention with online softmax, O(n) memory. Required for RoPE models at high throughput. FLASHINFER: Highly optimized for decode-heavy workloads, better parallelism for small decode batches. Often best for latency-sensitive serving. TORCH_SDPA: PyTorch native, universal hardware support (CPU, MPS, ROCm). Slower than Flash kernels. TRITON: Good for ALiBi positional encodings. xFormers: Legacy, use Flash when possible.",
    tradeoffs: [
      { label: "FLASH_ATTN",   val: "Best for large prefill batches, high throughput" },
      { label: "FLASHINFER",   val: "Best for decode-heavy, low-latency serving" },
      { label: "TORCH_SDPA",   val: "Universal but slowest. Use for non-NVIDIA hardware." },
    ],
    when: {
      throughput:  "FLASH_ATTN for highest throughput on NVIDIA. FlashInfer for decode-heavy.",
      latency:     "FLASHINFER often best for minimize per-token latency.",
      comparison:  "Fix to FLASH_ATTN for all comparisons.",
      memory:      "FlashAttention uses O(n) memory vs O(n²) — critical for long context.",
      cost:        "Best kernel = best efficiency = lowest cost.",
    },
    example: "VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve <model>",
    recipe: { throughput: "FLASH_ATTN", latency: "FLASHINFER", memory: "FLASH_ATTN (O(n) memory)" },
  },

  // ── SPECULATIVE DECODING ─────────────────────────────────────────────────
  {
    name: "--speculative-model",
    alias: "",
    category: "Speculative",
    type: "string",
    default: "None",
    focus: ["latency", "throughput"],
    impact: "high",
    summary: "Draft model for speculative decoding — dramatically reduces latency.",
    detail: "Speculative decoding uses a small draft model to predict N tokens ahead, then verifies them all in a single forward pass of the large target model. If all tokens are accepted, you get N+1 tokens per target forward pass. Acceptance rate depends on draft/target alignment — same-family models work best (Llama 3 8B drafting for Llama 3 70B). Use --num-speculative-tokens to set draft length (3–5 is typical). vLLM supports: separate draft model or [ngram] for N-gram speculation (no draft model needed).",
    tradeoffs: [
      { label: "↑ Latency (TTFT unchanged)", val: "2–3× lower inter-token latency when acceptance rate >70%" },
      { label: "↓ Throughput",               val: "Lower batch utilization — better for latency, not throughput" },
      { label: "↓ Memory",                   val: "Requires loading both draft and target model" },
    ],
    when: {
      latency:     "Best tool for reducing inter-token latency. Use same-family draft (e.g., 8B for 70B).",
      throughput:  "Avoid — speculative decoding trades throughput for latency.",
      memory:      "Avoid if VRAM is tight — needs two models loaded.",
      cost:        "Neutral to negative — lower throughput means more GPU time per batch.",
      comparison:  "Disable for model comparison benchmarks.",
    },
    example: "vllm serve meta-llama/Llama-3.1-70B-Instruct --speculative-model meta-llama/Llama-3.2-1B-Instruct --num-speculative-tokens 5",
    recipe: { latency: "Same-family small model, 3–5 speculative tokens" },
  },
  {
    name: "--num-speculative-tokens",
    alias: "",
    category: "Speculative",
    type: "int",
    default: "None",
    focus: ["latency"],
    impact: "medium",
    summary: "Number of tokens the draft model generates ahead per speculation step.",
    detail: "More speculative tokens = more potential speedup if all accepted, but lower acceptance rate (probability drops exponentially with sequence length). Optimal K balances acceptance rate × tokens_per_step. For chat/QA workloads: 3–5 works well. For code completion with high alignment: 5–8. Monitor acceptance_rate metric — if below 60%, reduce K. If above 85%, increase K.",
    tradeoffs: [
      { label: "K=3",   val: "Safe default. Good acceptance rate on most workloads." },
      { label: "K=5",   val: "Good for same-family models. Balance of speed and acceptance." },
      { label: "K=8+",  val: "Only if acceptance rate stays >80% for your specific workload." },
    ],
    when: {
      latency:     "Start at 5 with same-family draft. Monitor acceptance rate, tune accordingly.",
      throughput:  "N/A — only relevant with speculative model enabled.",
      comparison:  "Disable.",
      memory:      "N/A.",
      cost:        "N/A.",
    },
    example: "vllm serve <model> --speculative-model <draft> --num-speculative-tokens 5",
    recipe: { latency: "5 for same-family, 3 for cross-family" },
  },

  // ── MODEL LOADING ────────────────────────────────────────────────────────
  {
    name: "--load-format",
    alias: "",
    category: "Model Loading",
    type: "enum",
    options: "auto | pt | safetensors | npcache | dummy | gguf | bitsandbytes | mistral",
    default: "auto",
    focus: ["cost", "comparison"],
    impact: "low",
    summary: "Controls how model weights are deserialized from disk.",
    detail: "safetensors: Fastest, most memory-safe (mmap-friendly). npcache: Converts to numpy format on first load, caches to disk — faster cold starts on subsequent runs. dummy: Random weights, no download — pure latency profiling. pt: PyTorch pickle format — slower and less safe. gguf/bitsandbytes: For pre-quantized GGUF or bitsandbytes models.",
    tradeoffs: [
      { label: "safetensors",  val: "Fastest general-purpose loading" },
      { label: "npcache",      val: "Fastest after first run. Extra ~2× disk space." },
      { label: "dummy",        val: "For benchmarking infrastructure without model download" },
    ],
    when: {
      cost:        "npcache if you restart frequently with same model — saves GPU startup cost.",
      comparison:  "Use dummy to benchmark vLLM overhead without model-specific variation.",
      throughput:  "Minimal impact after model is loaded.",
      latency:     "Minimal impact after model is loaded.",
      memory:      "safetensors enables direct mmap — lower peak RAM during load.",
    },
    example: "vllm serve <model> --load-format npcache",
    recipe: { cost: "npcache for frequent restarts", comparison: "dummy for infrastructure profiling" },
  },
  {
    name: "--tokenizer-pool-size",
    alias: "",
    category: "Model Loading",
    type: "int",
    default: "0 (synchronous)",
    focus: ["throughput", "latency"],
    impact: "low",
    summary: "Async tokenizer worker pool to overlap tokenization with inference.",
    detail: "With pool_size > 0, tokenization is offloaded to background processes, preventing slow tokenizers from blocking the GPU inference loop. Relevant at very high QPS (>100 req/s) where tokenizer latency becomes a measurable fraction of request time. The pool uses Python multiprocessing, so overhead is non-trivial at low QPS.",
    tradeoffs: [
      { label: "↑ Throughput at high QPS",  val: "Hides tokenizer latency" },
      { label: "↓ Low QPS overhead",         val: "Process pool adds latency at <50 req/s" },
    ],
    when: {
      throughput:  "Enable (4–8 workers) if QPS > 100 and tokenizer is slow.",
      latency:     "Minimal impact unless tokenizer is the bottleneck.",
      cost:        "Minor improvement at high QPS.",
      comparison:  "Keep at 0 for consistent benchmarks.",
      memory:      "Each worker uses ~100MB RAM.",
    },
    example: "vllm serve <model> --tokenizer-pool-size 4",
    recipe: { throughput: "4 workers at high QPS" },
  },
];

// ─── COMPONENT ────────────────────────────────────────────────────────────────

const IMPACT_STYLES = {
  critical: { color: "#ef4444", bg: "#450a0a", label: "CRITICAL" },
  high:     { color: "#f59e0b", bg: "#451a03", label: "HIGH" },
  medium:   { color: "#38bdf8", bg: "#082f49", label: "MEDIUM" },
  low:      { color: "#6b7280", bg: "#1f2937", label: "LOW" },
};

function Tag({ children, bg, color, style }) {
  return (
    <span style={{
      background: bg, color, padding: "2px 8px", borderRadius: 4,
      fontSize: 10, fontWeight: 700, letterSpacing: "0.06em",
      whiteSpace: "nowrap", ...style,
    }}>{children}</span>
  );
}

function RecipeBox({ param, focus }) {
  const rec = param.recipe?.[focus];
  if (!rec) return null;
  const fm = FOCUS_META[focus];
  return (
    <div style={{ marginTop: 8, padding: "8px 12px", background: "#0d1117", borderRadius: 6, borderLeft: `3px solid ${fm.color}` }}>
      <span style={{ fontSize: 10, color: fm.color, fontWeight: 700, letterSpacing: "0.08em" }}>{fm.icon} {fm.label.toUpperCase()} RECIPE: </span>
      <code style={{ fontSize: 12, color: "#86efac" }}>{rec}</code>
    </div>
  );
}

export default function App() {
  const [activeFocus, setActiveFocus] = useState("throughput");
  const [activeCategory, setActiveCategory] = useState("All");
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState(null);
  const [impactFilter, setImpactFilter] = useState("all");
  const [showAll, setShowAll] = useState(false);

  const fm = FOCUS_META[activeFocus];

  const filtered = useMemo(() => {
    return PARAMS.filter(p => {
      const matchFocus = showAll || p.focus.includes(activeFocus);
      const matchCat = activeCategory === "All" || p.category === activeCategory;
      const matchImp = impactFilter === "all" || p.impact === impactFilter;
      const q = search.toLowerCase();
      const matchSearch = !q ||
        p.name.toLowerCase().includes(q) ||
        p.summary.toLowerCase().includes(q) ||
        p.category.toLowerCase().includes(q) ||
        (p.alias && p.alias.toLowerCase().includes(q));
      return matchFocus && matchCat && matchImp && matchSearch;
    }).sort((a, b) => {
      const order = { critical: 0, high: 1, medium: 2, low: 3 };
      return order[a.impact] - order[b.impact];
    });
  }, [activeFocus, activeCategory, search, impactFilter, showAll]);

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
      background: "#080c12",
      minHeight: "100vh",
      color: "#cbd5e1",
    }}>
      {/* Top header */}
      <div style={{ background: "#0a0e17", borderBottom: "1px solid #0f172a", padding: "28px 32px 20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
          <div style={{ width: 3, height: 32, background: fm.color, borderRadius: 2, transition: "background 0.3s" }} />
          <div>
            <div style={{ fontSize: 10, letterSpacing: "0.18em", color: "#334155", fontWeight: 600 }}>VLLM CONFIG REFERENCE v2</div>
            <h1 style={{ margin: "2px 0 0", fontSize: 22, fontWeight: 800, color: "#f1f5f9", letterSpacing: "-0.03em" }}>
              Parameter Explorer
            </h1>
          </div>
        </div>

        {/* Focus selector */}
        <div style={{ marginTop: 20, display: "flex", gap: 8, flexWrap: "wrap" }}>
          {Object.entries(FOCUS_META).map(([key, meta]) => {
            const active = activeFocus === key;
            return (
              <button
                key={key}
                onClick={() => { setActiveFocus(key); setExpanded(null); setShowAll(false); }}
                style={{
                  padding: "8px 16px", borderRadius: 8, cursor: "pointer",
                  border: `1px solid ${active ? meta.color : "#1e293b"}`,
                  background: active ? meta.bg : "transparent",
                  color: active ? meta.color : "#475569",
                  fontSize: 12, fontWeight: 700, letterSpacing: "0.04em",
                  transition: "all 0.2s", display: "flex", alignItems: "center", gap: 6,
                }}
              >
                <span>{meta.icon}</span> {meta.label}
              </button>
            );
          })}
        </div>

        {/* Focus description */}
        <div style={{
          marginTop: 10, padding: "10px 14px", borderRadius: 6,
          background: fm.bg, borderLeft: `3px solid ${fm.color}`,
          fontSize: 12, color: fm.color, transition: "all 0.3s",
        }}>
          {fm.icon} <strong>{fm.label}</strong> — {fm.desc}
          <span style={{ marginLeft: 12, fontSize: 11, color: "#475569" }}>
            {filtered.length} parameter{filtered.length !== 1 ? "s" : ""} relevant to this goal
          </span>
        </div>
      </div>

      {/* Filter bar */}
      <div style={{ padding: "12px 32px", borderBottom: "1px solid #0f172a", display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center", background: "#090d15" }}>
        {/* Search */}
        <div style={{ position: "relative", flex: "1 1 200px" }}>
          <span style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "#334155", fontSize: 13 }}>⌕</span>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search..."
            style={{
              width: "100%", boxSizing: "border-box",
              padding: "7px 10px 7px 28px", background: "#111827",
              border: "1px solid #1e293b", borderRadius: 6,
              color: "#e2e8f0", fontSize: 12, outline: "none",
            }}
          />
        </div>

        {/* Category pills */}
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              style={{
                padding: "4px 10px", borderRadius: 12, fontSize: 10, fontWeight: 700,
                letterSpacing: "0.06em", cursor: "pointer",
                border: `1px solid ${activeCategory === cat ? fm.color : "#1e293b"}`,
                background: activeCategory === cat ? fm.bg : "transparent",
                color: activeCategory === cat ? fm.color : "#475569",
              }}
            >{cat.toUpperCase()}</button>
          ))}
        </div>

        {/* Impact */}
        <div style={{ display: "flex", gap: 4 }}>
          {["all", "critical", "high", "medium", "low"].map(imp => {
            const s = imp === "all" ? { color: "#94a3b8", bg: "#1e293b" } : IMPACT_STYLES[imp];
            return (
              <button
                key={imp}
                onClick={() => setImpactFilter(imp)}
                style={{
                  padding: "4px 8px", borderRadius: 6, fontSize: 10, cursor: "pointer",
                  border: `1px solid ${impactFilter === imp ? s.color : "#1e293b"}`,
                  background: impactFilter === imp ? s.bg : "transparent",
                  color: impactFilter === imp ? s.color : "#334155",
                  fontWeight: 700,
                }}
              >{imp.toUpperCase()}</button>
            );
          })}
        </div>

        <label style={{ fontSize: 11, color: "#475569", cursor: "pointer", display: "flex", alignItems: "center", gap: 6, marginLeft: "auto" }}>
          <input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)}
            style={{ accentColor: fm.color }} />
          Show all parameters
        </label>
      </div>

      {/* Param list */}
      <div style={{ padding: "12px 32px 48px" }}>
        {filtered.map(param => {
          const isOpen = expanded === param.name;
          const imp = IMPACT_STYLES[param.impact];
          const whenText = param.when[activeFocus];

          return (
            <div
              key={param.name}
              style={{
                background: isOpen ? "#0e1420" : "#0b0f1a",
                border: `1px solid ${isOpen ? fm.color : "#111827"}`,
                borderRadius: 8, marginBottom: 6, overflow: "hidden",
                transition: "border-color 0.15s",
              }}
            >
              {/* Row */}
              <div
                onClick={() => setExpanded(isOpen ? null : param.name)}
                style={{ padding: "12px 16px", cursor: "pointer", display: "flex", alignItems: "flex-start", gap: 12 }}
              >
                {/* Impact bar */}
                <div style={{ width: 3, borderRadius: 2, background: imp.color, flexShrink: 0, alignSelf: "stretch", minHeight: 20 }} />

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <code style={{ fontSize: 13, fontWeight: 800, color: fm.color }}>{param.name}</code>
                    {param.alias && <code style={{ fontSize: 11, color: "#334155" }}>{param.alias}</code>}
                    <Tag bg={imp.bg} color={imp.color}>{imp.label}</Tag>
                    <Tag bg="#111827" color="#475569">{param.category}</Tag>
                    {param.focus.filter(f => f !== activeFocus).map(f => (
                      <Tag key={f} bg={FOCUS_META[f].bg} color={FOCUS_META[f].color} style={{ fontSize: 9 }}>
                        {FOCUS_META[f].icon}
                      </Tag>
                    ))}
                    <Tag bg="#0f172a" color="#334155" style={{ fontSize: 9 }}>{param.type}</Tag>
                  </div>
                  <div style={{ marginTop: 4, fontSize: 12, color: "#64748b", lineHeight: 1.5 }}>{param.summary}</div>

                  {/* Focus-specific when text */}
                  {whenText && !isOpen && (
                    <div style={{ marginTop: 6, fontSize: 11, color: fm.color, background: fm.bg, padding: "4px 8px", borderRadius: 4, display: "inline-block" }}>
                      {fm.icon} {whenText}
                    </div>
                  )}
                </div>

                <div style={{ color: "#1e293b", fontSize: 16, flexShrink: 0 }}>{isOpen ? "▲" : "▼"}</div>
              </div>

              {/* Expanded panel */}
              {isOpen && (
                <div style={{ borderTop: "1px solid #111827", padding: "16px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  {/* Left */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <Section title="HOW IT WORKS" color="#38bdf8">
                      <p style={{ margin: 0, fontSize: 12, color: "#94a3b8", lineHeight: 1.8 }}>{param.detail}</p>
                    </Section>

                    <Section title="TRADEOFFS" color="#f59e0b">
                      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                        {param.tradeoffs.map((t, i) => (
                          <div key={i} style={{ display: "flex", gap: 8, fontSize: 12 }}>
                            <code style={{ color: "#f59e0b", flexShrink: 0, minWidth: 140 }}>{t.label}</code>
                            <span style={{ color: "#64748b" }}>{t.val}</span>
                          </div>
                        ))}
                      </div>
                    </Section>
                  </div>

                  {/* Right */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <Section title={`${fm.icon} WHEN TO USE — ${fm.label.toUpperCase()}`} color={fm.color}>
                      <p style={{ margin: 0, fontSize: 12, color: fm.color, lineHeight: 1.7, background: fm.bg, padding: "8px 10px", borderRadius: 6 }}>
                        {param.when[activeFocus] || "—"}
                      </p>

                      {/* All focus guidance */}
                      <div style={{ marginTop: 10 }}>
                        {Object.entries(param.when).filter(([k]) => k !== activeFocus).map(([k, v]) => {
                          const f = FOCUS_META[k];
                          return (
                            <div key={k} style={{ marginBottom: 5, fontSize: 11, color: "#334155" }}>
                              <span style={{ color: f.color }}>{f.icon} {f.label}: </span>{v}
                            </div>
                          );
                        })}
                      </div>
                    </Section>

                    <Section title="DEFAULT & OPTIONS" color="#a78bfa">
                      <div style={{ fontSize: 12 }}>
                        <span style={{ color: "#475569" }}>default: </span>
                        <code style={{ color: "#c4b5fd" }}>{param.default}</code>
                      </div>
                      {param.options && (
                        <div style={{ marginTop: 4, fontSize: 11, color: "#334155" }}>options: {param.options}</div>
                      )}
                    </Section>

                    <Section title="EXAMPLE" color="#34d399">
                      <code style={{ display: "block", fontSize: 11, color: "#86efac", background: "#0a0f1a", padding: "8px 10px", borderRadius: 6, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-all", lineHeight: 1.6 }}>
                        {param.example}
                      </code>
                    </Section>

                    {param.recipe && Object.keys(param.recipe).length > 0 && (
                      <Section title="RECIPES BY GOAL" color="#f472b6">
                        {Object.entries(param.recipe).map(([k, v]) => {
                          const f = FOCUS_META[k];
                          return (
                            <div key={k} style={{ marginBottom: 5 }}>
                              <span style={{ fontSize: 10, color: f.color, fontWeight: 700 }}>{f.icon} {f.label.toUpperCase()}: </span>
                              <code style={{ fontSize: 11, color: "#86efac" }}>{v}</code>
                            </div>
                          );
                        })}
                      </Section>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {filtered.length === 0 && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "#1e293b" }}>
            <div style={{ fontSize: 36 }}>∅</div>
            <div style={{ marginTop: 8, fontSize: 13 }}>No parameters match</div>
          </div>
        )}
      </div>

      {/* Bottom recipe strip */}
      <div style={{ position: "sticky", bottom: 0, background: "#080c12", borderTop: "1px solid #0f172a", padding: "14px 32px" }}>
        <div style={{ fontSize: 10, letterSpacing: "0.12em", color: fm.color, marginBottom: 8, fontWeight: 700 }}>
          {fm.icon} {fm.label.toUpperCase()} — RECOMMENDED CONFIGURATION
        </div>
        <div style={{ overflowX: "auto" }}>
          {activeFocus === "throughput" && <Recipe code={`vllm serve <model> --tensor-parallel-size <N> --quantization awq --enable-prefix-caching --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.93 --kv-cache-dtype fp8_e4m3 --dtype bfloat16`} color={fm.color} />}
          {activeFocus === "latency" && <Recipe code={`vllm serve <model> --max-num-seqs 32 --max-num-batched-tokens 1024 --enable-prefix-caching --speculative-model <draft-model> --num-speculative-tokens 5 --attention-backend FLASHINFER --scheduler-delay-factor 0.0`} color={fm.color} />}
          {activeFocus === "memory" && <Recipe code={`vllm serve <model> --tensor-parallel-size <N> --quantization awq --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.92 --max-model-len 4096 --block-size 8 --swap-space 16`} color={fm.color} />}
          {activeFocus === "cost" && <Recipe code={`vllm serve <model> --quantization awq --enable-prefix-caching --max-num-seqs 512 --max-num-batched-tokens 16384 --scheduler-delay-factor 0.5 --gpu-memory-utilization 0.93 --load-format npcache`} color={fm.color} />}
          {activeFocus === "comparison" && <Recipe code={`vllm serve <model> --dtype bfloat16 --max-num-seqs 128 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.90 --max-model-len 4096 --swap-space 0 --scheduler-delay-factor 0.0 --enforce-eager false`} color={fm.color} />}
        </div>
      </div>
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div>
      <div style={{ fontSize: 9, letterSpacing: "0.14em", color, fontWeight: 800, marginBottom: 6 }}>{title}</div>
      {children}
    </div>
  );
}

function Recipe({ code, color }) {
  return (
    <code style={{
      display: "block", fontSize: 11, color: "#86efac",
      background: "#0a0f1a", padding: "10px 14px", borderRadius: 6,
      borderLeft: `3px solid ${color}`,
      whiteSpace: "pre-wrap", wordBreak: "break-all", lineHeight: 2,
    }}>{code}</code>
  );
}