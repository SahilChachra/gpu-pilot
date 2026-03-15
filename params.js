// vLLM Parameter data — shared across the app
window.VLLM_PARAMS = [
  {
    name: "--tensor-parallel-size", alias: "-tp", category: "Parallelism", type: "int", default: "1",
    focus: ["throughput","memory","cost"], impact: "critical",
    summary: "Split model weights across N GPUs using Megatron column/row parallelism.",
    detail: "Each GPU holds 1/N of every attention and FFN weight matrix. Forward pass requires all-reduce collectives to synchronize activations. Best on NVLink (A100/H100 SXM). PCIe GPUs incur significant communication overhead above TP=2.",
    tradeoffs: [{l:"↑ Throughput",v:"Linear scaling on NVLink, ~70% on PCIe"},{l:"↑ Memory",v:"Enables models larger than single GPU VRAM"},{l:"↓ Latency",v:"All-reduce adds 1–5ms per layer on PCIe"}],
    when: { throughput:"Set to number of GPUs. Best on NVLink. Cap at 2 on PCIe.", latency:"Avoid if model fits on 1 GPU — all-reduce adds overhead.", memory:"Required when model > single GPU VRAM. 70B needs TP≥4 on 80GB GPUs.", cost:"TP=4 on 4×A10G often cheaper than 1×A100 for same throughput.", comparison:"Fix TP=1 across all models being compared." },
    example: "vllm serve meta-llama/Llama-3.1-70B --tensor-parallel-size 4",
    recipe: { throughput:"Match GPU count", latency:"1 if model fits", memory:"ceil(model_gb / gpu_vram)" }
  },
  {
    name: "--pipeline-parallel-size", alias: "-pp", category: "Parallelism", type: "int", default: "1",
    focus: ["memory","throughput"], impact: "high",
    summary: "Split transformer layers across GPUs in pipeline stages.",
    detail: "Each GPU hosts a contiguous slice of layers. Micro-batches fill the pipeline to reduce bubble overhead. Combine with TP: tp=4 pp=2 → 8 GPUs total.",
    tradeoffs: [{l:"↑ Memory scale",v:"Enables 100B+ models across nodes"},{l:"↓ Latency",v:"Pipeline bubble adds idle time per batch"},{l:"↓ Complexity",v:"Harder to tune, more failure modes"}],
    when: { throughput:"Use only when model doesn't fit on TP alone.", memory:"Combine with TP: total GPUs = tp × pp.", latency:"Avoid — pipeline bubble directly hurts latency.", cost:"Can reduce cost on multi-node setups.", comparison:"Keep fixed across runs." },
    example: "vllm serve <model> --tensor-parallel-size 4 --pipeline-parallel-size 2",
    recipe: { memory:"tp=node_gpus, pp=num_nodes" }
  },
  {
    name: "--quantization", alias: "-q", category: "Quantization", type: "enum",
    options: "awq | gptq | fp8 | int8 | gguf | bitsandbytes", default: "None (fp16/bf16)",
    focus: ["throughput","memory","cost","comparison"], impact: "critical",
    summary: "Reduce weight precision to shrink model size and accelerate matrix multiplications.",
    detail: "AWQ (4-bit): Best quality at 4-bit, fastest via fused kernels. GPTQ (4-bit): Layer-wise quantization, slightly lower quality. FP8: Weights AND activations — requires H100/Ada, near fp16 quality with ~2× throughput. INT8: SmoothQuant, broader GPU support.",
    tradeoffs: [{l:"AWQ 4-bit",v:"2× VRAM reduction, 1.5–2× faster decode"},{l:"GPTQ 4-bit",v:"Similar VRAM, slightly slower kernels"},{l:"FP8 (H100)",v:"1.5–2× faster with near fp16 quality"},{l:"INT8",v:"~1.3× speedup, widely supported"}],
    when: { throughput:"FP8 on H100 > AWQ > GPTQ > INT8.", memory:"AWQ/GPTQ cut memory ~50%. Essential for 70B on fewer GPUs.", cost:"AWQ on smaller GPU often beats fp16 on larger GPU.", latency:"FP8 and AWQ have good latency.", comparison:"Benchmark same model at fp16 vs awq vs fp8." },
    example: "vllm serve TheBloke/Llama-2-13B-AWQ --quantization awq",
    recipe: { throughput:"fp8 on H100, awq elsewhere", memory:"awq or gptq for 4-bit", cost:"awq on smaller/cheaper GPU" }
  },
  {
    name: "--dtype", alias: "", category: "Quantization", type: "enum",
    options: "auto | half (fp16) | bfloat16 | float32", default: "auto",
    focus: ["throughput","memory","comparison"], impact: "high",
    summary: "Base floating point dtype for weights and activations.",
    detail: "bfloat16: Better dynamic range, fewer NaN/Inf issues. Preferred for modern LLMs. fp16: Better precision but can overflow. float32: Never use — doubles VRAM, halves throughput.",
    tradeoffs: [{l:"bf16 vs fp16",v:"bf16 more stable, fp16 marginally faster on old GPUs"},{l:"fp32",v:"2× memory cost, near-zero benefit for inference"}],
    when: { throughput:"Use bf16 for modern models, fp16 for Volta/Turing GPUs.", memory:"Both fp16/bf16 identical footprint. Avoid fp32.", comparison:"Fix to bfloat16 for all comparisons.", cost:"bf16/fp16 equivalent cost. Never fp32.", latency:"fp16 marginally faster on older hardware." },
    example: "vllm serve <model> --dtype bfloat16",
    recipe: { comparison:"Fix to bfloat16 for all runs" }
  },
  {
    name: "--enable-prefix-caching", alias: "", category: "KV Cache", type: "bool", default: "false",
    focus: ["throughput","latency","cost"], impact: "critical",
    summary: "Cache and reuse KV tensors for shared prompt prefixes across requests.",
    detail: "vLLM hashes token sequences using a radix tree. Shared prefixes (system prompts, few-shot examples) reuse cached KV blocks — no recomputation. Most effective for chat APIs with long system prompts.",
    tradeoffs: [{l:"↑ Throughput",v:"Up to 5× speedup with 80%+ shared prefix"},{l:"↑ TTFT",v:"Prefill of shared tokens is skipped"},{l:"↓ Memory",v:"Slight fragmentation in edge cases"}],
    when: { throughput:"Always enable for chat APIs. Dramatic gains with system prompts > 200 tokens.", latency:"Enable — reduces TTFT by skipping prefill on shared tokens.", cost:"More requests/GPU-hour by reducing compute per request.", memory:"Neutral to slightly negative.", comparison:"Disable for apples-to-apples model comparison." },
    example: "vllm serve <model> --enable-prefix-caching",
    recipe: { throughput:"Always on", latency:"Always on", cost:"Always on", comparison:"Disable" }
  },
  {
    name: "--kv-cache-dtype", alias: "", category: "KV Cache", type: "enum",
    options: "auto | fp8 | fp8_e5m2 | fp8_e4m3", default: "auto",
    focus: ["throughput","memory","cost"], impact: "high",
    summary: "Quantize KV cache tensors to fp8 — halves KV memory usage.",
    detail: "KV cache typically consumes 30–60% of GPU memory. fp8_e4m3 quantizes K and V tensors, freeing ~50% of KV memory. Allows either more concurrent sequences or longer context. Requires H100 or Ada Lovelace.",
    tradeoffs: [{l:"↑ 2× KV capacity",v:"Double max_num_seqs in same VRAM"},{l:"↑ Throughput",v:"Higher batch size → more tok/s"},{l:"↓ Quality (minor)",v:"~0.1–0.3 perplexity increase on long sequences"},{l:"H100/Ada only",v:"Falls back to fp16 on unsupported hardware"}],
    when: { throughput:"Enable on H100 — more KV space = bigger batches = more tok/s.", memory:"Essential when hitting KV OOM at large batch sizes.", cost:"More seq/GPU = higher utilization = lower cost per request.", latency:"Neutral.", comparison:"Disable — only available on H100+." },
    example: "vllm serve <model> --kv-cache-dtype fp8_e4m3",
    recipe: { throughput:"fp8_e4m3 on H100", memory:"fp8_e4m3 to double capacity" }
  },
  {
    name: "--block-size", alias: "", category: "KV Cache", type: "int",
    options: "8 | 16 | 32", default: "16",
    focus: ["memory","throughput"], impact: "medium",
    summary: "Token granularity of PagedAttention KV blocks.",
    detail: "Smaller blocks: less fragmentation, more overhead. Larger blocks: better GPU memory access patterns, more waste for short sequences. 16 is the sweet spot for most workloads.",
    tradeoffs: [{l:"block=8",v:"Best for very short sequences"},{l:"block=16",v:"Optimal for mixed workloads"},{l:"block=32",v:"Best for long-context workloads"}],
    when: { memory:"Use 8 for short-seq workloads. Use 32 for RAG/long-context.", throughput:"32 improves GPU memory access for long sequences.", latency:"Minimal impact.", comparison:"Keep at default 16.", cost:"Right size reduces fragmentation → better batch size." },
    example: "vllm serve <model> --block-size 32",
    recipe: { memory:"8 for short, 32 for long context" }
  },
  {
    name: "--swap-space", alias: "", category: "KV Cache", type: "float (GiB)", default: "4",
    focus: ["memory","throughput"], impact: "low",
    summary: "CPU RAM reserved for swapping KV blocks when GPU cache is exhausted.",
    detail: "When GPU KV cache is full, vLLM offloads lower-priority sequences to CPU RAM instead of aborting them. High swap space prevents aborts under burst traffic but introduces PCIe latency when triggered.",
    tradeoffs: [{l:"↑ Fewer aborts",v:"Handles traffic spikes without dropping requests"},{l:"↓ Latency spike",v:"PCIe swap adds 10–100ms when triggered"}],
    when: { memory:"Increase to 16–32 GiB if you see high abort rates.", throughput:"Prevents throughput collapse under bursts.", latency:"Keep low — swaps cause P99 spikes.", cost:"Higher swap → fewer dropped requests.", comparison:"Set to 0 for clean benchmarks." },
    example: "vllm serve <model> --swap-space 16",
    recipe: { memory:"16-32 for production", comparison:"0 for benchmarks" }
  },
  {
    name: "--max-num-seqs", alias: "", category: "Batching", type: "int", default: "256",
    focus: ["throughput","latency","cost"], impact: "critical",
    summary: "Maximum number of sequences in a single decode iteration (effective batch size).",
    detail: "Controls how many requests are processed simultaneously during decode. vLLM uses continuous batching — new requests slot in as others finish. Higher = better GPU utilization but more queuing latency.",
    tradeoffs: [{l:"↑ Throughput",v:"Higher GPU utilization → more tok/s"},{l:"↑ Cost eff.",v:"More requests per GPU-hour"},{l:"↓ Latency",v:"More queuing → higher average response time"},{l:"↓ Memory",v:"Each sequence consumes KV blocks"}],
    when: { throughput:"Set as high as KV memory allows. Start at 256.", latency:"Lower (32–64) for tight latency SLAs.", cost:"High (256–512) for batch/offline workloads.", memory:"Lower if hitting KV cache OOM.", comparison:"Fix to same value across model comparisons." },
    example: "vllm serve <model> --max-num-seqs 512",
    recipe: { throughput:"256–512", latency:"32–64", cost:"512+", comparison:"fix at 128" }
  },
  {
    name: "--max-num-batched-tokens", alias: "", category: "Batching", type: "int", default: "max_model_len",
    focus: ["throughput","latency"], impact: "critical",
    summary: "Max total tokens processed in one forward pass — primary throughput lever.",
    detail: "Higher values pack more prefill tokens per step, improving matmul efficiency. Bounded by GPU SRAM for attention computation. Setting too high causes OOM. Start at 4096, double until OOM.",
    tradeoffs: [{l:"↑ Throughput",v:"Larger matrices = better GPU FLOP utilization"},{l:"↑ Prefill speed",v:"More tokens per step = faster time-to-decode"},{l:"↓ Latency",v:"Large prefill batches delay decode steps"},{l:"↓ OOM risk",v:"Scales O(n²) for long contexts"}],
    when: { throughput:"Start at 4096, double to 8192 then 16384.", latency:"Lower (1024–2048) to reduce prefill HOL blocking.", cost:"High (8192+) for batch/offline inference.", comparison:"Fix to same value. Different models have different optima.", memory:"Reduce if hitting OOM during prefill." },
    example: "vllm serve <model> --max-num-batched-tokens 8192",
    recipe: { throughput:"8192–16384", latency:"1024–2048", cost:"16384+", comparison:"fix at 4096" }
  },
  {
    name: "--max-model-len", alias: "", category: "Batching", type: "int", default: "model's max context",
    focus: ["memory","throughput","comparison"], impact: "high",
    summary: "Truncate max context length to free KV cache memory for larger batches.",
    detail: "KV cache size scales linearly with max_model_len. Halving it roughly halves KV memory, allowing twice as many concurrent sequences. Essential when model supports 128K but your workload is 4K.",
    tradeoffs: [{l:"↑ More sequences",v:"Freed KV memory allows higher max_num_seqs"},{l:"↑ Throughput",v:"Bigger batches → better GPU utilization"},{l:"↓ Context length",v:"Long documents / conversations get truncated"}],
    when: { memory:"Set to 2–4× your p95 input length.", throughput:"Reduce if your workload is short-context.", comparison:"Fix to same value across models.", cost:"Shorter context = more requests per GPU.", latency:"Minimal direct impact." },
    example: "vllm serve <model> --max-model-len 4096",
    recipe: { memory:"2× your p95 prompt length", comparison:"fix to 4096 or 8192" }
  },
  {
    name: "--scheduler-delay-factor", alias: "", category: "Batching", type: "float", default: "0.0",
    focus: ["throughput","cost"], impact: "medium",
    summary: "Delay scheduling to allow larger prefill batches to accumulate.",
    detail: "Waits up to D × last_prefill_latency before dispatching. Allows more requests to arrive and be batched together for better GPU utilization. Tradeoff: higher TTFT.",
    tradeoffs: [{l:"↑ Throughput",v:"Larger prefill batches → better efficiency"},{l:"↑ Cost",v:"Higher utilization = more requests per GPU-hour"},{l:"↓ TTFT",v:"Intentional delay before first token"}],
    when: { throughput:"Set 0.3–0.7 for batch/offline workloads.", cost:"Higher delay → denser batches → lower cost.", latency:"Keep at 0.0.", comparison:"Keep at 0.0.", memory:"No impact." },
    example: "vllm serve <model> --scheduler-delay-factor 0.5",
    recipe: { throughput:"0.3–0.7 for offline", latency:"0.0", cost:"0.5+" }
  },
  {
    name: "--gpu-memory-utilization", alias: "", category: "Memory", type: "float (0–1)", default: "0.90",
    focus: ["throughput","memory","cost"], impact: "high",
    summary: "Fraction of GPU VRAM allocated to vLLM (weights + KV cache).",
    detail: "vLLM profiles activation memory at startup then allocates remaining × utilization for KV cache. Higher = more KV blocks = more concurrent sequences. Avoid 0.99+ as activation peaks can cause OOM.",
    tradeoffs: [{l:"↑ More KV cache",v:"Higher max_num_seqs and longer sequences"},{l:"↑ Throughput",v:"More batching → better GPU utilization"},{l:"↓ OOM risk",v:"Activation peaks can exceed budget"}],
    when: { throughput:"Start at 0.90, push to 0.93 if stable, then 0.95.", memory:"0.90–0.95 safe range.", cost:"Higher utilization = lower cost per request.", latency:"Minimal direct impact.", comparison:"Fix to 0.90 for fair comparisons." },
    example: "vllm serve <model> --gpu-memory-utilization 0.93",
    recipe: { throughput:"0.93–0.95", memory:"0.90 safe default", comparison:"fix at 0.90" }
  },
  {
    name: "--enforce-eager", alias: "", category: "Memory", type: "bool", default: "false",
    focus: ["memory","latency"], impact: "high",
    summary: "Disable CUDA graph capture — eager PyTorch mode.",
    detail: "CUDA graphs pre-compile fixed GPU kernel sequences, eliminating CPU kernel launch overhead. Disabling gives 20–40% lower decode throughput but faster startup and less VRAM.",
    tradeoffs: [{l:"true: ↑ Startup speed",v:"No graph capture = instant start"},{l:"true: ↑ Less VRAM",v:"Saves memory used by CUDA graphs"},{l:"true: ↓ Throughput",v:"20–40% lower — CPU kernel launch overhead"}],
    when: { memory:"Enable if running out of VRAM due to graph capture overhead.", latency:"Disable (default) — CUDA graphs reduce decode latency.", throughput:"Keep false. Never enable for production throughput.", cost:"Keep false — lower throughput = higher cost.", comparison:"Fix to false." },
    example: "vllm serve <model> --enforce-eager  # debugging only",
    recipe: { memory:"true only if OOM at startup", latency:"false (default)" }
  },
  {
    name: "--attention-backend", alias: "", category: "Attention", type: "enum",
    options: "FLASH_ATTN | FLASHINFER | TORCH_SDPA | TRITON | XFORMERS", default: "auto",
    focus: ["throughput","latency","comparison"], impact: "high",
    summary: "Attention kernel implementation — major impact on prefill and decode speed.",
    detail: "FLASH_ATTN: Best for most NVIDIA GPUs. Fused attention, O(n) memory, required for RoPE. FLASHINFER: Highly optimized for decode-heavy, low-latency serving. TORCH_SDPA: Universal but slowest.",
    tradeoffs: [{l:"FLASH_ATTN",v:"Best for large prefill batches, high throughput"},{l:"FLASHINFER",v:"Best for decode-heavy, low-latency serving"},{l:"TORCH_SDPA",v:"Universal but slowest. Use for non-NVIDIA."}],
    when: { throughput:"FLASH_ATTN for highest throughput on NVIDIA.", latency:"FLASHINFER often best for per-token latency.", comparison:"Fix to FLASH_ATTN.", memory:"FlashAttention O(n) memory — critical for long context.", cost:"Best kernel = best efficiency." },
    example: "VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve <model>",
    recipe: { throughput:"FLASH_ATTN", latency:"FLASHINFER", memory:"FLASH_ATTN (O(n) memory)" }
  },
  {
    name: "--speculative-model", alias: "", category: "Speculative", type: "string", default: "None",
    focus: ["latency","throughput"], impact: "high",
    summary: "Draft model for speculative decoding — dramatically reduces inter-token latency.",
    detail: "Small draft model predicts N tokens ahead, large model verifies in one pass. If accepted: N+1 tokens per target pass. Best with same-family models (Llama 3 8B drafting for 70B). Use [ngram] for no draft model.",
    tradeoffs: [{l:"↑ Latency",v:"2–3× lower inter-token latency when acceptance >70%"},{l:"↓ Throughput",v:"Lower batch utilization"},{l:"↓ Memory",v:"Requires loading both draft and target model"}],
    when: { latency:"Best tool for inter-token latency. Use same-family draft.", throughput:"Avoid — trades throughput for latency.", memory:"Avoid if VRAM is tight.", cost:"Neutral to negative.", comparison:"Disable." },
    example: "vllm serve meta-llama/Llama-3.1-70B --speculative-model meta-llama/Llama-3.2-1B --num-speculative-tokens 5",
    recipe: { latency:"Same-family small model, 3–5 speculative tokens" }
  },
  {
    name: "--num-speculative-tokens", alias: "", category: "Speculative", type: "int", default: "None",
    focus: ["latency"], impact: "medium",
    summary: "Number of tokens the draft model generates ahead per speculation step.",
    detail: "More tokens = more speedup if accepted, but lower acceptance rate (drops exponentially). Optimal K balances acceptance rate × tokens_per_step. Monitor acceptance_rate metric. If <60%, reduce K. If >85%, increase K.",
    tradeoffs: [{l:"K=3",v:"Safe default. Good acceptance rate."},{l:"K=5",v:"Good for same-family models."},{l:"K=8+",v:"Only if acceptance rate stays >80%."}],
    when: { latency:"Start at 5 with same-family draft. Monitor acceptance rate.", throughput:"N/A.", comparison:"Disable.", memory:"N/A.", cost:"N/A." },
    example: "vllm serve <model> --speculative-model <draft> --num-speculative-tokens 5",
    recipe: { latency:"5 for same-family, 3 for cross-family" }
  },
];

window.FOCUS_META = {
  throughput:  { label:"Throughput",      icon:"⚡", color:"#22d3ee", bg:"#083344",  desc:"Maximize tokens/sec & requests/sec",    recipe: `vllm serve <model> \\\n  --tensor-parallel-size <N> \\\n  --quantization awq \\\n  --enable-prefix-caching \\\n  --max-num-seqs 256 \\\n  --max-num-batched-tokens 8192 \\\n  --gpu-memory-utilization 0.93 \\\n  --kv-cache-dtype fp8_e4m3 \\\n  --dtype bfloat16` },
  latency:     { label:"Low Latency",     icon:"⏱",  color:"#fb923c", bg:"#431407",  desc:"Minimize TTFT & inter-token delay",      recipe: `vllm serve <model> \\\n  --max-num-seqs 32 \\\n  --max-num-batched-tokens 1024 \\\n  --enable-prefix-caching \\\n  --speculative-model <draft> \\\n  --num-speculative-tokens 5 \\\n  --scheduler-delay-factor 0.0` },
  memory:      { label:"Memory",          icon:"🗄",  color:"#a78bfa", bg:"#2e1065",  desc:"Fit larger models, reduce OOM risk",      recipe: `vllm serve <model> \\\n  --tensor-parallel-size <N> \\\n  --quantization awq \\\n  --kv-cache-dtype fp8_e4m3 \\\n  --gpu-memory-utilization 0.92 \\\n  --max-model-len 4096 \\\n  --block-size 8 \\\n  --swap-space 16` },
  cost:        { label:"Cost Efficiency", icon:"💰", color:"#4ade80", bg:"#052e16",  desc:"Best performance per GPU dollar",        recipe: `vllm serve <model> \\\n  --quantization awq \\\n  --enable-prefix-caching \\\n  --max-num-seqs 512 \\\n  --max-num-batched-tokens 16384 \\\n  --scheduler-delay-factor 0.5 \\\n  --gpu-memory-utilization 0.93 \\\n  --load-format npcache` },
  comparison:  { label:"Model Comparison",icon:"⚖", color:"#f472b6", bg:"#500724",  desc:"Fair apples-to-apples benchmarks",        recipe: `vllm serve <model> \\\n  --dtype bfloat16 \\\n  --max-num-seqs 128 \\\n  --max-num-batched-tokens 4096 \\\n  --gpu-memory-utilization 0.90 \\\n  --max-model-len 4096 \\\n  --swap-space 0 \\\n  --scheduler-delay-factor 0.0` },
};

window.IMPACT_META = {
  critical: { color:"#ef4444", bg:"rgba(239,68,68,0.1)",   label:"CRITICAL" },
  high:     { color:"#f59e0b", bg:"rgba(245,158,11,0.1)",  label:"HIGH" },
  medium:   { color:"#38bdf8", bg:"rgba(56,189,248,0.1)",  label:"MEDIUM" },
  low:      { color:"#6b7280", bg:"rgba(107,114,128,0.1)", label:"LOW" },
};

window.CATEGORIES = ["All","Parallelism","Quantization","KV Cache","Batching","Memory","Attention","Speculative"];
