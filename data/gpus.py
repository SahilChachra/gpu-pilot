# ── GPU Database ───────────────────────────────────────────────────────────────
# bw             = memory bandwidth in GB/s
# tflops_bf16    = peak BF16 tensor TFLOPS (dense, no structured sparsity)
# tflops_fp8     = peak FP8 tensor TFLOPS (dense)
# nvlink         = NVLink / NVSwitch interconnect (True = high multi-GPU BW)
# fp8            = hardware FP8 / Transformer Engine support
# arch           = GPU micro-architecture
# mig            = max MIG instance slices (0 = no MIG)
# runpod_hr / lambda_hr / vastai_hr = $/hr per GPU (None = unavailable)
# tier           = flagship / high / mid / budget / legacy

GPUS = {
    # ── Blackwell (NVIDIA 6th-gen datacenter) ────────────────────────────────
    "B200 SXM 192GB": {
        "vram": 192, "bw": 8000, "tflops_bf16": 2250, "tflops_fp8": 4500,
        "nvlink": True,  "fp8": True,  "arch": "Blackwell",            "mig": 7,
        "runpod_hr": None, "lambda_hr": None,  "vastai_hr": None,
        "tier": "flagship",
        "notes": "Latest NVIDIA flagship. 192 GB HBM3e, 8 TB/s BW. NVLink 5.0 (1.8 TB/s). FP4 Transformer Engine. Enterprise-only, limited cloud availability.",
    },
    "B300 SXM 288GB": {
        "vram": 288, "bw": 16000, "tflops_bf16": 4500, "tflops_fp8": 9000,
        "nvlink": True,  "fp8": True,  "arch": "Blackwell",            "mig": 7,
        "runpod_hr": None, "lambda_hr": None,  "vastai_hr": None,
        "tier": "flagship",
        "notes": "Next-gen Blackwell flagship announced GTC 2025. 288 GB HBM3e, 16 TB/s BW. FP4/FP8 Transformer Engine. Double the KV cache capacity of B200. Not yet widely available.",
    },
    "B100 SXM 192GB": {
        "vram": 192, "bw": 8000, "tflops_bf16": 1750, "tflops_fp8": 3500,
        "nvlink": True,  "fp8": True,  "arch": "Blackwell",            "mig": 7,
        "runpod_hr": None, "lambda_hr": None,  "vastai_hr": None,
        "tier": "flagship",
        "notes": "Blackwell mid-tier datacenter GPU. Same 192 GB HBM3e / 8 TB/s BW as B200, ~78% compute. H100 slot-compatible. Limited cloud availability.",
    },

    # ── Hopper (NVIDIA 5th-gen) ──────────────────────────────────────────────
    "H200 SXM 141GB": {
        "vram": 141, "bw": 4800, "tflops_bf16": 1979, "tflops_fp8": 3958,
        "nvlink": True,  "fp8": True,  "arch": "Hopper",               "mig": 7,
        "runpod_hr": 8.99, "lambda_hr": None,  "vastai_hr": 7.50,
        "tier": "flagship",
        "notes": "Best available. 141 GB HBM3e, 4.8 TB/s BW. FP8 Transformer Engine. 2× the KV cache of H100.",
    },
    "H100 SXM 80GB": {
        "vram": 80,  "bw": 3350, "tflops_bf16": 989,  "tflops_fp8": 1979,
        "nvlink": True,  "fp8": True,  "arch": "Hopper",               "mig": 7,
        "runpod_hr": 3.99, "lambda_hr": 2.99,  "vastai_hr": 2.80,
        "tier": "flagship",
        "notes": "NVLink 4.0 (900 GB/s). Premier inference GPU. FP8 = near-fp16 quality at 2× speed.",
    },
    "H100 NVL 94GB": {
        "vram": 94,  "bw": 3938, "tflops_bf16": 989,  "tflops_fp8": 1979,
        "nvlink": True,  "fp8": True,  "arch": "Hopper",               "mig": 7,
        "runpod_hr": 4.49, "lambda_hr": 3.99,  "vastai_hr": 3.00,
        "tier": "flagship",
        "notes": "NVLink variant with extra VRAM (94 GB HBM3). Pairs as 188 GB via NVLink. Same compute as SXM, slightly higher BW. Ideal for large KV caches.",
    },
    "H100 PCIe 80GB": {
        "vram": 80,  "bw": 2000, "tflops_bf16": 756,  "tflops_fp8": 1513,
        "nvlink": False, "fp8": True,  "arch": "Hopper",               "mig": 7,
        "runpod_hr": 2.99, "lambda_hr": 2.49,  "vastai_hr": 2.30,
        "tier": "flagship",
        "notes": "PCIe variant — 40% lower BW than SXM. Still excellent single-GPU option with FP8.",
    },

    # ── Ampere (NVIDIA 3rd-gen datacenter) ───────────────────────────────────
    "A100 SXM 80GB": {
        "vram": 80,  "bw": 2000, "tflops_bf16": 312,  "tflops_fp8": 312,
        "nvlink": True,  "fp8": False, "arch": "Ampere",               "mig": 7,
        "runpod_hr": 2.49, "lambda_hr": 1.99,  "vastai_hr": 1.75,
        "tier": "high",
        "notes": "NVLink 3.0 (600 GB/s). Workhorse for 70B-class models. No FP8 — use AWQ.",
    },
    "A100 PCIe 80GB": {
        "vram": 80,  "bw": 1935, "tflops_bf16": 312,  "tflops_fp8": 312,
        "nvlink": False, "fp8": False, "arch": "Ampere",               "mig": 7,
        "runpod_hr": 1.99, "lambda_hr": 1.79,  "vastai_hr": 1.50,
        "tier": "high",
        "notes": "PCIe variant. Good cost-per-GB option for 70B with AWQ/fp16.",
    },
    "A100 SXM 40GB": {
        "vram": 40,  "bw": 1555, "tflops_bf16": 312,  "tflops_fp8": 312,
        "nvlink": True,  "fp8": False, "arch": "Ampere",               "mig": 7,
        "runpod_hr": 1.64, "lambda_hr": 1.29,  "vastai_hr": 1.20,
        "tier": "high",
        "notes": "40 GB variant. Best for 13B fp16 or 70B AWQ with TP=2. MIG support.",
    },
    "A100 PCIe 40GB": {
        "vram": 40,  "bw": 1555, "tflops_bf16": 312,  "tflops_fp8": 312,
        "nvlink": False, "fp8": False, "arch": "Ampere",               "mig": 7,
        "runpod_hr": 1.49, "lambda_hr": 1.10,  "vastai_hr": 1.00,
        "tier": "high",
        "notes": "Budget A100. Solid for 13-34B models at fp16.",
    },
    "A40": {
        "vram": 48,  "bw": 696,  "tflops_bf16": 149,  "tflops_fp8": 149,
        "nvlink": False, "fp8": False, "arch": "Ampere",               "mig": 0,
        "runpod_hr": 0.79, "lambda_hr": 0.60,  "vastai_hr": 0.50,
        "tier": "mid",
        "notes": "48 GB is sweet spot for many models. Lower BW than A100 — use AWQ to compensate.",
    },
    "A30": {
        "vram": 24,  "bw": 933,  "tflops_bf16": 165,  "tflops_fp8": 165,
        "nvlink": True,  "fp8": False, "arch": "Ampere",               "mig": 4,
        "runpod_hr": 0.58, "lambda_hr": None,   "vastai_hr": 0.40,
        "tier": "mid",
        "notes": "High BW for its price. MIG support (4 slices). Good for 7B models.",
    },
    "A10G": {
        "vram": 24,  "bw": 600,  "tflops_bf16": 125,  "tflops_fp8": 125,
        "nvlink": False, "fp8": False, "arch": "Ampere",               "mig": 4,
        "runpod_hr": 0.44, "lambda_hr": 0.60,  "vastai_hr": 0.30,
        "tier": "mid",
        "notes": "AWS g5 standard. MIG support. Best for 7-13B models at modest cost.",
    },
    "A6000 (Ampere) 48GB": {
        "vram": 48,  "bw": 768,  "tflops_bf16": 159,  "tflops_fp8": 159,
        "nvlink": True,  "fp8": False, "arch": "Ampere",               "mig": 0,
        "runpod_hr": 0.49, "lambda_hr": 0.60,  "vastai_hr": 0.40,
        "tier": "mid",
        "notes": "Pro workstation Ampere (GA102). 48 GB GDDR6, NVLink bridge (2-GPU max). No FP8 — use AWQ. Cheaper alternative to A40 with similar VRAM.",
    },

    # ── Ada Lovelace (NVIDIA 4th-gen) ────────────────────────────────────────
    "L40S": {
        "vram": 48,  "bw": 864,  "tflops_bf16": 362,  "tflops_fp8": 733,
        "nvlink": False, "fp8": True,  "arch": "Ada Lovelace",         "mig": 0,
        "runpod_hr": 1.39, "lambda_hr": 1.19,  "vastai_hr": 1.00,
        "tier": "high",
        "notes": "Best Ada datacenter GPU. FP8 Transformer Engine. Excellent TFLOP/$ ratio.",
    },
    "L40": {
        "vram": 48,  "bw": 864,  "tflops_bf16": 181,  "tflops_fp8": 181,
        "nvlink": False, "fp8": False, "arch": "Ada Lovelace",         "mig": 0,
        "runpod_hr": 1.14, "lambda_hr": None,   "vastai_hr": 0.80,
        "tier": "mid",
        "notes": "Older Ada GPU without FP8 TE. High BW. Good for 34B AWQ.",
    },
    "A6000 Ada": {
        "vram": 48,  "bw": 960,  "tflops_bf16": 366,  "tflops_fp8": 733,
        "nvlink": False, "fp8": True,  "arch": "Ada Lovelace",         "mig": 0,
        "runpod_hr": 1.19, "lambda_hr": None,   "vastai_hr": 0.85,
        "tier": "high",
        "notes": "Pro workstation GPU. FP8 capable, high BW. Good alternative to L40S.",
    },
    "L4": {
        "vram": 24,  "bw": 300,  "tflops_bf16": 121,  "tflops_fp8": 242,
        "nvlink": False, "fp8": True,  "arch": "Ada Lovelace",         "mig": 0,
        "runpod_hr": 0.44, "lambda_hr": 0.38,  "vastai_hr": 0.25,
        "tier": "mid",
        "notes": "GCP g2 standard. Low power draw. FP8 support. Very low BW — latency suffers at 7B+.",
    },
    "RTX 4090": {
        "vram": 24,  "bw": 1008, "tflops_bf16": 330,  "tflops_fp8": 660,
        "nvlink": False, "fp8": True,  "arch": "Ada Lovelace",         "mig": 0,
        "runpod_hr": 0.74, "lambda_hr": None,   "vastai_hr": 0.45,
        "tier": "mid",
        "notes": "Consumer GPU. High peak throughput but throttles under sustained datacenter load.",
    },
    "RTX 4080 Super": {
        "vram": 16,  "bw": 736,  "tflops_bf16": 207,  "tflops_fp8": 414,
        "nvlink": False, "fp8": True,  "arch": "Ada Lovelace",         "mig": 0,
        "runpod_hr": 0.55, "lambda_hr": None,   "vastai_hr": 0.35,
        "tier": "budget",
        "notes": "16 GB limits to 7B fp16 or 13B AWQ. Budget option for small-model serving.",
    },

    # ── Blackwell (consumer) ─────────────────────────────────────────────────
    "RTX 5090": {
        "vram": 32,  "bw": 1792, "tflops_bf16": 838,  "tflops_fp8": 1676,
        "nvlink": False, "fp8": True,  "arch": "Blackwell (consumer)", "mig": 0,
        "runpod_hr": 2.49, "lambda_hr": None,   "vastai_hr": 1.80,
        "tier": "high",
        "notes": "Consumer Blackwell flagship. 32 GB GDDR7, 1.79 TB/s BW. FP8/FP4 Tensor Cores. Thermal throttling under sustained load vs datacenter cards.",
    },
    "RTX 5080": {
        "vram": 16,  "bw": 960,  "tflops_bf16": 440,  "tflops_fp8": 880,
        "nvlink": False, "fp8": True,  "arch": "Blackwell (consumer)", "mig": 0,
        "runpod_hr": 0.99, "lambda_hr": None,   "vastai_hr": 0.70,
        "tier": "mid",
        "notes": "16 GB limits model size. High BW for 7B fp16 or 13B AWQ. FP8 support.",
    },
    "RTX 5070 Ti": {
        "vram": 16,  "bw": 896,  "tflops_bf16": 352,  "tflops_fp8": 704,
        "nvlink": False, "fp8": True,  "arch": "Blackwell (consumer)", "mig": 0,
        "runpod_hr": 0.69, "lambda_hr": None,   "vastai_hr": 0.50,
        "tier": "mid",
        "notes": "16 GB Blackwell mid-range. Similar BW to 5080 at lower cost. FP8 support.",
    },
    "RTX 5070": {
        "vram": 12,  "bw": 672,  "tflops_bf16": 246,  "tflops_fp8": 492,
        "nvlink": False, "fp8": True,  "arch": "Blackwell (consumer)", "mig": 0,
        "runpod_hr": 0.49, "lambda_hr": None,   "vastai_hr": 0.35,
        "tier": "budget",
        "notes": "12 GB GDDR7. Sufficient only for ≤7B AWQ/INT4. Very limited for 7B fp16 (14 GB needed).",
    },
    "RTX Pro 6000 Blackwell": {
        "vram": 96,  "bw": 1792, "tflops_bf16": 890,  "tflops_fp8": 1780,
        "nvlink": False, "fp8": True,  "arch": "Blackwell (pro)",      "mig": 4,
        "runpod_hr": 1.89, "lambda_hr": None,   "vastai_hr": 1.20,
        "tier": "high",
        "notes": "Professional Blackwell workstation GPU. 96 GB GDDR7 ECC, 1.8 TB/s BW. 2nd-gen Transformer Engine (FP8/FP4). 4 MIG instances. No NVLink — PCIe only. Excellent VRAM-per-$ vs datacenter cards.",
    },
    "RTX 5060 Ti": {
        "vram": 16,  "bw": 448,  "tflops_bf16": 184,  "tflops_fp8": 368,
        "nvlink": False, "fp8": True,  "arch": "Blackwell (consumer)", "mig": 0,
        "runpod_hr": None, "lambda_hr": None,   "vastai_hr": 0.20,
        "tier": "budget",
        "notes": "16 GB GDDR7. Entry-level Blackwell. FP8 support. Suits 7B AWQ/INT4 or small 7B fp16. Not yet common in cloud deployments.",
    },

    # ── Previous-Gen / Legacy ────────────────────────────────────────────────
    "RTX 3090": {
        "vram": 24,  "bw": 936,  "tflops_bf16": 71,   "tflops_fp8": 71,
        "nvlink": False, "fp8": False, "arch": "Ampere (consumer)",    "mig": 0,
        "runpod_hr": 0.44, "lambda_hr": None,   "vastai_hr": 0.20,
        "tier": "budget",
        "notes": "Previous gen consumer. No FP8. High BW helps decode. Best for 7B AWQ.",
    },
    "T4": {
        "vram": 16,  "bw": 300,  "tflops_bf16": 65,   "tflops_fp8": 65,
        "nvlink": False, "fp8": False, "arch": "Turing",               "mig": 4,
        "runpod_hr": 0.20, "lambda_hr": 0.49,  "vastai_hr": 0.15,
        "tier": "legacy",
        "notes": "AWS g4dn / GCP n1. Cheapest option. Very slow for anything > 7B AWQ.",
    },
    "V100 SXM2 32GB": {
        "vram": 32,  "bw": 900,  "tflops_bf16": 125,  "tflops_fp8": 125,
        "nvlink": True,  "fp8": False, "arch": "Volta",                "mig": 0,
        "runpod_hr": 0.49, "lambda_hr": 0.80,  "vastai_hr": 0.25,
        "tier": "legacy",
        "notes": "Legacy datacenter GPU. No FP8. Surprisingly good BW. Avoid for new deployments.",
    },
}

# ── Provider name maps ─────────────────────────────────────────────────────────
# Maps raw provider GPU name strings → GPUS dict keys

RUNPOD_NAME_MAP = {
    # Blackwell
    "NVIDIA B300 SXM":             "B300 SXM 288GB",
    "NVIDIA B300 SXM 288GB":       "B300 SXM 288GB",
    "NVIDIA B200 SXM":             "B200 SXM 192GB",
    "NVIDIA B100 SXM":             "B100 SXM 192GB",
    # H200
    "H200 SXM5 141GB":             "H200 SXM 141GB",
    "NVIDIA H200":                 "H200 SXM 141GB",
    "NVIDIA H200 SXM":             "H200 SXM 141GB",
    # H100
    "NVIDIA H100 80GB HBM3":       "H100 SXM 80GB",
    "NVIDIA H100 SXM5 80GB":       "H100 SXM 80GB",
    "H100 SXM5 80GB HBM3":         "H100 SXM 80GB",
    "NVIDIA H100 NVL":             "H100 NVL 94GB",
    "NVIDIA H100 NVL 94GB":        "H100 NVL 94GB",
    "NVIDIA H100 PCIe":            "H100 PCIe 80GB",
    "NVIDIA H100 PCIe 80GB":       "H100 PCIe 80GB",
    # A100
    "NVIDIA A100-SXM4-80GB":       "A100 SXM 80GB",
    "NVIDIA A100 SXM 80GB":        "A100 SXM 80GB",
    "NVIDIA A100 80GB PCIe":       "A100 PCIe 80GB",
    "NVIDIA A100 PCIe 80GB":       "A100 PCIe 80GB",
    "NVIDIA A100-SXM4-40GB":       "A100 SXM 40GB",
    "NVIDIA A100 PCIe 40GB":       "A100 PCIe 40GB",
    # Ada / L-series
    "NVIDIA L40S":                 "L40S",
    "NVIDIA L40":                  "L40",
    "NVIDIA L4":                   "L4",
    "NVIDIA RTX A6000 Ada":        "A6000 Ada",
    "NVIDIA RTX 6000 Ada":         "A6000 Ada",
    # Ampere datacenter
    "NVIDIA A40":                  "A40",
    "NVIDIA A30":                  "A30",
    "NVIDIA A10G":                 "A10G",
    "NVIDIA RTX A6000":            "A6000 (Ampere) 48GB",
    # Consumer Blackwell
    "NVIDIA GeForce RTX 5090":     "RTX 5090",
    "NVIDIA RTX 5090":             "RTX 5090",
    "NVIDIA GeForce RTX 5080":     "RTX 5080",
    "NVIDIA RTX 5080":             "RTX 5080",
    "NVIDIA GeForce RTX 5070 Ti":  "RTX 5070 Ti",
    "NVIDIA GeForce RTX 5070":     "RTX 5070",
    "NVIDIA GeForce RTX 5060 Ti":  "RTX 5060 Ti",
    "NVIDIA RTX 5060 Ti":          "RTX 5060 Ti",
    # Pro Blackwell
    "NVIDIA RTX Pro 6000 Blackwell": "RTX Pro 6000 Blackwell",
    "RTX Pro 6000 Blackwell":        "RTX Pro 6000 Blackwell",
    # Consumer Ada
    "NVIDIA GeForce RTX 4090":     "RTX 4090",
    "NVIDIA RTX 4090":             "RTX 4090",
    "NVIDIA GeForce RTX 4080 SUPER": "RTX 4080 Super",
    "NVIDIA GeForce RTX 3090":     "RTX 3090",
    "NVIDIA RTX 3090":             "RTX 3090",
    # Legacy
    "Tesla T4":                    "T4",
    "NVIDIA Tesla T4":             "T4",
    "NVIDIA V100 SXM2 32GB":       "V100 SXM2 32GB",
}

VASTAI_NAME_MAP = {
    # Blackwell datacenter
    "B300_SXM_288GB":   "B300 SXM 288GB",
    "B200_SXM_192GB":   "B200 SXM 192GB",
    "B100_SXM_192GB":   "B100 SXM 192GB",
    # Hopper
    "H200_SXM5_141GB":  "H200 SXM 141GB",
    "H200_SXM":         "H200 SXM 141GB",
    "H100_SXM5_80GB":   "H100 SXM 80GB",
    "H100_SXM4_80GB":   "H100 SXM 80GB",
    "H100_NVL_94GB":    "H100 NVL 94GB",
    "H100_NVL":         "H100 NVL 94GB",
    "H100_PCIE_80GB":   "H100 PCIe 80GB",
    "H100_PCIE":        "H100 PCIe 80GB",
    # Ampere datacenter
    "A100_SXM4_80GB":   "A100 SXM 80GB",
    "A100_PCIE_80GB":   "A100 PCIe 80GB",
    "A100_SXM4_40GB":   "A100 SXM 40GB",
    "A100_PCIE_40GB":   "A100 PCIe 40GB",
    "A40":              "A40",
    "A30":              "A30",
    "A10G":             "A10G",
    "RTX_A6000":        "A6000 (Ampere) 48GB",
    # Ada
    "L40S":             "L40S",
    "L40":              "L40",
    "L4":               "L4",
    "RTX_A6000_ADA":    "A6000 Ada",
    # Consumer Blackwell
    "RTX_5090":         "RTX 5090",
    "RTX_5080":         "RTX 5080",
    "RTX_5070_TI":      "RTX 5070 Ti",
    "RTX_5070":         "RTX 5070",
    "RTX_5060_TI":      "RTX 5060 Ti",
    "RTX_PRO_6000_BLACKWELL": "RTX Pro 6000 Blackwell",
    # Consumer Ada
    "RTX_4090":         "RTX 4090",
    "RTX_4080_SUPER":   "RTX 4080 Super",
    "RTX_3090":         "RTX 3090",
    # Legacy
    "T4":               "T4",
    "V100_SXM2_32GB":   "V100 SXM2 32GB",
}
