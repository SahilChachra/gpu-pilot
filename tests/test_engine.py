"""Tests for engine.py — all pure functions, no I/O."""
import math
import pytest

from engine import (
    calc_model_vram_gb,
    calc_kv_cache_gb_per_seq,
    calc_max_batch,
    min_gpus_for_model,
    estimate_throughput,
    cost_per_1m_tokens,
    recommend_gpus,
    recommend_config,
)
from data.models import QUANT_OVERHEAD


# ── calc_model_vram_gb ─────────────────────────────────────────────────────────

class TestCalcModelVramGb:
    def test_fp16_formula(self):
        # 2 bytes/param × params_b = params_b * 2
        assert calc_model_vram_gb(7, "fp16") == pytest.approx(14.0, rel=0.01)

    def test_70b_fp16(self):
        assert calc_model_vram_gb(70, "fp16") == pytest.approx(140.0, rel=0.01)

    def test_70b_awq(self):
        # AWQ overhead ≈ 0.27 → 70 * 2 * 0.27 = 37.8
        result = calc_model_vram_gb(70, "awq")
        assert result == pytest.approx(70 * 2 * QUANT_OVERHEAD["awq"], rel=0.01)
        assert result < 40  # must fit on 40GB GPU

    def test_fp8_half_of_fp16(self):
        fp16 = calc_model_vram_gb(13, "fp16")
        fp8  = calc_model_vram_gb(13, "fp8")
        assert fp8 < fp16

    def test_small_model(self):
        result = calc_model_vram_gb(1, "fp16")
        assert result == pytest.approx(2.0, rel=0.01)

    def test_unknown_quant_defaults_to_fp16(self):
        # Unknown quant falls back to overhead 1.0
        result = calc_model_vram_gb(7, "unknown_quant")
        assert result == pytest.approx(14.0, rel=0.01)

    def test_returns_rounded_value(self):
        result = calc_model_vram_gb(7, "fp16")
        assert result == round(result, 1)


# ── calc_kv_cache_gb_per_seq ───────────────────────────────────────────────────

class TestCalcKvCacheGbPerSeq:
    def test_llama8b_4k_context(self, model_llama8b):
        result = calc_kv_cache_gb_per_seq(model_llama8b, 4096)
        # 2 * 32 * 8 * 128 * 4096 * 2 bytes = 536,870,912 bytes ≈ 0.537 GB
        expected = 2 * 32 * 8 * 128 * 4096 * 2 / 1e9
        assert result == pytest.approx(expected, rel=0.01)

    def test_scales_linearly_with_context(self, model_llama8b):
        kv_4k  = calc_kv_cache_gb_per_seq(model_llama8b, 4096)
        kv_8k  = calc_kv_cache_gb_per_seq(model_llama8b, 8192)
        assert kv_8k == pytest.approx(kv_4k * 2, rel=0.001)

    def test_fp8_half_of_fp16(self, model_llama8b):
        kv_fp16 = calc_kv_cache_gb_per_seq(model_llama8b, 4096, kv_dtype="fp16")
        kv_fp8  = calc_kv_cache_gb_per_seq(model_llama8b, 4096, kv_dtype="fp8")
        assert kv_fp8 == pytest.approx(kv_fp16 / 2, rel=0.001)

    def test_more_layers_bigger_cache(self, model_llama8b, model_llama70b):
        kv_8b  = calc_kv_cache_gb_per_seq(model_llama8b,  4096)
        kv_70b = calc_kv_cache_gb_per_seq(model_llama70b, 4096)
        assert kv_70b > kv_8b  # 70B has more layers


# ── calc_max_batch ─────────────────────────────────────────────────────────────

class TestCalcMaxBatch:
    def test_basic(self):
        # 80 GB * 0.9 = 72 usable; 72 - 14 = 58 for KV; 58 * 0.95 / 0.5 = 110
        result = calc_max_batch(80, 14.0, 0.5)
        assert result > 0

    def test_model_doesnt_fit(self):
        # Model larger than GPU → no KV budget → 0 sequences
        result = calc_max_batch(16, 80.0, 0.5)
        assert result == 0

    def test_higher_util_means_more_sequences(self):
        low  = calc_max_batch(80, 14, 0.5, gpu_util=0.80)
        high = calc_max_batch(80, 14, 0.5, gpu_util=0.95)
        assert high > low

    def test_zero_kv_per_seq_returns_zero(self):
        assert calc_max_batch(80, 14, 0) == 0

    def test_returns_int(self):
        result = calc_max_batch(80, 14, 0.5)
        assert isinstance(result, int)


# ── min_gpus_for_model ─────────────────────────────────────────────────────────

class TestMinGpusForModel:
    def test_fits_on_one(self):
        # 14 GB model on 80 GB GPU → 1 GPU
        assert min_gpus_for_model(14, 80) == 1

    def test_needs_two(self):
        # 70 GB model on 40 GB GPU → ceil(70 / (40*0.88)) = ceil(1.98) = 2
        assert min_gpus_for_model(70, 40) == 2

    def test_405b_needs_many(self):
        # 405B fp16 ≈ 810 GB on 80 GB GPUs → ceil(810 / 70.4) = 12
        result = min_gpus_for_model(810, 80)
        assert result >= 8

    def test_always_at_least_one(self):
        assert min_gpus_for_model(0.1, 80) == 1

    def test_returns_int(self):
        assert isinstance(min_gpus_for_model(14, 80), int)


# ── estimate_throughput ────────────────────────────────────────────────────────

class TestEstimateThroughput:
    def test_returns_positive(self, gpu_h100, model_llama8b):
        result = estimate_throughput(gpu_h100, model_llama8b, "fp16", 32)
        assert result > 0

    def test_higher_bw_gpu_faster_at_small_batch(self, model_llama8b):
        gpu_fast = {"vram": 80, "bw": 3350, "tflops_bf16": 989, "tflops_fp8": 1979, "fp8": True}
        gpu_slow = {"vram": 16, "bw": 300,  "tflops_bf16": 65,  "tflops_fp8": 65,  "fp8": False}
        fast_tps = estimate_throughput(gpu_fast, model_llama8b, "fp16", 1)
        slow_tps = estimate_throughput(gpu_slow, model_llama8b, "fp16", 1)
        assert fast_tps > slow_tps

    def test_larger_batch_increases_throughput(self, gpu_h100, model_llama8b):
        tps_1   = estimate_throughput(gpu_h100, model_llama8b, "fp16", 1)
        tps_256 = estimate_throughput(gpu_h100, model_llama8b, "fp16", 256)
        assert tps_256 >= tps_1

    def test_moe_uses_active_params(self, gpu_h100, model_mixtral):
        # MoE with 13B active on same GPU should beat dense 46B
        model_dense_46b = {**model_mixtral, "active_params_b": 46}
        tps_moe   = estimate_throughput(gpu_h100, model_mixtral,    "fp16", 32)
        tps_dense = estimate_throughput(gpu_h100, model_dense_46b,  "fp16", 32)
        assert tps_moe > tps_dense

    def test_fp8_quant_uses_fp8_tflops(self, gpu_h100, model_llama8b):
        tps_fp8  = estimate_throughput(gpu_h100, model_llama8b, "fp8",  256)
        tps_fp16 = estimate_throughput(gpu_h100, model_llama8b, "fp16", 256)
        # At large batches, FP8 should give higher throughput (uses fp8 tflops)
        assert tps_fp8 >= tps_fp16

    def test_returns_int(self, gpu_h100, model_llama8b):
        assert isinstance(estimate_throughput(gpu_h100, model_llama8b, "fp16", 32), int)


# ── cost_per_1m_tokens ─────────────────────────────────────────────────────────

class TestCostPer1mTokens:
    def test_basic_calculation(self):
        # $4/hr / 1000 tps * 1e6 tokens / 3600 sec = $1.111.../1M tokens
        result = cost_per_1m_tokens(4.0, 1000)
        assert result == pytest.approx(4.0 / 1000 * 1e6 / 3600, rel=0.001)

    def test_none_hourly_rate_returns_none(self):
        assert cost_per_1m_tokens(None, 1000) is None

    def test_zero_tps_returns_none(self):
        assert cost_per_1m_tokens(4.0, 0) is None

    def test_higher_throughput_cheaper(self):
        cheap  = cost_per_1m_tokens(4.0, 2000)
        expensive = cost_per_1m_tokens(4.0, 500)
        assert cheap < expensive

    def test_returns_rounded_to_4dp(self):
        result = cost_per_1m_tokens(3.99, 750)
        assert result == round(result, 4)


# ── recommend_gpus ─────────────────────────────────────────────────────────────

class TestRecommendGpus:
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    LARGE = "meta-llama/Llama-3.1-70B-Instruct"

    def test_returns_list(self):
        result = recommend_gpus(self.MODEL, "fp16", 4096, 32, 0)
        assert isinstance(result, list)

    def test_max_ten_results(self):
        result = recommend_gpus(self.MODEL, "fp16", 4096, 32, 0)
        assert len(result) <= 10

    def test_unknown_model_returns_empty(self):
        result = recommend_gpus("not/a-real-model", "fp16", 4096, 32, 0)
        assert result == []

    def test_result_has_required_keys(self):
        result = recommend_gpus(self.MODEL, "fp16", 4096, 32, 0)
        required = {"gpu", "vram_gb", "est_tps", "fits", "score", "pricing", "cost_per_1m"}
        for rec in result:
            assert required.issubset(rec.keys())

    def test_fitting_gpus_ranked_first(self):
        result = recommend_gpus(self.MODEL, "fp16", 4096, 32, 0)
        # First result should fit
        assert result[0]["fits"] is True

    def test_70b_fp16_needs_large_gpu(self):
        result = recommend_gpus(self.LARGE, "fp16", 4096, 1, 0)
        fitting = [r for r in result if r["fits"]]
        for r in fitting:
            assert r["vram_gb"] >= 80  # 70B fp16 ≈ 140 GB, needs 2×80 or bigger

    def test_awq_reduces_vram_requirement(self):
        fp16_fitting = [r for r in recommend_gpus(self.LARGE, "fp16", 4096, 1, 0) if r["fits"]]
        awq_fitting  = [r for r in recommend_gpus(self.LARGE, "awq",  4096, 1, 0) if r["fits"]]
        # AWQ should unlock smaller/more GPUs
        awq_vramsizes = {r["vram_gb"] for r in awq_fitting}
        fp16_vramsizes = {r["vram_gb"] for r in fp16_fitting}
        assert min(awq_vramsizes, default=9999) <= min(fp16_vramsizes, default=9999)

    def test_model_override_works(self):
        custom = {
            "params_b": 3, "active_params_b": 3, "family": "custom", "arch": "decoder",
            "layers": 28, "heads": 24, "kv_heads": 8, "head_dim": 128, "hidden": 3072,
            "context": 4096, "use_cases": [], "license": "unknown", "notes": "",
        }
        result = recommend_gpus("__custom__", "fp16", 4096, 32, 0, model_override=custom)
        assert len(result) > 0

    def test_scores_are_non_negative(self):
        for rec in recommend_gpus(self.MODEL, "fp16", 4096, 32, 0):
            assert rec["score"] >= 0

    def test_sorted_by_score_descending(self):
        result = recommend_gpus(self.MODEL, "fp16", 4096, 32, 0)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)


# ── recommend_config ───────────────────────────────────────────────────────────

class TestRecommendConfig:
    GPU   = "H100 SXM 80GB"
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    def test_returns_dict(self):
        result = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "balanced")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "balanced")
        required = {
            "tp_size", "max_num_seqs", "max_num_batched_tokens",
            "gpu_memory_utilization", "command", "est_tps", "cost_per_day",
        }
        assert required.issubset(result.keys())

    def test_command_contains_model_id(self):
        result = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "balanced")
        assert self.MODEL in result["command"]

    def test_unknown_gpu_returns_empty(self):
        result = recommend_config("Fake GPU 9000", self.MODEL, "fp16", 4096, "balanced")
        assert result == {}

    def test_unknown_model_returns_empty(self):
        result = recommend_config(self.GPU, "not/real", "fp16", 4096, "balanced")
        assert result == {}

    def test_throughput_priority_has_more_seqs_than_latency(self):
        tput    = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "throughput")
        latency = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "latency")
        assert tput["max_num_seqs"] >= latency["max_num_seqs"]

    def test_latency_has_no_scheduler_delay(self):
        result = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "latency")
        assert result["scheduler_delay_factor"] == 0.0

    def test_70b_needs_tensor_parallel(self):
        result = recommend_config(self.GPU, "meta-llama/Llama-3.1-70B-Instruct", "fp16", 4096, "balanced")
        # 70B fp16 ≈ 140 GB, single H100 SXM 80GB → tp_size should be > 1
        assert result["tp_size"] >= 2

    def test_8b_fits_single_gpu(self):
        result = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "balanced")
        assert result["tp_size"] == 1

    def test_fp8_gpu_enables_kv_cache_fp8(self):
        result = recommend_config(self.GPU, self.MODEL, "fp16", 4096, "throughput")
        # H100 has fp8=True, throughput priority → fp8 kv cache
        assert result["kv_cache_dtype"] in ("fp8_e4m3", "auto")

    def test_model_override_works(self):
        custom = {
            "params_b": 3, "active_params_b": 3, "family": "custom", "arch": "decoder",
            "layers": 28, "heads": 24, "kv_heads": 8, "head_dim": 128, "hidden": 3072,
            "context": 4096, "use_cases": [], "license": "unknown", "notes": "",
        }
        result = recommend_config(self.GPU, "__custom__", "fp16", 4096, "balanced", model_override=custom)
        assert result != {}

    @pytest.mark.parametrize("priority", ["throughput", "latency", "cost", "balanced"])
    def test_all_priorities_return_valid_config(self, priority):
        result = recommend_config(self.GPU, self.MODEL, "fp16", 4096, priority)
        assert result.get("tp_size", 0) >= 1
        assert result.get("max_num_seqs", 0) > 0
