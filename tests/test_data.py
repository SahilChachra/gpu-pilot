"""Data integrity tests — verify GPU and model databases are well-formed."""
import pytest

from data.gpus   import GPUS, RUNPOD_NAME_MAP, VASTAI_NAME_MAP
from data.models import MODELS, QUANT_BITS, QUANT_OVERHEAD, QUANT_USES_FP8_ENGINE


# ── GPUS database ──────────────────────────────────────────────────────────────

GPU_REQUIRED_KEYS = {
    "vram", "bw", "tflops_bf16", "tflops_fp8",
    "nvlink", "fp8", "arch", "mig",
    "runpod_hr", "lambda_hr", "vastai_hr",
    "tier", "notes",
}
VALID_TIERS = {"flagship", "high", "mid", "budget", "legacy"}


class TestGpuDatabase:
    def test_not_empty(self):
        assert len(GPUS) > 0

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_required_keys_present(self, gpu_name):
        missing = GPU_REQUIRED_KEYS - GPUS[gpu_name].keys()
        assert not missing, f"{gpu_name} is missing keys: {missing}"

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_vram_positive(self, gpu_name):
        assert GPUS[gpu_name]["vram"] > 0

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_bandwidth_positive(self, gpu_name):
        assert GPUS[gpu_name]["bw"] > 0

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_tflops_positive(self, gpu_name):
        assert GPUS[gpu_name]["tflops_bf16"] > 0
        assert GPUS[gpu_name]["tflops_fp8"] > 0

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_fp8_tflops_gte_bf16(self, gpu_name):
        assert GPUS[gpu_name]["tflops_fp8"] >= GPUS[gpu_name]["tflops_bf16"]

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_valid_tier(self, gpu_name):
        assert GPUS[gpu_name]["tier"] in VALID_TIERS, f"{gpu_name} has invalid tier"

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_boolean_flags(self, gpu_name):
        assert isinstance(GPUS[gpu_name]["nvlink"], bool)
        assert isinstance(GPUS[gpu_name]["fp8"], bool)

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_mig_non_negative(self, gpu_name):
        assert GPUS[gpu_name]["mig"] >= 0

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_prices_none_or_positive(self, gpu_name):
        g = GPUS[gpu_name]
        for field in ("runpod_hr", "lambda_hr", "vastai_hr"):
            price = g[field]
            assert price is None or price > 0, f"{gpu_name}.{field} = {price}"

    @pytest.mark.parametrize("gpu_name", list(GPUS.keys()))
    def test_notes_non_empty(self, gpu_name):
        assert len(GPUS[gpu_name]["notes"]) > 0

    def test_known_gpus_present(self):
        expected = {"H100 SXM 80GB", "A100 SXM 80GB", "RTX 4090", "T4"}
        assert expected.issubset(GPUS.keys())

    def test_new_blackwell_gpus_present(self):
        assert "RTX 5090" in GPUS
        assert "B200 SXM 192GB" in GPUS

    def test_b300_datacenter_present(self):
        assert "B300 SXM 288GB" in GPUS

    def test_rtx_5060_ti_present(self):
        assert "RTX 5060 Ti" in GPUS

    def test_rtx_pro_6000_blackwell_present(self):
        assert "RTX Pro 6000 Blackwell" in GPUS

    def test_no_duplicate_keys(self):
        # Python dicts can't have dupes, but check keys are unique strings
        assert len(GPUS) == len(set(GPUS.keys()))


class TestNameMaps:
    def test_runpod_map_values_are_valid_gpu_names(self):
        invalid = {v for v in RUNPOD_NAME_MAP.values() if v not in GPUS}
        assert not invalid, f"RunPod map points to unknown GPUs: {invalid}"

    def test_vastai_map_values_are_valid_gpu_names(self):
        invalid = {v for v in VASTAI_NAME_MAP.values() if v not in GPUS}
        assert not invalid, f"Vast.ai map points to unknown GPUs: {invalid}"

    def test_runpod_map_not_empty(self):
        assert len(RUNPOD_NAME_MAP) > 0

    def test_vastai_map_not_empty(self):
        assert len(VASTAI_NAME_MAP) > 0


# ── MODELS database ────────────────────────────────────────────────────────────

MODEL_REQUIRED_KEYS = {
    "params_b", "active_params_b", "family", "arch",
    "layers", "heads", "kv_heads", "head_dim", "hidden",
    "context", "use_cases", "license", "notes",
}
VALID_ARCHES   = {"decoder", "moe"}
VALID_LICENSES = {"llama3", "llama2", "apache2", "mistral", "gemma", "commercial", "mit", "unknown"}


class TestModelDatabase:
    def test_not_empty(self):
        assert len(MODELS) > 0

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_required_keys_present(self, model_id):
        missing = MODEL_REQUIRED_KEYS - MODELS[model_id].keys()
        assert not missing, f"{model_id} is missing keys: {missing}"

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_params_b_positive(self, model_id):
        assert MODELS[model_id]["params_b"] > 0

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_active_params_lte_total(self, model_id):
        m = MODELS[model_id]
        assert m["active_params_b"] <= m["params_b"]

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_valid_arch(self, model_id):
        assert MODELS[model_id]["arch"] in VALID_ARCHES

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_valid_license(self, model_id):
        assert MODELS[model_id]["license"] in VALID_LICENSES

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_architecture_positive(self, model_id):
        m = MODELS[model_id]
        assert m["layers"] > 0
        assert m["heads"] > 0
        assert m["kv_heads"] > 0
        assert m["head_dim"] > 0
        assert m["hidden"] > 0

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_kv_heads_lte_heads(self, model_id):
        m = MODELS[model_id]
        assert m["kv_heads"] <= m["heads"]

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_context_positive(self, model_id):
        assert MODELS[model_id]["context"] > 0

    @pytest.mark.parametrize("model_id", list(MODELS.keys()))
    def test_use_cases_is_list(self, model_id):
        assert isinstance(MODELS[model_id]["use_cases"], list)

    def test_known_models_present(self):
        expected = {
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }
        assert expected.issubset(MODELS.keys())

    def test_deepseek_r1_present(self):
        assert "deepseek-ai/DeepSeek-R1" in MODELS

    def test_moe_models_have_lower_active_params(self):
        moe_models = {k: v for k, v in MODELS.items() if v["arch"] == "moe"}
        assert len(moe_models) > 0
        for mid, m in moe_models.items():
            assert m["active_params_b"] < m["params_b"], f"{mid}: MoE active_params_b should be < params_b"


# ── Quant tables ───────────────────────────────────────────────────────────────

class TestQuantTables:
    def test_quant_bits_keys_match_overhead(self):
        assert set(QUANT_BITS.keys()) == set(QUANT_OVERHEAD.keys())

    def test_fp16_overhead_is_one(self):
        assert QUANT_OVERHEAD["fp16"] == 1.0

    def test_awq_less_than_int8(self):
        assert QUANT_OVERHEAD["awq"] < QUANT_OVERHEAD["int8"]

    def test_int8_less_than_fp16(self):
        assert QUANT_OVERHEAD["int8"] < QUANT_OVERHEAD["fp16"]

    def test_fp8_engine_is_subset_of_quant_bits(self):
        assert QUANT_USES_FP8_ENGINE.issubset(set(QUANT_BITS.keys()))
