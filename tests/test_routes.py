"""Flask route integration tests."""
import json
import pytest


# ── /api/gpus ─────────────────────────────────────────────────────────────────

class TestApiGpus:
    def test_returns_200(self, client):
        resp = client.get("/api/gpus")
        assert resp.status_code == 200

    def test_returns_json(self, client):
        resp = client.get("/api/gpus")
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_contains_known_gpu(self, client):
        data = client.get("/api/gpus").get_json()
        assert "H100 SXM 80GB" in data

    def test_gpu_has_vram_field(self, client):
        data = client.get("/api/gpus").get_json()
        for gpu in data.values():
            assert "vram" in gpu


# ── /api/models ───────────────────────────────────────────────────────────────

class TestApiModels:
    def test_returns_200(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_returns_json_dict(self, client):
        data = client.get("/api/models").get_json()
        assert isinstance(data, dict)

    def test_contains_llama(self, client):
        data = client.get("/api/models").get_json()
        assert any("llama" in k.lower() for k in data)


# ── /api/calculate ────────────────────────────────────────────────────────────

class TestApiCalculate:
    def _post(self, client, **kwargs):
        payload = {"params_b": 7, "quant": "fp16", "context_len": 4096,
                   "num_layers": 32, "kv_heads": 8, "head_dim": 128,
                   "gpu_vram": 80, "num_gpus": 1, "gpu_util": 0.90, **kwargs}
        return client.post("/api/calculate", json=payload)

    def test_returns_200(self, client):
        assert self._post(client).status_code == 200

    def test_has_required_fields(self, client):
        data = self._post(client).get_json()
        for key in ("model_vram_gb", "kv_per_seq_gb", "max_concurrent_seqs", "fits_on_gpu"):
            assert key in data

    def test_7b_fits_on_80gb(self, client):
        data = self._post(client, params_b=7, gpu_vram=80).get_json()
        assert data["fits_on_gpu"] is True

    def test_405b_fp16_does_not_fit_on_80gb(self, client):
        data = self._post(client, params_b=405, quant="fp16", gpu_vram=80).get_json()
        assert data["fits_on_gpu"] is False

    def test_fp8_kv_cache_smaller(self, client):
        fp16 = self._post(client, kv_dtype="fp16").get_json()["kv_per_seq_gb"]
        fp8  = self._post(client, kv_dtype="fp8").get_json()["kv_per_seq_gb"]
        assert fp8 < fp16

    def test_multi_gpu_increases_vram(self, client):
        single = self._post(client, num_gpus=1).get_json()["total_vram_gb"]
        dual   = self._post(client, num_gpus=2).get_json()["total_vram_gb"]
        assert dual == single * 2


# ── /api/recommend-gpu ────────────────────────────────────────────────────────

class TestApiRecommendGpu:
    PAYLOAD = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "quant": "fp16", "context_len": 4096,
        "target_batch": 32, "target_tps": 50, "num_gpus": 1,
    }

    def test_returns_200(self, client):
        resp = client.post("/api/recommend-gpu", json=self.PAYLOAD)
        assert resp.status_code == 200

    def test_has_recommendations_key(self, client):
        data = client.post("/api/recommend-gpu", json=self.PAYLOAD).get_json()
        assert "recommendations" in data

    def test_recommendations_is_list(self, client):
        data = client.post("/api/recommend-gpu", json=self.PAYLOAD).get_json()
        assert isinstance(data["recommendations"], list)

    def test_at_most_ten_recommendations(self, client):
        data = client.post("/api/recommend-gpu", json=self.PAYLOAD).get_json()
        assert len(data["recommendations"]) <= 10

    def test_custom_model_works(self, client):
        payload = {**self.PAYLOAD, "model_id": "__custom__", "custom_params_b": 7}
        resp = client.post("/api/recommend-gpu", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["recommendations"]) > 0

    def test_moe_flag_set_for_moe_model(self, client):
        payload = {**self.PAYLOAD, "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
        data = client.post("/api/recommend-gpu", json=payload).get_json()
        assert data["is_moe"] is True

    def test_dense_model_not_moe(self, client):
        data = client.post("/api/recommend-gpu", json=self.PAYLOAD).get_json()
        assert data["is_moe"] is False


# ── /api/recommend-config ─────────────────────────────────────────────────────

class TestApiRecommendConfig:
    PAYLOAD = {
        "gpu": "H100 SXM 80GB",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "quant": "fp16", "context_len": 4096, "priority": "balanced",
    }

    def test_returns_200(self, client):
        resp = client.post("/api/recommend-config", json=self.PAYLOAD)
        assert resp.status_code == 200

    def test_unknown_gpu_returns_400(self, client):
        payload = {**self.PAYLOAD, "gpu": "Fake GPU 9000"}
        resp = client.post("/api/recommend-config", json=payload)
        assert resp.status_code == 400

    def test_has_command_field(self, client):
        data = client.post("/api/recommend-config", json=self.PAYLOAD).get_json()
        assert "command" in data
        assert "vllm serve" in data["command"]

    def test_has_tp_size(self, client):
        data = client.post("/api/recommend-config", json=self.PAYLOAD).get_json()
        assert data["tp_size"] >= 1

    @pytest.mark.parametrize("priority", ["throughput", "latency", "cost", "balanced"])
    def test_all_priorities_200(self, client, priority):
        payload = {**self.PAYLOAD, "priority": priority}
        assert client.post("/api/recommend-config", json=payload).status_code == 200

    def test_custom_model_works(self, client):
        payload = {**self.PAYLOAD, "model_id": "__custom__", "custom_params_b": 3}
        resp = client.post("/api/recommend-config", json=payload)
        assert resp.status_code == 200


# ── /api/prices ───────────────────────────────────────────────────────────────

class TestApiPrices:
    def test_returns_200(self, client):
        assert client.get("/api/prices").status_code == 200

    def test_has_prices_key(self, client):
        data = client.get("/api/prices").get_json()
        assert "prices" in data

    def test_prices_is_dict(self, client):
        data = client.get("/api/prices").get_json()
        assert isinstance(data["prices"], dict)

    def test_each_price_entry_has_provider_keys(self, client):
        data = client.get("/api/prices").get_json()
        for gpu, prices in data["prices"].items():
            assert "runpod_hr" in prices
            assert "vastai_hr" in prices
            assert "lambda_hr" in prices

    def test_has_metadata_keys(self, client):
        data = client.get("/api/prices").get_json()
        assert "is_live" in data
        assert "errors" in data


# ── /api/refresh-prices ───────────────────────────────────────────────────────

class TestApiRefreshPrices:
    def test_returns_200(self, client, mocker):
        mocker.patch("pricing.fetch_runpod_prices", return_value=({}, None))
        mocker.patch("pricing.fetch_vastai_prices", return_value=({}, None))
        resp = client.post("/api/refresh-prices")
        assert resp.status_code == 200

    def test_returns_ok_true(self, client, mocker):
        mocker.patch("pricing.fetch_runpod_prices", return_value=({}, None))
        mocker.patch("pricing.fetch_vastai_prices", return_value=({}, None))
        data = client.post("/api/refresh-prices").get_json()
        assert data["ok"] is True


# ── /api/hf-model-info ────────────────────────────────────────────────────────

class TestApiHfModelInfo:
    def test_missing_model_id_returns_400(self, client):
        resp = client.get("/api/hf-model-info")
        assert resp.status_code == 400

    def test_bad_format_returns_400(self, client):
        resp = client.get("/api/hf-model-info?model_id=justname")
        assert resp.status_code == 400

    def test_mocked_success(self, client, requests_mock):
        hf_config = {
            "num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 8, "hidden_size": 4096,
            "intermediate_size": 11008, "vocab_size": 32000,
            "model_type": "llama", "hidden_act": "silu",
            "tie_word_embeddings": False,
        }
        requests_mock.get(
            "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/config.json",
            json=hf_config,
        )
        resp = client.get("/api/hf-model-info?model_id=meta-llama/Llama-3-8B")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "params_b" in data
        assert data["params_b"] > 0
        assert data["layers"] == 32

    def test_mocked_404_returns_404(self, client, requests_mock):
        requests_mock.get(
            "https://huggingface.co/org/nonexistent/resolve/main/config.json",
            status_code=404,
        )
        resp = client.get("/api/hf-model-info?model_id=org/nonexistent")
        assert resp.status_code == 404

    def test_mocked_gated_model_returns_403(self, client, requests_mock):
        requests_mock.get(
            "https://huggingface.co/meta-llama/Llama-3-70B/resolve/main/config.json",
            status_code=401,
        )
        resp = client.get("/api/hf-model-info?model_id=meta-llama/Llama-3-70B")
        assert resp.status_code == 403


# ── /api/hf-model-info — MoE param estimation ─────────────────────────────────

class TestApiHfMoeParams:
    """MoE param counts must account for num_experts × moe_intermediate_size,
       and configs nested inside text_config must be resolved correctly."""

    def _get(self, client, requests_mock, model_id, hf_config):
        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        requests_mock.get(url, json=hf_config)
        return client.get(f"/api/hf-model-info?model_id={model_id}")

    def test_nested_text_config_fields_resolved(self, client, requests_mock):
        """Fields inside text_config are used when absent at top level."""
        cfg = {
            "model_type": "qwen3_5_moe",
            "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "num_hidden_layers": 60,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "hidden_size": 4096,
                "head_dim": 256,
                "hidden_act": "silu",
                "moe_intermediate_size": 1024,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "shared_expert_intermediate_size": 1024,
                "vocab_size": 248320,
            },
        }
        data = self._get(client, requests_mock, "Qwen/Qwen3.5-397B-A17B", cfg).get_json()
        assert data["layers"] == 60
        assert data["kv_heads"] == 2
        assert data["hidden"] == 4096

    def test_moe_params_b_uses_all_experts(self, client, requests_mock):
        """Total params for MoE includes every expert's FFN weights."""
        cfg = {
            "model_type": "qwen3_5_moe",
            "tie_word_embeddings": False,
            "text_config": {
                "num_hidden_layers": 60, "num_attention_heads": 32,
                "num_key_value_heads": 2, "hidden_size": 4096, "head_dim": 256,
                "hidden_act": "silu", "vocab_size": 248320,
                "moe_intermediate_size": 1024, "num_experts": 512,
                "num_experts_per_tok": 10, "shared_expert_intermediate_size": 1024,
            },
        }
        data = self._get(client, requests_mock, "Qwen/Qwen3.5-397B-A17B", cfg).get_json()
        # Should be in the hundreds of billions, not single digits
        assert data["params_b"] > 100, f"Expected >100B but got {data['params_b']}B"
        assert data["is_moe"] is True

    def test_moe_active_params_b_returned(self, client, requests_mock):
        """active_params_b uses only num_experts_per_tok FFNs, much smaller than total."""
        cfg = {
            "model_type": "qwen3_5_moe",
            "tie_word_embeddings": False,
            "text_config": {
                "num_hidden_layers": 60, "num_attention_heads": 32,
                "num_key_value_heads": 2, "hidden_size": 4096, "head_dim": 256,
                "hidden_act": "silu", "vocab_size": 248320,
                "moe_intermediate_size": 1024, "num_experts": 512,
                "num_experts_per_tok": 10, "shared_expert_intermediate_size": 1024,
            },
        }
        data = self._get(client, requests_mock, "Qwen/Qwen3.5-397B-A17B", cfg).get_json()
        assert "active_params_b" in data
        assert data["active_params_b"] < data["params_b"]
        assert data["active_params_b"] < 50  # Active should be ~17B, not hundreds

    def test_moe_num_experts_per_tok_returned(self, client, requests_mock):
        cfg = {
            "model_type": "mixtral",
            "num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 8, "hidden_size": 4096,
            "intermediate_size": 14336, "vocab_size": 32000,
            "hidden_act": "silu", "tie_word_embeddings": False,
            "num_local_experts": 8, "num_experts_per_tok": 2,
        }
        data = self._get(client, requests_mock, "mistralai/Mixtral-8x7B", cfg).get_json()
        assert data["num_experts_per_tok"] == 2
        assert data["is_moe"] is True

    def test_dense_model_active_params_equals_total(self, client, requests_mock):
        """For non-MoE models, active_params_b == params_b."""
        cfg = {
            "num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 8, "hidden_size": 4096,
            "intermediate_size": 11008, "vocab_size": 32000,
            "model_type": "llama", "hidden_act": "silu", "tie_word_embeddings": False,
        }
        data = self._get(client, requests_mock, "org/dense-model", cfg).get_json()
        assert data["active_params_b"] == data["params_b"]

    def test_top_level_fields_override_text_config(self, client, requests_mock):
        """Top-level config keys take priority over text_config keys."""
        cfg = {
            "model_type": "some_vlm",
            "num_hidden_layers": 40,           # top-level — should win
            "tie_word_embeddings": False,
            "text_config": {
                "num_hidden_layers": 99,       # should be ignored
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "hidden_act": "silu",
            },
        }
        data = self._get(client, requests_mock, "org/some-model", cfg).get_json()
        assert data["layers"] == 40


# ── /api/hf-model-info — VLM detection ────────────────────────────────────────

class TestApiHfVlmDetection:
    """Verify that /api/hf-model-info detects VLM architectures and returns vision fields."""

    def _get(self, client, requests_mock, model_id, hf_config):
        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        requests_mock.get(url, json=hf_config)
        return client.get(f"/api/hf-model-info?model_id={model_id}")

    def test_regular_llm_is_vlm_false(self, client, requests_mock):
        cfg = {
            "model_type": "llama", "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "hidden_size": 4096, "intermediate_size": 11008,
            "vocab_size": 32000, "hidden_act": "silu", "tie_word_embeddings": False,
        }
        data = self._get(client, requests_mock, "meta-llama/Llama-3-8B", cfg).get_json()
        assert data["is_vlm"] is False

    def test_qwen2_vl_detected_as_vlm(self, client, requests_mock):
        cfg = {
            "model_type": "qwen2_vl", "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "hidden_size": 3584, "intermediate_size": 18944,
            "vocab_size": 152064, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"depth": 32, "embed_dim": 1280, "patch_size": 14},
            "spatial_merge_size": 2,
        }
        data = self._get(client, requests_mock, "Qwen/Qwen2-VL-7B-Instruct", cfg).get_json()
        assert data["is_vlm"] is True
        assert data["dynamic_res"] is True
        assert data["patch_size"] == 14
        assert data["img_token_merge"] == 2

    def test_mllama_detected_cross_attention(self, client, requests_mock):
        cfg = {
            "model_type": "mllama", "num_hidden_layers": 40,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "hidden_size": 4096, "intermediate_size": 14336,
            "vocab_size": 32000, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"image_size": 560, "patch_size": 14},
        }
        data = self._get(client, requests_mock, "meta-llama/Llama-3.2-11B-Vision-Instruct", cfg).get_json()
        assert data["is_vlm"] is True
        assert data["cross_attention_vision"] is True

    def test_llava_next_tile_based(self, client, requests_mock):
        cfg = {
            "model_type": "llava_next", "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "hidden_size": 4096, "intermediate_size": 11008,
            "vocab_size": 32000, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"image_size": 336, "patch_size": 14},
            "image_grid_pinpoints": [[336, 672], [672, 336], [672, 672]],
        }
        data = self._get(client, requests_mock, "llava-hf/llava-v1.6-mistral-7b-hf", cfg).get_json()
        assert data["is_vlm"] is True
        assert data["tile_based"] is True

    def test_llava_fixed_tokens(self, client, requests_mock):
        cfg = {
            "model_type": "llava", "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "hidden_size": 4096, "intermediate_size": 11008,
            "vocab_size": 32000, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"image_size": 336, "patch_size": 14},
        }
        data = self._get(client, requests_mock, "llava-hf/llava-1.5-7b-hf", cfg).get_json()
        assert data["is_vlm"] is True
        assert data["tile_based"] is False
        assert data["dynamic_res"] is False
        assert data["img_tokens_per_image"] == 576  # (336/14)^2

    def test_pixtral_dynamic_res(self, client, requests_mock):
        cfg = {
            "model_type": "pixtral", "num_hidden_layers": 40,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "hidden_size": 5120, "intermediate_size": 14336,
            "vocab_size": 32000, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"image_patch_size": 16},
        }
        data = self._get(client, requests_mock, "mistralai/Pixtral-12B-2409", cfg).get_json()
        assert data["is_vlm"] is True
        assert data["dynamic_res"] is True
        assert data["patch_size"] == 16

    def test_vlm_response_has_all_vision_fields(self, client, requests_mock):
        cfg = {
            "model_type": "qwen2_vl", "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "hidden_size": 3584, "intermediate_size": 18944,
            "vocab_size": 152064, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"depth": 32, "embed_dim": 1280, "patch_size": 14},
            "spatial_merge_size": 2,
        }
        data = self._get(client, requests_mock, "Qwen/Qwen2-VL-7B-Instruct", cfg).get_json()
        required = ("is_vlm", "vision_encoder", "vision_encoder_gb", "patch_size",
                    "dynamic_res", "tile_based", "cross_attention_vision", "img_token_merge",
                    "img_tokens_per_image", "img_size", "max_tiles")
        for field in required:
            assert field in data, f"Missing VLM field: {field}"

    def test_vision_config_only_triggers_vlm(self, client, requests_mock):
        """A config with only a vision_config key (no VLM model_type) is still detected as VLM."""
        cfg = {
            "model_type": "custom_vlm", "num_hidden_layers": 24,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "hidden_size": 2048, "intermediate_size": 8192,
            "vocab_size": 32000, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"image_size": 224, "patch_size": 16},
        }
        data = self._get(client, requests_mock, "org/some-vlm", cfg).get_json()
        assert data["is_vlm"] is True

    def test_encoder_gb_estimated_for_qwen2_vl(self, client, requests_mock):
        """Encoder VRAM is estimated from vision_config embed_dim."""
        cfg = {
            "model_type": "qwen2_vl", "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "hidden_size": 3584, "intermediate_size": 18944,
            "vocab_size": 152064, "hidden_act": "silu", "tie_word_embeddings": False,
            "vision_config": {"depth": 32, "embed_dim": 1280, "patch_size": 14},
            "spatial_merge_size": 2,
        }
        data = self._get(client, requests_mock, "Qwen/Qwen2-VL-7B-Instruct", cfg).get_json()
        assert data["vision_encoder_gb"] > 0


class TestApiRecommendGpuCustomVlm:
    """Verify that custom VLM params are forwarded through recommend-gpu."""

    PAYLOAD = {
        "model_id": "__custom__",
        "quant": "fp16", "context_len": 4096,
        "target_batch": 4, "num_gpus": 1,
        "custom_params_b": 7,
        "custom_is_vlm": True,
        "custom_vision_encoder_gb": 1.4,
        "custom_dynamic_res": True,
        "custom_patch_size": 14,
        "custom_img_token_merge": 2,
        "num_images": 4, "img_w": 336, "img_h": 336,
    }

    def test_custom_vlm_returns_200(self, client):
        resp = client.post("/api/recommend-gpu", json=self.PAYLOAD)
        assert resp.status_code == 200

    def test_custom_vlm_is_vlm_flag_true(self, client):
        data = client.post("/api/recommend-gpu", json=self.PAYLOAD).get_json()
        assert data["is_vlm"] is True

    def test_custom_vlm_img_tokens_populated(self, client):
        data = client.post("/api/recommend-gpu", json=self.PAYLOAD).get_json()
        assert data["img_tokens_per_image"] is not None
        assert data["img_tokens_per_image"] > 0

    def test_custom_vlm_encoder_vram_populated(self, client):
        data = client.post("/api/recommend-gpu", json=self.PAYLOAD).get_json()
        assert data["encoder_vram_gb"] == pytest.approx(1.4)


# ── / (index page) ────────────────────────────────────────────────────────────

class TestMoeCustomVram:
    """MoE custom models: VRAM must be based on total params, not active params."""

    def test_moe_vram_uses_total_params_not_active(self, client):
        """When custom_total_params_b is provided, model_vram_gb must reflect total params."""
        payload = {
            "model_id": "__custom__",
            "quant": "fp16", "context_len": 4096,
            "target_batch": 1, "num_gpus": 8,
            "custom_params_b": 14.62,          # active params
            "custom_total_params_b": 393.61,   # total params — VRAM should be based on this
        }
        data = client.post("/api/recommend-gpu", json=payload).get_json()
        # model_vram_gb for 393.61B fp16 ≈ 787 GB; for 14.62B ≈ 29 GB
        assert data["model_vram_gb"] > 100, (
            f"model_vram_gb={data['model_vram_gb']} — should use total params (393B), not active (14.62B)"
        )

    def test_moe_without_total_params_falls_back_to_custom_params_b(self, client):
        """Without custom_total_params_b, behavior is unchanged (backward compat)."""
        payload = {
            "model_id": "__custom__",
            "quant": "fp16", "context_len": 4096,
            "target_batch": 1, "num_gpus": 1,
            "custom_params_b": 7,
        }
        data = client.post("/api/recommend-gpu", json=payload).get_json()
        assert data["model_vram_gb"] < 50


class TestIndexRoute:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_returns_html(self, client):
        resp = client.get("/")
        assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data
