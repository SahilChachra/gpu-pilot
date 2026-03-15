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


# ── / (index page) ────────────────────────────────────────────────────────────

class TestIndexRoute:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_returns_html(self, client):
        resp = client.get("/")
        assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data
