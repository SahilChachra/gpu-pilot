"""Tests for pricing.py — name matching and price fetch logic (HTTP mocked)."""
import time
import pytest

from pricing import _match_gpu_name, fetch_runpod_prices, fetch_vastai_prices, refresh_prices, PRICE_CACHE
from data.gpus import RUNPOD_NAME_MAP, VASTAI_NAME_MAP


# ── _match_gpu_name ────────────────────────────────────────────────────────────

class TestMatchGpuName:
    def test_exact_match(self):
        name_map = {"NVIDIA H100 80GB HBM3": "H100 SXM 80GB"}
        assert _match_gpu_name("NVIDIA H100 80GB HBM3", name_map) == "H100 SXM 80GB"

    def test_case_insensitive_normalised_match(self):
        name_map = {"NVIDIA_H100_80GB": "H100 SXM 80GB"}
        assert _match_gpu_name("nvidia h100 80gb", name_map) == "H100 SXM 80GB"

    def test_no_match_returns_none(self):
        name_map = {"NVIDIA H100 80GB HBM3": "H100 SXM 80GB"}
        assert _match_gpu_name("TPU v4", name_map) is None

    def test_exact_takes_precedence_over_substring(self):
        name_map = {
            "RTX_4090":        "RTX 4090",
            "RTX_4090_SUPER":  "RTX 4090 Super",
        }
        result = _match_gpu_name("RTX_4090", name_map)
        assert result == "RTX 4090"

    def test_empty_string_returns_none(self):
        assert _match_gpu_name("", RUNPOD_NAME_MAP) is None

    def test_runpod_h100_sxm_maps_correctly(self):
        result = _match_gpu_name("NVIDIA H100 80GB HBM3", RUNPOD_NAME_MAP)
        assert result == "H100 SXM 80GB"

    def test_vastai_rtx_4090_maps_correctly(self):
        result = _match_gpu_name("RTX_4090", VASTAI_NAME_MAP)
        assert result == "RTX 4090"


# ── fetch_runpod_prices ────────────────────────────────────────────────────────

class TestFetchRunpodPrices:
    def test_http_error_returns_empty_dict_with_error(self, requests_mock):
        requests_mock.post("https://api.runpod.io/graphql", status_code=500)
        prices, err = fetch_runpod_prices()
        assert prices == {}
        assert err is not None
        assert "500" in err

    def test_success_returns_prices(self, requests_mock):
        requests_mock.post(
            "https://api.runpod.io/graphql",
            json={
                "data": {
                    "gpuTypes": [
                        {"id": "NVIDIA H100 80GB HBM3", "displayName": "H100 SXM", "memoryInGb": 80, "securePrice": "3.99", "communityPrice": "2.99"},
                        {"id": "NVIDIA RTX 4090",        "displayName": "RTX 4090", "memoryInGb": 24, "securePrice": "0.74", "communityPrice": "0.50"},
                    ]
                }
            },
        )
        prices, err = fetch_runpod_prices()
        assert err is None
        assert len(prices) > 0

    def test_zero_price_ignored(self, requests_mock):
        requests_mock.post(
            "https://api.runpod.io/graphql",
            json={"data": {"gpuTypes": [
                {"id": "NVIDIA H100 80GB HBM3", "displayName": "H100", "memoryInGb": 80, "securePrice": "0", "communityPrice": "0"},
            ]}},
        )
        prices, err = fetch_runpod_prices()
        assert prices == {}

    def test_connection_error_returns_error_string(self, requests_mock):
        import requests
        requests_mock.post("https://api.runpod.io/graphql", exc=requests.exceptions.ConnectionError)
        prices, err = fetch_runpod_prices()
        assert prices == {}
        assert err is not None

    def test_returns_tuple(self, requests_mock):
        requests_mock.post("https://api.runpod.io/graphql", status_code=200, json={"data": {"gpuTypes": []}})
        result = fetch_runpod_prices()
        assert isinstance(result, tuple)
        assert len(result) == 2


# ── fetch_vastai_prices ────────────────────────────────────────────────────────

class TestFetchVastaiPrices:
    VASTAI_URL = "https://console.vast.ai/api/v0/bundles/"

    def test_http_error_returns_empty_dict_with_error(self, requests_mock):
        requests_mock.get(self.VASTAI_URL, status_code=503)
        prices, err = fetch_vastai_prices()
        assert prices == {}
        assert err is not None

    def test_success_returns_prices(self, requests_mock):
        requests_mock.get(
            self.VASTAI_URL,
            json={"offers": [
                {"gpu_name": "RTX_4090", "dph_total": 0.45},
                {"gpu_name": "H100_SXM5_80GB", "dph_total": 2.80},
            ]},
        )
        prices, err = fetch_vastai_prices()
        assert err is None

    def test_keeps_minimum_price_per_gpu(self, requests_mock):
        requests_mock.get(
            self.VASTAI_URL,
            json={"offers": [
                {"gpu_name": "RTX_4090", "dph_total": 0.80},
                {"gpu_name": "RTX_4090", "dph_total": 0.45},  # cheaper → should win
                {"gpu_name": "RTX_4090", "dph_total": 1.20},
            ]},
        )
        prices, _ = fetch_vastai_prices()
        if "RTX 4090" in prices:
            assert prices["RTX 4090"] == pytest.approx(0.45, rel=0.001)

    def test_connection_error_returns_error_string(self, requests_mock):
        import requests
        requests_mock.get(self.VASTAI_URL, exc=requests.exceptions.Timeout)
        prices, err = fetch_vastai_prices()
        assert prices == {}
        assert err is not None

    def test_returns_tuple(self, requests_mock):
        requests_mock.get(self.VASTAI_URL, json={"offers": []})
        result = fetch_vastai_prices()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_uses_eq_format_for_rentable(self, requests_mock):
        """Vast.ai API requires rentable:{eq:true} not rentable:true (400 otherwise)."""
        import json as _json
        requests_mock.get(self.VASTAI_URL, json={"offers": []})
        fetch_vastai_prices()
        assert requests_mock.called
        qs = requests_mock.last_request.qs
        q_str = qs.get("q", [""])[0]
        q = _json.loads(q_str)
        assert isinstance(q.get("rentable"), dict), (
            "rentable should be a dict like {eq: true}, not a plain bool"
        )


# ── refresh_prices ─────────────────────────────────────────────────────────────

class TestRefreshPrices:
    def test_updates_price_cache_last_updated(self, mocker):
        mocker.patch("pricing.fetch_runpod_prices", return_value=({}, None))
        mocker.patch("pricing.fetch_vastai_prices", return_value=({}, None))
        before = PRICE_CACHE.get("last_updated")
        refresh_prices()
        assert PRICE_CACHE["last_updated"] is not None
        assert PRICE_CACHE["last_updated"] >= (before or 0)

    def test_errors_recorded_on_failure(self, mocker):
        mocker.patch("pricing.fetch_runpod_prices", return_value=({}, "RunPod down"))
        mocker.patch("pricing.fetch_vastai_prices", return_value=({}, "Vast.ai timeout"))
        refresh_prices()
        assert len(PRICE_CACHE["errors"]) > 0

    def test_no_errors_on_success(self, mocker):
        mocker.patch("pricing.fetch_runpod_prices", return_value=({"H100 SXM 80GB": 3.50}, None))
        mocker.patch("pricing.fetch_vastai_prices", return_value=({"RTX 4090": 0.45}, None))
        refresh_prices()
        assert PRICE_CACHE["errors"] == []

    def test_hit_counts_updated(self, mocker):
        mocker.patch("pricing.fetch_runpod_prices", return_value=({"H100 SXM 80GB": 3.50, "RTX 4090": 0.74}, None))
        mocker.patch("pricing.fetch_vastai_prices", return_value=({"L40S": 1.00}, None))
        refresh_prices()
        assert PRICE_CACHE["runpod_hits"] == 2
        assert PRICE_CACHE["vastai_hits"] == 1

    def test_gpus_dict_patched_in_place(self, mocker):
        from data.gpus import GPUS
        mocker.patch("pricing.fetch_runpod_prices", return_value=({"H100 SXM 80GB": 1.23}, None))
        mocker.patch("pricing.fetch_vastai_prices", return_value=({}, None))
        refresh_prices()
        assert GPUS["H100 SXM 80GB"]["runpod_hr"] == pytest.approx(1.23)
