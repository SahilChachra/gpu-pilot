"""Shared pytest fixtures."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ── Minimal GPU / model dicts reused across tests ─────────────────────────────

@pytest.fixture
def gpu_h100():
    return {
        "vram": 80, "bw": 3350, "tflops_bf16": 989, "tflops_fp8": 1979,
        "nvlink": True, "fp8": True, "arch": "Hopper", "mig": 7,
        "runpod_hr": 3.99, "lambda_hr": 2.99, "vastai_hr": 2.80,
        "tier": "flagship",
        "notes": "Test GPU fixture",
    }


@pytest.fixture
def gpu_t4():
    return {
        "vram": 16, "bw": 300, "tflops_bf16": 65, "tflops_fp8": 65,
        "nvlink": False, "fp8": False, "arch": "Turing", "mig": 4,
        "runpod_hr": 0.20, "lambda_hr": 0.49, "vastai_hr": 0.15,
        "tier": "legacy",
        "notes": "Test GPU fixture (legacy)",
    }


@pytest.fixture
def model_llama8b():
    return {
        "params_b": 8, "active_params_b": 8, "family": "llama3", "arch": "decoder",
        "layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128, "hidden": 4096,
        "context": 131072, "use_cases": ["chat"], "license": "llama3",
        "notes": "Test model fixture",
    }


@pytest.fixture
def model_llama70b():
    return {
        "params_b": 70, "active_params_b": 70, "family": "llama3", "arch": "decoder",
        "layers": 80, "heads": 64, "kv_heads": 8, "head_dim": 128, "hidden": 8192,
        "context": 131072, "use_cases": ["chat"], "license": "llama3",
        "notes": "Test model fixture",
    }


@pytest.fixture
def model_mixtral():
    """MoE model: 46B total, 13B active."""
    return {
        "params_b": 46, "active_params_b": 13, "family": "mixtral", "arch": "moe",
        "layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128, "hidden": 4096,
        "context": 32768, "use_cases": ["chat"], "license": "apache2",
        "notes": "Test MoE fixture",
    }
