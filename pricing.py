"""
Live GPU price fetching from RunPod (GraphQL) and Vast.ai (REST).
No API keys required — both use public endpoints.
"""
import time
import threading

import requests as http

from data.gpus import GPUS, RUNPOD_NAME_MAP, VASTAI_NAME_MAP

# ── Price cache ────────────────────────────────────────────────────────────────

PRICE_CACHE: dict = {
    "last_updated": None,   # epoch timestamp; None = never fetched
    "runpod_hits":  0,
    "vastai_hits":  0,
    "errors":       [],
    "live":         False,
}


# ── Name matching ──────────────────────────────────────────────────────────────

def _match_gpu_name(raw_name: str, name_map: dict):
    """Exact match first, then normalised substring match."""
    if raw_name in name_map:
        return name_map[raw_name]
    normalised = raw_name.upper().replace("-", "_").replace(" ", "_")
    for key, val in name_map.items():
        if key.upper().replace("-", "_").replace(" ", "_") in normalised:
            return val
    return None


# ── Provider fetchers ──────────────────────────────────────────────────────────

def fetch_runpod_prices():
    """Query RunPod's public GraphQL API for GPU spot prices. Returns (prices_dict, error_str|None)."""
    try:
        resp = http.post(
            "https://api.runpod.io/graphql",
            json={"query": "{ gpuTypes { id displayName memoryInGb securePrice communityPrice } }"},
            timeout=8,
        )
        if resp.status_code != 200:
            return {}, f"RunPod HTTP {resp.status_code}"

        prices = {}
        for g in resp.json().get("data", {}).get("gpuTypes", []):
            raw = g.get("id") or g.get("displayName") or ""
            mapped = _match_gpu_name(raw, RUNPOD_NAME_MAP) or \
                     _match_gpu_name(g.get("displayName", ""), RUNPOD_NAME_MAP)
            if mapped:
                price = g.get("securePrice") or g.get("communityPrice")
                if price and float(price) > 0:
                    prices[mapped] = round(float(price), 4)
        return prices, None
    except Exception as e:
        return {}, str(e)


def fetch_vastai_prices():
    """Query Vast.ai's public bundle API for minimum single-GPU spot prices. Returns (prices_dict, error_str|None)."""
    try:
        resp = http.get(
            "https://console.vast.ai/api/v0/bundles/",
            params={"q": '{"rentable":true,"num_gpus":{"gte":1,"lte":1},"order":[["dph_total","asc"]]}'},
            timeout=10,
        )
        if resp.status_code != 200:
            return {}, f"Vast.ai HTTP {resp.status_code}"

        prices = {}
        for offer in resp.json().get("offers", []):
            raw    = (offer.get("gpu_name") or "").replace(" ", "_").upper()
            mapped = _match_gpu_name(raw, VASTAI_NAME_MAP)
            if mapped:
                dph = offer.get("dph_total", 0)
                if dph and float(dph) > 0:
                    if mapped not in prices or dph < prices[mapped]:
                        prices[mapped] = round(float(dph), 4)
        return prices, None
    except Exception as e:
        return {}, str(e)


# ── Refresh logic ──────────────────────────────────────────────────────────────

def refresh_prices() -> None:
    """Fetch live prices from all providers and patch GPUS dict in-place."""
    errors = []

    runpod_prices, err_r = fetch_runpod_prices()
    if err_r:
        errors.append(f"RunPod: {err_r}")

    vastai_prices, err_v = fetch_vastai_prices()
    if err_v:
        errors.append(f"Vast.ai: {err_v}")

    for gpu_name in GPUS:
        if gpu_name in runpod_prices:
            GPUS[gpu_name]["runpod_hr"] = runpod_prices[gpu_name]
        if gpu_name in vastai_prices:
            GPUS[gpu_name]["vastai_hr"] = vastai_prices[gpu_name]

    # Mutate in-place so all import-site references see the update.
    # (Reassignment would only rebind the local name in this module.)
    PRICE_CACHE.update({
        "last_updated": time.time(),
        "runpod_hits":  len(runpod_prices),
        "vastai_hits":  len(vastai_prices),
        "errors":       errors,
        "live":         not bool(errors) or (len(runpod_prices) + len(vastai_prices) > 0),
    })


def _bg_price_refresh() -> None:
    """Background thread: fetch on startup then every 30 minutes."""
    time.sleep(3)   # let Flask finish starting
    while True:
        refresh_prices()
        time.sleep(1800)


def start_price_refresh_thread() -> None:
    """Start the background price refresh thread (call once at startup)."""
    t = threading.Thread(target=_bg_price_refresh, daemon=True)
    t.start()
