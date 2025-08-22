# app.py
from flask import Flask, jsonify, request
from instruments import (
    compute_index_expiries,
    refresh_cache,
    get_fetch_env_info,
)
from config import (
    DEFAULT_INDEX_UNDERLYINGS,
    DEFAULT_EXCHANGES,
    DEFAULT_INCLUDE_EXPIRED,
)

app = Flask(__name__)

@app.get("/")
def home():
    return jsonify(status="ok", message="Hello from Flask on Vercel!")

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/v1/refresh")
def refresh():
    refresh_cache()
    return jsonify({"status": "refreshed"}), 200

@app.get("/v1/expiries")
def expiries():
    underlyings = request.args.get("underlyings")
    underlyings = (
        [u.strip().upper() for u in underlyings.split(",")]
        if underlyings else [u.upper() for u in DEFAULT_INDEX_UNDERLYINGS]
    )

    exchanges = request.args.get("exchanges")
    exchanges = (
        [e.strip().upper() for e in exchanges.split(",")]
        if exchanges else [e.upper() for e in DEFAULT_EXCHANGES]
    )

    include_expired = request.args.get("include_expired")
    if include_expired is None:
        include_expired = DEFAULT_INCLUDE_EXPIRED
    else:
        include_expired = include_expired.lower() == "true"

    data = compute_index_expiries(
        underlyings=underlyings,
        exchanges=exchanges,
        include_expired=include_expired,
    )
    return jsonify(data), 200

@app.get("/v1/expiries/<underlying>")
def expiries_for_one(underlying: str):
    data = compute_index_expiries(underlyings=[underlying.upper()])
    return jsonify({
        "underlying": underlying.upper(),
        "fetched_at": data["fetched_at"],
        "source": data["source"],
        "expiries": data["expiries"].get(underlying.upper(), [])
    }), 200

@app.get("/v1/debug/env")
def debug_env():
    """
    Minimal, safe debug info (no secrets).
    Helps diagnose SSL/proxy/cert issues in corp networks.
    """
    return jsonify(get_fetch_env_info()), 200


