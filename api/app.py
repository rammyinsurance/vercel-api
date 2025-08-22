from flask import Flask, jsonify, request

# Absolute imports (ensure 'api' is the package, not a directory)
from api.instruments import (
    compute_index_expiries,
    refresh_cache,
    get_fetch_env_info,
)
from api.config import (
    DEFAULT_INDEX_UNDERLYINGS,
    DEFAULT_EXCHANGES,
    DEFAULT_INCLUDE_EXPIRED,
)

# Correct Flask initialization (was: Flask(**name**))
app = Flask(__name__)

@app.get("/")
def home():
    return jsonify(status="ok", message="Hello from Flask on Vercel!")

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/v1/refresh")
def refresh():
    try:
        refresh_cache()
        return jsonify({"status": "refreshed"}), 200
    except Exception as exc:
        app.logger.exception("refresh failed: %s", exc)
        return jsonify({"error": "refresh_failed", "detail": str(exc)}), 500

@app.get("/v1/expiries")
def expiries():
    underlyings_param = request.args.get("underlyings")
    if underlyings_param:
        underlyings = [u.strip().upper() for u in underlyings_param.split(",") if u.strip()]
    else:
        underlyings = [u.upper() for u in DEFAULT_INDEX_UNDERLYINGS]

    exchanges_param = request.args.get("exchanges")
    if exchanges_param:
        exchanges = [e.strip().upper() for e in exchanges_param.split(",") if e.strip()]
    else:
        exchanges = [e.upper() for e in DEFAULT_EXCHANGES]

    include_expired_param = request.args.get("include_expired")
    if include_expired_param is None:
        include_expired = DEFAULT_INCLUDE_EXPIRED
    else:
        include_expired = include_expired_param.lower() in {"true", "1", "yes", "y"}

    data = compute_index_expiries(
        underlyings=underlyings,
        exchanges=exchanges,
        include_expired=include_expired,
    )
    return jsonify(data), 200

@app.get("/v1/expiries/<underlying>")
def expiries_for_one(underlying: str):
    underlying_u = underlying.upper()
    data = compute_index_expiries(underlyings=[underlying_u])
    return jsonify({
        "underlying": underlying_u,
        "fetched_at": data["fetched_at"],
        "source": data["source"],
        "expiries": data["expiries"].get(underlying_u, [])
    }), 200

@app.get("/v1/debug/env")
def debug_env():
    return jsonify(get_fetch_env_info()), 200

# Vercel will serve the WSGI 'app' automatically.
