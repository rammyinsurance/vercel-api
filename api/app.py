# api/app.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from flask import Flask, jsonify, request

# If your 'instruments.py' and 'config.py' live at the project root (next to vercel.json),
# ensure the project root is on sys.path (Vercel's CWD is the project root, but this
# keeps imports robust when refactoring).  ref: vercel python runtime notes
# https://vercel.com/docs/functions/runtimes/python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from instruments import (  # noqa: E402
    compute_index_expiries,
    refresh_cache,
    get_fetch_env_info,
)
from config import (  # noqa: E402
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
    """
    Trigger a cache refresh. Keep this work fast for serverless
    (or move heavy operations to a background worker).
    """
    try:
        refresh_cache()
        return jsonify({"status": "refreshed"}), 200
    except Exception as exc:
        # Surface a JSON error without leaking secrets
        return jsonify({"error": "refresh_failed", "detail": str(exc)}), 500


@app.get("/v1/expiries")
def expiries():
    # underlyings
    underlyings_param = request.args.get("underlyings")
    if underlyings_param:
        underlyings = [u.strip().upper() for u in underlyings_param.split(",") if u.strip()]
    else:
        underlyings = [u.upper() for u in DEFAULT_INDEX_UNDERLYINGS]

    # exchanges
    exchanges_param = request.args.get("exchanges")
    if exchanges_param:
        exchanges = [e.strip().upper() for e in exchanges_param.split(",") if e.strip()]
    else:
        exchanges = [e.upper() for e in DEFAULT_EXCHANGES]

    # include_expired
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


# FIXED: use Flask's variable rule syntax, not HTML-escaped text
# Was: "/v1/expiries/&lt;underlying&gt;"
@app.get("/v1/expiries/<underlying>")
def expiries_for_one(underlying: str):
    data = compute_index_expiries(underlyings=[underlying.upper()])
    return jsonify(
        {
            "underlying": underlying.upper(),
            "fetched_at": data["fetched_at"],
            "source": data["source"],
            "expiries": data["expiries"].get(underlying.upper(), []),
        }
    ), 200


@app.get("/v1/debug/env")
def debug_env():
    """
    Minimal, safe debug info (no secrets).
    Useful to diagnose SSL/proxy/cert issues.
    """
    return jsonify(get_fetch_env_info()), 200


# Do NOT call app.run(); Vercel's Python runtime serves the WSGI `app` for you.
# ref: official Python runtime docs & Flask template
# https://vercel.com/docs/functions/runtimes/python
# https://vercel.com/templates/python/flask-hello-world
