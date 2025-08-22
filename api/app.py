import os
import base64
import datetime as dt
import json
import re
from typing import Any, Dict, List, Optional

import pyotp
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from SmartApi import SmartConnect


# ================== CONFIG ==================
API_KEY = "16l7qomQ"
USERNAME = "SNBM1003"
PASSWORD = "9043"
TOTP_SECRET = "G64WUKEIQHKBKNZKE4C4RKYS5U"
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
# ============================================

# NSE index tokens (per Angel forums/docs)
INDEX_TOKENS = {
    "NIFTY": "26000",
    "BANKNIFTY": "26009",
    "FINNIFTY": "26037",      # commonly used token
    "MIDCPNIFTY": "26074",    # commonly used token
}

# Exact index trading symbols Angel may expect in ltpData
INDEX_TRADING_SYMBOL = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
}

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


# ---- Session/cache ----
_sc: Optional[SmartConnect] = None
_sc_time: Optional[dt.datetime] = None
_jwt: Optional[str] = None
_feed_token: Optional[str] = None

_scrip_df: Optional[pd.DataFrame] = None
_scrip_time: Optional[dt.datetime] = None

# ---------- Helpers ----------
def _sanitize_totp(raw: str) -> str:
    cleaned = re.sub(r"[^A-Z2-7]", "", (raw or "").strip().upper())
    if not cleaned:
        raise ValueError("TOTP secret missing/invalid")
    pad = (-len(cleaned)) % 8
    base64.b32decode(cleaned + "=" * pad, casefold=True)  # raises if invalid
    return cleaned

def _format_expiry(d: dt.date) -> str:
    return d.strftime("%d%b%Y").upper()  # DDMMMYYYY

def _parse_expiry(s: str) -> dt.date:
    try:
        return dt.datetime.strptime(s.upper(), "%d%b%Y").date()
    except Exception:
        raise ValueError("Expiry must be DDMMMYYYY, e.g. 28AUG2025")

def _extract_option_type(sym: str) -> Optional[str]:
    s = str(sym).upper()
    return "CE" if s.endswith("CE") else ("PE" if s.endswith("PE") else None)

def _human_strike(x):
    try:
        v = float(x)
        return v / 100.0 if v >= 10000 else v
    except Exception:
        return x

def _fnum(x):
    try:
        return float(x)
    except Exception:
        return None

def _inum(x):
    try:
        return int(float(x))
    except Exception:
        return None

def _pick(d: Dict[str, Any], *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None

# ---------- Login & Scrip Master ----------
def login(force: bool = False) -> SmartConnect:
    """Login (or reuse) SmartAPI session and capture JWT + feed token."""
    global _sc, _sc_time, _jwt, _feed_token
    if (
        not force
        and _sc is not None
        and _sc_time is not None
        and (dt.datetime.utcnow() - _sc_time).total_seconds() < 8.5 * 3600
    ):
        return _sc
    secret = _sanitize_totp(TOTP_SECRET)
    otp = pyotp.TOTP(secret).now()
    sc = SmartConnect(api_key=API_KEY)
    ses = sc.generateSession(USERNAME, PASSWORD, otp)
    _jwt = (ses.get("data") or {}).get("jwtToken")
    _feed_token = sc.getfeedToken()
    _sc, _sc_time = sc, dt.datetime.utcnow()
    return _sc

def _headers_marketdata() -> Dict[str, str]:
    """Headers Angel expects for REST /market/v1/quote (Market Data API)."""
    return {
        "Authorization": f"Bearer {_jwt}",
        "x-api-key": API_KEY,
        "x-client-code": USERNAME,
        "x-feed-token": _feed_token or "",
        "X-SourceID": "WEB",
        "X-UserType": "USER",
        "X-ClientLocalIP": "127.0.0.1",
        "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "00:00:00:00:00:00",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

def load_scrip_master(force: bool = False) -> pd.DataFrame:
    global _scrip_df, _scrip_time
    now = dt.datetime.utcnow()
    if (
        not force
        and _scrip_df is not None
        and _scrip_time is not None
        and (now - _scrip_time).total_seconds() < 6 * 3600
    ):
        return _scrip_df
    r = requests.get(SCRIP_MASTER_URL, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    for c in ("exch_seg", "instrumenttype", "name", "symbol"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    _scrip_df, _scrip_time = df, now
    return df

# ---------- Expiries ----------
def list_expiries(index_name: str, instrumenttype: str = "OPTIDX") -> List[str]:
    df = load_scrip_master()
    today = dt.date.today()
    mask = (
        (df["name"] == index_name.upper())
        & (df["exch_seg"] == "NFO")
        & (df["instrumenttype"] == instrumenttype.upper())
        & df["expiry"].notna()
        & (df["expiry"] >= today)
    )
    exps = sorted(set(df.loc[mask, "expiry"]))
    return [_format_expiry(e) for e in exps]

# ---------- Strike window selection ----------
def _nearest_index(values: List[float], x: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - x))

def _select_window_strikes(strikes_sorted: List[float], center_val: float, half_width: int = 10) -> List[float]:
    if not strikes_sorted:
        return []
    i = _nearest_index(strikes_sorted, center_val)
    lo = max(0, i - half_width)
    hi = min(len(strikes_sorted), i + half_width + 1)
    return strikes_sorted[lo:hi]

def _fast_spot(index_name: str) -> Optional[float]:
    """Quick SDK spot (may fail) using exact Angel display name + token."""
    sc = login()
    tok = INDEX_TOKENS.get(index_name.upper())
    if not tok:
        return None
    try:
        t_sym = INDEX_TRADING_SYMBOL.get(index_name.upper(), index_name.upper())
        q = sc.ltpData("NSE", t_sym, str(tok))
        d = (q or {}).get("data") or {}
        return _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
    except Exception:
        return None

# ---------- Market data via SDK & REST ----------
def fetch_full_marketdata(tokens: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    sc = login()
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if not tokens:
        return out
    for i in range(0, len(tokens), 50):
        chunk = [str(t) for t in tokens[i : i + 50]]
        try:
            res = sc.getMarketData("FULL", {"NFO": chunk})
        except Exception:
            res = {}
        data = (res or {}).get("data") or {}
        fetched = data.get("fetched") or []
        if not isinstance(fetched, list):
            fetched = []
        for row in fetched:
            tok = str(row.get("symbolToken") or row.get("token") or "")
            if not tok:
                continue
            ltp = _fnum(_pick(row, "ltp", "lastTradedPrice", "lastPrice"))
            vol = _inum(_pick(row, "tradeVolume", "totalTradedVolume", "volume", "vtt"))
            oi = _inum(_pick(row, "opnInterest", "openInterest", "oi"))
            und = _fnum(_pick(row, "underlyingValue", "underlyingSpot"))
            out[tok] = {"ltp": ltp, "volume": vol, "oi": oi, "underlying": und}
    return out

def fetch_quote_single(tradingsymbol: str, token: str) -> Dict[str, Optional[float]]:
    sc = login()
    try:
        res = sc.quote({"exchange": "NFO", "tradingsymbol": tradingsymbol, "symboltoken": str(token)})
        data = (res or {}).get("data") or {}
    except Exception:
        data = {}
    ltp = _fnum(_pick(data, "ltp", "lastTradedPrice", "lastPrice"))
    vol = _inum(_pick(data, "tradeVolume", "totalTradedVolume", "volume", "vtt"))
    oi = _inum(_pick(data, "opnInterest", "openInterest", "oi"))
    und = _fnum(_pick(data, "underlyingValue", "underlyingSpot"))
    return {"ltp": ltp, "volume": vol, "oi": oi, "underlying": und}

def _rest_quote_ltp_nse(tokens: List[str]) -> Optional[float]:
    """
    LTP via REST: POST /market/v1/quote with {"mode":"LTP","exchangeTokens":{"NSE":[...]}}
    Uses JWT + x-api-key + x-client-code + x-feed-token.
    """
    login()  # ensure jwt/feed token present
    if not tokens:
        return None
    url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/market/v1/quote"
    payload = {"mode": "LTP", "exchangeTokens": {"NSE": [str(t) for t in tokens]}}
    try:
        r = requests.post(url, headers=_headers_marketdata(), data=json.dumps(payload), timeout=15)
        j = r.json() or {}
        data = j.get("data")
        fetched = (data.get("fetched") if isinstance(data, dict) else data) or []
        if isinstance(fetched, list) and fetched:
            row = fetched[0]
            ltp = row.get("ltp") or row.get("lastTradedPrice") or row.get("lastPrice")
            return float(ltp) if ltp is not None else None
    except Exception:
        return None
    return None

# ---------- Chain builder ----------
def build_option_chain(index_name: str, expiry_str: str, instrumenttype: str = "OPTIDX", half_width: int = 10) -> Dict[str, Any]:
    expiry_date = _parse_expiry(expiry_str)
    df = load_scrip_master()

    mask = (
        (df["name"] == index_name.upper())
        & (df["exch_seg"] == "NFO")
        & (df["instrumenttype"] == instrumenttype.upper())
        & (df["expiry"] == expiry_date)
    )
    rows_all = df.loc[mask, ["symbol", "strike", "token"]].dropna(subset=["symbol", "token"]).copy()
    if rows_all.empty:
        return {
            "name": index_name.upper(),
            "instrumenttype": instrumenttype.upper(),
            "expiry": expiry_str,
            "spot": None,
            "rows": [],
        }

    rows_all["option_type"] = rows_all["symbol"].apply(_extract_option_type)
    rows_all = rows_all.dropna(subset=["option_type"]).copy()
    rows_all["strike"] = rows_all["strike"].apply(_human_strike)

    # pick ATM ± half_width
    spot_hint = _fast_spot(index_name)
    uniq_strikes = sorted(set(float(s) for s in rows_all["strike"]))
    center = spot_hint if spot_hint is not None else (uniq_strikes[len(uniq_strikes) // 2] if uniq_strikes else 0.0)
    win_strikes = _select_window_strikes(uniq_strikes, center, half_width=half_width)

    rows = rows_all[rows_all["strike"].astype(float).isin(set(win_strikes))].copy()
    rows = rows.sort_values(["strike", "option_type"]).reset_index(drop=True)

    tokens = [str(int(float(t))) for t in rows["token"]]
    book = fetch_full_marketdata(tokens)

    # per-symbol fallback if any token missing all three
    for t in tokens:
        rec = book.get(t) or {}
        if rec.get("ltp") is None and rec.get("oi") is None and rec.get("volume") is None:
            sym = rows.loc[rows["token"].astype(float).astype(int).astype(str) == t, "symbol"].iloc[0]
            book[t] = fetch_quote_single(sym, t)

    # spot from underlying, else REST LTP, else hint
    spot_candidates = [v.get("underlying") for v in book.values() if v.get("underlying") is not None]
    spot = None
    if spot_candidates:
        try:
            spot = sum(float(x) for x in spot_candidates) / len(spot_candidates)
        except Exception:
            spot = None
    if spot is None:
        tok = INDEX_TOKENS.get(index_name.upper())
        if tok:
            rest_spot = _rest_quote_ltp_nse([str(tok)])
            if rest_spot is not None:
                spot = rest_spot
    if spot is None:
        spot = spot_hint

    # merge CE/PE by strike
    chain_map: Dict[float, Dict[str, Any]] = {}
    for _, r in rows.iterrows():
        strike = float(r["strike"])
        side = r["option_type"]  # CE/PE
        tok = str(int(float(r["token"])))
        q = book.get(tok) or {}

        ltp = _fnum(q.get("ltp"))
        vol = _inum(q.get("volume"))
        oi = _inum(q.get("oi"))

        if strike not in chain_map:
            chain_map[strike] = {
                "strike": strike,
                "ce": {"ltp": 0.0, "oi": 0, "volume": 0},
                "pe": {"ltp": 0.0, "oi": 0, "volume": 0},
            }
        chain_map[strike][side.lower()] = {
            "ltp": float(ltp) if ltp is not None else 0.0,
            "oi": int(oi) if oi is not None else 0,
            "volume": int(vol) if vol is not None else 0,
        }

    out_rows = [chain_map[k] for k in sorted(chain_map.keys(), key=float)]
    if len(out_rows) > (2 * half_width + 1):
        keep = set(
            _select_window_strikes(
                sorted({r["strike"] for r in out_rows}),
                spot if spot is not None else center,
                half_width=half_width,
            )
        )
        out_rows = [r for r in out_rows if r["strike"] in keep]

    return {
        "name": index_name.upper(),
        "instrumenttype": instrumenttype.upper(),
        "expiry": expiry_str,
        "spot": float(spot) if spot is not None else None,
        "rows": out_rows,
    }

# ---------- Robust spot ----------
def get_spot_price(symbol: str) -> Optional[float]:
    """
    Best-effort spot for an index:
    1) REST LTP via JWT (+ x-api-key/x-client-code/x-feed-token)  ✅ reliable
    2) SDK getMarketData('LTP') via token
    3) SDK ltpData('NSE', trading symbol, token)
    4) SDK getMarketData('FULL')
    5) Derive from options' underlyingValue
    """
    login()
    sym = (symbol or "").upper()
    tok = INDEX_TOKENS.get(sym)
    if not tok:
        return None

    # (1) REST first (most reliable per recent API behavior)
    rest = _rest_quote_ltp_nse([str(tok)])
    if rest is not None:
        return rest

    # (2) SDK: getMarketData LTP
    try:
        res = _sc.getMarketData("LTP", {"NSE": [str(tok)]})
        data = (res or {}).get("data") or {}
        fetched = data.get("fetched") or []
        if isinstance(fetched, list) and fetched:
            ltp = _fnum(_pick(fetched[0], "ltp", "lastTradedPrice", "lastPrice"))
            if ltp is not None:
                return ltp
    except Exception:
        pass

    # (3) SDK: ltpData with exact index name
    try:
        t_sym = INDEX_TRADING_SYMBOL.get(sym, sym)
        q = _sc.ltpData("NSE", t_sym, str(tok))
        d = (q or {}).get("data") or {}
        ltp = _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
        if ltp is not None:
            return ltp
    except Exception:
        pass

    # (4) SDK: FULL
    try:
        res = _sc.getMarketData("FULL", {"NSE": [str(tok)]})
        data = (res or {}).get("data") or {}
        fetched = data.get("fetched") or []
        if isinstance(fetched, list) and fetched:
            ltp = _fnum(_pick(fetched[0], "ltp", "lastTradedPrice", "lastPrice"))
            if ltp is not None:
                return ltp
    except Exception:
        pass

    # (5) Derive from options' underlying
    try:
        df = load_scrip_master()
        today = dt.date.today()
        mask_exp = (
            (df.get("name", "").str.upper() == sym)
            & (df.get("exch_seg", "").str.upper() == "NFO")
            & (df.get("instrumenttype", "").str.upper() == "OPTIDX")
            & df["expiry"].notna()
            & (df["expiry"] >= today)
        )
        exps = sorted(set(df.loc[mask_exp, "expiry"]))
        if not exps:
            return None
        nearest = exps[0]
        mask_opt = (
            (df["name"] == sym)
            & (df["exch_seg"] == "NFO")
            & (df["instrumenttype"] == "OPTIDX")
            & (df["expiry"] == nearest)
        )
        subset = df.loc[mask_opt, ["symbol", "token"]].dropna().head(50)
        tokens = [str(int(float(t))) for t in subset["token"]]
        book = fetch_full_marketdata(tokens)
        candidates = [v.get("underlying") for v in book.values() if v.get("underlying") is not None]
        if candidates:
            return sum(float(x) for x in candidates) / len(candidates)
    except Exception:
        pass

    return None

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


@app.route("/expiry_dates/<index>", methods=["GET"])
def http_expiries(index):
    it = request.args.get("instrumenttype", "OPTIDX")
    try:
        dates = list_expiries(index, instrumenttype=it)
        return jsonify({"success": True, "index": index.upper(), "instrumenttype": it.upper(), "expiry_dates": dates})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/option_chain", methods=["GET"])
def http_chain():
    symbol = request.args.get("symbol")          # e.g., NIFTY
    expiry = request.args.get("expiry")          # e.g., 28AUG2025
    it = request.args.get("instrumenttype", "OPTIDX")
    width = request.args.get("width", type=int, default=10)  # ATM ±width

    if not symbol or not expiry:
        return jsonify({"success": False, "error": "query params 'symbol' and 'expiry' are required (expiry as DDMMMYYYY)"}), 400
    try:
        data = build_option_chain(symbol, expiry, instrumenttype=it, half_width=width)
        return jsonify({
            "success": True,
            "params": {"symbol": symbol.upper(), "expiry": expiry, "instrumenttype": it.upper(), "width": width},
            "data": data
        })
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/spot", methods=["GET"])
def http_spot():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"success": False, "error": "query param 'symbol' required"}), 400
    try:
        spot = get_spot_price(symbol)
        return jsonify({"success": True, "symbol": symbol.upper(), "spot": float(spot) if spot is not None else None})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# Vercel will serve the WSGI 'app' automatically.
