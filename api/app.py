# api/index.py  (Vercel auto-serves /api)
import os
import base64
import datetime as dt
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pyotp
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from SmartApi import SmartConnect

# ================== CONFIG (use env on Vercel) ==================
API_KEY = os.getenv("ANGEL_API_KEY", "").strip() or "16l7qomQ"
USERNAME = os.getenv("ANGEL_USERNAME", "").strip() or "SNBM1003"
PASSWORD = os.getenv("ANGEL_PASSWORD", "").strip() or "9043"
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "").strip() or "G64WUKEIQHKBKNZKE4C4RKYS5U"
SCRIP_MASTER_URL = os.getenv(
    "SCRIP_MASTER_URL",
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
).strip()
# ===============================================================

# NSE index tokens (per Angel docs/forums)
INDEX_TOKENS = {
    "NIFTY": "26000",
    "BANKNIFTY": "26009",
    "FINNIFTY": "26037",
    "MIDCPNIFTY": "26074",
}

# Display names SmartAPI expects for NSE indices
INDEX_TRADING_SYMBOL = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"])

# -------- Session / cache --------
_sc: Optional[SmartConnect] = None
_sc_time: Optional[dt.datetime] = None
_jwt: Optional[str] = None
_feed_token: Optional[str] = None

_scrip_cache: Optional[List[Dict[str, Any]]] = None
_scrip_time: Optional[dt.datetime] = None

# -------- Helpers --------
def _sanitize_totp(raw: str) -> str:
    cleaned = re.sub(r"[^A-Z2-7]", "", (raw or "").strip().upper())
    if not cleaned:
        raise ValueError("TOTP secret missing/invalid")
    pad = (-len(cleaned)) % 8
    base64.b32decode(cleaned + "=" * pad, casefold=True)  # raises if invalid
    return cleaned

def _norm_str(v: Any) -> str:
    return str(v or "").strip().upper()

def _format_expiry(d: dt.date) -> str:
    return d.strftime("%d%b%Y").upper()  # DDMMMYYYY

def _parse_expiry_input(s: str) -> dt.date:
    try:
        return dt.datetime.strptime(s.upper(), "%d%b%Y").date()
    except Exception:
        raise ValueError("Expiry must be DDMMMYYYY, e.g. 28AUG2025")

_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

def _parse_expiry_any(raw: Any) -> Optional[dt.date]:
    """Parse ISO, DD-MMM-YYYY, DDMMMYYYY, or variants; else None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() == "NAN":
        return None
    # ISO yyyy-mm-dd or yyyy-mm-ddTHH:MM:SS
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        pass
    # DD-MMM-YYYY or DD MMM YYYY (any case)
    m = re.match(r"^(\d{1,2})[-\s]?([A-Za-z]{3})[-\s]?(\d{4})$", s)
    if m:
        d, mm3, y = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        if mm3 in _MONTHS:
            try:
                return dt.date(y, _MONTHS[mm3], d)
            except Exception:
                return None
    # DDMMMYYYY (e.g., 28AUG2025)
    m = re.match(r"^(\d{2})([A-Za-z]{3})(\d{4})$", s)
    if m:
        d, mm3, y = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        if mm3 in _MONTHS:
            try:
                return dt.date(y, _MONTHS[mm3], d)
            except Exception:
                return None
    return None

def _extract_expiry_from_symbol(sym: str) -> Optional[dt.date]:
    """Find 28AUG2025 or 28-AUG-2025 inside symbol and parse it."""
    s = _norm_str(sym)
    m = re.search(r"(\d{2}[A-Z]{3}\d{4})", s)
    if m:
        return _parse_expiry_any(m.group(1))
    m = re.search(r"(\d{2}-[A-Z]{3}-\d{4})", s)
    if m:
        return _parse_expiry_any(m.group(1))
    return None

def _extract_option_type(sym: str) -> Optional[str]:
    s = _norm_str(sym)
    if s.endswith("CE"): return "CE"
    if s.endswith("PE"): return "PE"
    return None

def _human_strike(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v / 100.0 if v >= 10000 else v
    except Exception:
        return None

def _fnum(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _inum(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None

def _pick(d: Dict[str, Any], *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None

# -------- Login & headers --------
def login(force: bool = False) -> SmartConnect:
    global _sc, _sc_time, _jwt, _feed_token
    if (
        not force and _sc is not None and _sc_time is not None
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

# -------- Scrip master --------
def load_scrip_master(force: bool = False) -> List[Dict[str, Any]]:
    global _scrip_cache, _scrip_time
    now = dt.datetime.utcnow()
    if (
        not force and _scrip_cache is not None and _scrip_time is not None
        and (now - _scrip_time).total_seconds() < 6 * 3600
    ):
        return _scrip_cache

    r = requests.get(SCRIP_MASTER_URL, timeout=60)
    r.raise_for_status()
    raw = r.json()
    out: List[Dict[str, Any]] = []

    for row in raw:
        rec = dict(row) if isinstance(row, dict) else {}
        rec["exch_seg"] = _norm_str(rec.get("exch_seg"))
        rec["instrumenttype"] = _norm_str(rec.get("instrumenttype"))
        rec["name"] = _norm_str(rec.get("name"))
        rec["symbol"] = _norm_str(rec.get("symbol"))

        # expiry: prefer field, else derive from symbol
        exp = _parse_expiry_any(rec.get("expiry"))
        if exp is None:
            exp = _extract_expiry_from_symbol(rec.get("symbol"))
        rec["expiry"] = exp

        # token (string) if valid
        tok = rec.get("token")
        try:
            rec["token"] = str(int(float(tok)))
        except Exception:
            rec["token"] = None

        # strike kept raw; normalize later
        out.append(rec)

    _scrip_cache, _scrip_time = out, now
    return out

# -------- Expiry list --------
def list_expiries(index_name: str, instrumenttype: str = "OPTIDX") -> List[str]:
    data = load_scrip_master()
    today = dt.date.today()
    want_name = _norm_str(index_name)
    want_inst = _norm_str(instrumenttype)
    exps: set = set()
    for r in data:
        if (
            r.get("name") == want_name
            and r.get("exch_seg") == "NFO"
            and r.get("instrumenttype") == want_inst
        ):
            e = r.get("expiry") or _extract_expiry_from_symbol(r.get("symbol"))
            if isinstance(e, dt.date) and e >= today:
                exps.add(e)
    return [_format_expiry(e) for e in sorted(exps)]

# -------- Strike window helpers --------
def _nearest_index(values: List[float], x: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - x)) if values else 0

def _select_window_strikes(strikes_sorted: List[float], center_val: float, half_width: int = 10) -> List[float]:
    if not strikes_sorted:
        return []
    i = _nearest_index(strikes_sorted, center_val)
    lo = max(0, i - half_width)
    hi = min(len(strikes_sorted), i + half_width + 1)
    return strikes_sorted[lo:hi]

def _fast_spot(index_name: str) -> Optional[float]:
    sc = login()
    tok = INDEX_TOKENS.get(_norm_str(index_name))
    if not tok:
        return None
    try:
        t_sym = INDEX_TRADING_SYMBOL.get(_norm_str(index_name), _norm_str(index_name))
        q = sc.ltpData("NSE", t_sym, str(tok))
        d = (q or {}).get("data") or {}
        return _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
    except Exception:
        return None

# -------- Market data via SDK & REST --------
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
    """REST quote LTP: {"mode":"LTP","exchangeTokens":{"NSE":[...]}}"""
    login()
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

# -------- Option-chain helpers --------
def _same_expiry_row(r: Dict[str, Any], want_date: dt.date) -> bool:
    e = r.get("expiry")
    if not isinstance(e, dt.date):
        e = _extract_expiry_from_symbol(r.get("symbol"))
    return isinstance(e, dt.date) and e == want_date

def _filter_rows_scrip(
    data: List[Dict[str, Any]], name: str, instrumenttype: str, expiry_date: dt.date
) -> List[Dict[str, Any]]:
    out = []
    N, IT = _norm_str(name), _norm_str(instrumenttype)
    for r in data:
        if (
            r.get("name") == N
            and r.get("exch_seg") == "NFO"
            and r.get("instrumenttype") == IT
            and _same_expiry_row(r, expiry_date)
        ):
            sym = r.get("symbol")
            tok = r.get("token")
            if sym and tok:
                out.append({"symbol": sym, "strike": r.get("strike"), "token": tok})
    return out

def build_option_chain(index_name: str, expiry_str: str, instrumenttype: str = "OPTIDX", half_width: int = 10) -> Dict[str, Any]:
    expiry_date = _parse_expiry_input(expiry_str)
    data = load_scrip_master()

    rows_all = _filter_rows_scrip(data, index_name, instrumenttype, expiry_date)
    if not rows_all:
        return {
            "name": _norm_str(index_name),
            "instrumenttype": _norm_str(instrumenttype),
            "expiry": expiry_str,
            "spot": None,
            "rows": [],
        }

    # Prepare: detect CE/PE & numeric strike
    prepared: List[Dict[str, Any]] = []
    for r in rows_all:
        opt = _extract_option_type(r["symbol"])
        if opt is None:
            continue
        strike = _human_strike(r.get("strike"))
        if strike is None:
            # fallback: try extracting strike from symbol (e.g., ...C25000 / P25000)
            m = re.search(r"(\d{4,6})[CP]E?$", r["symbol"])
            if m:
                strike = _human_strike(m.group(1))
        if strike is None:
            continue
        prepared.append({
            "symbol": r["symbol"],
            "token": str(r["token"]),
            "option_type": opt,
            "strike": float(strike),
        })

    if not prepared:
        return {
            "name": _norm_str(index_name),
            "instrumenttype": _norm_str(instrumenttype),
            "expiry": expiry_str,
            "spot": None,
            "rows": [],
        }

    # ATM window
    spot_hint = _fast_spot(index_name)
    uniq_strikes = sorted({p["strike"] for p in prepared})
    center = spot_hint if spot_hint is not None else (uniq_strikes[len(uniq_strikes)//2] if uniq_strikes else 0.0)
    win_strikes = set(_select_window_strikes(uniq_strikes, center, half_width=half_width))

    filtered = [p for p in prepared if p["strike"] in win_strikes]
    filtered.sort(key=lambda x: (x["strike"], x["option_type"]))

    tokens = [p["token"] for p in filtered]
    book = fetch_full_marketdata(tokens)

    # fallback per token if empty
    token_to_symbol = {p["token"]: p["symbol"] for p in filtered}
    for t in tokens:
        rec = book.get(t) or {}
        if rec.get("ltp") is None and rec.get("oi") is None and rec.get("volume") is None:
            sym = token_to_symbol.get(t)
            if sym:
                book[t] = fetch_quote_single(sym, t)

    # spot from underlying -> REST -> hint
    spot_candidates = [v.get("underlying") for v in book.values() if v.get("underlying") is not None]
    spot = None
    if spot_candidates:
        try:
            spot = sum(float(x) for x in spot_candidates) / len(spot_candidates)
        except Exception:
            spot = None
    if spot is None:
        tok = INDEX_TOKENS.get(_norm_str(index_name))
        if tok:
            rest_spot = _rest_quote_ltp_nse([str(tok)])
            if rest_spot is not None:
                spot = rest_spot
    if spot is None:
        spot = spot_hint

    # merge CE/PE rows
    chain_map: Dict[float, Dict[str, Any]] = {}
    for p in filtered:
        strike = p["strike"]
        side = p["option_type"]  # CE/PE
        q = book.get(p["token"]) or {}
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

    out_rows = [chain_map[k] for k in sorted(chain_map.keys())]
    if len(out_rows) > (2 * half_width + 1):
        keep = set(_select_window_strikes([r["strike"] for r in out_rows], spot if spot is not None else center, half_width))
        out_rows = [r for r in out_rows if r["strike"] in keep]

    return {
        "name": _norm_str(index_name),
        "instrumenttype": _norm_str(instrumenttype),
        "expiry": expiry_str,
        "spot": float(spot) if spot is not None else None,
        "rows": out_rows,
    }

# -------- Robust spot --------
def _first_future_expiry_for_symbol(data: List[Dict[str, Any]], sym: str) -> Optional[dt.date]:
    today = dt.date.today()
    exps = []
    for r in data:
        if (
            r.get("name") == _norm_str(sym)
            and r.get("exch_seg") == "NFO"
            and r.get("instrumenttype") == "OPTIDX"
        ):
            e = r.get("expiry") or _extract_expiry_from_symbol(r.get("symbol"))
            if isinstance(e, dt.date) and e >= today:
                exps.append(e)
    exps = sorted(set(exps))
    return exps[0] if exps else None

def _subset_tokens_for_expiry(
    data: List[Dict[str, Any]], sym: str, expiry: dt.date, limit: int = 50
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for r in data:
        if (
            r.get("name") == _norm_str(sym)
            and r.get("exch_seg") == "NFO"
            and r.get("instrumenttype") == "OPTIDX"
            and _same_expiry_row(r, expiry)
        ):
            symb = r.get("symbol")
            tok = r.get("token")
            if symb and tok:
                out.append((symb, tok))
                if len(out) >= limit:
                    break
    return out

def get_spot_price(symbol: str) -> Optional[float]:
    login()
    sym = _norm_str(symbol)
    tok = INDEX_TOKENS.get(sym)
    if not tok:
        return None

    # 1) REST
    rest = _rest_quote_ltp_nse([str(tok)])
    if rest is not None:
        return rest

    # 2) SDK LTP
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

    # 3) SDK ltpData
    try:
        t_sym = INDEX_TRADING_SYMBOL.get(sym, sym)
        q = _sc.ltpData("NSE", t_sym, str(tok))
        d = (q or {}).get("data") or {}
        ltp = _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
        if ltp is not None:
            return ltp
    except Exception:
        pass

    # 4) SDK FULL
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

    # 5) Derive from options' underlying
    try:
        data = load_scrip_master()
        nearest = _first_future_expiry_for_symbol(data, sym)
        if nearest is None:
            return None
        subset = _subset_tokens_for_expiry(data, sym, nearest, limit=50)
        tokens = [str(tok) for _, tok in subset]
        book = fetch_full_marketdata(tokens)
        candidates = [v.get("underlying") for v in book.values() if v.get("underlying") is not None]
        if candidates:
            return sum(float(x) for x in candidates) / len(candidates)
    except Exception:
        pass

    return None


# ---------- HTTP ----------
@app.get("/")
def home():
    return jsonify(status="ok", message="Hello from Flask on Vercel!")


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


# -------- HTTP --------
@app.route("/expiry_dates/<index>", methods=["GET"])
def http_expiries(index):
    it = request.args.get("instrumenttype", "OPTIDX")
    try:
        dates = list_expiries(index, instrumenttype=it)
        return jsonify({
            "success": True,
            "index": _norm_str(index),
            "instrumenttype": _norm_str(it),
            "expiry_dates": dates
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/option_chain", methods=["GET"])
def http_chain():
    symbol = request.args.get("symbol")
    expiry = request.args.get("expiry")
    it = request.args.get("instrumenttype", "OPTIDX")
    width = request.args.get("width", type=int, default=10)

    if not symbol or not expiry:
        return jsonify({"success": False, "error": "query params 'symbol' and 'expiry' are required (expiry as DDMMMYYYY)"}), 400
    try:
        data = build_option_chain(symbol, expiry, instrumenttype=it, half_width=width)
        return jsonify({
            "success": True,
            "params": {"symbol": _norm_str(symbol), "expiry": expiry, "instrumenttype": _norm_str(it), "width": width},
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
        return jsonify({"success": True, "symbol": _norm_str(symbol), "spot": float(spot) if spot is not None else None})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# NOTE: Do NOT call app.run(); Vercel serves WSGI `app`.
