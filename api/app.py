# api/index.py  (Vercel auto-serves /api)
import os
import base64
import datetime as dt
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# --- make logging writable in serverless ---
try:
    os.makedirs("/tmp/logs", exist_ok=True)
    os.chdir("/tmp")
except Exception:
    # if it fails, we still continue; SmartApi may fall back to stdout
    pass
# ------------------------------------------

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
INDEX_TOKENS = {"NIFTY": "26000", "BANKNIFTY": "26009", "FINNIFTY": "26037", "MIDCPNIFTY": "26074"}
INDEX_TRADING_SYMBOL = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"])

# ---- Session/cache ----
_sc: Optional[SmartConnect] = None
_sc_time: Optional[dt.datetime] = None
_jwt: Optional[str] = None
_feed_token: Optional[str] = None

_scrip_rows: Optional[List[Dict[str, Any]]] = None
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
    return d.strftime("%d%b%Y").upper()

def _parse_expiry(s: str) -> dt.date:
    try:
        return dt.datetime.strptime(s.upper(), "%d%b%Y").date()
    except Exception:
        raise ValueError("Expiry must be DDMMMYYYY, e.g. 28AUG2025")

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

# ----- Symbol parsing (handles 2- or 4-digit year & various layouts) -----
_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
# Examples matched:
#  NIFTY28AUG24CE, NIFTY28AUG2025CE, NIFTY28AUG24 20000CE, NIFTY28AUG24C20000, ... etc.
_EXPIRY_YR2_OR_YR4 = re.compile(r"(\d{1,2})([A-Z]{3})(\d{2,4})")
_STRIKE_AND_OT_RE   = re.compile(r"(\d{3,6})\s*(CE|PE|C|P)\b")

def _normalize_year(y_txt: str) -> int:
    y = int(y_txt)
    if len(y_txt) == 2:
        y += 2000
    if y < 2000 or y > 2099:
        y = 2000 + (y % 100)
    return y

def _human_strike(x):
    try:
        v = float(x)
        return v / 100.0 if v >= 10000 else v
    except Exception:
        return None

def _parse_symbol_fields(symbol: str) -> Tuple[Optional[dt.date], Optional[float], Optional[str]]:
    """
    Returns (expiry_date, strike, option_type) from trading symbol, if present.
    option_type in {"CE","PE"} or None.
    """
    if not symbol:
        return None, None, None
    s = str(symbol).upper()

    # expiry
    exp = None
    m = _EXPIRY_YR2_OR_YR4.search(s)
    if m:
        d, mon_abbr, y = m.groups()
        mon = _MONTHS.get(mon_abbr)
        if mon:
            try:
                exp = dt.date(_normalize_year(y), mon, int(d))
            except Exception:
                exp = None

    # option type + strike (try a few layouts)
    ot = None
    strike = None

    # pattern like "... 20000CE" or "...C20000"
    m2 = _STRIKE_AND_OT_RE.search(s)
    if m2:
        strike = _human_strike(m2.group(1))
        ot_raw = m2.group(2)
        ot = "CE" if ot_raw in ("CE", "C") else ("PE" if ot_raw in ("PE","P") else None)

    # fallback: if we couldn't find strike+OT in one go, try to detect OT then nearest number
    if strike is None:
        m_ot = re.search(r"\b(CE|PE|C|P)\b", s)
        if m_ot:
            ot_raw = m_ot.group(1)
            ot = "CE" if ot_raw in ("CE","C") else "PE"
            # try last 5-6 digit number as strike
            m_num = re.findall(r"(\d{4,6})", s)
            if m_num:
                strike = _human_strike(m_num[-1])

    return exp, strike, ot

# ---------- Login & headers ----------
def login(force: bool = False) -> SmartConnect:
    global _sc, _sc_time, _jwt, _feed_token
    missing = [n for n, v in {
        "ANGEL_API_KEY": API_KEY,
        "ANGEL_USERNAME": USERNAME,
        "ANGEL_PASSWORD": PASSWORD,
        "ANGEL_TOTP_SECRET": TOTP_SECRET,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    if (
        not force and _sc is not None and _sc_time is not None
        and (dt.datetime.utcnow() - _sc_time).total_seconds() < 8.5*3600
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

# ---------- Scrip Master (no pandas) ----------
def _to_date_safe(value) -> Optional[dt.date]:
    if value in (None, "", "NaT"):
        return None
    s = str(value).strip()
    fmts = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y", "%d%b%Y", "%d-%b-%y"]
    for fmt in fmts:
        try:
            return dt.datetime.strptime(s[:len(fmt)], fmt).date()
        except Exception:
            pass
    return None

def load_scrip_master(force: bool = False) -> List[Dict[str, Any]]:
    global _scrip_rows, _scrip_time
    now = dt.datetime.utcnow()
    if not force and _scrip_rows is not None and _scrip_time and (now - _scrip_time).total_seconds() < 6*3600:
        return _scrip_rows

    r = requests.get(SCRIP_MASTER_URL, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        for key in ("data","rows","result"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        _scrip_rows, _scrip_time = [], now
        return _scrip_rows

    norm: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol","")).strip().upper()
        if not symbol:
            continue
        exch = str(row.get("exch_seg","")).strip().upper()
        inst = str(row.get("instrumenttype","")).strip().upper()
        name = str(row.get("name","")).strip().upper()

        # parse fields if not provided
        expiry = _to_date_safe(row.get("expiry")) or _parse_symbol_fields(symbol)[0]
        strike = None
        if row.get("strike") not in (None, "", "NaN"):
            strike = _human_strike(row.get("strike"))
        if strike is None:
            strike = _parse_symbol_fields(symbol)[1]
        opt_type = None
        if symbol.endswith(("CE","PE")) or re.search(r"\b(CE|PE|C|P)\b", symbol):
            ot = _parse_symbol_fields(symbol)[2]
            opt_type = ot

        token = row.get("token")
        if token in (None, "", "NaN"):
            continue

        norm.append({
            "symbol": symbol,
            "exch_seg": exch,
            "instrumenttype": inst,
            "name": name,
            "expiry": expiry,
            "strike": strike,
            "option_type": opt_type,
            "token": token,
        })
    _scrip_rows, _scrip_time = norm, now
    return norm

# ---------- Expiries ----------
def list_expiries(index_name: str, instrumenttype: str = "OPTIDX") -> List[str]:
    rows = load_scrip_master()
    today = dt.date.today()
    idx = (index_name or "").upper()
    inst = (instrumenttype or "").upper()

    exp_set: set[dt.date] = set()
    for r in rows:
        if r.get("exch_seg") != "NFO":
            continue
        # be lenient: instrumenttype startswith 'OPT' if exact doesn't match
        inst_ok = (r.get("instrumenttype") == inst) or (inst.startswith("OPT") and str(r.get("instrumenttype","")).startswith("OPT"))
        if not inst_ok:
            continue
        # match by 'name' OR symbol prefix
        name_ok = (r.get("name") == idx)
        sym_ok = str(r.get("symbol","")).startswith(idx)
        if not (name_ok or sym_ok):
            continue

        exp = r.get("expiry")
        if isinstance(exp, dt.date) and exp >= today:
            exp_set.add(exp)

    return [_format_expiry(e) for e in sorted(exp_set)]

# ---------- Strike window selection ----------
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

# ---------- Market data ----------
def fetch_full_marketdata(tokens: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    sc = login()
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if not tokens:
        return out
    for i in range(0, len(tokens), 50):
        chunk = [str(t) for t in tokens[i:i+50]]
        try:
            res = sc.getMarketData("FULL", {"NFO": chunk})
        except Exception:
            res = {}
        data = (res or {}).get("data") or {}
        fetched = data.get("fetched") or []
        if not isinstance(fetched, list):
            continue
        for row in fetched:
            tok = str(row.get("symbolToken") or row.get("token") or "")
            if not tok:
                continue
            ltp = _fnum(_pick(row, "ltp", "lastTradedPrice", "lastPrice"))
            vol = _inum(_pick(row, "tradeVolume", "totalTradedVolume", "volume", "vtt"))
            oi  = _inum(_pick(row, "opnInterest", "openInterest", "oi"))
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
    oi  = _inum(_pick(data, "opnInterest", "openInterest", "oi"))
    und = _fnum(_pick(data, "underlyingValue", "underlyingSpot"))
    return {"ltp": ltp, "volume": vol, "oi": oi, "underlying": und}

def _rest_quote_ltp_nse(tokens: List[str]) -> Optional[float]:
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

# ---------- Chain builder ----------
def build_option_chain(index_name: str, expiry_str: str, instrumenttype: str = "OPTIDX", half_width: int = 10) -> Dict[str, Any]:
    expiry_date = _parse_expiry(expiry_str)
    idx = index_name.upper()
    inst = instrumenttype.upper()

    # collect relevant rows (lenient matching & symbol parsing fallback)
    rows_all = []
    for r in load_scrip_master():
        if r.get("exch_seg") != "NFO":
            continue
        inst_ok = (r.get("instrumenttype") == inst) or (inst.startswith("OPT") and str(r.get("instrumenttype","")).startswith("OPT"))
        if not inst_ok:
            continue
        if not (r.get("name") == idx or str(r.get("symbol","")).startswith(idx)):
            continue

        # ensure expiry present
        exp = r.get("expiry")
        if not isinstance(exp, dt.date):
            exp = _parse_symbol_fields(r.get("symbol",""))[0]
        if exp != expiry_date:
            continue

        opt_type = r.get("option_type")
        if opt_type not in ("CE","PE"):
            # try parse again if missing
            opt_type = _parse_symbol_fields(r.get("symbol",""))[2]
        strike = r.get("strike")
        if strike is None:
            strike = _parse_symbol_fields(r.get("symbol",""))[1]

        if opt_type not in ("CE","PE") or strike is None:
            continue

        token = r.get("token")
        if token in (None,"","NaN"):
            continue

        rows_all.append({"symbol": r["symbol"], "strike": float(strike), "token": token, "option_type": opt_type})

    if not rows_all:
        return {"name": idx, "instrumenttype": inst, "expiry": expiry_str, "spot": None, "rows": []}

    # pick ATM ± window
    spot_hint = _fast_spot(idx)
    uniq_strikes = sorted(set(float(r["strike"]) for r in rows_all))
    center = spot_hint if spot_hint is not None else (uniq_strikes[len(uniq_strikes)//2] if uniq_strikes else 0.0)
    win_strikes = set(_select_window_strikes(uniq_strikes, center, half_width=half_width))

    rows = [r for r in rows_all if float(r["strike"]) in win_strikes]
    rows.sort(key=lambda x: (float(x["strike"]), x["option_type"]))
    tokens = [str(int(float(r["token"]))) for r in rows]

    book = fetch_full_marketdata(tokens)
    # fallback per symbol if needed
    for t in tokens:
        rec = book.get(t) or {}
        if rec.get("ltp") is None and rec.get("oi") is None and rec.get("volume") is None:
            sym = next((r["symbol"] for r in rows if str(int(float(r["token"]))) == t), None)
            if sym:
                book[t] = fetch_quote_single(sym, t)

    # robust spot
    spot_candidates = [v.get("underlying") for v in book.values() if v.get("underlying") is not None]
    spot: Optional[float] = None
    if spot_candidates:
        try:
            spot = sum(float(x) for x in spot_candidates) / len(spot_candidates)
        except Exception:
            spot = None
    if spot is None:
        tok = INDEX_TOKENS.get(idx)
        if tok:
            rest_spot = _rest_quote_ltp_nse([str(tok)])
            if rest_spot is not None:
                spot = rest_spot
    if spot is None:
        spot = spot_hint

    # merge CE/PE by strike
    chain_map: Dict[float, Dict[str, Any]] = {}
    for r in rows:
        strike = float(r["strike"])
        side = r["option_type"]
        tok = str(int(float(r["token"])))
        q = book.get(tok) or {}
        ltp = _fnum(q.get("ltp"))
        vol = _inum(q.get("volume"))
        oi  = _inum(q.get("oi"))
        if strike not in chain_map:
            chain_map[strike] = {"strike": strike, "ce": {"ltp": 0.0, "oi": 0, "volume": 0}, "pe": {"ltp": 0.0, "oi": 0, "volume": 0}}
        chain_map[strike][side.lower()] = {
            "ltp": float(ltp) if ltp is not None else 0.0,
            "oi": int(oi) if oi is not None else 0,
            "volume": int(vol) if vol is not None else 0,
        }

    out_rows = [chain_map[k] for k in sorted(chain_map.keys(), key=float)]
    if len(out_rows) > (2*half_width + 1):
        keep = set(_select_window_strikes(sorted({r["strike"] for r in out_rows}), spot if spot is not None else center, half_width=half_width))
        out_rows = [r for r in out_rows if r["strike"] in keep]

    return {"name": idx, "instrumenttype": inst, "expiry": expiry_str, "spot": float(spot) if spot is not None else None, "rows": out_rows}

# ---------- Spot ----------
def get_spot_price(symbol: str) -> Optional[float]:
    login()
    sym = (symbol or "").upper()
    tok = INDEX_TOKENS.get(sym)
    if not tok:
        return None
    rest = _rest_quote_ltp_nse([str(tok)])
    if rest is not None:
        return rest
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
    try:
        t_sym = INDEX_TRADING_SYMBOL.get(sym, sym)
        q = _sc.ltpData("NSE", t_sym, str(tok))
        d = (q or {}).get("data") or {}
        ltp = _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
        if ltp is not None:
            return ltp
    except Exception:
        pass
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
    return None

# ---------- HTTP ----------
@app.get("/")
def home():
    return jsonify(status="ok", message="Hello from Flask on Vercel!")

@app.get("/health")
def health():
    rows = load_scrip_master()
    return jsonify({"status":"ok","rows_cached":len(rows)}), 200

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
    symbol = request.args.get("symbol")
    expiry = request.args.get("expiry")
    it = request.args.get("instrumenttype", "OPTIDX")
    width = request.args.get("width", type=int, default=10)
    if not symbol or not expiry:
        return jsonify({"success": False, "error": "query params 'symbol' and 'expiry' are required (expiry as DDMMMYYYY)"}), 400
    try:
        data = build_option_chain(symbol, expiry, instrumenttype=it, half_width=width)
        return jsonify({"success": True, "params": {"symbol": symbol.upper(), "expiry": expiry, "instrumenttype": it.upper(), "width": width}, "data": data})
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

# No app.run(); Vercel serves WSGI `app`.
