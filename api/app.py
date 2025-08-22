# app.py
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

# ================== CONFIG ==================
API_KEY = "16l7qomQ"
USERNAME = "SNBM1003"
PASSWORD = "9043"
TOTP_SECRET = "G64WUKEIQHKBKNZKE4C4RKYS5U"
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
# ============================================

SUPPORTED_EXCH = {"NFO", "BFO"}

# NSE tokens (REST LTP for NSE indices uses these)
INDEX_TOKENS = {
    "NIFTY": "26000",
    "BANKNIFTY": "26009",
    "FINNIFTY": "26037",
    "MIDCPNIFTY": "26074",
    # For SENSEX/BANKEX we discover via Scrip Master (BSE), do not hardcode
    # "SENSEX": ...,
    # "BANKEX": ...,
}

# Display names used by SDK ltpData for indices
INDEX_TRADING_SYMBOL = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
    "SENSEX": "SENSEX",
    "BANKEX": "BSE BANKEX",
}

# Aliases so Scrip Master name mismatches don't break matching (esp. BSE)
NAME_ALIASES = {
    "SENSEX": {"SENSEX", "S&P BSE SENSEX", "BSE SENSEX"},
    "BANKEX": {"BANKEX", "S&P BSE BANKEX", "BSE BANKEX"},
    "NIFTY": {"NIFTY", "NIFTY 50"},
    "BANKNIFTY": {"BANKNIFTY", "NIFTY BANK"},
    "FINNIFTY": {"FINNIFTY", "NIFTY FIN SERVICE", "NIFTY FINANCIAL SERVICES"},
    "MIDCPNIFTY": {"MIDCPNIFTY", "NIFTY MIDCAP SELECT"},
}

SUPPORTED_INDICES = set(list(INDEX_TOKENS.keys()) + ["SENSEX", "BANKEX"])

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

def _name_match(target: str, row_name: str) -> bool:
    t = _norm_str(target)
    r = _norm_str(row_name)
    if t == r:
        return True
    aliases = NAME_ALIASES.get(t, set())
    if r in {_norm_str(x) for x in aliases}:
        return True
    # a little loose for BSE indices
    if t in ("SENSEX", "BANKEX"):
        return t in r or "BSE" in r
    return False

def _format_expiry(d: dt.date) -> str:
    return d.strftime("%d%b%Y").upper()  # DDMMMYYYY

def _parse_expiry_input(s: str) -> dt.date:
    try:
        return dt.datetime.strptime(s.upper(), "%d%b%Y").date()
    except Exception:
        raise ValueError("Expiry must be DDMMMYYYY, e.g. 28AUG2025")

_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

def _parse_expiry_any(raw: Any) -> Optional[dt.date]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() == "NAN":
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        pass
    m = re.match(r"^(\d{1,2})[-\s]?([A-Za-z]{3})[-\s]?(\d{4})$", s)
    if m:
        d, mm3, y = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        if mm3 in _MONTHS:
            try:
                return dt.date(y, _MONTHS[mm3], d)
            except Exception:
                return None
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

def _parse_exchanges_param(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["NFO", "BFO"]
    parts = [p.strip().upper() for p in str(raw).split(",")]
    out = [p for p in parts if p in SUPPORTED_EXCH]
    return out or ["NFO", "BFO"]

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

        exp = _parse_expiry_any(rec.get("expiry"))
        if exp is None:
            exp = _extract_expiry_from_symbol(rec.get("symbol"))
        rec["expiry"] = exp

        tok = rec.get("token")
        try:
            rec["token"] = str(int(float(tok)))
        except Exception:
            rec["token"] = None

        out.append(rec)

    _scrip_cache, _scrip_time = out, now
    return out

# -------- Expiry list (limit to next 4) --------
def list_expiries(index_name: str, instrumenttype: str, exchanges: List[str], limit: int = 4) -> List[str]:
    data = load_scrip_master()
    today = dt.date.today()
    want_inst = _norm_str(instrumenttype)
    ex_set = set(exchanges)
    exps: set = set()

    for r in data:
        if (
            _name_match(index_name, r.get("name"))
            and r.get("exch_seg") in ex_set
            and r.get("instrumenttype") == want_inst
        ):
            e = r.get("expiry") or _extract_expiry_from_symbol(r.get("symbol"))
            if isinstance(e, dt.date) and e >= today:
                exps.add(e)

    sorted_exps = sorted(exps)[: max(0, int(limit or 4))]
    return [_format_expiry(e) for e in sorted_exps]

# -------- Market data via SDK & REST --------
def fetch_full_marketdata_by_exchange(req: Dict[str, List[str]]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """
    req = { "NFO": [token1, token2,...], "BFO": [token3,...] }
    return: { exch: { token: {ltp, volume, oi, underlying} } }
    """
    sc = login()
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {ex: {} for ex in req.keys()}

    for exch, tokens in req.items():
        if not tokens:
            continue
        for i in range(0, len(tokens), 50):
            chunk = [str(t) for t in tokens[i : i + 50]]
            try:
                res = sc.getMarketData("FULL", {exch: chunk})
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
                und = _fnum(_pick(
                    row,
                    "underlyingValue",
                    "underlyingSpot",
                    "underlyingPrice",
                    "underlying",
                    "underlying_index",
                    "underlyingIndex",
                    "underlying_value",
                ))
                out[exch][tok] = {"ltp": ltp, "volume": vol, "oi": oi, "underlying": und}
    return out

def fetch_quote_single(exch: str, tradingsymbol: str, token: str) -> Dict[str, Optional[float]]:
    sc = login()
    try:
        res = sc.quote({"exchange": exch, "tradingsymbol": tradingsymbol, "symboltoken": str(token)})
        data = (res or {}).get("data") or {}
    except Exception:
        data = {}
    ltp = _fnum(_pick(data, "ltp", "lastTradedPrice", "lastPrice"))
    vol = _inum(_pick(data, "tradeVolume", "totalTradedVolume", "volume", "vtt"))
    oi = _inum(_pick(data, "opnInterest", "openInterest", "oi"))
    und = _fnum(_pick(
        data,
        "underlyingValue",
        "underlyingSpot",
        "underlyingPrice",
        "underlying",
        "underlying_index",
        "underlyingIndex",
        "underlying_value",
    ))
    return {"ltp": ltp, "volume": vol, "oi": oi, "underlying": und}

def _rest_quote_ltp(exchange: str, tokens: List[str]) -> Optional[float]:
    """Generic Angel REST LTP for NSE/BSE indices via /rest/secure/.../quote"""
    login()
    if not tokens:
        return None
    url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/market/v1/quote"
    payload = {"mode": "LTP", "exchangeTokens": {exchange: [str(t) for t in tokens]}}
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

def _collect_rows_for_exchanges(
    data: List[Dict[str, Any]], name: str, instrumenttype: str, expiry_date: dt.date, exchanges: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {ex: [] for ex in exchanges}
    IT = _norm_str(instrumenttype)
    ex_set = set(exchanges)
    for r in data:
        if (
            _name_match(name, r.get("name"))
            and r.get("exch_seg") in ex_set
            and r.get("instrumenttype") == IT
            and _same_expiry_row(r, expiry_date)
        ):
            sym = r.get("symbol")
            tok = r.get("token")
            if sym and tok:
                out[r["exch_seg"]].append({"symbol": sym, "strike": r.get("strike"), "token": tok})
    return out

def _nearest_index(values: List[float], x: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - x)) if values else 0

def _select_window_strikes(strikes_sorted: List[float], center_val: float, half_width: int) -> List[float]:
    if not strikes_sorted:
        return []
    i = _nearest_index(strikes_sorted, center_val)
    lo = max(0, i - half_width)
    hi = min(len(strikes_sorted), i + half_width + 1)
    return strikes_sorted[lo:hi]

def _select_exact_strikes(strikes_sorted: List[float], center_val: float, count: int) -> List[float]:
    """
    Pick exactly `count` unique strikes nearest to `center_val` (spot).
    Expands symmetrically left/right from the nearest strike.
    """
    n = len(strikes_sorted)
    if n == 0 or count <= 0:
        return []
    if count >= n:
        return strikes_sorted[:]
    center_idx = _nearest_index(strikes_sorted, center_val)
    left = center_idx - 1
    right = center_idx + 1
    chosen = [strikes_sorted[center_idx]]
    while len(chosen) < count and (left >= 0 or right < n):
        left_dist = abs(center_val - strikes_sorted[left]) if left >= 0 else float("inf")
        right_dist = abs(center_val - strikes_sorted[right]) if right < n else float("inf")
        if left_dist <= right_dist:
            if left >= 0:
                chosen.append(strikes_sorted[left]); left -= 1
            elif right < n:
                chosen.append(strikes_sorted[right]); right += 1
        else:
            if right < n:
                chosen.append(strikes_sorted[right]); right += 1
            elif left >= 0:
                chosen.append(strikes_sorted[left]); left -= 1
    chosen.sort()
    return chosen

def _estimate_spot_from_strikes(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Median of available strikes as a crude last-resort spot."""
    strikes = []
    for r in rows:
        s = _human_strike(r.get("strike"))
        if s is not None:
            try:
                strikes.append(float(s))
            except Exception:
                pass
    if not strikes:
        return None
    strikes.sort()
    n = len(strikes)
    if n % 2 == 1:
        return strikes[n // 2]
    return (strikes[n // 2 - 1] + strikes[n // 2]) / 2.0

# -------- Index discovery (for SENSEX/BANKEX etc.) --------
def _find_index_meta(index_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Return (exchange, token, trading_symbol) for an index.
    Priority: try NSE first for NIFTY-family, else BSE for SENSEX/BANKEX.
    Scans Scrip Master to avoid brittle hardcoding.
    """
    data = load_scrip_master()
    name = _norm_str(index_name)

    # If we already have a known NSE token (NIFTY family), use NSE path
    if name in INDEX_TOKENS:
        token = INDEX_TOKENS[name]
        return ("NSE", token, INDEX_TRADING_SYMBOL.get(name, name))

    # Otherwise discover (BSE-friendly for SENSEX/BANKEX)
    # Look for rows with exch_seg in NSE/BSE, instrumenttype likely "INDEX" (be lenient),
    # and name/symbol matching our aliases.
    best: Optional[Tuple[str, str, str]] = None
    for r in data:
        exch = r.get("exch_seg")
        if exch not in {"NSE", "BSE"}:
            continue
        nm = r.get("name")
        sym = r.get("symbol")
        tok = r.get("token")
        if not tok:
            continue
        if _name_match(name, nm) or _name_match(name, sym):
            # Prefer BSE for SENSEX/BANKEX, otherwise first match
            t_sym = INDEX_TRADING_SYMBOL.get(name, nm or sym or name)
            cand = (exch, str(tok), t_sym)
            if name in {"SENSEX", "BANKEX"}:
                if exch == "BSE":
                    return cand
                # else keep as fallback if no BSE found
                if best is None:
                    best = cand
            else:
                # for other indices, return first match
                return cand
    return best

# -------- Build option chain (exact N strikes around FINAL spot) --------
def build_option_chain(
    index_name: str,
    expiry_str: str,
    exchanges: List[str],
    instrumenttype: str = "OPTIDX",
    half_width: int = 20,
    count: Optional[int] = 20,  # TOTAL strikes to return (default 20)
) -> Dict[str, Any]:
    expiry_date = _parse_expiry_input(expiry_str)
    data = load_scrip_master()

    rows_by_exch = _collect_rows_for_exchanges(data, index_name, instrumenttype, expiry_date, exchanges)
    if all(not rows_by_exch[ex] for ex in exchanges):
        return {
            "name": _norm_str(index_name),
            "instrumenttype": _norm_str(instrumenttype),
            "expiry": expiry_str,
            "exchanges": exchanges,
            "spot": None,
            "rows": [],
        }

    # Prepare per exchange
    prepared_by_exch: Dict[str, List[Dict[str, Any]]] = {ex: [] for ex in exchanges}
    for ex in exchanges:
        for r in rows_by_exch.get(ex, []):
            opt = _extract_option_type(r["symbol"])
            if opt is None:
                continue
            strike = _human_strike(r.get("strike"))
            if strike is None:
                m = re.search(r"(\d{4,6})[CP]E?$", r["symbol"])
                if m:
                    strike = _human_strike(m.group(1))
            if strike is None:
                continue
            prepared_by_exch[ex].append({
                "exchange": ex,
                "symbol": r["symbol"],
                "token": str(r["token"]),
                "option_type": opt,
                "strike": float(strike),
            })

    # Unified strike universe and pre-window (to limit fetch)
    all_strikes = sorted({p["strike"] for ex in exchanges for p in prepared_by_exch[ex]})
    spot_hint = _fast_spot_any(index_name)  # may be None for SENSEX/BANKEX
    temp_center = spot_hint if spot_hint is not None else (all_strikes[len(all_strikes)//2] if all_strikes else 0.0)
    pre_half_width = max(half_width + 10, 30)
    pre_keep = set(_select_window_strikes(all_strikes, temp_center, half_width=pre_half_width))

    # Filter per exchange by pre-window
    filtered_by_exch: Dict[str, List[Dict[str, Any]]] = {ex: [] for ex in exchanges}
    for ex in exchanges:
        filtered_by_exch[ex] = [p for p in prepared_by_exch[ex] if p["strike"] in pre_keep]
        filtered_by_exch[ex].sort(key=lambda x: (x["strike"], x["option_type"]))

    # Fetch market data per exchange
    token_req: Dict[str, List[str]] = {ex: [p["token"] for p in filtered_by_exch[ex]] for ex in exchanges}
    book_by_exch = fetch_full_marketdata_by_exchange(token_req)

    # Fallback per token via quote()
    for ex in exchanges:
        token_to_symbol = {p["token"]: p["symbol"] for p in filtered_by_exch[ex]}
        for t in token_req[ex]:
            rec = book_by_exch.get(ex, {}).get(t) or {}
            if rec.get("ltp") is None and rec.get("oi") is None and rec.get("volume") is None:
                sym = token_to_symbol.get(t)
                if sym:
                    book_by_exch[ex][t] = fetch_quote_single(ex, sym, t)

    # Compute FINAL spot: avg underlying across exchanges -> REST index (NSE/BSE) -> hint
    spot_candidates = []
    for ex in exchanges:
        for v in (book_by_exch.get(ex, {}) or {}).values():
            if v and v.get("underlying") is not None:
                fv = _fnum(v.get("underlying"))
                if fv is not None:
                    spot_candidates.append(fv)

    spot: Optional[float] = None
    if spot_candidates:
        try:
            spot = sum(spot_candidates) / len(spot_candidates)
        except Exception:
            spot = None

    if spot is None:
        # Try REST LTP for known/discovered index meta
        meta = _find_index_meta(index_name)
        if meta:
            ex_meta, tok_meta, _ = meta
            rest_spot = _rest_quote_ltp(ex_meta, [str(tok_meta)])
            if rest_spot is not None:
                spot = rest_spot

    if spot is None:
        spot = spot_hint

    # Merge CE/PE into per-exchange buckets by strike
    chain_map: Dict[float, Dict[str, Any]] = {}
    for ex in exchanges:
        for p in filtered_by_exch[ex]:
            strike = p["strike"]
            q = (book_by_exch.get(ex, {}) or {}).get(p["token"]) or {}
            ltp = _fnum(q.get("ltp"))
            vol = _inum(q.get("volume"))
            oi = _inum(q.get("oi"))
            if strike not in chain_map:
                chain_map[strike] = {"strike": strike}
            if ex not in chain_map[strike]:
                chain_map[strike][ex] = {
                    "ce": {"ltp": 0.0, "oi": 0, "volume": 0},
                    "pe": {"ltp": 0.0, "oi": 0, "volume": 0},
                }
            side = p["option_type"].lower()  # 'ce' or 'pe'
            chain_map[strike][ex][side] = {
                "ltp": float(ltp) if ltp is not None else 0.0,
                "oi": int(oi) if oi is not None else 0,
                "volume": int(vol) if vol is not None else 0,
            }

    out_rows = [chain_map[k] for k in sorted(chain_map.keys())]

    # Final exact-N window around FINAL spot
    strikes_sorted = sorted([r["strike"] for r in out_rows])
    center_for_final = spot if spot is not None else temp_center
    final_count = int(count or 20)
    final_strikes = _select_exact_strikes(strikes_sorted, center_for_final, final_count)
    if final_strikes:
        keep = set(final_strikes)
        out_rows = [r for r in out_rows if r["strike"] in keep]

    return {
        "name": _norm_str(index_name),
        "instrumenttype": _norm_str(instrumenttype),
        "expiry": expiry_str,
        "exchanges": exchanges,
        "spot": float(spot) if spot is not None else None,
        "rows": out_rows,
    }

# -------- Spot helpers --------
def _fast_spot_any(index_name: str) -> Optional[float]:
    """Try fast SDK ltpData for index if we can determine exchange/token quickly."""
    sc = login()
    sym = _norm_str(index_name)

    # Known NSE tokens (NIFTY family)
    tok = INDEX_TOKENS.get(sym)
    if tok:
        try:
            t_sym = INDEX_TRADING_SYMBOL.get(sym, sym)
            q = sc.ltpData("NSE", t_sym, str(tok))
            d = (q or {}).get("data") or {}
            return _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
        except Exception:
            pass

    # Discover (works for SENSEX/BANKEX)
    meta = _find_index_meta(sym)
    if not meta:
        return None
    ex_meta, tok_meta, t_sym = meta
    try:
        q = sc.ltpData(ex_meta, t_sym, str(tok_meta))
        d = (q or {}).get("data") or {}
        return _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
    except Exception:
        return None

def get_spot_price(symbol: str) -> Optional[float]:
    """Best-effort spot for NSE/BSE indices with robust fallbacks."""
    login()
    sym = _norm_str(symbol)

    # 1) REST LTP using known/discovered meta (handles SENSEX/BANKEX via BSE)
    meta = _find_index_meta(sym)
    if meta:
        ex_meta, tok_meta, _ = meta
        rest = _rest_quote_ltp(ex_meta, [str(tok_meta)])
        if rest is not None:
            return rest

    # 2) SDK ltpData fallback if we have meta
    if meta:
        ex_meta, tok_meta, t_sym = meta
        try:
            q = _sc.ltpData(ex_meta, t_sym, str(tok_meta))
            d = (q or {}).get("data") or {}
            ltp = _fnum(_pick(d, "ltp", "lastPrice", "lastTradedPrice"))
            if ltp is not None:
                return ltp
        except Exception:
            pass

    # 3) Derive from nearest options' underlying across both exchanges
    try:
        data = load_scrip_master()
        today = dt.date.today()
        exps = []
        for r in data:
            if _name_match(sym, r.get("name")) and r.get("exch_seg") in SUPPORTED_EXCH and r.get("instrumenttype") == "OPTIDX":
                e = r.get("expiry") or _extract_expiry_from_symbol(r.get("symbol"))
                if isinstance(e, dt.date) and e >= today:
                    exps.append(e)
        exps = sorted(set(exps))
        if not exps:
            return None
        nearest = exps[0]

        tokens_by_ex: Dict[str, List[str]] = {"NFO": [], "BFO": []}
        rows_for_nearest = []
        for r in data:
            if (
                _name_match(sym, r.get("name"))
                and r.get("exch_seg") in SUPPORTED_EXCH
                and r.get("instrumenttype") == "OPTIDX"
                and _same_expiry_row(r, nearest)
            ):
                rows_for_nearest.append(r)
                if r.get("token"):
                    ex = r["exch_seg"]
                    if len(tokens_by_ex[ex]) < 60:
                        tokens_by_ex[ex].append(str(r["token"]))

        book_by_ex = fetch_full_marketdata_by_exchange(tokens_by_ex)
        cand = []
        for ex in tokens_by_ex.keys():
            for v in (book_by_ex.get(ex, {}) or {}).values():
                if v and v.get("underlying") is not None:
                    fv = _fnum(v.get("underlying"))
                    if fv is not None:
                        cand.append(fv)
        if cand:
            return sum(cand) / len(cand)

        # Last resort: estimate from strikes (median)
        est = _estimate_spot_from_strikes(rows_for_nearest)
        if est is not None:
            return est
    except Exception:
        pass

    return None

# -------- HTTP --------
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
    exchanges = _parse_exchanges_param(request.args.get("exchange"))
    limit = request.args.get("limit", type=int, default=4)
    try:
        dates = list_expiries(index, instrumenttype=it, exchanges=exchanges, limit=limit)
        return jsonify({
            "success": True,
            "index": _norm_str(index),
            "instrumenttype": _norm_str(it),
            "exchanges": exchanges,
            "limit": limit,
            "expiry_dates": dates
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/option_chain", methods=["GET"])
def http_chain():
    symbol = request.args.get("symbol")
    expiry = request.args.get("expiry")
    it = request.args.get("instrumenttype", "OPTIDX")
    exchanges = _parse_exchanges_param(request.args.get("exchange"))
    count = request.args.get("count", type=int, default=20)  # exact total strikes to return
    width = request.args.get("width", type=int, default=20)  # used only for pre-window sizing

    if not symbol or not expiry:
        return jsonify({"success": False, "error": "query params 'symbol' and 'expiry' are required (expiry as DDMMMYYYY)"}), 400
    try:
        data = build_option_chain(
            symbol, expiry, exchanges=exchanges, instrumenttype=it, half_width=width, count=count
        )
        return jsonify({
            "success": True,
            "params": {
                "symbol": _norm_str(symbol),
                "expiry": expiry,
                "instrumenttype": _norm_str(it),
                "exchanges": exchanges,
                "count": count,
                "width": width
            },
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

