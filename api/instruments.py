import json
import os
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import certifi
import requests
from dateutil import parser as dateparser
from requests.exceptions import SSLError, ProxyError, ConnectionError, Timeout
from api.config import INSTRUMENTS_URL, CACHE_TTL

# --------------- Thread-safe in-memory cache ----------------
_lock = threading.Lock()
_cached = {
    "fetched_at": None,   # datetime
    "data": None          # list[dict]
}

# Local file cache (serves stale data if network/SSL fails)
CACHE_FILE = Path("cache_OpenAPIScripMaster.json")

# --------------- SSL/Proxy knobs via env --------------------
CA_BUNDLE_PATH = os.getenv("CA_BUNDLE_PATH")  # e.g., C:\\certs\\corp-root-bundle.pem
INSECURE_SSL   = os.getenv("INSECURE_SSL", "false").lower() == "true"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
USER_AGENT = os.getenv("USER_AGENT", "angel-expiries/1.0 (+https://github.com/your-org/angel-expiries)")

def _parse_expiry_date(raw: str):
    """
    Parse Angel One 'expiry' variants into ISO date (YYYY-MM-DD).
    Seen formats: '27MAY2025', '27 MAY 2025', '2025-05-27', etc.
    Returns None if unparsable.
    """
    if not raw:
        return None
    s = raw.strip()
    # Normalize '27MAY2025' => '27 MAY 2025'
    m = re.match(r"^(\d{1,2})([A-Z]{3})(\d{4})$", s)
    if m:
        s = f"{m.group(1)} {m.group(2)} {m.group(3)}"
    try:
        dt = dateparser.parse(s, dayfirst=True, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return None

def _get_verify_param():
    """
    Determine what Requests should use for certificate verification.
    Priority:
    1) INSECURE_SSL=true  -> verify=False  (diagnostics only)
    2) CA_BUNDLE_PATH set -> verify=that file (corp proxy/SSL inspection)
    3) Default            -> verify=certifi.where() (Mozilla CA bundle)
    """
    if INSECURE_SSL:
        return False
    if CA_BUNDLE_PATH and Path(CA_BUNDLE_PATH).exists():
        return CA_BUNDLE_PATH
    return certifi.where()

def get_fetch_env_info():
    """Return safe diagnostics of SSL/proxy setup (no secrets)."""
    verify = _get_verify_param()
    return {
        "insecure_ssl": INSECURE_SSL,
        "verify": verify if verify is not False else "False (INSECURE)",
        "timeout_sec": REQUEST_TIMEOUT,
        "instruments_url": INSTRUMENTS_URL,
        "cache_file": str(CACHE_FILE.resolve()),
    }

def _try_save_cache(data: list):
    try:
        CACHE_FILE.write_text(json.dumps(data))
    except Exception:
        pass  # don't fail the request due to cache write

def _load_cache_if_exists():
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text())
    except Exception:
        return None

def _fetch_instruments_from_network():
    verify = _get_verify_param()
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(
        INSTRUMENTS_URL,
        timeout=REQUEST_TIMEOUT,
        verify=verify,
        headers=headers
    )
    resp.raise_for_status()
    return resp.json()

def _fetch_instruments():
    """
    Fetch instrument master with SSL hardening and cache fallback:
      - If network fails, serve last good snapshot (if available).
    """
    try:
        data = _fetch_instruments_from_network()
        _try_save_cache(data)
        return data
    except (SSLError, ProxyError, ConnectionError, Timeout) as e:
        # Serve stale cache if present
        cached = _load_cache_if_exists()
        if cached:
            return cached
        # Otherwise, raise helpful error
        raise RuntimeError(
            "Failed to download Angel One instrument master over HTTPS. "
            "If you're behind a corporate proxy with SSL inspection, export the proxy's "
            "root certificate (PEM) and set CA_BUNDLE_PATH or REQUESTS_CA_BUNDLE. "
            "You may also need HTTPS_PROXY/HTTP_PROXY environment variables."
        ) from e

def get_instruments(force: bool = False):
    now = datetime.now(timezone.utc)
    with _lock:
        if (
            not force
            and _cached["data"]
            and _cached["fetched_at"]
            and (now - _cached["fetched_at"]) < CACHE_TTL
        ):
            return _cached["data"], _cached["fetched_at"].isoformat()
        data = _fetch_instruments()
        _cached["data"] = data
        _cached["fetched_at"] = now
        return data, now.isoformat()

def compute_index_expiries(
    underlyings=None,
    exchanges=("NFO", "BFO"),
    include_expired=False,
):
    """
    Returns:
    {
      "source": <url>,
      "fetched_at": <iso datetime>,
      "expiries": { "NIFTY": ["YYYY-MM-DD", ...], ... }
    }
    """
    instruments, fetched_at = get_instruments()
    today_iso = datetime.now().date().isoformat()
    groups = defaultdict(set)
    for item in instruments:
        try:
            exch = (item.get("exch_seg") or item.get("exchange") or "").upper()
            if exch not in exchanges:
                continue
            inst_type = (item.get("instrumenttype") or "").upper()
            # Options only; OPTIDX is index options (OPTSTK is stock options).
            if not inst_type.startswith("OPT"):
                continue
            # Underlying "name" hosts NIFTY/BANKNIFTY/FINNIFTY/etc. in instrument file.
            underlying = (item.get("name") or "").upper().strip()
            if not underlying:
                continue
            if underlyings and underlying not in underlyings:
                continue
            expiry_raw = (item.get("expiry") or "").strip()
            expiry_iso = _parse_expiry_date(expiry_raw)
            if not expiry_iso:
                continue
            if not include_expired and expiry_iso < today_iso:
                continue
            groups[underlying].add(expiry_iso)
        except Exception:
            continue  # skip malformed rows
    return {
        "source": INSTRUMENTS_URL,   # community-known master location
        "fetched_at": fetched_at,
        "expiries": {k: sorted(v) for k, v in groups.items()},
    }

def refresh_cache():
    get_instruments(force=True)
