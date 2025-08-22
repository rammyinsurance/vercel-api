import os
from datetime import timedelta

# Public Angel One instrument master (community uses this URL frequently)
# If Angel One relocates it, override via env: INSTRUMENTS_URL
INSTRUMENTS_URL = os.getenv(
    "INSTRUMENTS_URL",
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
)

# Cache TTL (minutes) for re-downloading instrument file
CACHE_TTL = timedelta(minutes=int(os.getenv("CACHE_TTL_MIN", "15")))

# Default exchanges: NFO (NSE derivatives), BFO (BSE derivatives)
DEFAULT_EXCHANGES = os.getenv("DEFAULT_EXCHANGES", "NFO,BFO").upper().split(",")

# Underlyings to treat as "indices" by default
DEFAULT_INDEX_UNDERLYINGS = os.getenv(
    "DEFAULT_INDEX_UNDERLYINGS",
    "NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY,NIFTYNXT50,SENSEX,SENSEX50,BANKEX"
).upper().split(",")

# By default, exclude expired contracts
DEFAULT_INCLUDE_EXPIRED = os.getenv("DEFAULT_INCLUDE_EXPIRED", "false").lower() == "true"
