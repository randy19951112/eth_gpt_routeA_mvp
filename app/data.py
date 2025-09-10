import ccxt
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440
}

def _parse_iso_to_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z","+00:00"))
    return int(dt.timestamp() * 1000)

class MarketDataProvider:
    def __init__(self, exchange_id: str = "binance"):
        ex_cls = getattr(ccxt, exchange_id)
        self.exchange = ex_cls({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: Optional[str]=None, end: Optional[str]=None, limit: int=1500) -> pd.DataFrame:
        since = _parse_iso_to_ms(start)
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            raise RuntimeError("No OHLCV returned; check symbol/timeframe.")
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        return df
