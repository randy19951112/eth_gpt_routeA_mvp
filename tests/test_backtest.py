import pandas as pd
import numpy as np
from app.backtest import backtest_ema_long_only

def test_backtest_basic():
    idx = pd.date_range("2023-01-01", periods=500, freq="H")
    price = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    df = pd.DataFrame({"close": price, "open": price, "high": price*1.01, "low": price*0.99, "volume": 1000})
    res = backtest_ema_long_only(df, fast=12, slow=26, fee=0.0, timeframe="1h")
    assert "metrics" in res and res["metrics"]["total_return"] > 0
