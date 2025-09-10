import pandas as pd
import numpy as np

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def crossover(series_fast: pd.Series, series_slow: pd.Series) -> pd.Series:
    cross = (series_fast > series_slow) & (series_fast.shift(1) <= series_slow.shift(1))
    return cross

def crossunder(series_fast: pd.Series, series_slow: pd.Series) -> pd.Series:
    cross = (series_fast < series_slow) & (series_fast.shift(1) >= series_slow.shift(1))
    return cross

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity/roll_max - 1.0
    return dd.min() if len(dd) else 0.0


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rolling_high(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).max()

def rolling_low(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).min()
