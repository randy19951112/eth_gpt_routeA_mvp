import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .indicators import ema, crossover, crossunder, max_drawdown

BARS_PER_YEAR_BY_TF = {
    "1m": 525600, "3m": 175200, "5m": 105120, "15m": 35040, "30m": 17520,
    "1h": 8760, "2h": 4380, "4h": 2190, "6h": 1460, "12h": 730,
    "1d": 365
}

def backtest_ema_long_only(df: pd.DataFrame, fast: int=12, slow: int=26, fee: float=0.001, timeframe: str="1h") -> Dict[str, Any]:
    close = df["close"].astype(float).copy()
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    long_signal = (ema_fast > ema_slow).astype(int)
    position = long_signal.shift(1).fillna(0)

    ret = close.pct_change().fillna(0)
    strat_ret = ret * position

    pos_change = position.diff().abs().fillna(position)
    fee_series = pos_change * fee
    strat_ret_after_fee = strat_ret - fee_series

    equity = (1 + strat_ret_after_fee).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)
    bars_per_year = BARS_PER_YEAR_BY_TF.get(timeframe, 8760)
    n = len(equity)
    cagr = float((equity.iloc[-1]) ** (bars_per_year / max(n,1)) - 1.0) if n > 0 else 0.0
    sharpe = 0.0
    std = float(strat_ret_after_fee.std())
    if std > 0:
        sharpe = float(strat_ret_after_fee.mean() / std * np.sqrt(bars_per_year))
    mdd = float(max_drawdown(equity))

    entries = (crossover(ema_fast, ema_slow)).astype(int)
    exits = (crossunder(ema_fast, ema_slow)).astype(int)
    trade_indices = entries[entries==1].index
    wins = 0
    trades = 0
    last_entry_price = None
    for ts in df.index:
        if ts in trade_indices:
            last_entry_price = close.loc[ts]
            trades += 1
        if last_entry_price is not None and crossunder(ema_fast, ema_slow).astype(int).shift(0).reindex([ts]).fillna(0).iloc[0]==1:
            pnl = (close.loc[ts] / last_entry_price) - 1.0 - 2*fee
            if pnl > 0:
                wins += 1
            last_entry_price = None
    win_rate = float(wins / trades) if trades>0 else 0.0

    return {
        "metrics": {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "mdd": mdd,
            "trades": int(trades),
            "win_rate": win_rate
        },
        "equity_curve": equity.tolist(),
        "signals": position.astype(int).tolist(),
        "ema_fast": ema_fast.tolist(),
        "ema_slow": ema_slow.tolist()
    }
