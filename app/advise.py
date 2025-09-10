from typing import Dict, Any, Optional
from .data import MarketDataProvider
from .backtest import backtest_ema_long_only
from .knowledge import search_knowledge
from .indicators import atr, rolling_high, rolling_low

def _risk_to_position_pct(risk_level: str) -> float:
    mapping = {"low": 0.2, "medium": 0.5, "high": 1.0}
    return mapping.get(risk_level.lower(), 0.5)

def _position_size(capital: Optional[float], pos_pct: float, entry: float, stop: float) -> Optional[float]:
    if not capital or not entry or not stop:
        return None
    # MVP: 直接用資金 * 部位% / 進場價 當建議數量
    qty = (capital * pos_pct) / entry
    return float(qty)

def make_live_advice(user_id: str, symbol: str, exchange: str, timeframe: str, risk_level: str, horizon_days: int, params: Dict[str, Any], capital: Optional[float]=None) -> Dict[str, Any]:
    mdp = MarketDataProvider(exchange_id=exchange)
    df = mdp.fetch_ohlcv(symbol, timeframe=timeframe, limit=800)
    res = backtest_ema_long_only(df, fast=params.get("fast",12), slow=params.get("slow",26), fee=params.get("fee",0.001), timeframe=timeframe)

    ema_fast_last = res["ema_fast"][-1]
    ema_slow_last = res["ema_slow"][-1]
    close_last = float(df['close'].iloc[-1])

    a = atr(df, 14).iloc[-1]
    if not a or a != a:
        a = max(0.001 * close_last, 1.0)

    diff = (ema_fast_last - ema_slow_last) / ema_slow_last if ema_slow_last else 0.0
    action = "HOLD"
    confidence = min(max(abs(diff) * 20, 0), 1.0)
    if ema_fast_last > ema_slow_last:
        action = "BUY"
    elif ema_fast_last < ema_slow_last:
        action = "SELL"

    rr = 1.5
    if action == "BUY":
        entry = close_last
        stop = min(ema_slow_last, entry - 1.5 * a)
        take = entry + rr * (entry - stop)
    else:
        recent_high = float(rolling_high(df['high'], 20).iloc[-2]) if len(df) > 20 else close_last
        entry = float(max(ema_fast_last, recent_high))
        stop = float(ema_slow_last) if ema_slow_last else entry - 2 * a
        take = entry + rr * (entry - stop)

    pos_pct = _risk_to_position_pct(risk_level)
    qty = _position_size(capital, pos_pct, entry, stop)

    notes = search_knowledge(user_id=user_id, query=None, tags=None, limit=5)

    evid = {
        "timeframe": timeframe,
        "close": close_last,
        "ema_fast": ema_fast_last,
        "ema_slow": ema_slow_last,
        "ema_diff_ratio": diff,
        "atr": float(a),
        "suggested_entry": float(entry),
        "suggested_stop": float(stop),
        "suggested_take": float(take),
        "position_pct": pos_pct,
        "suggested_qty": qty,
        "knowledge_snippets": notes
    }
    summary = f"{symbol} 在 {timeframe} 週期目前為 {action}，建議進場價 {entry:.2f}、停損 {stop:.2f}、獲利目標 {take:.2f}；建議部位 {int(pos_pct*100)}% ，信心 {confidence:.2f}。"
    disclaimer = "僅供教育研究，非投資建議；加密資產風險極高，請自行評估。"
    return {
        "summary": summary,
        "action": action,
        "confidence": confidence,
        "evidence": evid,
        "risk_disclaimer": disclaimer
    }
