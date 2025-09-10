from __future__ import annotations
from typing import Optional, List, Dict
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, Session

from .config import settings
from .data import MarketDataProvider

Base = declarative_base()

class AdviceLog(Base):
    __tablename__ = "advice_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    symbol = Column(String, nullable=False)
    exchange = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    risk_level = Column(String, nullable=False)
    horizon_days = Column(Integer, nullable=False)

    action = Column(String, nullable=False)          # BUY / SELL / HOLD
    entry = Column(Float, nullable=False)            # 建議進場
    stop = Column(Float, nullable=False)
    take = Column(Float, nullable=False)
    position_pct = Column(Float, nullable=False)
    qty = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # 評估結果
    evaluated = Column(Boolean, default=False)
    triggered = Column(Boolean, nullable=True)       # 是否有觸發進場
    exit_reason = Column(String, nullable=True)      # TP / SL / HORIZON / OPEN / NOT_TRIGGERED
    exit_price = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    evaluated_at = Column(DateTime(timezone=True), nullable=True)

engine = create_engine(settings.db_url, echo=False, future=True)
Base.metadata.create_all(engine)

def save_log(payload: Dict) -> int:
    with Session(engine) as sess:
        log = AdviceLog(**payload)
        sess.add(log)
        sess.commit()
        return log.id

def list_recent_logs(user_id: str, limit: int = 5) -> List[AdviceLog]:
    with Session(engine) as sess:
        q = sess.query(AdviceLog).filter(AdviceLog.user_id == user_id).order_by(AdviceLog.created_at.desc()).limit(limit)
        return q.all()

def evaluate_one(log: AdviceLog) -> AdviceLog:
    """
    以 timeframe 推估需要的根數來評估，不再依賴 time 欄位或 start 參數。
    規則：先觸發 entry，之後先到 TP 或 SL；若整段沒結論則判定 NOT_TRIGGERED/OPEN。
    """
    mdp = MarketDataProvider(exchange_id=log.exchange)

    tf_map = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
              "1h": 60, "2h": 120, "4h": 240, "6h": 360,
              "12h": 720, "1d": 1440}
    tf_minutes = tf_map.get(log.timeframe, 60)

    import math
    bars_needed = int(math.ceil(log.horizon_days * (1440 / tf_minutes)) + 200)
    bars_needed = max(300, min(bars_needed, 2000))

    df = mdp.fetch_ohlcv(log.symbol, timeframe=log.timeframe, limit=bars_needed)
    if df is None or len(df) == 0:
        with Session(engine) as sess:
            rec = sess.query(AdviceLog).get(log.id)
            rec.evaluated = True
            rec.triggered = False
            rec.exit_reason = "OPEN"
            rec.exit_price = None
            rec.pnl_pct = None
            rec.evaluated_at = datetime.now(timezone.utc)
            sess.commit()
            sess.refresh(rec)
            return rec

    df = df.reset_index(drop=True)

    triggered = False
    exit_reason = None
    exit_price = None

    for _, row in df.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if not triggered:
            if high >= log.entry:
                triggered = True
                if high >= log.take:
                    exit_reason = "TP"
                    exit_price = log.take
                    break
                if low <= log.stop:
                    exit_reason = "SL"
                    exit_price = log.stop
                    break
        else:
            if high >= log.take:
                exit_reason = "TP"
                exit_price = log.take
                break
            if low <= log.stop:
                exit_reason = "SL"
                exit_price = log.stop
                break

    if exit_reason is None:
        if not triggered:
            exit_reason = "NOT_TRIGGERED"
            exit_price = None
        else:
            exit_reason = "OPEN"
            exit_price = float(df["close"].iloc[-1])

    pnl_pct = None
    if triggered and exit_price is not None and log.entry:
        pnl_pct = (exit_price - log.entry) / log.entry * 100.0

    with Session(engine) as sess:
        rec = sess.query(AdviceLog).get(log.id)
        rec.evaluated = True
        rec.triggered = triggered
        rec.exit_reason = exit_reason
        rec.exit_price = exit_price
        rec.pnl_pct = pnl_pct
        rec.evaluated_at = datetime.now(timezone.utc)
        sess.commit()
        sess.refresh(rec)
        return rec
from datetime import timedelta

def compute_review_stats(
    user_id: str,
    lookback_days: int | None = None,
    limit: int | None = 200,
    symbol: str | None = None,
    timeframe: str | None = None,
):
    now = datetime.now(timezone.utc)
    with Session(engine) as sess:
        q = sess.query(AdviceLog).filter(AdviceLog.user_id == user_id)
        if symbol:
            q = q.filter(AdviceLog.symbol == symbol)
        if timeframe:
            q = q.filter(AdviceLog.timeframe == timeframe)
        if lookback_days:
            q = q.filter(AdviceLog.created_at >= now - timedelta(days=int(lookback_days)))
        q = q.order_by(AdviceLog.created_at.desc())
        if limit:
            q = q.limit(int(limit))
        rows = q.all()

    total = len(rows)
    triggered = [r for r in rows if r.triggered]
    untriggered = [r for r in rows if not r.triggered]
    wins = [r for r in triggered if r.exit_reason == "TP"]
    losses = [r for r in triggered if r.exit_reason == "SL"]
    open_trades = [r for r in triggered if r.exit_reason == "OPEN"]

    # 只統計有出場價或能計算損益的
    pnl_list = [float(r.pnl_pct) for r in triggered if r.pnl_pct is not None]
    avg_pnl = sum(pnl_list) / len(pnl_list) if pnl_list else None
    med_pnl = sorted(pnl_list)[len(pnl_list)//2] if pnl_list else None

    best = max(triggered, key=lambda r: (r.pnl_pct if r.pnl_pct is not None else -1e9), default=None)
    worst = min(triggered, key=lambda r: (r.pnl_pct if r.pnl_pct is not None else 1e9), default=None)

    # 估算平均R（以 (entry-stop)/entry 的百分比當 1R）
    def r_multiple(r):
        if r.pnl_pct is None or r.entry is None or r.stop is None:
            return None
        r_pct = (r.entry - r.stop) / r.entry * 100.0 if r.entry else None
        if not r_pct or r_pct == 0:
            return None
        return r.pnl_pct / r_pct

    r_list = [r_multiple(r) for r in triggered]
    r_list = [x for x in r_list if x is not None]
    avg_r = sum(r_list)/len(r_list) if r_list else None

    win_rate = (len(wins) / (len(wins) + len(losses))) if (len(wins)+len(losses)) > 0 else None

    out = {
        "filters": {
            "user_id": user_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback_days": lookback_days,
            "limit": limit
        },
        "counts": {
            "total_logs": total,
            "triggered": len(triggered),
            "untriggered": len(untriggered),
            "wins": len(wins),
            "losses": len(losses),
            "open": len(open_trades)
        },
        "performance": {
            "win_rate": win_rate,            # 0~1 之間
            "avg_pnl_pct": avg_pnl,          # 平均報酬（%）
            "median_pnl_pct": med_pnl,       # 中位數報酬（%）
            "avg_R": avg_r                   # 平均R倍數
        },
        "best_trade": ({
            "id": best.id,
            "created_at": best.created_at.isoformat() if best and best.created_at else None,
            "pnl_pct": best.pnl_pct,
            "entry": best.entry, "stop": best.stop, "take": best.take
        } if best else None),
        "worst_trade": ({
            "id": worst.id,
            "created_at": worst.created_at.isoformat() if worst and worst.created_at else None,
            "pnl_pct": worst.pnl_pct,
            "entry": worst.entry, "stop": worst.stop, "take": worst.take
        } if worst else None)
    }
    return out

