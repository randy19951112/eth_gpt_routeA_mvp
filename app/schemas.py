from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class BacktestParams(BaseModel):
    fast: int = 12
    slow: int = 26
    fee: float = 0.001  # 0.1%

class BacktestRequest(BaseModel):
    symbol: str = "ETH/USDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    start: Optional[str] = None  # ISO8601
    end: Optional[str] = None
    params: BacktestParams = BacktestParams()

class BacktestMetrics(BaseModel):
    total_return: float
    cagr: float
    sharpe: float
    mdd: float
    trades: int
    win_rate: float

class BacktestResponse(BaseModel):
    metrics: BacktestMetrics
    equity_curve: Optional[List[float]] = None
    signals: Optional[List[int]] = None

class LiveSignalRequest(BaseModel):
    symbol: str = "ETH/USDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    lookback_bars: int = 400
    params: BacktestParams = BacktestParams()

class LiveSignalResponse(BaseModel):
    action: str
    confidence: float
    evidence: Dict[str, Any]

class AdviseRequest(BaseModel):
    user_id: str = "default_user"
    goal: str = Field("swing", description="intraday/swing/position")
    risk_level: str = Field("medium", description="low/medium/high")
    horizon_days: int = 7
    capital: Optional[float] = None
    symbol: str = "ETH/USDT"
    exchange: str = "binance"
    timeframe: str = "1h"

class AdviseResponse(BaseModel):
    summary: str
    action: str
    confidence: float
    evidence: Dict[str, Any]
    risk_disclaimer: str

class MemoryUpsertRequest(BaseModel):
    user_id: str
    key: str
    value: Dict[str, Any]

class MemoryQueryRequest(BaseModel):
    user_id: str
    key: str

class MemoryQueryResponse(BaseModel):
    user_id: str
    key: str
    value: Optional[Dict[str, Any]] = None


class KnowledgeUpsertRequest(BaseModel):
    user_id: str
    title: str
    content: str
    tags: Optional[List[str]] = None
    active: bool = True

class KnowledgeSearchRequest(BaseModel):
    user_id: str
    query: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 5

class KnowledgeItem(BaseModel):
    id: int
    title: str
    content: str
    tags: List[str] = []
    created_at: str

class KnowledgeSearchResponse(BaseModel):
    items: List[KnowledgeItem]
