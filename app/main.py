## app/main.py — 完整可用版本（沿用既有模組）
from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, HttpUrl, conint

# 專案內模組（沿用你原本的專案）
from .schemas import (
    BacktestRequest, BacktestResponse, BacktestMetrics,
    LiveSignalRequest, LiveSignalResponse,
    AdviseRequest, AdviseResponse,
    MemoryUpsertRequest, MemoryQueryResponse,
    KnowledgeUpsertRequest, KnowledgeSearchRequest, KnowledgeSearchResponse, KnowledgeItem
)
from .data import MarketDataProvider
from .backtest import backtest_ema_long_only
from .memory import upsert_memory, query_memory
from .advise import make_live_advice
from .knowledge import upsert_knowledge, search_knowledge
from .uploader import extract_text_by_ext, chunk_text, fetch_bytes_from_url, ocr_image_bytes, ocr_pdf_bytes
from .logs import save_log, list_recent_logs, evaluate_one, compute_review_stats
from .config import settings
import ccxt
import ta  # replaced talib with ta
import numpy as np
import pandas as pd


exchange = ccxt.binance({"enableRateLimit": True})

   
# ================== [A] 取得 OHLCV 資料（含錯誤處理） ==================
def fetch_ohlcv(symbol="ETH/USDT", timeframe="15m", limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        # 回傳空的 DataFrame；由呼叫端判斷 df.empty 來給友善錯誤訊息
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ================== [B] 計算技術指標（沿用你的寫法） ==================
def calculate_indicators(df: pd.DataFrame):
    result = {}

    # === KDJ (以 Stochastic 近似) ===
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3
    )
    result["KDJ"] = {
        "K": float(stoch.stoch().iloc[-1]),
        "D": float(stoch.stoch_signal().iloc[-1]),
        "J": float(3 * stoch.stoch().iloc[-1] - 2 * stoch.stoch_signal().iloc[-1]),
    }

    # === MACD ===
    macd = ta.trend.MACD(close=df["close"])
    result["MACD"] = {
        "DIF": float(macd.macd().iloc[-1]),
        "DEA": float(macd.macd_signal().iloc[-1]),
        "hist": float(macd.macd_diff().iloc[-1]),
    }

      # === 布林帶 (BOLL) ===
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    boll = {
        "upper": float(bb.bollinger_hband().iloc[-1]),
        "middle": float(bb.bollinger_mavg().iloc[-1]),
        "lower": float(bb.bollinger_lband().iloc[-1]),
    }
    # 正式名稱
    result["BOLL"] = boll
    # 相容舊版（可用一段時間後移除）
    result["BB"] = boll


    # === 均線 (MA) ===
    result["MA"] = {
        "MA5": float(df["close"].rolling(5).mean().iloc[-1]),
        "MA20": float(df["close"].rolling(20).mean().iloc[-1]),
        "MA60": float(df["close"].rolling(60).mean().iloc[-1]) if len(df) >= 60 else None,
    }

    return result


# =========================
# 其他補充用 Schema（解 OpenAPI 驗證錯誤）
# =========================
class RootResponse(BaseModel):
    name: str = "ETH/USDT Research API (Route A)"
    version: str = "0.3.0"
    docs: str = "/docs"
    openapi: str = "/openapi.json"
    privacy: str = "/privacy"
    terms: str = "/terms"

class HealthResponse(BaseModel):
    status: str = "ok"
    time: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

class GenericOK(BaseModel):
    status: str = "ok"
    detail: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class KnowledgeUpsertOK(BaseModel):
    status: str = "ok"
    knowledge_id: int

class UploadResult(BaseModel):
    status: str = "ok"
    chunks_saved: int
    ids: List[int]

class IngestURLRequest(BaseModel):
    user_id: str
    url: HttpUrl
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    chunk_size: conint(ge=200, le=5000) = 1200
    overlap: conint(ge=0, le=500) = 120

class EvaluatedLogItem(BaseModel):
    id: int
    created_at: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    action: Optional[str] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    take: Optional[float] = None
    evaluated: Optional[bool] = None
    triggered: Optional[bool] = None
    exit_reason: Optional[str] = None  # TP / SL / OPEN / NOT_TRIGGERED
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None

class OneClickReview(BaseModel):
    evaluated_recent: List[EvaluatedLogItem]

class OneClickRequest(BaseModel):
    user_id: str
    symbol: str = "ETH/USDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    risk_level: str = "medium"
    horizon_days: conint(ge=1, le=90) = 7
    capital: Optional[float] = None
    review_last: conint(ge=1, le=200) = 5

class OneClickResponse(BaseModel):
    now_advice: AdviseResponse
    log_id: Optional[int] = None
    review: OneClickReview

class ReviewSummaryRequest(BaseModel):
    user_id: str
    lookback_days: Optional[int] = None
    limit: conint(ge=1, le=1000) = 200
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

class ReviewStats(BaseModel):
    win_rate: Optional[float] = None
    avg_return: Optional[float] = None
    median_return: Optional[float] = None
    avg_R: Optional[float] = None
    best_trade: Optional[Dict[str, Any]] = None
    worst_trade: Optional[Dict[str, Any]] = None
    filter: Optional[Dict[str, Any]] = None

class ReviewSummaryResponse(BaseModel):
    stats: ReviewStats


# =========================
# App & CORS
# =========================
app = FastAPI(
    title="ETH/USDT Research API (Route A)",
    version="0.3.0",
    description="Backtest / live signal / knowledge / memory / assistant API for ETH/USDT (Route A)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 需要更嚴格可改成你的前端網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API Key Middleware（白名單路徑免金鑰）
# =========================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
OPEN_PATHS = {
    "/", "/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico",
    "/privacy", "/terms", "/__routes"
}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if (path in OPEN_PATHS) or path.startswith("/static"):
        return await call_next(request)

    token_env = (getattr(settings, "api_token", "") or "").strip()
    # 未設定 API_TOKEN → 不啟用驗證（方便本地/測試）
    if not token_env:
        return await call_next(request)

    key = request.headers.get("X-API-Key")
    if key != token_env:
        return JSONResponse({"detail": "Missing or invalid API key"}, status_code=401)

    return await call_next(request)

# =========================
# OpenAPI customization：加入 securitySchemes 與 servers
# =========================
PUBLIC_BASE_URL = (getattr(settings, "public_base_url", "") or
                   "https://eth-gpt-routea-mvp.onrender.com").rstrip("/")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description=app.description,
    )
    # securitySchemes（讓 /docs 有 Authorize）
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["APIKeyHeader"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Paste the value of API_TOKEN (without the 'API_TOKEN=' prefix)",
    }
    schema["security"] = [{"APIKeyHeader": []}]
    # servers（避免「Could not find a valid URL in servers」）
    schema["servers"] = [{"url": PUBLIC_BASE_URL}]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

# =========================
# 小工具
# =========================
def _ensure_tags_to_list(tags: Optional[Any]) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    return [t.strip() for t in str(tags).split(",") if t.strip()]

def _force_binance(exchange_id: Optional[str]) -> str:
    # 為避免「不是幣安數據」，統一強制使用 binance
    return "binance"

# =========================
# 公開頁面：政策 / 條款（給 GPT 公開連結）
# =========================
PRIVACY_HTML = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>Privacy Policy - ETH/USDT Research API (Route A)</title>
<style>body{max-width:860px;margin:40px auto;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC","Helvetica Neue",Arial,sans-serif;padding:0 16px;color:#222}</style>
<h1>隱私權政策</h1>
<p>本服務由開發者自架於 <code>eth-gpt-routea-mvp.onrender.com</code>。當您透過 GPT 或直接呼叫 API 時，我們可能處理請求內容與必要技術記錄。</p>
<p>不販售個資，不作廣告追蹤；行情來源為 Binance（透過 CCXT）。</p>
<p>聯絡：<a href="mailto:support@example.com">support@example.com</a></p>
</html>"""

TERMS_HTML = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>Terms of Use - ETH/USDT Research API (Route A)</title>
<style>body{max-width:860px;margin:40px auto;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC","Helvetica Neue",Arial,sans-serif;padding:0 16px;color:#222}</style>
<h1>使用條款</h1>
<ul>
  <li>本服務僅供研究與教育，不構成投資建議。</li>
  <li>請遵守交易所與法域規範；盈虧自負。</li>
  <li>服務以「現狀」提供，不保證不中斷或完全準確。</li>
</ul>
<p>聯絡：<a href="mailto:support@example.com">support@example.com</a></p>
</html>"""

@app.get("/privacy", include_in_schema=False, response_class=HTMLResponse)
def privacy():
    return PRIVACY_HTML

@app.get("/terms", include_in_schema=False, response_class=HTMLResponse)
def terms():
    return TERMS_HTML

@app.get("/__routes", include_in_schema=False)
def list_routes():
    return [{"path": r.path, "name": getattr(r, "name", "")} for r in app.router.routes]

# =========================
# 基礎路由
# =========================
@app.get("/", response_model=RootResponse, summary="Root")
def root():
    return RootResponse(
        docs="/docs",
        openapi="/openapi.json",
        privacy="/privacy",
        terms="/terms",
    )

@app.get("/health", response_model=HealthResponse, summary="Health")
def health():
    return HealthResponse()

# =========================
# 訊號 & 回測（強制使用 binance）
# =========================
@app.post("/signals/backtest", response_model=BacktestResponse, summary="Post Backtest")
def post_backtest(req: BacktestRequest):
    ex = _force_binance(req.exchange)
    mdp = MarketDataProvider(exchange_id=ex)
    df = mdp.fetch_ohlcv(
        req.symbol,
        timeframe=req.timeframe,
        start=req.start,
        end=req.end,
        limit=1500
    )
    r = backtest_ema_long_only(
        df, fast=req.params.fast, slow=req.params.slow,
        fee=req.params.fee, timeframe=req.timeframe
    )
    metrics = BacktestMetrics(**r["metrics"])
    return BacktestResponse(metrics=metrics, equity_curve=r["equity_curve"], signals=r["signals"])

@app.post("/signals/live", response_model=LiveSignalResponse, summary="Post Live")
def post_live(req: LiveSignalRequest):
    ex = _force_binance(req.exchange)
    mdp = MarketDataProvider(exchange_id=ex)
    df = mdp.fetch_ohlcv(req.symbol, timeframe=req.timeframe, limit=req.lookback_bars)
    r = backtest_ema_long_only(
        df, fast=req.params.fast, slow=req.params.slow,
        fee=req.params.fee, timeframe=req.timeframe
    )
    ema_fast_last = r["ema_fast"][-1]
    ema_slow_last = r["ema_slow"][-1]
    action = "HOLD"
    if ema_fast_last > ema_slow_last:
        action = "BUY"
    elif ema_fast_last < ema_slow_last:
        action = "SELL"
    diff = (ema_fast_last - ema_slow_last) / ema_slow_last if ema_slow_last else 0.0
    confidence = min(max(abs(diff) * 20, 0), 1.0)
    evidence = {
        "ema_fast": ema_fast_last,
        "ema_slow": ema_slow_last,
        "ema_diff_ratio": diff,
        "exchange": ex,
    }
    return LiveSignalResponse(action=action, confidence=confidence, evidence=evidence)

# =========================
# 建議（即時） & 記憶
# =========================
@app.post("/advise", response_model=AdviseResponse, summary="Post Advise")
def post_advise(req: AdviseRequest):
    prefs = query_memory(req.user_id, "preferences")
    risk = req.risk_level or (prefs.get("risk") if prefs else "medium")
    tf = req.timeframe or (prefs.get("default_timeframe") if prefs else "1h")
    cap = req.capital or (prefs.get("capital") if prefs else None)
    ex = _force_binance(req.exchange)
    params = {"fast": 12, "slow": 26, "fee": 0.001}
    advice = make_live_advice(
        req.user_id, req.symbol, ex,
        tf, risk, req.horizon_days, params, capital=cap
    )
    return AdviseResponse(**advice)

@app.post("/memory/upsert", response_model=GenericOK, summary="Post Memory Upsert")
def post_memory_upsert(req: MemoryUpsertRequest):
    upsert_memory(req.user_id, req.key, req.value)
    return GenericOK(status="ok", detail="upserted")

@app.get("/memory/query", response_model=MemoryQueryResponse, summary="Get Memory Query")
def get_memory_query(user_id: str, key: str):
    val = query_memory(user_id, key)
    return MemoryQueryResponse(user_id=user_id, key=key, value=val)

# =========================
# 知識庫（純文字 / 搜尋 / 上傳 / OCR / 由 URL 匯入）
# =========================
@app.post("/knowledge/upsert", response_model=KnowledgeUpsertOK, summary="Post Knowledge Upsert")
def post_knowledge_upsert(req: KnowledgeUpsertRequest):
    kid = upsert_knowledge(
        user_id=req.user_id,
        title=req.title,
        content=req.content,
        tags=_ensure_tags_to_list(req.tags),
        active=req.active
    )
    return KnowledgeUpsertOK(status="ok", knowledge_id=kid)

@app.post("/knowledge/search", response_model=KnowledgeSearchResponse, summary="Post Knowledge Search")
def post_knowledge_search(req: KnowledgeSearchRequest):
    items = search_knowledge(
        user_id=req.user_id,
        query=req.query,
        tags=_ensure_tags_to_list(req.tags),
        limit=req.limit
    )
    return KnowledgeSearchResponse(items=[KnowledgeItem(**it) for it in items])

@app.post("/knowledge/upload", response_model=UploadResult, summary="Knowledge Upload")
async def knowledge_upload(
    user_id: str = Form(...),
    title: str = Form(...),
    tags: str = Form("upload"),
    chunk_size: int = Form(1200),
    overlap: int = Form(120),
    file: UploadFile = File(...)
):
    fb = await file.read()
    text = extract_text_by_ext(fb, file.filename or (file.content_type or ""))
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    tag_list = _ensure_tags_to_list(tags)
    saved_ids: List[int] = []
    for idx, ch in enumerate(chunks, start=1):
        kid = upsert_knowledge(
            user_id=user_id,
            title=f"{title} [chunk {idx}]",
            content=ch,
            tags=tag_list,
            active=True
        )
        saved_ids.append(kid)
    return UploadResult(status="ok", chunks_saved=len(saved_ids), ids=saved_ids)

@app.post("/knowledge/ingest_url", response_model=UploadResult, summary="Knowledge Ingest Url")
def knowledge_ingest_url(req: IngestURLRequest):
    fb, ct = fetch_bytes_from_url(str(req.url))
    text = extract_text_by_ext(fb, (ct or str(req.url)))
    title = req.title or str(req.url)
    chunks = chunk_text(text, chunk_size=req.chunk_size, overlap=req.overlap)
    tags = _ensure_tags_to_list(req.tags or ["url"])
    saved_ids: List[int] = []
    for idx, ch in enumerate(chunks, start=1):
        kid = upsert_knowledge(
            user_id=req.user_id,
            title=f"{title} [chunk {idx}]",
            content=ch,
            tags=tags,
            active=True
        )
        saved_ids.append(kid)
    return UploadResult(status="ok", chunks_saved=len(saved_ids), ids=saved_ids)

@app.post("/knowledge/upload_ocr", response_model=UploadResult, summary="Knowledge Upload OCR")
async def knowledge_upload_ocr(
    user_id: str = Form(...),
    title: str = Form(...),
    tags: str = Form("upload,ocr"),
    lang: str = Form("eng"),
    chunk_size: int = Form(1200),
    overlap: int = Form(120),
    poppler_path: str = Form(""),
    file: UploadFile = File(...)
):
    fb = await file.read()
    filename = (file.filename or "").lower()
    if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        text = ocr_image_bytes(fb, lang=lang)
    elif filename.endswith(".pdf"):
        text = ocr_pdf_bytes(fb, lang=lang, poppler_path=(poppler_path or None))
    else:
        text = extract_text_by_ext(fb, filename or (file.content_type or ""))

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    tag_list = _ensure_tags_to_list(tags)
    saved_ids: List[int] = []
    for idx, ch in enumerate(chunks, start=1):
        kid = upsert_knowledge(
            user_id=user_id,
            title=f"{title} [chunk {idx}]",
            content=ch,
            tags=tag_list,
            active=True
        )
        saved_ids.append(kid)
    return UploadResult(status="ok", chunks_saved=len(saved_ids), ids=saved_ids)

# =========================
# 一鍵助理（讀偏好 → 產生建議 → 寫日誌 → 檢討）
# =========================
@app.post("/assistant/one_click", response_model=OneClickResponse, summary="Assistant One Click",
          description="讀偏好（風險/timeframe/capital）→ 產生最新建議 → 寫入建議日誌 → 回傳最近 N 筆評估")
def assistant_one_click(req: OneClickRequest):
    symbol = req.symbol or "ETH/USDT"
    exchange = _force_binance(req.exchange)
    timeframe = req.timeframe or "1h"
    risk_level = req.risk_level or "medium"
    horizon_days = req.horizon_days

    # 讀偏好覆蓋
    prefs = query_memory(req.user_id, "preferences")
    capital = req.capital
    if prefs:
        risk_level = req.risk_level or prefs.get("risk") or risk_level
        timeframe  = req.timeframe  or prefs.get("default_timeframe") or timeframe
        capital    = req.capital    or prefs.get("capital")

    params = {"fast": 12, "slow": 26, "fee": 0.001}
    advice_dict = make_live_advice(
        req.user_id, symbol, exchange, timeframe, risk_level, horizon_days,
        params, capital=capital
    )
    advice = AdviseResponse(**advice_dict)

    ev = advice.evidence
    log_id = save_log({
        "user_id": req.user_id,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "risk_level": risk_level,
        "horizon_days": horizon_days,
        "action": advice.action,
        "entry": float(ev.get("suggested_entry")) if ev.get("suggested_entry") is not None else None,
        "stop": float(ev.get("suggested_stop")) if ev.get("suggested_stop") is not None else None,
        "take": float(ev.get("suggested_take")) if ev.get("suggested_take") is not None else None,
        "position_pct": float(ev.get("position_pct")) if ev.get("position_pct") is not None else None,
        "qty": float(ev.get("suggested_qty")) if ev.get("suggested_qty") is not None else None,
    })

    recent = list_recent_logs(user_id=req.user_id, limit=req.review_last)
    evaluated: List[EvaluatedLogItem] = []
    for r in recent:
        rec = evaluate_one(r)
        evaluated.append(EvaluatedLogItem(
            id=rec.id,
            created_at=rec.created_at.isoformat() if getattr(rec, "created_at", None) else None,
            symbol=rec.symbol,
            timeframe=rec.timeframe,
            action=rec.action,
            entry=rec.entry,
            stop=rec.stop,
            take=rec.take,
            evaluated=rec.evaluated,
            triggered=rec.triggered,
            exit_reason=rec.exit_reason,
            exit_price=rec.exit_price,
            pnl_pct=rec.pnl_pct,
        ))

    return OneClickResponse(
        now_advice=advice,
        log_id=log_id,
        review=OneClickReview(evaluated_recent=evaluated)
    )

# =========================
# 長期檢討統計
# =========================
@app.post("/review/summary", response_model=ReviewSummaryResponse, summary="Review Summary",
          description="回傳長期檢討統計：勝率、平均/中位數報酬、平均R、最佳/最差交易；可用 lookback_days 或 limit 篩選，也可指定 symbol / timeframe。")
def review_summary(req: ReviewSummaryRequest):
    stats_dict = compute_review_stats(
        user_id=req.user_id,
        lookback_days=req.lookback_days,
        limit=req.limit,
        symbol=req.symbol,
        timeframe=req.timeframe
    ) or {}
    # 確保轉成定義好的 ReviewStats
    return ReviewSummaryResponse(stats=ReviewStats(**stats_dict))
# ================== [B] 單一 timeframe 版（改為同步函式 + 保護） ==================
@app.get("/signals/full_analysis")
def full_analysis(symbol: str = "ETH/USDT", timeframe: str = "5m", limit: int = 200):
    try:
        # 即時價（可選；若出錯不影響後續）
        try:
            ticker = exchange.fetch_ticker(symbol)
            last_price = float(ticker.get("last")) if ticker and ticker.get("last") is not None else None
        except Exception:
            last_price = None

        # K 線資料
        df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        if df.empty:
            # 與 multi 版一致：遇到空資料時回傳清楚訊息
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": "no ohlcv data",
                "indicators": None,
                "ohlcv": []
            }

        # 指標計算（延用你已經定義好的 calculate_indicators）
        indicators = calculate_indicators(df)

        # 只回最近 5 根 K 線作檢查（和你原檔行為一致）
        ohlcv_tail = df.tail(5)[["timestamp", "open", "high", "low", "close", "volume"]]
        ohlcv_list = [
            [
                int(ts.value // 10**6),  # 轉回毫秒時間戳（可與前端常見格式對齊）
                float(o), float(h), float(l), float(c), float(v)
            ]
            for ts, o, h, l, c, v in ohlcv_tail.itertuples(index=False)
        ]

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "last_price": last_price,
            "indicators": {
                # 與 multi 版 key 命名風格一致
                "MACD": indicators["MACD"],
                "KDJ": indicators["KDJ"],
                "BOLL": indicators["BOLL"],
                "MA": indicators["MA"],
            },
            "ohlcv": ohlcv_list
        }

    except Exception as e:
        # 最外層保險：回傳清楚錯誤，不讓整個 API 500 掉
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": f"{type(e).__name__}: {str(e)}"
        }


# ================== [C] 多時間框架版 ==================
@app.get("/signals/full_analysis_multi")
def full_analysis_multi(symbol: str = "ETH/USDT"):
    timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
    output = {}

    for tf in timeframes:
        df = fetch_ohlcv(symbol, timeframe=tf, limit=200)
        if df.empty:
            output[tf] = {"error": "no data"}
            continue

        indicators = calculate_indicators(df)
        last_row = df.iloc[-1]

        output[tf] = {
            "last_candle": {
                "time": str(last_row["timestamp"]),
                "open": float(last_row["open"]),
                "high": float(last_row["high"]),
                "low": float(last_row["low"]),
                "close": float(last_row["close"]),
                "volume": float(last_row["volume"]),
            },
            "indicators": indicators,
        }

    return {"symbol": symbol, "analysis": output}
