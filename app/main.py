# app/main.py  — 完整可用版本
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi

# 專案內模組
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

# ---------------------------
# 建立 FastAPI App
# ---------------------------
app = FastAPI(
    title="ETH/USDT Research API (Route A)",
    version="0.3.0",
)

# CORS（保持寬鬆，之後你要可再收斂）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 如需更安全可指定前端網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# API Key Middleware（白名單路徑免金鑰，其餘需帶 X-API-Key）
# ---------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
OPEN_PATHS = {
    "/", "/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"
}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    # 允許預檢 & 白名單
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if (path in OPEN_PATHS) or path.startswith("/static"):
        return await call_next(request)

    token_env = (getattr(settings, "api_token", "") or "").strip()
    # 未設定 API_TOKEN → 視為不啟用驗證（本地方便）
    if not token_env:
        return await call_next(request)

    key = request.headers.get("X-API-Key")
    if key != token_env:
        return JSONResponse({"detail": "Missing or invalid API key"}, status_code=401)

    return await call_next(request)

# 讓 /docs 出現 Authorize 按鈕（OpenAPI 加入 apiKey 安全機制）
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description=getattr(app, "description", None),
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["APIKeyHeader"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Paste the value of API_TOKEN (without the 'API_TOKEN=' prefix)"
    }
    schema["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

# ---------------------------
# 小工具
# ---------------------------
def _ensure_tags_to_list(tags: Optional[Any]) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    # 字串以逗號分隔
    return [t.strip() for t in str(tags).split(",") if t.strip()]

# ---------------------------
# 基礎路由
# ---------------------------
@app.get("/")
def root():
    # 直接導到 /docs
    return RedirectResponse(url="/docs", status_code=302)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# 訊號 & 回測
# ---------------------------
@app.post("/signals/backtest", response_model=BacktestResponse)
def post_backtest(req: BacktestRequest):
    mdp = MarketDataProvider(exchange_id=req.exchange)
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

@app.post("/signals/live", response_model=LiveSignalResponse)
def post_live(req: LiveSignalRequest):
    mdp = MarketDataProvider(exchange_id=req.exchange)
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
        "ema_diff_ratio": diff
    }
    return LiveSignalResponse(action=action, confidence=confidence, evidence=evidence)

# ---------------------------
# 建議（即時） & 記憶
# ---------------------------
@app.post("/advise", response_model=AdviseResponse)
def post_advise(req: AdviseRequest):
    prefs = query_memory(req.user_id, "preferences")
    risk = req.risk_level or (prefs.get("risk") if prefs else "medium")
    tf = req.timeframe or (prefs.get("default_timeframe") if prefs else "1h")
    cap = req.capital or (prefs.get("capital") if prefs else None)
    params = {"fast": 12, "slow": 26, "fee": 0.001}
    advice = make_live_advice(
        req.user_id, req.symbol, req.exchange,
        tf, risk, req.horizon_days, params, capital=cap
    )
    return AdviseResponse(**advice)

@app.post("/memory/upsert")
def post_memory_upsert(req: MemoryUpsertRequest):
    upsert_memory(req.user_id, req.key, req.value)
    return {"status": "ok"}

@app.get("/memory/query", response_model=MemoryQueryResponse)
def get_memory_query(user_id: str, key: str):
    val = query_memory(user_id, key)
    return MemoryQueryResponse(user_id=user_id, key=key, value=val)

# ---------------------------
# 知識庫（純文字 / 搜尋 / 上傳 OCR / 由 URL 匯入）
# ---------------------------
@app.post("/knowledge/upsert")
def post_knowledge_upsert(req: KnowledgeUpsertRequest):
    kid = upsert_knowledge(
        user_id=req.user_id,
        title=req.title,
        content=req.content,
        tags=_ensure_tags_to_list(req.tags),
        active=req.active
    )
    return {"status": "ok", "knowledge_id": kid}

@app.post("/knowledge/search", response_model=KnowledgeSearchResponse)
def post_knowledge_search(req: KnowledgeSearchRequest):
    items = search_knowledge(
        user_id=req.user_id,
        query=req.query,
        tags=_ensure_tags_to_list(req.tags),
        limit=req.limit
    )
    return KnowledgeSearchResponse(items=[KnowledgeItem(**it) for it in items])

@app.post("/knowledge/upload")
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
    saved_ids = []
    for idx, ch in enumerate(chunks, start=1):
        kid = upsert_knowledge(
            user_id=user_id,
            title=f"{title} [chunk {idx}]",
            content=ch,
            tags=tag_list,
            active=True
        )
        saved_ids.append(kid)
    return {"status": "ok", "chunks_saved": len(saved_ids), "ids": saved_ids}

@app.post("/knowledge/ingest_url")
def knowledge_ingest_url(req: Dict[str, Any]):
    user_id = req.get("user_id")
    url = req.get("url")
    title = req.get("title") or url
    tags = _ensure_tags_to_list(req.get("tags") or ["url"])
    chunk_size = int(req.get("chunk_size") or 1200)
    overlap = int(req.get("overlap") or 120)
    assert user_id and url, "user_id 與 url 必填"

    fb, ct = fetch_bytes_from_url(url)
    text = extract_text_by_ext(fb, (ct or url))
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    saved_ids = []
    for idx, ch in enumerate(chunks, start=1):
        kid = upsert_knowledge(
            user_id=user_id,
            title=f"{title} [chunk {idx}]",
            content=ch,
            tags=tags,
            active=True
        )
        saved_ids.append(kid)
    return {"status": "ok", "chunks_saved": len(saved_ids), "ids": saved_ids}

@app.post("/knowledge/upload_ocr")
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

    # 圖片 / PDF / 其他文本 類型自動辨識
    if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        text = ocr_image_bytes(fb, lang=lang)
    elif filename.endswith(".pdf"):
        text = ocr_pdf_bytes(fb, lang=lang, poppler_path=(poppler_path or None))
    else:
        text = extract_text_by_ext(fb, filename or (file.content_type or ""))

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    tag_list = _ensure_tags_to_list(tags)
    saved_ids = []
    for idx, ch in enumerate(chunks, start=1):
        kid = upsert_knowledge(
            user_id=user_id,
            title=f"{title} [chunk {idx}]",
            content=ch,
            tags=tag_list,
            active=True
        )
        saved_ids.append(kid)
    return {"status": "ok", "chunks_saved": len(saved_ids), "ids": saved_ids}

# ---------------------------
# 一鍵助理（讀偏好→產生建議→寫日誌→自動檢討）
# ---------------------------
@app.post("/assistant/one_click")
def assistant_one_click(req: Dict[str, Any]):
    """
    一句話搞定：
    - 讀偏好（風險、timeframe、capital）
    - 產生最新建議（entry/stop/take/position）
    - 寫入建議日誌
    - 自動評估最近 N 筆歷史建議（檢討）
    """
    user_id = req.get("user_id")
    symbol = (req.get("symbol") or "ETH/USDT")
    exchange = (req.get("exchange") or settings.exchange)
    timeframe = (req.get("timeframe") or "1h")
    risk_level = (req.get("risk_level") or "medium")
    horizon_days = int(req.get("horizon_days") or 7)

    # 讀偏好覆蓋
    prefs = query_memory(user_id, "preferences")
    if prefs:
        risk_level = req.get("risk_level") or prefs.get("risk") or risk_level
        timeframe  = req.get("timeframe")  or prefs.get("default_timeframe") or timeframe
        capital    = req.get("capital")    or prefs.get("capital")
    else:
        capital    = req.get("capital")

    params = {"fast": 12, "slow": 26, "fee": 0.001}
    advice = make_live_advice(
        user_id, symbol, exchange, timeframe, risk_level, horizon_days,
        params, capital=capital
    )

    # 寫入日誌
    ev = advice["evidence"]
    log_id = save_log({
        "user_id": user_id,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "risk_level": risk_level,
        "horizon_days": horizon_days,
        "action": advice["action"],
        "entry": float(ev["suggested_entry"]),
        "stop": float(ev["suggested_stop"]),
        "take": float(ev["suggested_take"]),
        "position_pct": float(ev["position_pct"]),
        "qty": float(ev["suggested_qty"] or 0) if ev.get("suggested_qty") is not None else None,
    })

    # 自動評估最近 N 筆（含剛存的）
    recent = list_recent_logs(user_id=user_id, limit=int(req.get("review_last") or 5))
    evaluated = []
    for r in recent:
        rec = evaluate_one(r)  # 評估一次後再取欄位
        evaluated.append({
            "id": rec.id,
            "created_at": rec.created_at.isoformat() if getattr(rec, "created_at", None) else None,
            "symbol": rec.symbol,
            "timeframe": rec.timeframe,
            "action": rec.action,
            "entry": rec.entry,
            "stop": rec.stop,
            "take": rec.take,
            "evaluated": rec.evaluated,
            "triggered": rec.triggered,
            "exit_reason": rec.exit_reason,   # TP / SL / OPEN / NOT_TRIGGERED
            "exit_price": rec.exit_price,
            "pnl_pct": rec.pnl_pct,
        })

    return {
        "now_advice": advice,
        "log_id": log_id,
        "review": {"evaluated_recent": evaluated}
    }

# ---------------------------
# 長期檢討統計
# ---------------------------
@app.post("/review/summary")
def review_summary(req: Dict[str, Any]):
    """
    回傳長期檢討統計：勝率、平均/中位數報酬、平均R、最佳/最差交易；
    可用 lookback_days 或 limit 篩選，也可指定 symbol / timeframe。
    """
    user_id = req.get("user_id")
    lookback_days = req.get("lookback_days")
    limit = req.get("limit", 200)
    symbol = req.get("symbol")
    timeframe = req.get("timeframe")
    return compute_review_stats(
        user_id=user_id,
        lookback_days=lookback_days,
        limit=limit,
        symbol=symbol,
        timeframe=timeframe
    )

