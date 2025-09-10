# ETH/USDT Research API (Route A · Custom GPT + Backend)

一個可直接啟動的 FastAPI 專案，提供：

- `/signals/backtest`：回測（預設 EMA 黃金交叉，長/短參數可調）
- `/signals/live`：即時訊號（取交易所資料、產出 BUY/SELL/HOLD + 信心分數）
- `/advise`：依據策略表現、風險等級與當前市場，輸出「建議 + 證據」
- `/memory/*`：長期記憶（偏好、風險承受度、常用時間框等）

> MVP 避免安裝障礙，**回測以 pandas/numpy 實作**（無需 vectorbt）。未來可把回測引擎換成 vectorbt 或 Backtrader（保留了擴充位）。

---

## 快速開始

1) 建環境
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
2) 啟動 API
```bash
uvicorn app.main:app --reload --port 8787
```
打開瀏覽器訪問 `http://127.0.0.1:8787/docs` 可看互動式接口。

3) 在 ChatGPT 的 **自訂 GPT → Actions** 匯入 `openapi.yaml`，
   - 把 `servers.url` 改成你的實際 API 網域（例如 `https://your-domain.com` 或本機測試 `http://127.0.0.1:8787`）。
   - 儲存後即可在自訂 GPT 對話中呼叫這些端點。

---

## 端點摘要

- `POST /signals/backtest`
  - Body: `symbol` (預設 `ETH/USDT`), `exchange` (預設 `binance`), `timeframe` (例如 `1h`), `start`, `end` (ISO8601), `params`（策略參數，如 `{"fast": 12, "slow": 26, "fee": 0.001}`）
  - 回傳：{
      "metrics": {"total_return","cagr","sharpe","mdd","trades","win_rate"},
      "equity_curve": [...], "signals": [...]
    }
- `POST /signals/live`
  - Body: `symbol`, `exchange`, `timeframe`，可選 `lookback_bars`、策略參數
  - 回傳：{
      "action": "BUY/SELL/HOLD",
      "confidence": 0~1,
      "evidence": {"ema_fast","ema_slow","ema_diff","trend_bars"}
    }
- `POST /advise`
  - Body: `goal`（例如 "swing"/"intraday"）、`risk_level`（"low"/"medium"/"high"）、`horizon_days`、`capital`（可選）、`user_id`
  - 內部會抓 `/signals/live`、讀取 `/memory/query`，回傳建議（含風險聲明與可執行選項）。
- `POST /memory/upsert`、`GET /memory/query`

---

## 環境設定 `.env`

- `EXCHANGE=binance`
- `DEFAULT_SYMBOL=ETH/USDT`
- `DEFAULT_TIMEFRAME=1h`
- `DB_URL=sqlite:///./eth_agent.db`

> 現階段僅用公有行情（無需 API Key）。若將來要查私有帳戶或下單，再加入金鑰。

---

## 策略說明（MVP）

- **EMA 黃金交叉**：`fast` 上穿 `slow` 做多；下穿轉為空手（不做放空）
- 手續費：`fee`（預設 0.1%）在訊號切換點扣除
- 指標與績效：Sharpe 以每年 bar 數估計（依 timeframe 推算），MDD、勝率等均提供。

> 後續可在 `app/backtest.py` 中：
> - 加入 ATR 停損/停利、倉位風險固定%、槓桿、滑點
> - 接入 `vectorbt` 或 `Backtrader` 作高擬真事件級回測

---

## 測試

```bash
pytest -q
```

---

## 合規與風險聲明

本專案僅供教育與研究用途，**不構成投資建議**。加密資產波動極大，可能發生重大損失。請自我評估風險，必要時諮詢持證專業人士。
