# --- 基底映像 ---
FROM python:3.11-slim

# --- 安裝系統套件：tesseract + 繁中字庫 & poppler ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-chi-tra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Tesseract 語料位置（多數環境不必設，但設了較保險）
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# --- 建置 Python 環境 ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 服務埠
ENV PORT=8787
ENV PYTHONUNBUFFERED=1

# 如果你的 SQLite 在 ./data/ 底下，先確保資料夾存在
RUN mkdir -p /app/data

# 對外開放埠（平台通常會自動映射）
EXPOSE 8787

# 啟動指令（使用平台提供的 $PORT，如無則8866）
CMD ["sh","-c","uvicorn --app-dir . app.main:app --host 0.0.0.0 --port ${PORT:-8787}"]
