FROM python:3.11

# 安裝系統依賴：poppler & tesseract（含繁中語言包）
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-tra \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SQLite 儲存位置
RUN mkdir -p /var/data
ENV DB_URL=sqlite:////var/data/app.db

COPY . .

# 用 sh -c 讓 ${PORT} 能被展開（Render 會塞 PORT）
CMD ["sh","-c","python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8787}"]


