# syntax=docker/dockerfile:1
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 建資料夾給 SQLite（Render 的 Persistent Disk 也會掛在 /var/data）
RUN mkdir -p /var/data

# 如果你沒在 Render 設 DB_URL，這個 ENV 當預設也 OK（和 config 預設一致）
ENV DB_URL=sqlite:////var/data/app.db

# 用 sh -c 讓 ${PORT} 能被展開
CMD ["sh","-c","python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8787}"]


COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ=Asia/Taipei

EXPOSE 8787
CMD ["python","-c","import os,uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=int(os.getenv('PORT','8787')))"]


