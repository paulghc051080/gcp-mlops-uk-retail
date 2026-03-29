# 1. 使用官方輕量 Python 鏡像
FROM python:3.10-slim

# 2. 設定工作目錄
WORKDIR /app

# 3. 安裝系統套件 (Git 是 dbt 運作必備)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製所有程式碼 (包含 dbt 資料夾與 Python 腳本)
COPY . .

# 6. 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV DBT_PROFILES_DIR=.

# 這裡不設定 ENTRYPOINT，因為我們之後在 Pipeline 裡會動態下指令