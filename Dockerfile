FROM python:3.9-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製所有 Python 檔案
COPY *.py ./
COPY requirements.txt ./

# 複製模板資料夾
COPY templates/ ./templates/

# 複製模型和資料檔案
COPY best_model1000.pt ./
COPY processed_data.csv ./

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 開放連接埠
EXPOSE 8080

# 啟動應用程式
CMD ["python", "app.py"]
