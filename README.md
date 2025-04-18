# BERT 商品即時推薦系統

<div align="center">
  <br>
  <p><strong>基於語義分析的即時商品推薦引擎</strong></p>
  <p>
    <a href="#系統特色">特色</a> •
    <a href="#架構說明">架構</a> •
    <a href="#快速開始">快速開始</a> •
    <a href="#技術細節">技術細節</a> •
    <a href="#展示">展示</a>
  </p>
</div>

## 專案概述

這是一個基本的商品推薦系統，基於BERT中文預訓練模型，實現了真正的**即時向量化**和**動態相似度計算**。不同於傳統推薦系統預先計算並儲存所有向量，本系統在用戶請求時才即時計算向量並分析相似度，大幅提高推薦的靈活性和準確性。

系統主要應用於電子商務平台，可根據用戶當前瀏覽的商品，立即分析並推薦語義相關且符合用戶潛在需求的其他商品，同時提供直觀的計算過程視覺化展示。

## 系統特色

### 核心功能

- **即時向量化計算** - 不依賴預先計算的結果，每次推薦都進行實時分析
- **動態相似度評估** - 結合語義理解和標籤匹配進行多維度相似度計算
- **視覺化計算過程** - 透過進度條和狀態提示直觀展示計算階段
- **高度自適應** - 能夠立即反映商品資料的變更，無需重新訓練整個系統

### 技術亮點

- **深度語義理解** - 使用 BERT 深度學習模型對商品文本進行語義分析
- **多層次相似度** - 結合向量餘弦相似度、屬性匹配和標籤 Jaccard 相似度
- **自動硬體加速** - 智能檢測並使用可用的 GPU 或 Apple Silicon (MPS) 加速計算
- **異步處理機制** - 採用前後端分離設計，優化使用者體驗

## 架構說明

### 系統架構圖

![系統架構圖](/images/framework.png)

### 主要組件

1. **資料處理層**
   - 商品資料載入與預處理
   - 文本標記化與特徵提取
   - 標籤和屬性分析

2. **模型計算層**
   - BERT 模型載入與推理
   - 向量表示生成
   - 相似度計算與排序

3. **應用邏輯層**
   - 推薦算法實現
   - API 服務設計
   - 計算進度跟踪

4. **表現層**
   - 前端界面設計
   - 推薦結果展示
   - 計算過程視覺化

## 快速開始

### 環境需求

- Docker 與 Docker Compose
- 或 Python 3.9+ 環境
- 對於 GPU 加速：NVIDIA CUDA 支援或 Apple Silicon (M1/M2/M3) 

### Docker 部署

最簡單的方式是通過 Docker 部署，這不需要手動設置 Python 環境：

```bash
# 複製專案
git clone https://github.com/EricLee1114/bert-recommender.git
cd bert-recommender

# 啟動 Docker 容器
docker-compose up -d

# 檢查容器狀態
docker ps

# 查看日誌
docker logs bert-recommender-bert-recommender-1
```

應用將在 http://localhost:8080 運行。

### 本地部署

如果您想直接在本地環境運行，請按照以下步驟：

```bash
# 複製專案
git clone https://github.com/EricLee1114/bert-recommender.git
cd bert-recommender

# 創建並啟用虛擬環境 (可選但推薦)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或使用 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt

# 運行應用
python app.py
```

應用將在 http://localhost:8080 運行。

### 使用 GPU 加速

若要在 Docker 中使用 GPU 加速，請修改 docker-compose.yml：

```yaml
services:
  bert-recommender:
    build: .
    ports:
      - "8080:8080"
    restart: unless-stopped
    environment:
      - FLASK_ENV=production
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

然後重新啟動容器：

```bash
docker-compose down
docker-compose up -d
```

## 技術細節

### 資料格式

系統使用的 CSV 檔案格式如下：

```csv
商品名稱,屬性,標籤
超輕量運動鞋,鞋類,運動,輕量,透氣
專業跑步護膝,配件,運動,防護,彈性
...
```

各欄位說明：
- **商品名稱**：商品的完整名稱，用於識別和展示
- **屬性**：商品的主要類別，影響基礎相似度權重
- **標籤**：商品的特徵標籤，多個標籤以逗號分隔

### 向量計算方法

本系統採用 BERT 模型提取文本特徵，具體過程如下：

1. **文本預處理**：將商品名稱、屬性和標籤合併為統一文本
2. **標記化處理**：使用 BERT 分詞器將文本轉換為模型輸入格式
3. **特徵提取**：通過 BERT 獲取 token 的最終隱藏層表示作為向量
4. **相似度計算**：結合向量餘弦相似度、類別匹配度和標籤相似度

計算公式：
```
最終相似度 = 0.9 * 向量相似度 * 類別權重 + 0.1 * 標籤Jaccard相似度
```

### BERT模型訓練詳解

本系統使用的BERT模型需要針對商品推薦任務進行特殊調整。完整的訓練流程如下：

#### 模型架構

- 基礎模型：BERT-base-chinese 
- 微調層：自定義的相似度學習層
- 訓練目標：最大化相關商品之間的向量相似度

#### 訓練參數

![訓練參數](/images/framework.png)

#### 訓練環境

由於模型訓練需要大量計算資源，我們提供了Google Colab環境的完整訓練筆記本：

[![Open In Colab](https://colab.research.google.com/drive/1pb8UmzFWEdQnNUQHbqdSh69jcDUFkjVg#scrollTo=x_8_8ljKEeN1))


#### 評估與優化

筆記本中還包含模型評估工具，可以測試以下指標：
- 推薦準確率與召回率
- 向量空間的視覺化分析
- 各類商品的推薦效果對比

### API 參考

系統提供以下 API 端點：

#### 獲取商品推薦
```
GET /api/realtime-recommendations/{product_id}?count={count}
```

參數：
- `product_id`: 商品 ID (整數)
- `count`: 推薦數量，預設為 5 (可選)

回應示例：
```json
{
  "product_id": 123,
  "product_name": "商品名稱",
  "recommendations": [
    {
      "id": 456,
      "name": "推薦商品1",
      "category": "類別",
      "tags": ["標籤1", "標籤2"],
      "similarity": 0.92
    },
    ...
  ],
  "computation_time": 1.234,
  "timestamp": "2025-04-17 10:30:45"
}
```

### 健康檢查 API
```
GET /health
```

回應示例：
```json
{
  "status": "healthy"
}
```

## 調優與配置

### 效能調優

可以通過調整 `Config` 類中的參數優化系統表現：

```python
class Config:
    MAX_LEN = 128            # BERT 輸入序列長度
    CATEGORY_WEIGHTS = {}    # 類別權重字典
```

### 相似度權重調整

調整相似度計算中的權重比例：

```python
# 調整向量相似度和標籤相似度的權重
final_similarity = 0.9 * base_similarity * attribute_factor + 0.1 * tag_similarity
```


## 展示

### 系統截圖

![推薦系統主界面](/images/index.png)

*商品詳情頁面與即時推薦結果*

![計算過程展示](/images/calculating.png)

*即時計算過程的視覺化展示*

### 演示

[觀看演示視頻](https://www.youtube.com/watch?v=OYCc_2yO7uk)



## 未來規劃

- [ ] 添加用戶行為分析
- [ ] 整合更多推薦策略
- [ ] 優化大規模商品目錄的處理效率
- [ ] 提供更多自定義配置選項
- [ ] 支援分布式部署
