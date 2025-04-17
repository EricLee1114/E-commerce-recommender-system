from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import time


app = Flask(__name__)

# 全局變量
model = None
tokenizer = None
data = None
product_vectors = None
device = None

# 超參數設定
class Config:
    MAX_LEN = 128
    # 初始化權重字典
    CATEGORY_WEIGHTS = {}

# 初始化模型和數據
def init_model():
    global model, tokenizer, data, device
    
    # 確保模型正確初始化
    try:
        # 載入數據
        data = pd.read_csv('processed_data.csv')
        data['標籤'] = data['標籤'].fillna('').astype(str)
        data['屬性'] = data['屬性'].fillna('').astype(str)
        data['combined_features'] = data['屬性'] + ' ' + data['標籤']
        
        # 從屬性列提取所有標籤
        all_attributes = []
        for attr_string in data['屬性']:
            attributes = [attr.strip() for attr in attr_string.split(',') if attr.strip()]
            all_attributes.extend(attributes)

        # 獲取唯一屬性值
        unique_attributes = sorted(list(set(all_attributes)))
        
        # 為所有屬性標籤設置權重1.1
        for attr in unique_attributes:
            Config.CATEGORY_WEIGHTS[attr] = 1.1
        
        # 設定正確的標籤數量
        num_labels = 266  # 根據模型的實際訓練參數
        
        # 初始化模型
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=num_labels
        )
        
        # 初始化分詞器
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 將模型移到正確的設備上
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        
        # 載入訓練好的權重
        model.load_state_dict(torch.load('best_model1000.pt', map_location=device))
        model.eval()  # 設置為評估模式
        
        print(f"使用設備: {device}")
        print(f"已載入模型，可進行即時向量化推薦")
        
        # 移除預先計算向量的部分
        print("初始化完成！系統將在請求時進行即時向量化")

    except Exception as e:
        print(f"模型初始化錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        model = None
        tokenizer = None
        data = None
# 計算兩個商品之間的余弦相似度 - 即時向量化版本
def calculate_cosine_similarity_realtime(idx1, idx2):
    # 即時獲取向量
    vec1 = get_product_vector(data.iloc[idx1]['combined_features'], model)
    vec2 = get_product_vector(data.iloc[idx2]['combined_features'], model)
    
    # 計算余弦相似度
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]
# 生成商品向量
def get_product_vector(text, model):
    # 添加錯誤檢查
    if model is None:
        raise ValueError("模型未正確初始化")
        
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
        # 使用最後一層的 CLS token 表示
        return output.hidden_states[-1][:, 0, :].cpu().numpy()[0]

# 計算兩個商品之間的余弦相似度
def calculate_cosine_similarity(idx1, idx2):
    vec1 = product_vectors[idx1].reshape(1, -1)
    vec2 = product_vectors[idx2].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# 計算加權相似度 - 即時向量化版本
def calculate_weighted_similarity_realtime(idx1, idx2):
    # 即時計算余弦相似度
    base_similarity = calculate_cosine_similarity_realtime(idx1, idx2)

    # 屬性權重
    attribute_factor = 1.0
    if data.iloc[idx1]['屬性'] == data.iloc[idx2]['屬性']:
        attribute = data.iloc[idx1]['屬性']
        attribute_factor = Config.CATEGORY_WEIGHTS.get(attribute, 1.0)

    # 標籤相似度分析
    tags1 = set(data.iloc[idx1]['標籤'].split(','))
    tags2 = set(data.iloc[idx2]['標籤'].split(','))

    # 計算標籤的Jaccard相似度
    tag_similarity = 0.0
    if tags1 and tags2:
        intersection = len(tags1.intersection(tags2))
        union = len(tags1.union(tags2))
        tag_similarity = intersection / union if union > 0 else 0

    # 組合基本相似度、屬性權重和標籤相似度
    final_similarity = 0.7 * base_similarity * attribute_factor + 0.3 * tag_similarity

    return min(final_similarity, 1.0)

# 獲取推薦 - 即時向量化版本
def get_recommendations_realtime(product_index, top_n=5):
    # 記錄計算時間
    start_time = time.time()
    
    print(f"開始為商品 {product_index} 進行即時向量化和相似度計算...")
    
    # 計算與其他所有商品的相似度
    similarities = []
    total_products = len(data)
    
    # 由於每次需要即時向量化，考慮減少處理的商品數量
    # 可以先用簡單的過濾方法縮小候選範圍
    # 例如：只考慮相同或相關屬性的商品
    
    # 獲取當前商品的屬性
    current_category = data.iloc[product_index]['屬性']
    
    # 建立候選商品索引列表 - 可以根據需要調整策略
    # 這裡使用一個簡單策略：優先考慮相同類別的商品，然後再加入部分其他類別
    candidate_indices = []
    
    # 首先添加相同類別的商品
    for i in range(total_products):
        if i != product_index and data.iloc[i]['屬性'] == current_category:
            candidate_indices.append(i)
    
    # 然後從其他類別中隨機抽樣一些商品
    # 如果需要更高效率，可以設定一個較小的數量
    import random
    other_indices = [i for i in range(total_products) 
                     if i != product_index and data.iloc[i]['屬性'] != current_category]
    
    # 限制其他類別的商品數量，以提高效率
    max_other_products = min(100, len(other_indices))  # 最多處理100個其他類別商品
    if len(other_indices) > max_other_products:
        other_indices = random.sample(other_indices, max_other_products)
    
    # 合併候選商品
    candidate_indices.extend(other_indices)
    
    # 計算候選商品的相似度
    print(f"將計算 {len(candidate_indices)} 個候選商品的相似度...")
    
    for i, idx in enumerate(candidate_indices):
        # 每處理10個商品輸出一次進度
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"即時計算進度: {i}/{len(candidate_indices)}, 已耗時: {elapsed:.2f}秒")
        
        similarity = calculate_weighted_similarity_realtime(product_index, idx)
        similarities.append((idx, similarity))
    
    # 根據相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 只取前top_n個
    recommendations = similarities[:top_n]
    
    # 計算耗時
    computation_time = time.time() - start_time
    print(f"即時推薦計算完成！總耗時: {computation_time:.4f}秒，為商品 {product_index} 找到 {len(recommendations)} 個相似商品")
    
    return recommendations, computation_time


def format_recommendations(recommendations):
    formatted = []
    for idx, score in recommendations:
        formatted.append({
            'index': idx,
            'name': data.iloc[idx]['商品名稱'],
            'category': data.iloc[idx]['屬性'],
            'tags': data.iloc[idx]['標籤'],
            'similarity': f'{score:.2f}'
        })
    return formatted

# 路由定義
@app.route('/product/<int:product_id>')
def product_detail(product_id):
    # 檢查商品ID是否有效
    if product_id < 0 or product_id >= len(data):
        return render_template('error.html', message='商品不存在')
    
    # 獲取商品信息
    product = data.iloc[product_id].to_dict()
    
    # 提供設備信息
    device_info = str(device)
    if 'cuda' in device_info:
        device_info += " (GPU加速即時計算)"
    elif 'mps' in device_info:
        device_info += " (Apple Silicon加速即時計算)"
    else:
        device_info += " (CPU模式即時計算)"
    
    # 沒有推薦和計算時間，這些將透過 AJAX 獲取
    return render_template('product.html', 
                          product=product, 
                          product_id=product_id,
                          device=device_info,
                          data=data,
                          is_realtime_calculation=True)

@app.route('/api/realtime-recommendations/<int:product_id>')
def realtime_recommendations(product_id):
    # 檢查商品ID是否有效
    if product_id < 0 or product_id >= len(data):
        return jsonify({'error': '商品不存在'}), 404
    
    # 獲取推薦數量
    count = request.args.get('count', 5, type=int)
    
    # 記錄API請求
    print(f"API請求：為商品 {product_id} 即時計算 {count} 個推薦項目")
    
    # 獲取推薦 - 使用即時向量化
    recommendations, computation_time = get_recommendations_realtime(product_id, top_n=count)
    recommended_products = []
    
    for idx, score in recommendations:
        recommended_products.append({
            'index': idx,
            'name': data.iloc[idx]['商品名稱'],
            'category': data.iloc[idx]['屬性'],
            'tags': data.iloc[idx]['標籤'].split(','),
            'similarity': float(score)
        })
    
    # 在API回應中加入更多資訊
    return jsonify({
        'product_id': product_id,
        'product_name': data.iloc[product_id]['商品名稱'],
        'recommendations': recommended_products,
        'computation_time': computation_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify({'results': []})
    
    # 簡單的搜尋邏輯 - 可以根據需要改進
    matches = data[data['商品名稱'].str.contains(query, case=False, na=False)]
    
    # 限制返回結果數量
    matches = matches.head(10)
    
    results = []
    for idx, row in matches.iterrows():
        results.append({
            'index': idx,
            'name': row['商品名稱'],
            'category': row['屬性'],
            'tags': row['標籤']
        })
    
    return jsonify({'results': results})

@app.route('/api/recommendations/<int:product_id>')
def api_recommendations(product_id):
    # 檢查商品ID是否有效
    if product_id < 0 or product_id >= len(data):
        return jsonify({'error': '商品不存在'}), 404
    
    # 獲取推薦數量
    count = request.args.get('count', 5, type=int)
    
    # 獲取推薦
    recommendations, computation_time = get_recommendations_realtime(product_id, top_n=count)
    recommended_products = []
    
    for idx, score in recommendations:
        recommended_products.append({
            'id': int(idx),
            'name': data.iloc[idx]['商品名稱'],
            'category': data.iloc[idx]['屬性'],
            'tags': data.iloc[idx]['標籤'],
            'similarity': float(score)
        })
    
    return jsonify({
        'product_id': product_id,
        'product_name': data.iloc[product_id]['商品名稱'],
        'recommendations': recommended_products,
        'computation_time_seconds': computation_time
    })

# 添加健康檢查端點
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/')
def index():
    # 獲取分頁參數
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # 計算分頁
    total_items = len(data)
    total_pages = (total_items + per_page - 1) // per_page
    offset = (page - 1) * per_page
    
    # 獲取當前頁的商品
    current_products = data.iloc[offset:offset+per_page].copy()
    products_list = current_products.to_dict('records')
    
    # 添加索引信息
    for i, product in enumerate(products_list):
        product['index'] = offset + i
    
    return render_template('index.html', 
                          products=products_list, 
                          page=page, 
                          total_pages=total_pages)

if __name__ == '__main__':
    # 在啟動伺服器前初始化模型和資料
    init_model()
    app.run(debug=True, host='0.0.0.0', port=8080)