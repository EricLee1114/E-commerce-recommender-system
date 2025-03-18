from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# 全局變量
model = None
tokenizer = None
data = None
product_vectors = None
base_similarity_matrix = None
device = None

# 超參數設定
class Config:
    MAX_LEN = 128
    # 初始化權重字典
    CATEGORY_WEIGHTS = {}

# 初始化模型和數據
def init_model():
    global model, tokenizer, data, product_vectors, base_similarity_matrix, device
    
    # 設備配置
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 載入BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 載入數據
    data = pd.read_csv('processed_data.csv')
    data['標籤'] = data['標籤'].fillna('').astype(str)
    data['屬性'] = data['屬性'].fillna('').astype(str)
    data['combined_features'] = data['屬性'] + ' ' + data['標籤']
    
    # 從屬性列提取所有標籤
    all_attributes = []
    for attr_string in data['屬性']:
        # 分割每個屬性字符串，處理有多個標籤的情況
        attributes = [attr.strip() for attr in attr_string.split(',') if attr.strip()]
        # 將分割後的標籤添加到總列表
        all_attributes.extend(attributes)

    # 獲取唯一屬性值
    unique_attributes = sorted(list(set(all_attributes)))
    
    # 為所有屬性標籤設置權重1.1
    for attr in unique_attributes:
        Config.CATEGORY_WEIGHTS[attr] = 1.1
    
    # 載入預訓練模型
    num_labels = len(data['屬性'].unique())
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=num_labels
    ).to(device)
    
    # 加載保存的模型權重
    model.load_state_dict(torch.load('best_model1000.pt', map_location=device))
    model.eval()
    
    # 生成並緩存所有商品向量
    print("生成商品向量...")
    # 檢查是否存在預先計算的向量文件
    if os.path.exists('product_vectors.npy'):
        print("從文件載入向量...")
        product_vectors = np.load('product_vectors.npy')
    else:
        print("計算商品向量...")
        vectors = []
        for text in data['combined_features']:
            vector = get_product_vector(text)
            vectors.append(vector)
        product_vectors = np.vstack(vectors)
        # 儲存向量到文件以便下次快速載入
        np.save('product_vectors.npy', product_vectors)
    
    # 預先計算相似度矩陣
    print("計算相似度矩陣...")
    base_similarity_matrix = cosine_similarity(product_vectors)
    
    print("初始化完成！")

# 生成商品向量
def get_product_vector(text):
    encoded_input = tokenizer(text, return_tensors='pt',
                            max_length=512,
                            truncation=True,
                            padding='max_length').to(device)
    with torch.no_grad():
        output = model.bert(**encoded_input)
    return output.last_hidden_state[:, 0, :].cpu().numpy()[0]  # 返回 1D array

# 計算加權相似度
def calculate_weighted_similarity(idx1, idx2):
    base_similarity = base_similarity_matrix[idx1, idx2]

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
    # 調整權重可以改變各部分的影響
    final_similarity = 0.7 * base_similarity * attribute_factor + 0.3 * tag_similarity

    return min(final_similarity, 1.0)

# 獲取推薦
def get_recommendations(product_index, top_n=5):
    similarities = np.array([
        calculate_weighted_similarity(product_index, i)
        for i in range(len(data))
    ])

    similarities[product_index] = -1  # 排除自己
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(idx, similarities[idx]) for idx in top_indices]

# 路由定義
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
                          total_pages=total_pages,
                          max=max,  # 傳遞 max 函數
                          min=min)

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

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    # 檢查商品ID是否有效
    if product_id < 0 or product_id >= len(data):
        return render_template('error.html', message='商品不存在')
    
    # 獲取商品信息
    product = data.iloc[product_id].to_dict()
    
    # 獲取推薦
    recommendations = get_recommendations(product_id, top_n=5)
    recommended_products = []
    
    for idx, score in recommendations:
        recommended_products.append({
            'index': idx,
            'name': data.iloc[idx]['商品名稱'],
            'category': data.iloc[idx]['屬性'],
            'tags': data.iloc[idx]['標籤'],
            'similarity': f'{score:.2f}'
        })
    
    return render_template('product.html', 
                          product=product, 
                          product_id=product_id, 
                          recommendations=recommended_products)

@app.route('/api/recommendations/<int:product_id>')
def api_recommendations(product_id):
    # 檢查商品ID是否有效
    if product_id < 0 or product_id >= len(data):
        return jsonify({'error': '商品不存在'}), 404
    
    # 獲取推薦數量
    count = request.args.get('count', 5, type=int)
    
    # 獲取推薦
    recommendations = get_recommendations(product_id, top_n=count)
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
        'recommendations': recommended_products
    })

if __name__ == '__main__':
    # 在啟動伺服器前初始化模型和資料
    init_model()
    app.run(debug=True, host='0.0.0.0', port=8080)