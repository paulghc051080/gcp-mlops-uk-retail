import mlflow
import mlflow.xgboost

import google.cloud.aiplatform as aiplatform

import os
import datetime
import pandas as pd
import numpy as np
from google.cloud import bigquery
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 1. 初始化 BigQuery 客戶端 (會自動讀取你的 ADC 憑證)
client = bigquery.Client(location="europe-west2")

# 2. 從 dbt 產出的「乾淨表」撈取資料 (限制 100 萬筆以確保筆電執行流暢)
query = """
SELECT 
    supermarket_name, 
    category_name, 
    is_own_brand, 
    price 
FROM `ml-time-series.hmm_retail_analysis_uk.stg_uk_retail`
WHERE price > 0
LIMIT 1000000
"""
# 設定檔案路徑
CACHE_FILE = "data_sample.parquet"
USE_CACHE = True  # 測試時設為 True，跑正式流程時設為 False

if USE_CACHE and os.path.exists(CACHE_FILE):
    print("📦 正在從本地快取讀取數據 (跳過 BigQuery)...")
    df = pd.read_parquet(CACHE_FILE)
else:
    print("🚀 正在從 BigQuery 下載 1,000,000 筆清洗後的數據...")
    df = client.query(query).to_dataframe()

    # 第一次下載後存起來，下次就快了
    df.to_parquet(CACHE_FILE)
    print(f"💾 已儲存快取至 {CACHE_FILE}")

# 3. 特徵工程：將類別變數轉為數值 (One-Hot Encoding)
# 這是處理「超市名稱」和「類別名稱」標準做法
df = pd.get_dummies(df, columns=['supermarket_name', 'category_name'])

# 4. 準備訓練數據
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 配置 XGBoost 迴歸模型
# 設定 tree_method='hist' 可以加速處理百萬級數據
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    tree_method='hist',
    random_state=42
)

print("🧠 開始訓練 XGBoost 模型...")
model.fit(X_train, y_train)

# 6. 模型評估 (計算 RMSE)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)

print("-" * 30)
print(f"📊 評估結果 (Evaluation Metrics):")
print(f"✅ RMSE: £{rmse:.4f}")
print(f"✅ MAPE: {mape:.2%}")
print("-" * 30)

# 7. 視覺化：預測值 vs 實際值 (面試展示用)
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:100], preds[:100], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')
plt.title('HMM Retail Price Prediction: Actual vs Predicted')
plt.savefig('prediction_results.png')
print("📸 預測結果圖已儲存為 prediction_results.png")

# 初始化 Vertex AI 實驗環境
aiplatform.init(
    project="ml-time-series",
    location="europe-west2",
    experiment="uk-retail-analysis"
)

# 設定實驗名稱 (如果不存在會自動建立)
mlflow.set_experiment("UK_Retail_Analysis")

with mlflow.start_run(run_name="XGBoost_Baseline_v1"):
    # 紀錄你設定的超參數
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist"
    })

    # 紀錄計算出來的指標
    mlflow.log_metrics({
        "rmse": rmse,
        "mape": mape
    })

    # 紀錄模型本身 (將來可以一鍵部署)
    mlflow.xgboost.log_model(model, "model")

    # 紀錄剛剛產出的那張 PNG 圖
    mlflow.log_artifact("prediction_results.png")

    # --- Vertex AI 部分 (新增這段) ---
    # 使用與 run_name 一致的名稱，方便在 GCP 找資料
    # 產生如 xgboost-baseline-20260329-0830 的名稱
    run_id = f"xgboost-baseline-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with aiplatform.start_run(run=run_id):
        aiplatform.log_metrics({"rmse": rmse, "mape": mape})
        aiplatform.log_params({
            "n_estimators": 100, 
            "max_depth": 6,
            "learning_rate": 0.1
        })
    
    # 這是分類模型才會用的寫法（你不適用，但供參考）
    #aiplatform.log_classification_metrics(
    #    labels=['Low Price', 'High Price'],
    #    matrix=[[45, 5], [10, 40]],  # 預測對錯的次數
    #    display_name="my-confusion-matrix"
    #)
    # 如果要上傳圖片，通常是透過 log_model 或手動傳到 GCS 關聯
        
    print("✅ 實驗數據已同步至 MLflow 與 Vertex AI Experiments！")

