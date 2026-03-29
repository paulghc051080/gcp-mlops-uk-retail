import mlflow
import mlflow.xgboost
import google.cloud.aiplatform as aiplatform
import os
import datetime
import pandas as pd
import numpy as np
from google.cloud import bigquery, storage
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 0. 全域配置 ---
PROJECT_ID = "ml-time-series"
REGION = "europe-west2"
BUCKET_NAME = "ml-time-series-london"
EXPERIMENT_NAME = "uk-retail-analysis"

# 環境判定：更強健的偵測方式
IS_CLOUD = (os.getenv('CLOUD_ML_JOB_ID') is not None) or (os.getenv('AIP_MODEL_DIR') is not None)
CACHE_FILE = "data_sample.parquet"

# 生成唯一的執行識別碼
job_id = os.getenv('CLOUD_ML_JOB_ID', 'local-run')
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
run_id = f"xgboost-{job_id}-{timestamp}"

# 1. 初始化 BigQuery
bq_client = bigquery.Client(project=PROJECT_ID, location=REGION)

query = """
        SELECT supermarket_name, category_name, is_own_brand, price 
        FROM `ml-time-series.hmm_retail_analysis_uk.stg_uk_retail`
        WHERE price > 0 LIMIT 1000000
        """

# 2. 數據獲取：實現數據版本控制與效能優化 [cite: 15, 16]
if IS_CLOUD:
    print("雲端環境：從 BigQuery 獲取百萬級數據...")
    df = bq_client.query(query).to_dataframe()
else:
    print("本地環境：啟用快取機制以加速開發...")
    if os.path.exists(CACHE_FILE):
        df = pd.read_parquet(CACHE_FILE)
    else:
        df = bq_client.query(query).to_dataframe()
        df.to_parquet(CACHE_FILE)

# 3. 特徵工程：滿足 JD 對特徵優化的要求 
df = pd.get_dummies(df, columns=['supermarket_name', 'category_name'])
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 模型訓練：使用 hist 提升百萬級數據處理速度 [cite: 13, 24]
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, tree_method='hist', random_state=42)
print("開始訓練 XGBoost 模型...")
model.fit(X_train, y_train)

# 5. 評估指標：直接對應 JD 的評估要求 [cite: 23]
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)
print(f"評估結果 - RMSE: £{rmse:.4f}, MAPE: {mape:.2%}")

#6. 產出預測圖表
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:100], preds[:100], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.savefig('prediction_results.png')
plt.close()
print("預測結果圖已儲存為 prediction_results.png")

# --- 6. 實驗追蹤與模型註冊：分流處理  ---

if not IS_CLOUD:
    # 本地模式：使用輕量的 MLflow
    mlflow.set_experiment(f"{EXPERIMENT_NAME}-local")
    with mlflow.start_run(run_name=f"dev-{timestamp}"):
        mlflow.log_params({"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "tree_method": "hist"})
        mlflow.log_metrics({"rmse": rmse, "mape": mape})
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact("prediction_results.png")
    print("✅ MLflow 本地紀錄完成。")
else:
    # 雲端模式：整合 Vertex AI 完整生命週期 [cite: 15, 21]
    print("🚀 啟動 Vertex AI 實驗追蹤與自動化註冊...")
    aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)
    
    with aiplatform.start_run(run=run_id):
        aiplatform.log_metrics({"rmse": rmse, "mape": mape})
        aiplatform.log_params({"n_estimators": 100, "max_depth": 6})

        # 模型持久化儲存至 GCS
        model.save_model("model.bst")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        model_blob = bucket.blob(f"model-artifacts/{run_id}/model.bst")
        model_blob.upload_from_filename("model.bst")
        
        # 同步圖表至 GCS 以供長期審計
        plot_blob = bucket.blob(f"plots/{run_id}/prediction_results.png")
        plot_blob.upload_from_filename("prediction_results.png")

        # 自動註冊模型，準備進行 Online Serving 
        print("📦 註冊模型至 Model Registry...")
        registered_model = aiplatform.Model.upload(
            display_name=f"uk-retail-price-predictor{run_id}",
            artifact_uri=f"gs://{BUCKET_NAME}/model-artifacts/{run_id}/",
            serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"
        )
        print(f"🚀 模型註冊成功！版本: {registered_model.version_id}")