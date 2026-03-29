import pandas as pd
import os

# 1. 取得目前這個 Python 檔所在的絕對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接成完整的檔案路徑
file_path = os.path.join(current_dir, 'uk_retail_data.parquet')

# 3. 讀取
df = pd.read_parquet(file_path)

# 2. 檢查基本資訊與缺失值
print(df.info())
print(df.isnull().sum())

# 3. 核心檢查：確認超市類別與時間範圍
print("超市清單:", df['supermarket_name'].unique()) # 應包含 Tesco, Aldi 等 [cite: 13]
print("時間起訖:", df['capture_date'].min(), "到", df['capture_date'].max())

# 4. 轉換時間格式 (確保為 Time Series 準備)
df['capture_date'] = pd.to_datetime(df['capture_date'])