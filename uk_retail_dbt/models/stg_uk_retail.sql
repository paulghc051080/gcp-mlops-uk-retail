-- models/stg_uk_retail.sql
{{ config(materialized='table') }}

SELECT
    supermarket_name,
    product_name,
    category_name,
    -- 強制轉換型別，確保機器學習模型讀取正確
    SAFE_CAST(price_gbp AS FLOAT64) as price,
    CAST(is_own_brand AS BOOL) as is_own_brand,
    capture_date
FROM `ml-time-series.hmm_retail_analysis_uk.uk_retail_data`
WHERE price_gbp IS NOT NULL -- 排除缺失值