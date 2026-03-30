import os
from kfp import dsl, compiler
from google.cloud import aiplatform
# --- 新增這行 ---
from google_cloud_pipeline_components.v1.custom_job import CustomContainerTrainingJobRunOp
# --- 如果下方部署也要用組件，建議一併導入 ---
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp
import google.auth
import datetime

# --- 自動抓取 Project ID ---
# 優先嘗試環境變數，若無則透過 ADC (預設憑證) 抓取
try:
    _, PROJECT_ID = google.auth.default()
except:
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "ml-time-series")

REGION = "europe-west2"
BUCKET_NAME = f"gs://{PROJECT_ID}-london" # 假設你的 Bucket 命名也遵循專案名

# 初始化 AI Platform
aiplatform.init(project=PROJECT_ID, location=REGION)


# 產生帶有時間戳記的名稱，實現真正的版本控制
# 使用年月日時分，確保唯一性與可追溯性
VERSION_TAG = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# 注意：這行移出函數外，確保編譯時 project_id 已經是正確的字串
my_dataset = aiplatform.TabularDataset.create(
    display_name=f"uk_retail_{VERSION_TAG}",
    bq_source=f"bq://ml-time-series.hmm_retail_analysis_uk.stg_uk_retail",
    labels={"env": "prod", "version": "v2"}
)

@dsl.pipeline(name="uk-retail-kfp-pipeline")
def retail_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION
):  

    # 2. 訓練組件：動態組裝鏡像路徑
    # 注意：這裡的 'trainer' 是根據你 cloudbuild.yaml 裡的鏡像名稱
    container_uri = f"{region}-docker.pkg.dev/{project_id}/uk-retail-repo/trainer:latest"

    train_job_op = CustomContainerTrainingJobRunOp(
        display_name=f"uk-retail-training-{VERSION_TAG}",
        container_uri=container_uri,
        model_serving_container_image_uri=f"{region}-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        model_display_name=f"uk-retail-price-model-{VERSION_TAG}",
        base_output_dir=f"gs://ml-time-series-london/pipeline-artifacts/{VERSION_TAG}",
        staging_bucket=f"gs://ml-time-series-london",
        # 4. 加上 Labels (進階)：這能讓你透過篩選器快速找到特定版本的 Job
        model_labels={
        "data_version": "v2",
        "pipeline_run": VERSION_TAG,
        "framework": "xgboost_tf_serving"
        }
    )
"""
    # 3. 部署
    deploy_op = ModelDeployOp(
         model=train_job_op.outputs["model"],
        endpoint_display_name=f"uk-retail-endpoint-{VERSION_TAG}",
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        # 流量分配：新部署的模型預設接收 100% 流量
        traffic_split={"0": 100},
    )
"""
# --- 編譯與提交 ---
compiler.Compiler().compile(pipeline_func=retail_pipeline, package_path="pipeline.json")

"""
# 提交 Pipeline Job 到雲端執行
pipeline_job = aiplatform.PipelineJob(
    display_name=f"uk-automated-run-{VERSION_TAG}",
    template_path="pipeline.json",
    pipeline_root=f"gs://{PROJECT_ID}-london/pipeline_root/{VERSION_TAG}",
    enable_caching=True,
    # 5. 加上標籤：方便後續大數據審計與過濾
    labels={
        "env": "production-ready",
        "author": "hmm_engineer",
        "version": VERSION_TAG
    }
)
pipeline_job.submit()
"""