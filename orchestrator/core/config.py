from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # APIs
    GROQ_API_KEY: str
    HF_TOKEN: str

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "data_flywheel"

    # Elasticsearch
    ES_HOST: str = "http://localhost:9200"
    ES_INDEX_LOGS: str = "inference_logs"
    ES_INDEX_DATASETS: str = "curated_datasets"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "data-flywheel"

    # Configs
    FLYWHEEL_CONFIG: str = "configs/flywheel.yaml"
    MODELS_CONFIG: str = "configs/models.yaml"
    EVAL_CRITERIA_CONFIG: str = "configs/eval_criteria.yaml"


settings = Settings()
