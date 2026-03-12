from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel


class ExperimentType(str, Enum):
    ICL = "icl"
    LORA_SFT = "lora_sft"


class ExperimentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentSummary(BaseModel):
    experiment_id: str
    run_id: str
    model_id: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    accuracy: float | None = None
    latency_p95_ms: float | None = None
    cost_per_1k_tokens: float | None = None
    promoted: bool = False
    created_at: datetime


class ExperimentDetail(ExperimentSummary):
    metrics: dict[str, Any] = {}
    artifacts: list[str] = []
    mlflow_run_id: str | None = None
