from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class FlywheelStatus(str, Enum):
    PENDING = "pending"
    CURATING = "curating"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    FAILED = "failed"


class FlywheelRunRequest(BaseModel):
    run_icl: bool = True
    run_lora_sft: bool = True
    dry_run: bool = False          # curate + eval only, skip fine-tuning


class FlywheelRunResponse(BaseModel):
    run_id: str
    status: FlywheelStatus
    started_at: datetime
    message: str


class FlywheelStatusResponse(BaseModel):
    run_id: str
    status: FlywheelStatus
    started_at: datetime
    updated_at: datetime
    stages: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
