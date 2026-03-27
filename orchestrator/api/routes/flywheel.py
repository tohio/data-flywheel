from datetime import datetime, timezone
import uuid
from fastapi import APIRouter, HTTPException
from orchestrator.api.models.flywheel import (
    FlywheelRunRequest,
    FlywheelRunResponse,
    FlywheelStatus,
    FlywheelStatusResponse,
)
from orchestrator.core.database import get_mongo_db
from orchestrator.core.flywheel_loop import start_flywheel_run, resume_flywheel_run
from orchestrator.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/run", response_model=FlywheelRunResponse)
async def run_flywheel(request: FlywheelRunRequest = FlywheelRunRequest()):
    """Trigger a full flywheel cycle — curate → train → evaluate → promote."""
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    db = get_mongo_db()
    await db.flywheel_runs.insert_one({
        "_id": run_id,
        "status": FlywheelStatus.PENDING,
        "started_at": started_at,
        "updated_at": started_at,
        "config": request.model_dump(),
        "stages": {},
        "error": None,
    })

    start_flywheel_run.delay(run_id, request.model_dump())
    logger.info("flywheel_run_started", run_id=run_id, dry_run=request.dry_run)

    return FlywheelRunResponse(
        run_id=run_id,
        status=FlywheelStatus.PENDING,
        started_at=started_at,
        message="Flywheel run started",
    )


@router.post("/resume/{run_id}", response_model=FlywheelRunResponse)
async def resume_run(run_id: str):
    """Resume a failed or incomplete flywheel run from its last completed stage."""
    db = get_mongo_db()
    doc = await db.flywheel_runs.find_one({"_id": run_id})

    if not doc:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if doc["status"] == "completed":
        raise HTTPException(status_code=400, detail=f"Run {run_id} already completed")

    resume_flywheel_run.delay(run_id, doc.get("config", {}))
    logger.info("flywheel_run_resuming", run_id=run_id)

    return FlywheelRunResponse(
        run_id=run_id,
        status=FlywheelStatus.PENDING,
        started_at=doc["started_at"],
        message="Flywheel run resuming from last completed stage",
    )


@router.get("/status/{run_id}", response_model=FlywheelStatusResponse)
async def get_run_status(run_id: str):
    """Get the current status of a flywheel run."""
    db = get_mongo_db()
    doc = await db.flywheel_runs.find_one({"_id": run_id})

    if not doc:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return FlywheelStatusResponse(
        run_id=run_id,
        status=doc["status"],
        started_at=doc["started_at"],
        updated_at=doc["updated_at"],
        stages=doc.get("stages", {}),
        error=doc.get("error"),
    )


@router.get("/runs")
async def list_runs(limit: int = 20, status: str = None):
    """List recent flywheel runs, optionally filtered by status."""
    db = get_mongo_db()
    query = {}
    if status:
        query["status"] = status

    cursor = db.flywheel_runs.find(
        query, {"_id": 1, "status": 1, "started_at": 1, "updated_at": 1}
    ).sort("started_at", -1).limit(limit)

    runs = await cursor.to_list(length=limit)
    for r in runs:
        r["run_id"] = r.pop("_id")

    return {"runs": runs}