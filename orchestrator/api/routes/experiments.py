from fastapi import APIRouter, HTTPException

from orchestrator.api.models.experiment import ExperimentDetail, ExperimentSummary
from orchestrator.core.database import get_mongo_db

router = APIRouter()


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(run_id: str | None = None, limit: int = 50):
    """List experiments, optionally filtered by flywheel run."""
    db = get_mongo_db()
    query = {"run_id": run_id} if run_id else {}
    cursor = db.experiments.find(query).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["experiment_id"] = str(d.pop("_id"))
    return docs


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: str):
    """Get full detail for a single experiment."""
    db = get_mongo_db()
    doc = await db.experiments.find_one({"_id": experiment_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    doc["experiment_id"] = str(doc.pop("_id"))
    return doc
