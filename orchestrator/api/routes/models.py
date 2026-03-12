from fastapi import APIRouter, HTTPException

from orchestrator.core.database import get_mongo_db
from orchestrator.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("")
async def list_models():
    """List all models in the registry with their current status."""
    db = get_mongo_db()
    cursor = db.model_registry.find({}).sort("promoted_at", -1)
    models = await cursor.to_list(length=100)
    for m in models:
        m["id"] = str(m.pop("_id"))
    return {"models": models}


@router.get("/active")
async def get_active_model():
    """Return the currently promoted production model."""
    db = get_mongo_db()
    doc = await db.model_registry.find_one({"status": "production"})
    if not doc:
        raise HTTPException(status_code=404, detail="No production model found")
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.post("/{model_id}/promote")
async def promote_model(model_id: str):
    """Manually promote a model to production (demotes current)."""
    db = get_mongo_db()

    candidate = await db.model_registry.find_one({"_id": model_id})
    if not candidate:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Demote current production model
    await db.model_registry.update_many(
        {"status": "production"}, {"$set": {"status": "archived"}}
    )
    # Promote candidate
    await db.model_registry.update_one(
        {"_id": model_id}, {"$set": {"status": "production"}}
    )

    logger.info("model_manually_promoted", model_id=model_id)
    return {"message": f"Model {model_id} promoted to production"}
