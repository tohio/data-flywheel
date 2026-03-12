"""
Stage 1 — Data Curation
Pulls raw inference logs from Elasticsearch, runs the curation
pipeline, and writes a clean dataset to MongoDB.
"""
from datetime import datetime, timezone

from orchestrator.core.celery_app import celery_app
from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


def _update_run_status(run_id: str, status: str, stage_result: dict) -> None:
    from pymongo import MongoClient
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    db.flywheel_runs.update_one(
        {"_id": run_id},
        {"$set": {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
            "stages.curation": stage_result,
        }}
    )
    client.close()


@celery_app.task(name="flywheel.curate", bind=True)
def run_curation(self, run_id: str, config: dict) -> dict:
    """
    Pull logs → filter → dedup → save dataset.
    Returns a summary dict passed forward in the Celery chain.
    """
    logger.info("curation_started", run_id=run_id, task_id=self.request.id)
    _update_run_status(run_id, "curating", {
        "status": "running",
        "started_at": str(datetime.now(timezone.utc)),
    })

    try:
        from orchestrator.services.curator.pipeline import CurationPipeline
        result = CurationPipeline().run(run_id, config)

        _update_run_status(run_id, "curating", {
            **result,
            "completed_at": str(datetime.now(timezone.utc)),
        })
        logger.info("curation_completed", run_id=run_id, **result)
        return result

    except Exception as exc:
        _update_run_status(run_id, "failed", {"status": "failed", "error": str(exc)})
        logger.error("curation_failed", run_id=run_id, error=str(exc))
        raise
