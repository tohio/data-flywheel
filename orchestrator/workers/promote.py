"""
Stage 4 — Promotion
Final gate. Checks best candidate against promotion criteria.
If it passes, promotes to production and closes the flywheel loop.
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
            "stages.promotion": stage_result,
        }}
    )
    client.close()


@celery_app.task(name="flywheel.promote", bind=True)
def run_promotion(self, eval_result: dict, run_id: str, config: dict) -> dict:
    """
    Receives eval_result from the evaluation stage.
    Checks best candidate against criteria in eval_criteria.yaml.
    Promotes if passing, records outcome either way.
    """
    logger.info("promotion_started", run_id=run_id, task_id=self.request.id)

    _update_run_status(run_id, "promoting", {
        "status": "running",
        "started_at": str(datetime.now(timezone.utc)),
    })

    try:
        from orchestrator.services.deployment.manager import DeploymentManager

        manager = DeploymentManager()
        result = manager.maybe_promote(
            run_id=run_id,
            best_candidate=eval_result.get("best_candidate"),
            all_scored=eval_result.get("scored_experiments", []),
        )
        manager.close()

        final_result = {
            **result,
            "completed_at": str(datetime.now(timezone.utc)),
        }

        _update_run_status(run_id, "completed", final_result)

        if result["promoted"]:
            logger.info("flywheel_loop_complete_with_promotion",
                        run_id=run_id,
                        promoted_model=result["promoted_model_name"],
                        accuracy=result["metrics"]["accuracy"],
                        latency_p95_ms=result["metrics"]["latency_p95_ms"],
                        cost_per_1k=result["metrics"]["cost_per_1k_tokens"])
        else:
            logger.info("flywheel_loop_complete_no_promotion",
                        run_id=run_id,
                        reason=result["reason"])

        return final_result

    except Exception as exc:
        _update_run_status(run_id, "failed", {"status": "failed", "error": str(exc)})
        logger.error("promotion_failed", run_id=run_id, error=str(exc))
        raise
