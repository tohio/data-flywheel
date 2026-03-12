"""
Stage 3 — Evaluation
Scores every experiment from the finetuning stage using
the LLM judge, latency metrics, and cost estimates.
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
            "stages.evaluation": stage_result,
        }}
    )
    client.close()


@celery_app.task(name="flywheel.evaluate", bind=True)
def run_evaluation(self, finetune_result: dict, run_id: str, config: dict) -> dict:
    """
    Score all experiments produced by the finetuning stage.
    Receives finetune_result from the Celery chain.
    Returns scored results for the promotion stage.
    """
    logger.info("evaluation_started", run_id=run_id, task_id=self.request.id)

    experiment_ids = finetune_result.get("experiments", [])
    if not experiment_ids:
        logger.warning("evaluation_skipped", run_id=run_id, reason="no_experiments")
        _update_run_status(run_id, "evaluating", {
            "status": "skipped",
            "reason": "no_experiments_from_finetuning",
        })
        return {"status": "skipped", "scored_experiments": [], "best_candidate": None}

    _update_run_status(run_id, "evaluating", {
        "status": "running",
        "started_at": str(datetime.now(timezone.utc)),
        "experiment_count": len(experiment_ids),
    })

    try:
        from orchestrator.services.evaluator.benchmarks import EvaluationSuite
        scored = EvaluationSuite().evaluate_all(experiment_ids)

        # Pick the best candidate — highest accuracy, then lowest cost as tiebreaker
        best = None
        if scored:
            best = max(
                scored,
                key=lambda x: (x["accuracy"], -x["cost_per_1k_tokens"])
            )

        result = {
            "status": "completed",
            "scored_experiments": scored,
            "best_candidate": best,
            "completed_at": str(datetime.now(timezone.utc)),
        }

        _update_run_status(run_id, "evaluating", result)
        logger.info("evaluation_completed",
                    run_id=run_id,
                    experiments_scored=len(scored),
                    best_model=best["model_id"] if best else None,
                    best_accuracy=best["accuracy"] if best else None)
        return result

    except Exception as exc:
        _update_run_status(run_id, "failed", {"status": "failed", "error": str(exc)})
        logger.error("evaluation_failed", run_id=run_id, error=str(exc))
        raise
