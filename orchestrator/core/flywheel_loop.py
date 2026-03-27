"""
flywheel_loop.py
----------------
Orchestrates the full flywheel cycle as a Celery chain:
  curate → finetune (ICL + LoRA) → evaluate → promote

Each stage updates the run document in MongoDB so status
can be polled via GET /flywheel/status/{run_id}.

Resume support:
  If a run fails mid-cycle, use resume_flywheel_run() to
  pick up from the last completed stage rather than starting over.
"""
from celery import chain
from orchestrator.core.celery_app import celery_app
from orchestrator.workers.curate import run_curation
from orchestrator.workers.finetune import run_finetuning
from orchestrator.workers.evaluate import run_evaluation
from orchestrator.workers.promote import run_promotion
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(name="flywheel.start", bind=True)
def start_flywheel_run(self, run_id: str, config: dict) -> None:
    """
    Entry point — builds and dispatches the full Celery task chain.
    The chain passes its result forward at each step:
      curate result → finetune → evaluate → promote
    """
    logger.info("flywheel_chain_starting", run_id=run_id, task_id=self.request.id)

    task_chain = chain(
        run_curation.s(run_id, config),
        run_finetuning.s(run_id, config),
        run_evaluation.s(run_id, config),
        run_promotion.s(run_id, config),
    )
    task_chain.apply_async()
    logger.info("flywheel_chain_dispatched", run_id=run_id)


@celery_app.task(name="flywheel.resume", bind=True)
def resume_flywheel_run(self, run_id: str, config: dict) -> None:
    """
    Resume a failed or incomplete flywheel run from the last
    completed stage. Skips stages that already succeeded.
    """
    from pymongo import MongoClient
    from orchestrator.core.config import settings
    from datetime import datetime, timezone

    logger.info("flywheel_resume_starting", run_id=run_id, task_id=self.request.id)

    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    run = db.flywheel_runs.find_one({"_id": run_id})
    client.close()

    if not run:
        logger.error("flywheel_resume_run_not_found", run_id=run_id)
        return

    stages = run.get("stages", {})
    curation = stages.get("curation", {})
    finetuning = stages.get("finetuning", {})
    evaluation = stages.get("evaluation", {})

    # Build chain starting from the first incomplete stage
    tasks = []

    if curation.get("status") != "completed":
        logger.info("flywheel_resume_from_curation", run_id=run_id)
        tasks = [
            run_curation.s(run_id, config),
            run_finetuning.s(run_id, config),
            run_evaluation.s(run_id, config),
            run_promotion.s(run_id, config),
        ]
    elif finetuning.get("status") not in ("completed", "skipped"):
        logger.info("flywheel_resume_from_finetuning", run_id=run_id)
        # Pass curation result forward
        curation_result = {
            "status": "completed",
            "logs_pulled": curation.get("logs_pulled", 0),
            "samples_after_filter": curation.get("samples_after_filter", 0),
            "samples_after_dedup": curation.get("samples_after_dedup", 0),
            "dataset_id": curation.get("dataset_id"),
        }
        tasks = [
            run_finetuning.s(curation_result, run_id, config),
            run_evaluation.s(run_id, config),
            run_promotion.s(run_id, config),
        ]
    elif evaluation.get("status") not in ("completed", "skipped"):
        logger.info("flywheel_resume_from_evaluation", run_id=run_id)
        finetuning_result = {
            "status": finetuning.get("status", "completed"),
            "experiments": finetuning.get("experiments", []),
        }
        tasks = [
            run_evaluation.s(finetuning_result, run_id, config),
            run_promotion.s(run_id, config),
        ]
    else:
        logger.info("flywheel_resume_from_promotion", run_id=run_id)
        evaluation_result = {
            "status": evaluation.get("status", "completed"),
            "scored_experiments": evaluation.get("scored_experiments", []),
            "best_candidate": evaluation.get("best_candidate"),
        }
        tasks = [
            run_promotion.s(evaluation_result, run_id, config),
        ]

    # Mark run as resuming
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    db.flywheel_runs.update_one(
        {"_id": run_id},
        {"$set": {
            "status": "resuming",
            "updated_at": datetime.now(timezone.utc),
        }}
    )
    client.close()

    if tasks:
        chain(*tasks).apply_async()
        logger.info("flywheel_resume_chain_dispatched", run_id=run_id, stages=len(tasks))