"""
Stage 2 — Model Experimentation
Runs ICL and LoRA SFT experiments for all candidate models.
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
            "stages.finetuning": stage_result,
        }}
    )
    client.close()


def _load_dataset_samples(dataset_id: str) -> list[dict]:
    """Fetch curated samples from MongoDB for upload to HF."""
    from pymongo import MongoClient
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    doc = db.datasets.find_one({"_id": dataset_id})
    client.close()
    if not doc:
        raise ValueError(f"Dataset {dataset_id} not found in MongoDB")
    return doc.get("samples", [])


@celery_app.task(name="flywheel.finetune", bind=True)
def run_finetuning(self, curation_result: dict, run_id: str, config: dict) -> dict:
    """
    For each candidate model, run ICL and/or LoRA SFT.
    Receives curation_result from the previous Celery chain step.
    Returns experiment IDs for the evaluation stage.
    """
    logger.info("finetuning_started", run_id=run_id, task_id=self.request.id)

    # Skip if curation found nothing
    dataset_id = curation_result.get("dataset_id")
    if not dataset_id:
        logger.warning("finetuning_skipped", run_id=run_id,
                       reason=curation_result.get("reason", "no_dataset"))
        _update_run_status(run_id, "training", {
            "status": "skipped",
            "reason": "no_dataset_from_curation",
        })
        return {"status": "skipped", "experiments": []}

    _update_run_status(run_id, "training", {
        "status": "running",
        "started_at": str(datetime.now(timezone.utc)),
        "dataset_id": dataset_id,
    })

    try:
        samples = _load_dataset_samples(dataset_id)

        from orchestrator.services.customizer.lora_sft import LoRASFTService
        experiment_ids = LoRASFTService().run_all_experiments(
            run_id=run_id,
            dataset_id=dataset_id,
            samples=samples,
            config=config,
        )

        result = {
            "status": "completed",
            "dataset_id": dataset_id,
            "experiments": experiment_ids,
            "completed_at": str(datetime.now(timezone.utc)),
        }
        _update_run_status(run_id, "training", result)
        logger.info("finetuning_completed",
                    run_id=run_id, experiments=len(experiment_ids))
        return result

    except Exception as exc:
        _update_run_status(run_id, "failed", {"status": "failed", "error": str(exc)})
        logger.error("finetuning_failed", run_id=run_id, error=str(exc))
        raise
