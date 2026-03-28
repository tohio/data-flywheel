"""
scheduled.py
------------
Celery Beat entry point for the automated flywheel cycle.
Beat fires `scheduled_run` on the cron defined in flywheel.yaml.
Before kicking off a full cycle the task checks whether enough
new logs have accumulated since the last run (min_new_logs).
If not, it exits early — avoiding wasted API calls and compute.
"""
import uuid
from datetime import datetime, timezone

import yaml
from elasticsearch import Elasticsearch
from pymongo import MongoClient

from orchestrator.core.celery_app import celery_app
from orchestrator.core.config import settings
from orchestrator.core.flywheel_loop import start_flywheel_run
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


def _load_schedule_config() -> dict:
    with open(settings.FLYWHEEL_CONFIG) as f:
        return yaml.safe_load(f).get("schedule", {})


def _count_new_logs() -> int:
    """Count uncurated inference logs in Elasticsearch."""
    try:
        es = Elasticsearch(settings.ES_HOST)
        resp = es.count(
            index=settings.ES_INDEX_LOGS,
            body={"query": {"bool": {"must_not": [
                {"exists": {"field": "curated_in_run"}}
            ]}}}
        )
        es.close()
        return resp["count"]
    except Exception as e:
        logger.warning("log_count_failed", error=str(e))
        return 0


@celery_app.task(name="flywheel.scheduled_run")
def scheduled_run() -> dict:
    """
    Fired by Celery Beat on the configured cron schedule.
    Checks min_new_logs before starting a full flywheel cycle.
    """
    cfg = _load_schedule_config()
    min_new_logs = cfg.get("min_new_logs", 500)
    new_log_count = _count_new_logs()

    logger.info("scheduled_run_triggered",
                new_logs=new_log_count,
                min_required=min_new_logs)

    if new_log_count < min_new_logs:
        logger.info("scheduled_run_skipped",
                    reason="insufficient_logs",
                    new_logs=new_log_count,
                    min_required=min_new_logs)
        return {
            "status": "skipped",
            "reason": "insufficient_logs",
            "new_logs": new_log_count,
            "min_required": min_new_logs,
        }

    # Enough logs — kick off a full cycle
    run_id = str(uuid.uuid4())
    config = {"run_icl": True, "run_lora_sft": True, "dry_run": False}

    # Record the run in MongoDB before dispatching
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    now = datetime.now(timezone.utc)
    db.flywheel_runs.insert_one({
        "_id": run_id,
        "status": "pending",
        "started_at": now,
        "updated_at": now,
        "config": config,
        "triggered_by": "celery_beat",
        "stages": {},
        "error": None,
    })
    client.close()

    start_flywheel_run.delay(run_id, config)
    logger.info("scheduled_run_started",
                run_id=run_id,
                new_logs=new_log_count)

    return {
        "status": "started",
        "run_id": run_id,
        "new_logs": new_log_count,
    }