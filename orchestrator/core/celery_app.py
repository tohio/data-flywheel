from celery import Celery
from celery.schedules import crontab
import yaml

from orchestrator.core.config import settings


def _load_schedule() -> dict:
    """Read cron and min_new_logs from flywheel.yaml."""
    with open(settings.FLYWHEEL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("schedule", {})


def _parse_crontab(cron: str) -> crontab:
    """
    Parse a standard 5-field cron string into a Celery crontab.
    e.g. "0 */6 * * *" → crontab(minute=0, hour="*/6")
    """
    minute, hour, day_of_month, month, day_of_week = cron.split()
    return crontab(
        minute=minute,
        hour=hour,
        day_of_month=day_of_month,
        month_of_year=month,
        day_of_week=day_of_week,
    )


celery_app = Celery(
    "data_flywheel",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "orchestrator.workers.curate",
        "orchestrator.workers.finetune",
        "orchestrator.workers.evaluate",
        "orchestrator.workers.promote",
        "orchestrator.workers.scheduled",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
)

# ── Beat schedule ─────────────────────────────────────────────────────────
# Loaded from configs/flywheel.yaml at startup.
# Beat fires the scheduled task; the task itself checks min_new_logs
# before deciding whether to kick off a full cycle.

_schedule = _load_schedule()
_cron = _schedule.get("cron", "0 */6 * * *")

celery_app.conf.beat_schedule = {
    "flywheel-scheduled-run": {
        "task": "flywheel.scheduled_run",
        "schedule": _parse_crontab(_cron),
    },
}