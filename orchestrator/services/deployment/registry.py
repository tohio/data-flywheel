"""
ModelRegistry
-------------
MongoDB-backed registry tracking every model that has passed
through the flywheel. Each record captures:

  - model identity (id, name, provider)
  - experiment lineage (which run, which experiment produced it)
  - current status: candidate | production | archived
  - evaluation metrics at time of promotion
  - promotion / archival timestamps

At any point in time exactly ONE model has status == "production".
All others are either candidates (evaluated, not promoted) or
archived (previously promoted, superseded).
"""
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient

from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:

    def __init__(self):
        self._client = MongoClient(settings.MONGO_URI)
        self._db = self._client[settings.MONGO_DB]
        self._col = self._db.model_registry

    def close(self):
        self._client.close()

    # ── Read ──────────────────────────────────────────────────────────────

    def get_production_model(self) -> dict | None:
        return self._col.find_one({"status": "production"})

    def get_model(self, model_id: str) -> dict | None:
        return self._col.find_one({"_id": model_id})

    def list_models(self, status: str | None = None) -> list[dict]:
        query = {"status": status} if status else {}
        return list(self._col.find(query).sort("created_at", -1))

    # ── Write ─────────────────────────────────────────────────────────────

    def register_candidate(
        self,
        model_id: str,
        model_name: str,
        run_id: str,
        experiment_id: str,
        metrics: dict,
    ) -> str:
        """Add a newly evaluated candidate to the registry."""
        now = datetime.now(timezone.utc)
        self._col.update_one(
            {"_id": model_id},
            {"$set": {
                "_id": model_id,
                "model_name": model_name,
                "provider": "groq",
                "run_id": run_id,
                "experiment_id": experiment_id,
                "status": "candidate",
                "metrics": metrics,
                "created_at": now,
                "updated_at": now,
                "promoted_at": None,
                "archived_at": None,
            }},
            upsert=True,
        )
        logger.info("candidate_registered", model_id=model_id)
        return model_id

    def promote(self, model_id: str) -> None:
        """
        Promote model_id to production.
        Atomically archives any current production model first.
        """
        now = datetime.now(timezone.utc)

        # Archive current production model
        result = self._col.update_many(
            {"status": "production"},
            {"$set": {"status": "archived", "archived_at": now, "updated_at": now}},
        )
        if result.modified_count:
            logger.info("previous_model_archived", count=result.modified_count)

        # Promote new model
        self._col.update_one(
            {"_id": model_id},
            {"$set": {
                "status": "production",
                "promoted_at": now,
                "updated_at": now,
            }},
        )
        logger.info("model_promoted", model_id=model_id)

    def archive(self, model_id: str) -> None:
        """Manually archive a model without promoting another."""
        now = datetime.now(timezone.utc)
        self._col.update_one(
            {"_id": model_id},
            {"$set": {"status": "archived", "archived_at": now, "updated_at": now}},
        )
        logger.info("model_archived", model_id=model_id)
