"""
CurationPipeline
----------------
Orchestrates the full curation flow:
  1. Pull raw inference logs from Elasticsearch
  2. Filter (quality, PII, toxicity, length)
  3. Near-deduplicate (MinHash)
  4. Cap to max_samples if configured
  5. Save curated dataset to MongoDB + return dataset_id
"""
from datetime import datetime, timezone
from typing import Any

import yaml

from orchestrator.core.config import settings
from orchestrator.services.curator.filters import FilterPipeline
from orchestrator.services.curator.dedup import MinHashDeduplicator
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


def _load_config() -> dict:
    with open(settings.FLYWHEEL_CONFIG) as f:
        return yaml.safe_load(f)


class CurationPipeline:

    def __init__(self):
        cfg = _load_config()["curation"]
        self.min_quality_score = cfg["min_quality_score"]
        self.max_token_length = cfg["max_token_length"]
        self.dedup_threshold = cfg["dedup_threshold"]
        self.remove_pii = cfg["remove_pii"]
        self.min_response_length = cfg["min_response_length"]
        self.max_samples = cfg.get("max_samples")  # optional cap

        self.filter_pipeline = FilterPipeline(
            min_quality_score=self.min_quality_score,
            max_token_length=self.max_token_length,
            min_response_length=self.min_response_length,
            remove_pii=self.remove_pii,
        )
        self.deduplicator = MinHashDeduplicator(threshold=self.dedup_threshold)

    def run(self, run_id: str, config: dict) -> dict:
        """
        Full curation run. Returns a summary dict with counts and dataset_id.
        Synchronous — called from within a Celery worker.
        """
        from elasticsearch import Elasticsearch
        from pymongo import MongoClient

        es = Elasticsearch(settings.ES_HOST)
        mongo = MongoClient(settings.MONGO_URI)
        db = mongo[settings.MONGO_DB]

        # ── 1. Pull logs ──────────────────────────────────────────────────
        raw_logs = self._pull_logs(es, run_id)
        logger.info("logs_pulled", run_id=run_id, count=len(raw_logs))

        if not raw_logs:
            return {
                "status": "skipped",
                "reason": "no_logs_available",
                "logs_pulled": 0,
                "samples_after_filter": 0,
                "samples_after_dedup": 0,
                "dataset_id": None,
            }

        # ── 2. Filter ─────────────────────────────────────────────────────
        filtered = self.filter_pipeline.run(raw_logs)
        logger.info("filtering_done", run_id=run_id,
                    before=len(raw_logs), after=len(filtered))

        # ── 3. Deduplicate ────────────────────────────────────────────────
        deduped = self.deduplicator.run(filtered)
        logger.info("dedup_done", run_id=run_id,
                    before=len(filtered), after=len(deduped))

        # ── 4. Cap samples ────────────────────────────────────────────────
        if self.max_samples and len(deduped) > self.max_samples:
            deduped = deduped[:self.max_samples]
            logger.info("samples_capped", run_id=run_id,
                        max_samples=self.max_samples, after=len(deduped))

        # ── 5. Save dataset ───────────────────────────────────────────────
        dataset_id = self._save_dataset(db, run_id, deduped)
        logger.info("dataset_saved", run_id=run_id, dataset_id=dataset_id)

        es.close()
        mongo.close()

        return {
            "status": "completed",
            "logs_pulled": len(raw_logs),
            "samples_after_filter": len(filtered),
            "samples_after_dedup": len(deduped),
            "dataset_id": dataset_id,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _pull_logs(self, es, run_id: str) -> list[dict]:
        """
        Fetch inference logs from Elasticsearch that haven't been
        used in a previous curation run.
        """
        query = {
            "query": {
                "bool": {
                    "must_not": [
                        {"exists": {"field": "curated_in_run"}}
                    ]
                }
            },
            "size": 10_000,
            "_source": ["prompt", "response", "model", "latency_ms",
                        "timestamp", "session_id"],
        }
        resp = es.search(index=settings.ES_INDEX_LOGS, body=query)
        logs = [hit["_source"] | {"_id": hit["_id"]}
                for hit in resp["hits"]["hits"]]

        # Mark logs as consumed so they aren't pulled again
        if logs:
            ops = []
            for log in logs:
                ops.append({"update": {"_index": settings.ES_INDEX_LOGS,
                                       "_id": log["_id"]}})
                ops.append({"doc": {"curated_in_run": run_id}})
            es.bulk(body=ops)

        return logs

    def _save_dataset(self, db, run_id: str, samples: list[dict]) -> str:
        """Persist curated samples to MongoDB, return dataset_id."""
        import uuid
        dataset_id = str(uuid.uuid4())
        db.datasets.insert_one({
            "_id": dataset_id,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc),
            "sample_count": len(samples),
            "samples": samples,
        })
        return dataset_id