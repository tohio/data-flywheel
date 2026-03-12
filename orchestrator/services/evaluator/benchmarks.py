"""
EvaluationSuite
---------------
Orchestrates the full evaluation run for a single experiment:

  1. Load experiment + eval samples from MongoDB
  2. Collect latency + cost metrics (MetricsCollector)
  3. Score with LLM judge (LLMJudge)
  4. Write results back to MongoDB experiment record
  5. Log everything to MLflow

Returns a scored dict used by the promotion stage.
"""
import dataclasses
from datetime import datetime, timezone
from typing import Any

import yaml

from orchestrator.core.config import settings
from orchestrator.services.evaluator.judge import LLMJudge
from orchestrator.services.evaluator.metrics import MetricsCollector
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

# How many samples to use for evaluation (random subset of curated dataset)
_EVAL_SAMPLE_LIMIT = 100


class EvaluationSuite:

    def __init__(self):
        self.judge = LLMJudge()
        self.metrics = MetricsCollector()

    def evaluate_experiment(self, experiment_id: str) -> dict:
        """
        Run full evaluation for one experiment.
        Returns a scored dict with all metrics.
        """
        from pymongo import MongoClient
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB]

        # Load experiment record
        exp = db.experiments.find_one({"_id": experiment_id})
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Load eval samples from curated dataset
        dataset = db.datasets.find_one({"_id": exp["dataset_id"]})
        if not dataset:
            raise ValueError(f"Dataset {exp['dataset_id']} not found")

        samples = dataset["samples"][:_EVAL_SAMPLE_LIMIT]
        prompts = [s["prompt"] for s in samples]

        client.close()

        model_name = exp["model_name"]
        model_id = exp["model_id"]

        logger.info("evaluation_started",
                    experiment_id=experiment_id,
                    model_id=model_id,
                    sample_count=len(samples))

        # ── 1. Latency + cost metrics ─────────────────────────────────────
        model_metrics = self.metrics.measure(model_name, prompts)

        # ── 2. LLM judge accuracy ─────────────────────────────────────────
        judge_summary = self.judge.evaluate(
            experiment_id=experiment_id,
            model_id=model_id,
            model_name=model_name,
            eval_samples=samples,
            adapter_repo_id=exp.get("adapter_repo_id"),
        )

        # ── 3. Compile result ─────────────────────────────────────────────
        result = {
            "experiment_id": experiment_id,
            "model_id": model_id,
            "model_name": model_name,
            "experiment_type": exp["experiment_type"],
            "accuracy": judge_summary.win_rate,
            "mean_score": judge_summary.mean_score,
            "score_distribution": judge_summary.score_distribution,
            "latency_p95_ms": model_metrics.latency.p95_ms,
            "latency_p50_ms": model_metrics.latency.p50_ms,
            "cost_per_1k_tokens": model_metrics.cost.cost_per_1k_tokens_usd,
            "total_cost_usd": model_metrics.cost.total_cost_usd,
            "sample_count": len(samples),
        }

        # ── 4. Persist to MongoDB ─────────────────────────────────────────
        self._save_results(experiment_id, result)

        # ── 5. Log to MLflow ──────────────────────────────────────────────
        self._log_to_mlflow(exp, result)

        logger.info("evaluation_completed",
                    experiment_id=experiment_id,
                    accuracy=result["accuracy"],
                    latency_p95_ms=result["latency_p95_ms"],
                    cost_per_1k=result["cost_per_1k_tokens"])

        return result

    def evaluate_all(self, experiment_ids: list[str]) -> list[dict]:
        """Evaluate all experiments and return scored list."""
        results = []
        for exp_id in experiment_ids:
            try:
                result = self.evaluate_experiment(exp_id)
                results.append(result)
            except Exception as e:
                logger.error("experiment_eval_failed",
                             experiment_id=exp_id, error=str(e))
        return results

    # ── Private helpers ───────────────────────────────────────────────────

    def _save_results(self, experiment_id: str, result: dict) -> None:
        from pymongo import MongoClient
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB]
        db.experiments.update_one(
            {"_id": experiment_id},
            {"$set": {
                "status": "completed",
                "metrics": result,
                "evaluated_at": datetime.now(timezone.utc),
            }}
        )
        client.close()

    def _log_to_mlflow(self, exp: dict, result: dict) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=f"{exp['model_id']}-{exp['experiment_type']}"):
                mlflow.log_params({
                    "model_id": exp["model_id"],
                    "model_name": exp["model_name"],
                    "experiment_type": exp["experiment_type"],
                    "dataset_id": exp["dataset_id"],
                    "adapter_repo_id": exp.get("adapter_repo_id", "none"),
                })
                mlflow.log_metrics({
                    "accuracy": result["accuracy"],
                    "mean_score": result["mean_score"],
                    "latency_p95_ms": result["latency_p95_ms"],
                    "latency_p50_ms": result["latency_p50_ms"],
                    "cost_per_1k_tokens": result["cost_per_1k_tokens"],
                    "total_cost_usd": result["total_cost_usd"],
                    "sample_count": result["sample_count"],
                })
                mlflow.set_tag("run_id", exp["run_id"])
                mlflow.set_tag("experiment_id", exp["_id"])
        except Exception as e:
            logger.warning("mlflow_logging_failed", error=str(e))
