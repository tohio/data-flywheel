"""
benchmarks.py
-------------
Orchestrates a full evaluation run for a given experiment.
Combines LLM judge scoring and latency/cost metrics, then logs
everything to MLflow. No external eval framework required.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import mlflow

from orchestrator.services.evaluator.judge import LLMJudge, JudgeSummary
from orchestrator.services.evaluator.metrics import MetricsCollector, MetricsSummary
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    experiment_id: str
    model_id: str
    model_name: str
    run_id: str

    # Judge
    accuracy: float          # win-rate (fraction scoring >= 4)
    mean_score: float
    score_distribution: dict

    # Metrics
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cost_per_1k_tokens_usd: float
    total_cost_usd: float

    # Meta
    sample_count: int
    evaluated_at: datetime

    def passes_criteria(self, criteria: dict) -> tuple[bool, dict]:
        failures = {}
        if self.accuracy < criteria.get("min_accuracy", 0.85):
            failures["accuracy"] = {
                "required": criteria["min_accuracy"],
                "actual": self.accuracy,
            }
        if self.latency_p95_ms > criteria.get("max_latency_p95_ms", 800):
            failures["latency"] = {
                "required": criteria["max_latency_p95_ms"],
                "actual": self.latency_p95_ms,
            }
        if self.cost_per_1k_tokens_usd > criteria.get("max_cost_per_1k_tokens", 0.02):
            failures["cost"] = {
                "required": criteria["max_cost_per_1k_tokens"],
                "actual": self.cost_per_1k_tokens_usd,
            }
        if self.sample_count < criteria.get("min_eval_sample_size", 100):
            failures["sample_size"] = {
                "required": criteria["min_eval_sample_size"],
                "actual": self.sample_count,
            }
        return len(failures) == 0, failures


class EvaluationSuite:
    """
    Runs the full evaluation pipeline for a single experiment or
    a batch of experiments.
    """

    def __init__(self):
        self.judge = LLMJudge()
        self.metrics = MetricsCollector()

    def evaluate_all(self, experiment_ids: list[str]) -> list[dict]:
        """
        Evaluate all experiments by ID. Loads each experiment from
        MongoDB, runs inference + judge scoring, and returns a list
        of result dicts for the promotion stage.
        """
        from pymongo import MongoClient
        from orchestrator.core.config import settings

        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB]

        results = []
        for experiment_id in experiment_ids:
            exp = db.experiments.find_one({"_id": experiment_id})
            if not exp:
                logger.warning("experiment_not_found", experiment_id=experiment_id)
                continue

            if exp.get("status") not in ("pending_eval",):
                logger.warning("experiment_skipped",
                               experiment_id=experiment_id,
                               status=exp.get("status"))
                continue

            try:
                # Load samples for this dataset
                dataset = db.datasets.find_one({"_id": exp["dataset_id"]})
                if not dataset or not dataset.get("samples"):
                    logger.warning("dataset_not_found",
                                   experiment_id=experiment_id,
                                   dataset_id=exp.get("dataset_id"))
                    continue

                samples = dataset["samples"]
                prompts = [s["prompt"] for s in samples]

                result = self.run(
                    experiment_id=experiment_id,
                    model_id=exp["model_id"],
                    model_name=exp["model_name"],
                    run_id=exp["run_id"],
                    prompts=prompts,
                    eval_samples=samples,
                )

                # Update experiment status in MongoDB
                db.experiments.update_one(
                    {"_id": experiment_id},
                    {"$set": {
                        "status": "completed",
                        "metrics": {
                            "accuracy": result.accuracy,
                            "mean_score": result.mean_score,
                            "latency_p95_ms": result.latency_p95_ms,
                            "cost_per_1k_tokens_usd": result.cost_per_1k_tokens_usd,
                        },
                        "updated_at": datetime.now(timezone.utc),
                    }}
                )

                results.append({
                    "experiment_id": experiment_id,
                    "model_id": result.model_id,
                    "model_name": result.model_name,
                    "experiment_type": exp.get("experiment_type"),
                    "accuracy": result.accuracy,
                    "mean_score": result.mean_score,
                    "latency_p95_ms": result.latency_p95_ms,
                    "cost_per_1k_tokens": result.cost_per_1k_tokens_usd,
                    "total_cost_usd": result.total_cost_usd,
                    "sample_count": result.sample_count,
                    "adapter_repo_id": exp.get("adapter_repo_id"),
                })

            except Exception as exc:
                logger.error("experiment_evaluation_failed",
                             experiment_id=experiment_id,
                             error=str(exc))
                db.experiments.update_one(
                    {"_id": experiment_id},
                    {"$set": {
                        "status": "eval_failed",
                        "error": str(exc),
                        "updated_at": datetime.now(timezone.utc),
                    }}
                )

        client.close()
        logger.info("evaluate_all_complete",
                    total=len(experiment_ids),
                    scored=len(results))
        return results

    def run(
        self,
        experiment_id: str,
        model_id: str,
        model_name: str,
        run_id: str,
        prompts: list[str],
        eval_samples: list[dict],
        mlflow_experiment: Optional[str] = None,
    ) -> EvaluationResult:
        logger.info("evaluation_started",
                    experiment_id=experiment_id,
                    model_name=model_name,
                    sample_count=len(prompts))

        # ── 1. Latency + cost metrics ──────────────────────────────────
        metrics_summary: MetricsSummary = self.metrics.measure(
            model_name=model_name,
            prompts=prompts,
        )

        # ── 2. LLM judge scoring ───────────────────────────────────────
        # Judge uses eval_samples directly — generates teacher responses internally
        judge_summary: JudgeSummary = self.judge.evaluate(
            experiment_id=experiment_id,
            model_id=model_id,
            model_name=model_name,
            eval_samples=eval_samples,
        )

        # ── 3. Assemble result ─────────────────────────────────────────
        result = EvaluationResult(
            experiment_id=experiment_id,
            model_id=model_id,
            model_name=model_name,
            run_id=run_id,
            accuracy=judge_summary.win_rate,
            mean_score=judge_summary.mean_score,
            score_distribution=judge_summary.score_distribution,
            latency_p50_ms=metrics_summary.p50_ms,
            latency_p95_ms=metrics_summary.p95_ms,
            latency_p99_ms=metrics_summary.p99_ms,
            cost_per_1k_tokens_usd=metrics_summary.cost_per_1k_tokens_usd,
            total_cost_usd=metrics_summary.total_cost_usd,
            sample_count=judge_summary.sample_count,
            evaluated_at=datetime.now(timezone.utc),
        )

        # ── 4. Log to MLflow ───────────────────────────────────────────
        self._log_to_mlflow(result, mlflow_experiment or "data-flywheel")

        logger.info("evaluation_complete",
                    experiment_id=experiment_id,
                    model_name=model_name,
                    accuracy=result.accuracy,
                    latency_p95_ms=result.latency_p95_ms,
                    cost_per_1k=result.cost_per_1k_tokens_usd)

        return result

    def _log_to_mlflow(self, result: EvaluationResult, experiment_name: str) -> None:
        try:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"{result.model_name}_{result.experiment_id[:8]}"):
                mlflow.set_tags({
                    "experiment_id": result.experiment_id,
                    "model_id": result.model_id,
                    "model_name": result.model_name,
                    "run_id": result.run_id,
                    "evaluated_at": result.evaluated_at.isoformat(),
                })
                mlflow.log_metrics({
                    "accuracy": result.accuracy,
                    "mean_score": result.mean_score,
                    "latency_p50_ms": result.latency_p50_ms,
                    "latency_p95_ms": result.latency_p95_ms,
                    "latency_p99_ms": result.latency_p99_ms,
                    "cost_per_1k_tokens_usd": result.cost_per_1k_tokens_usd,
                    "total_cost_usd": result.total_cost_usd,
                    "sample_count": result.sample_count,
                    **{f"score_{k}": v
                       for k, v in result.score_distribution.items()},
                })
        except Exception as e:
            logger.warning("mlflow_log_failed",
                           experiment_id=result.experiment_id,
                           error=str(e))