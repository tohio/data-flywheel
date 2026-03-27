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
        """
        Check whether this result passes promotion criteria.
        Returns (passed, failures) where failures maps criterion → details.
        """
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
    Runs the full evaluation pipeline for a single experiment:
      1. Collect latency + cost metrics via MetricsCollector
      2. Score responses via LLMJudge
      3. Log everything to MLflow
      4. Return a structured EvaluationResult
    """

    def __init__(self):
        self.judge = LLMJudge()
        self.metrics = MetricsCollector()

    def run(
        self,
        experiment_id: str,
        model_id: str,
        model_name: str,
        run_id: str,
        prompts: list[str],
        reference_responses: list[str],
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
        candidate_responses = metrics_summary.responses
        judge_summary: JudgeSummary = self.judge.evaluate(
            experiment_id=experiment_id,
            model_name=model_name,
            prompts=prompts,
            reference_responses=reference_responses,
            candidate_responses=candidate_responses,
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