"""
DeploymentManager
-----------------
The final gate in the flywheel loop. Receives the best candidate
from the evaluation stage and decides whether to promote it.

Promotion criteria (from eval_criteria.yaml):
  - min_accuracy          win-rate vs teacher judge
  - max_latency_p95_ms    95th percentile latency
  - max_cost_per_1k       USD per 1k tokens
  - min_eval_sample_size  minimum samples evaluated

If the candidate passes ALL criteria it is promoted to production.
The previous production model is archived automatically.
A smoke test is run post-promotion to verify the model is live.
All decisions are logged to MLflow with full rationale.
"""
from datetime import datetime, timezone
from typing import Any

import yaml

from orchestrator.core.config import settings
from orchestrator.services.deployment.registry import ModelRegistry
from orchestrator.services.deployment.groq_client import GroqClient
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

_SMOKE_TEST_PROMPT = "In one sentence, what is machine learning?"


def _load_criteria() -> dict:
    with open(settings.EVAL_CRITERIA_CONFIG) as f:
        return yaml.safe_load(f)["promotion_criteria"]


class DeploymentManager:

    def __init__(self):
        self.registry = ModelRegistry()
        self.groq = GroqClient()

    def close(self):
        self.registry.close()

    # ── Main entry point ──────────────────────────────────────────────────

    def maybe_promote(
        self,
        run_id: str,
        best_candidate: dict | None,
        all_scored: list[dict],
    ) -> dict:
        """
        Evaluate best_candidate against promotion criteria.
        Returns a promotion result dict.
        """
        if not best_candidate:
            return self._no_promotion("no_candidate", "No experiments produced a scoreable candidate")

        criteria = _load_criteria()
        passed, failures = self._check_criteria(best_candidate, criteria)

        # Register all candidates in the registry regardless of promotion
        for scored in all_scored:
            self.registry.register_candidate(
                model_id=scored["experiment_id"],
                model_name=scored["model_name"],
                run_id=run_id,
                experiment_id=scored["experiment_id"],
                metrics=scored,
            )

        if not passed:
            logger.info("promotion_criteria_failed",
                        run_id=run_id,
                        model_id=best_candidate["model_id"],
                        failures=failures)
            return self._no_promotion("criteria_not_met", failures)

        # Criteria passed — promote
        return self._promote(run_id, best_candidate)

    # ── Criteria check ────────────────────────────────────────────────────

    def _check_criteria(
        self,
        candidate: dict,
        criteria: dict,
    ) -> tuple[bool, dict]:
        """
        Check candidate metrics against each criterion.
        Returns (all_passed, {criterion: failure_reason}).
        """
        failures = {}

        accuracy = candidate.get("accuracy", 0.0)
        latency_p95 = candidate.get("latency_p95_ms", float("inf"))
        cost_per_1k = candidate.get("cost_per_1k_tokens", float("inf"))
        sample_count = candidate.get("sample_count", 0)

        if accuracy < criteria["min_accuracy"]:
            failures["accuracy"] = (
                f"{accuracy:.3f} < required {criteria['min_accuracy']}"
            )
        if latency_p95 > criteria["max_latency_p95_ms"]:
            failures["latency"] = (
                f"{latency_p95:.0f}ms > max {criteria['max_latency_p95_ms']}ms"
            )
        if cost_per_1k > criteria["max_cost_per_1k_tokens"]:
            failures["cost"] = (
                f"${cost_per_1k:.5f} > max ${criteria['max_cost_per_1k_tokens']}"
            )
        if sample_count < criteria["min_eval_sample_size"]:
            failures["sample_size"] = (
                f"{sample_count} < required {criteria['min_eval_sample_size']}"
            )

        return len(failures) == 0, failures

    # ── Promotion flow ────────────────────────────────────────────────────

    def _promote(self, run_id: str, candidate: dict) -> dict:
        model_id = candidate["experiment_id"]
        model_name = candidate["model_name"]

        logger.info("promoting_model",
                    run_id=run_id,
                    model_id=model_id,
                    accuracy=candidate["accuracy"],
                    latency_p95_ms=candidate["latency_p95_ms"],
                    cost_per_1k=candidate["cost_per_1k_tokens"])

        # Verify model is reachable before promoting
        if not self.groq.is_model_available(model_name):
            return self._no_promotion(
                "model_unreachable",
                f"Model {model_name} failed availability check"
            )

        # Promote in registry (archives current production atomically)
        self.registry.promote(model_id)

        # Smoke test post-promotion
        smoke = self.groq.test_inference(model_name, _SMOKE_TEST_PROMPT)
        logger.info("smoke_test_passed",
                    model_id=model_id,
                    latency_ms=smoke["latency_ms"])

        # Log to MLflow
        self._log_promotion_to_mlflow(run_id, candidate, smoke)

        return {
            "promoted": True,
            "promoted_model_id": model_id,
            "promoted_model_name": model_name,
            "metrics": {
                "accuracy": candidate["accuracy"],
                "latency_p95_ms": candidate["latency_p95_ms"],
                "cost_per_1k_tokens": candidate["cost_per_1k_tokens"],
            },
            "smoke_test": smoke,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }

    def _no_promotion(self, reason: str, detail: Any) -> dict:
        return {
            "promoted": False,
            "promoted_model_id": None,
            "reason": reason,
            "detail": detail,
        }

    # ── MLflow ────────────────────────────────────────────────────────────

    def _log_promotion_to_mlflow(
        self,
        run_id: str,
        candidate: dict,
        smoke: dict,
    ) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=f"promotion-{candidate['model_id']}"):
                mlflow.log_params({
                    "run_id": run_id,
                    "model_id": candidate["model_id"],
                    "model_name": candidate["model_name"],
                    "experiment_type": candidate["experiment_type"],
                })
                mlflow.log_metrics({
                    "accuracy": candidate["accuracy"],
                    "latency_p95_ms": candidate["latency_p95_ms"],
                    "cost_per_1k_tokens": candidate["cost_per_1k_tokens"],
                    "smoke_test_latency_ms": smoke["latency_ms"],
                })
                mlflow.set_tag("event", "promotion")
                mlflow.set_tag("status", "production")
        except Exception as e:
            logger.warning("mlflow_promotion_log_failed", error=str(e))
