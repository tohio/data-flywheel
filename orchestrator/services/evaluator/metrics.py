"""
MetricsCollector
----------------
Measures latency and cost for a candidate model alongside
the judge accuracy score. These three signals together drive
the promotion decision:

  - accuracy  (from LLMJudge win_rate)
  - latency   (p95 across eval samples, measured live)
  - cost      (estimated from token counts + Groq pricing)
"""
import time
import statistics
from dataclasses import dataclass

from groq import Groq

from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

# Groq pricing as of 2025 (USD per 1M tokens) — update as needed
_GROQ_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile":  {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":     {"input": 0.05, "output": 0.08},
    "llama-3.2-3b-preview":     {"input": 0.06, "output": 0.06},
    "llama-3.2-1b-preview":     {"input": 0.04, "output": 0.04},
}
_DEFAULT_PRICING = {"input": 0.10, "output": 0.10}


@dataclass
class LatencyMetrics:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


@dataclass
class CostMetrics:
    total_input_tokens: int
    total_output_tokens: int
    cost_per_1k_tokens_usd: float
    total_cost_usd: float


@dataclass
class ModelMetrics:
    model_name: str
    sample_count: int
    latency: LatencyMetrics
    cost: CostMetrics


class MetricsCollector:

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def measure(
        self,
        model_name: str,
        prompts: list[str],
    ) -> ModelMetrics:
        """
        Run inference on all prompts and collect latency + token metrics.
        Returns a ModelMetrics dataclass.
        """
        latencies_ms = []
        total_input_tokens = 0
        total_output_tokens = 0

        for prompt in prompts:
            t0 = time.time()
            resp = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            latency_ms = (time.time() - t0) * 1000
            latencies_ms.append(latency_ms)

            usage = resp.usage
            total_input_tokens += usage.prompt_tokens
            total_output_tokens += usage.completion_tokens

            time.sleep(0.05)  # small rate limit buffer

        latency_metrics = self._compute_latency(latencies_ms)
        cost_metrics = self._compute_cost(
            model_name, total_input_tokens, total_output_tokens, len(prompts)
        )

        logger.info("metrics_collected",
                    model=model_name,
                    samples=len(prompts),
                    p95_ms=latency_metrics.p95_ms,
                    cost_per_1k=cost_metrics.cost_per_1k_tokens_usd)

        return ModelMetrics(
            model_name=model_name,
            sample_count=len(prompts),
            latency=latency_metrics,
            cost=cost_metrics,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _compute_latency(self, latencies_ms: list[float]) -> LatencyMetrics:
        sorted_l = sorted(latencies_ms)
        n = len(sorted_l)

        def percentile(p: float) -> float:
            idx = int(p / 100 * n)
            return round(sorted_l[min(idx, n - 1)], 2)

        return LatencyMetrics(
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            mean_ms=round(statistics.mean(latencies_ms), 2),
        )

    def _compute_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        sample_count: int,
    ) -> CostMetrics:
        pricing = _GROQ_PRICING.get(model_name, _DEFAULT_PRICING)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        total_tokens = input_tokens + output_tokens
        cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens else 0.0

        return CostMetrics(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            cost_per_1k_tokens_usd=round(cost_per_1k, 6),
            total_cost_usd=round(total_cost, 6),
        )
