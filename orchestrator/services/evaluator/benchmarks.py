"""
LLMJudge
--------
Uses Groq (Llama 3.3 70B) as an automated judge to score candidate
model responses against teacher model responses.

Scoring approach:
  For each sample in the eval set:
    1. Get teacher response (Llama 3.3 70B) — ground truth
    2. Get candidate response (base or LoRA-adapted model)
    3. Ask judge to score candidate vs teacher on a 1-5 scale
    4. Aggregate into a win-rate / accuracy score

The judge prompt is designed to be position-unbiased and
criterion-specific (factual accuracy, completeness, coherence).
"""
import time
from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are an impartial AI evaluator. Your task is to compare two responses to the same prompt and score the Candidate response relative to the Reference response.

Score the Candidate on a scale of 1-5:
  5 - Equivalent or better than Reference: fully accurate, complete, and coherent
  4 - Mostly equivalent: minor omissions or wording differences only
  3 - Partially equivalent: correct direction but missing key details
  2 - Weak: significant inaccuracies or missing important content
  1 - Poor: incorrect, incoherent, or completely off-topic

Respond with ONLY a JSON object in this exact format:
{"score": <1-5>, "reason": "<one sentence explanation>"}"""

_JUDGE_USER_TEMPLATE = """Prompt: {prompt}

Reference response: {reference}

Candidate response: {candidate}

Score the Candidate response."""


@dataclass
class JudgeResult:
    prompt: str
    reference: str
    candidate: str
    score: int           # 1-5
    reason: str
    latency_ms: float


@dataclass
class JudgeSummary:
    experiment_id: str
    model_id: str
    sample_count: int
    mean_score: float
    win_rate: float          # fraction of samples scoring >= 4
    score_distribution: dict[int, int] = field(default_factory=dict)
    results: list[JudgeResult] = field(default_factory=list)


class LLMJudge:

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.teacher_model = self._load_teacher_model()
        self.judge_model = "llama-3.3-70b-versatile"   # same model judges

    def _load_teacher_model(self) -> str:
        import yaml
        with open(settings.MODELS_CONFIG) as f:
            cfg = yaml.safe_load(f)
        return cfg["teacher"]["model"]

    # ── Main entry point ──────────────────────────────────────────────────

    def evaluate(
        self,
        experiment_id: str,
        model_id: str,
        model_name: str,
        eval_samples: list[dict],
        adapter_repo_id: str | None = None,
    ) -> JudgeSummary:
        """
        Score a candidate model against the teacher on eval_samples.
        Returns a JudgeSummary with win_rate and full per-sample results.
        """
        logger.info("judge_eval_started",
                    experiment_id=experiment_id,
                    model_id=model_id,
                    samples=len(eval_samples))

        results = []
        for sample in eval_samples:
            result = self._score_sample(
                prompt=sample["prompt"],
                candidate_model=model_name,
            )
            results.append(result)
            time.sleep(0.1)   # rate limit buffer

        summary = self._summarise(experiment_id, model_id, results)
        logger.info("judge_eval_completed",
                    experiment_id=experiment_id,
                    win_rate=summary.win_rate,
                    mean_score=summary.mean_score)
        return summary

    # ── Per-sample scoring ────────────────────────────────────────────────

    def _score_sample(self, prompt: str, candidate_model: str) -> JudgeResult:
        """Get teacher + candidate responses, then ask judge to score."""

        # 1. Teacher response (ground truth)
        reference = self._call_model(self.teacher_model, prompt)

        # 2. Candidate response
        t0 = time.time()
        candidate = self._call_model(candidate_model, prompt)
        latency_ms = (time.time() - t0) * 1000

        # 3. Judge score
        score, reason = self._judge(prompt, reference, candidate)

        return JudgeResult(
            prompt=prompt,
            reference=reference,
            candidate=candidate,
            score=score,
            reason=reason,
            latency_ms=latency_ms,
        )

    def _call_model(self, model: str, prompt: str) -> str:
        """Call a Groq-hosted model and return the text response."""
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def _judge(self, prompt: str, reference: str, candidate: str) -> tuple[int, str]:
        """Ask the judge model to score candidate vs reference. Returns (score, reason)."""
        import json

        user_msg = _JUDGE_USER_TEMPLATE.format(
            prompt=prompt,
            reference=reference,
            candidate=candidate,
        )

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                parsed = json.loads(raw)
                score = int(parsed["score"])
                reason = parsed.get("reason", "")
                return max(1, min(5, score)), reason   # clamp to 1-5

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("judge_parse_failed",
                               attempt=attempt, error=str(e), raw=raw[:200])
                time.sleep(1)

        # Fallback if all attempts fail
        logger.error("judge_failed_all_attempts", prompt=prompt[:100])
        return 1, "parse_error"

    # ── Summarisation ─────────────────────────────────────────────────────

    def _summarise(
        self,
        experiment_id: str,
        model_id: str,
        results: list[JudgeResult],
    ) -> JudgeSummary:
        if not results:
            return JudgeSummary(
                experiment_id=experiment_id,
                model_id=model_id,
                sample_count=0,
                mean_score=0.0,
                win_rate=0.0,
            )

        scores = [r.score for r in results]
        distribution = {i: scores.count(i) for i in range(1, 6)}
        mean_score = sum(scores) / len(scores)
        win_rate = sum(1 for s in scores if s >= 4) / len(scores)

        return JudgeSummary(
            experiment_id=experiment_id,
            model_id=model_id,
            sample_count=len(results),
            mean_score=round(mean_score, 4),
            win_rate=round(win_rate, 4),
            score_distribution=distribution,
            results=results,
        )
