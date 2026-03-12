"""
FilterPipeline
--------------
Runs a sequence of filters over raw inference logs.
Each filter returns a (kept, reason) decision per sample.

Filters applied in order:
  1. Length filter     — too short / too long
  2. Quality filter    — heuristic score (punctuation, alpha ratio, etc.)
  3. PII filter        — detect and optionally redact PII (Presidio)
  4. Toxicity filter   — keyword blocklist (lightweight, no model needed)
"""
import re
from dataclasses import dataclass
from typing import Any

from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

# Lightweight toxicity blocklist — extend as needed
_TOXIC_PATTERNS = [
    r"\b(kill|murder|rape|terrorist|bomb|suicide)\b",
]
_TOXIC_RE = re.compile("|".join(_TOXIC_PATTERNS), re.IGNORECASE)


@dataclass
class FilterResult:
    kept: list[dict]
    dropped: dict[str, int]   # reason → count


class FilterPipeline:

    def __init__(
        self,
        min_quality_score: float = 0.7,
        max_token_length: int = 2048,
        min_response_length: int = 20,
        remove_pii: bool = True,
    ):
        self.min_quality_score = min_quality_score
        self.max_token_length = max_token_length
        self.min_response_length = min_response_length
        self.remove_pii = remove_pii

        self._pii_analyzer = None
        if remove_pii:
            self._pii_analyzer = self._init_pii_analyzer()

    def run(self, samples: list[dict]) -> list[dict]:
        kept = []
        dropped: dict[str, int] = {}

        for sample in samples:
            reason = self._filter(sample)
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                kept.append(sample)

        logger.info("filter_summary", kept=len(kept), dropped=dropped)
        return kept

    # ── Per-sample decision ───────────────────────────────────────────────

    def _filter(self, sample: dict) -> str | None:
        """Return a drop reason string, or None if the sample should be kept."""
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        text = f"{prompt} {response}"

        # 1. Length checks
        if len(response.split()) < self.min_response_length:
            return "too_short"

        approx_tokens = len(text.split()) * 1.3
        if approx_tokens > self.max_token_length:
            return "too_long"

        # 2. Quality score
        if self._quality_score(text) < self.min_quality_score:
            return "low_quality"

        # 3. Toxicity
        if _TOXIC_RE.search(text):
            return "toxic"

        # 4. PII — redact in-place rather than drop
        if self.remove_pii and self._pii_analyzer:
            sample["prompt"] = self._redact_pii(prompt)
            sample["response"] = self._redact_pii(response)

        return None

    # ── Quality heuristics ────────────────────────────────────────────────

    def _quality_score(self, text: str) -> float:
        """
        Fast heuristic quality score in [0, 1].
        Combines alpha ratio, punctuation density, and repetition penalty.
        No model required — runs in microseconds.
        """
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Alpha ratio — penalise gibberish
        alpha_chars = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_chars / max(len(text), 1)

        # Punctuation density — some is good, too much is bad
        punct_chars = sum(c in ".!?,;:" for c in text)
        punct_ratio = punct_chars / max(len(text), 1)
        punct_score = 1.0 - abs(punct_ratio - 0.05) * 10  # sweet spot ~5%
        punct_score = max(0.0, min(1.0, punct_score))

        # Repetition penalty — detect repeated n-grams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if bigrams:
            unique_ratio = len(set(bigrams)) / len(bigrams)
        else:
            unique_ratio = 1.0

        score = (alpha_ratio * 0.5) + (punct_score * 0.2) + (unique_ratio * 0.3)
        return round(score, 4)

    # ── PII helpers ───────────────────────────────────────────────────────

    def _init_pii_analyzer(self):
        try:
            from presidio_analyzer import AnalyzerEngine
            return AnalyzerEngine()
        except ImportError:
            logger.warning("presidio_not_installed", msg="PII filtering disabled")
            return None

    def _redact_pii(self, text: str) -> str:
        if not self._pii_analyzer or not text:
            return text
        try:
            results = self._pii_analyzer.analyze(text=text, language="en")
            redacted = text
            # Replace from end to preserve offsets
            for r in sorted(results, key=lambda x: x.start, reverse=True):
                redacted = redacted[:r.start] + f"[{r.entity_type}]" + redacted[r.end:]
            return redacted
        except Exception as e:
            logger.warning("pii_redaction_failed", error=str(e))
            return text
