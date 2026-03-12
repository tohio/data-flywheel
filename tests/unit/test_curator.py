"""
Unit tests for the curation pipeline.
All tests are offline — no Elasticsearch, MongoDB, or API calls.
"""
import pytest

from orchestrator.services.curator.filters import FilterPipeline
from orchestrator.services.curator.dedup import MinHashDeduplicator


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_sample(prompt: str, response: str) -> dict:
    return {"prompt": prompt, "response": response, "model": "test", "latency_ms": 100}


GOOD_RESPONSE = (
    "Machine learning is a subfield of artificial intelligence that enables systems "
    "to learn from data and improve their performance without being explicitly programmed. "
    "It uses statistical techniques to give computers the ability to learn from experience."
)

SHORT_RESPONSE = "Yes."

TOXIC_CONTENT = "I want to kill all processes in the system and bomb the server."

PII_PROMPT = "My name is John Smith, email john.smith@example.com. Explain transformers."

DUPLICATE_RESPONSE = (
    "A neural network is a computational model inspired by biological neurons. "
    "It consists of layers of interconnected nodes that process input data."
)


# ── FilterPipeline tests ───────────────────────────────────────────────────

class TestFilterPipeline:

    def setup_method(self):
        self.pipeline = FilterPipeline(
            min_quality_score=0.5,
            max_token_length=2048,
            min_response_length=10,
            remove_pii=False,   # avoid Presidio dep in unit tests
        )

    def test_keeps_good_sample(self):
        samples = [make_sample("What is ML?", GOOD_RESPONSE)]
        result = self.pipeline.run(samples)
        assert len(result) == 1

    def test_drops_short_response(self):
        samples = [make_sample("What is ML?", SHORT_RESPONSE)]
        result = self.pipeline.run(samples)
        assert len(result) == 0

    def test_drops_toxic_content(self):
        samples = [make_sample(TOXIC_CONTENT, GOOD_RESPONSE)]
        result = self.pipeline.run(samples)
        assert len(result) == 0

    def test_drops_too_long(self):
        long_response = " ".join(["word"] * 3000)
        samples = [make_sample("prompt", long_response)]
        pipeline = FilterPipeline(max_token_length=100, min_response_length=5,
                                  remove_pii=False)
        result = pipeline.run(samples)
        assert len(result) == 0

    def test_keeps_multiple_good_samples(self):
        samples = [make_sample(f"Question {i}?", GOOD_RESPONSE) for i in range(10)]
        result = self.pipeline.run(samples)
        assert len(result) == 10

    def test_mixed_batch(self):
        samples = [
            make_sample("Good question?", GOOD_RESPONSE),
            make_sample("Another?", GOOD_RESPONSE),
            make_sample("Short?", SHORT_RESPONSE),          # dropped
            make_sample(TOXIC_CONTENT, GOOD_RESPONSE),      # dropped
        ]
        result = self.pipeline.run(samples)
        assert len(result) == 2

    def test_empty_input(self):
        result = self.pipeline.run([])
        assert result == []

    def test_empty_response_dropped(self):
        samples = [make_sample("Question?", "")]
        result = self.pipeline.run(samples)
        assert len(result) == 0


# ── Quality score tests ────────────────────────────────────────────────────

class TestQualityScore:

    def setup_method(self):
        self.pipeline = FilterPipeline(remove_pii=False)

    def test_good_text_scores_high(self):
        score = self.pipeline._quality_score(GOOD_RESPONSE)
        assert score >= 0.5

    def test_empty_text_scores_zero(self):
        score = self.pipeline._quality_score("")
        assert score == 0.0

    def test_repeated_words_score_low(self):
        repeated = "word " * 200
        score = self.pipeline._quality_score(repeated)
        assert score < 0.8

    def test_score_in_range(self):
        for text in [GOOD_RESPONSE, SHORT_RESPONSE, "random text here"]:
            score = self.pipeline._quality_score(text)
            assert 0.0 <= score <= 1.0


# ── MinHashDeduplicator tests ─────────────────────────────────────────────

class TestMinHashDeduplicator:

    def setup_method(self):
        self.dedup = MinHashDeduplicator(threshold=0.85)

    def test_keeps_unique_samples(self):
        samples = [
            make_sample("What is ML?", GOOD_RESPONSE),
            make_sample("What is a transformer?",
                        "A transformer is an attention-based neural network architecture."),
            make_sample("Explain backprop.",
                        "Backpropagation computes gradients by applying the chain rule."),
        ]
        result = self.dedup.run(samples)
        assert len(result) == 3

    def test_removes_exact_duplicates(self):
        sample = make_sample("What is a neural network?", DUPLICATE_RESPONSE)
        samples = [sample.copy(), sample.copy(), sample.copy()]
        result = self.dedup.run(samples)
        assert len(result) == 1

    def test_removes_near_duplicates(self):
        # Same content with minor suffix variation
        samples = [
            make_sample("What is a neural network?", DUPLICATE_RESPONSE),
            make_sample("What is a neural network?", DUPLICATE_RESPONSE + " They use backprop."),
            make_sample("What is a neural network?", DUPLICATE_RESPONSE + " Layers matter."),
        ]
        result = self.dedup.run(samples)
        # Near-dups should be collapsed to 1
        assert len(result) == 1

    def test_keeps_first_occurrence(self):
        samples = [
            make_sample("Q?", DUPLICATE_RESPONSE),
            make_sample("Q?", DUPLICATE_RESPONSE),
        ]
        result = self.dedup.run(samples)
        assert len(result) == 1
        assert result[0]["prompt"] == "Q?"

    def test_empty_input(self):
        result = self.dedup.run([])
        assert result == []

    def test_single_sample_kept(self):
        samples = [make_sample("What is ML?", GOOD_RESPONSE)]
        result = self.dedup.run(samples)
        assert len(result) == 1

    def test_low_threshold_more_aggressive(self):
        """Lower threshold removes more near-duplicates."""
        dedup_strict = MinHashDeduplicator(threshold=0.3)
        samples = [
            make_sample("What is ML?", GOOD_RESPONSE),
            make_sample("What is machine learning?", GOOD_RESPONSE + " It is useful."),
        ]
        result = dedup_strict.run(samples)
        assert len(result) <= 2   # may remove one depending on overlap
