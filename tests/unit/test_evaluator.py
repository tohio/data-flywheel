"""
Unit tests for the evaluator services.
All external calls (Groq API, MongoDB) are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch

from orchestrator.services.evaluator.metrics import MetricsCollector, _GROQ_PRICING
from orchestrator.services.evaluator.judge import LLMJudge, JudgeResult


# ── MetricsCollector tests ────────────────────────────────────────────────

class TestMetricsCollector:

    def test_latency_percentiles_correct(self):
        collector = MetricsCollector.__new__(MetricsCollector)
        latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        result = collector._compute_latency(latencies)
        assert result.p50_ms == latencies[5]       # 60th index approx
        assert result.p95_ms == latencies[9]       # 100th index
        assert result.mean_ms == 550.0

    def test_latency_single_sample(self):
        collector = MetricsCollector.__new__(MetricsCollector)
        result = collector._compute_latency([250.0])
        assert result.p50_ms == 250.0
        assert result.p95_ms == 250.0
        assert result.p99_ms == 250.0

    def test_cost_calculation_known_model(self):
        collector = MetricsCollector.__new__(MetricsCollector)
        result = collector._compute_cost(
            model_name="llama-3.1-8b-instant",
            input_tokens=10_000,
            output_tokens=5_000,
            sample_count=10,
        )
        pricing = _GROQ_PRICING["llama-3.1-8b-instant"]
        expected_input = (10_000 / 1_000_000) * pricing["input"]
        expected_output = (5_000 / 1_000_000) * pricing["output"]
        expected_total = expected_input + expected_output
        assert abs(result.total_cost_usd - expected_total) < 1e-8
        assert result.total_input_tokens == 10_000
        assert result.total_output_tokens == 5_000

    def test_cost_calculation_unknown_model_uses_default(self):
        collector = MetricsCollector.__new__(MetricsCollector)
        result = collector._compute_cost(
            model_name="unknown-model-xyz",
            input_tokens=1_000,
            output_tokens=1_000,
            sample_count=5,
        )
        # Should not raise, should use default pricing
        assert result.cost_per_1k_tokens_usd >= 0
        assert result.total_cost_usd >= 0

    def test_cost_zero_tokens(self):
        collector = MetricsCollector.__new__(MetricsCollector)
        result = collector._compute_cost("llama-3.1-8b-instant", 0, 0, 0)
        assert result.cost_per_1k_tokens_usd == 0.0
        assert result.total_cost_usd == 0.0

    @patch("orchestrator.services.evaluator.metrics.Groq")
    def test_measure_calls_groq_for_each_prompt(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        # Mock Groq response
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "test response"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("orchestrator.core.config.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "test-key"
            collector = MetricsCollector()
            collector.client = mock_client

        prompts = ["prompt one", "prompt two", "prompt three"]
        result = collector.measure("llama-3.1-8b-instant", prompts)

        assert mock_client.chat.completions.create.call_count == 3
        assert result.sample_count == 3
        assert result.model_name == "llama-3.1-8b-instant"


# ── LLMJudge tests ────────────────────────────────────────────────────────

class TestLLMJudge:

    def test_summarise_empty(self):
        judge = LLMJudge.__new__(LLMJudge)
        summary = judge._summarise("exp-1", "model-1", [])
        assert summary.sample_count == 0
        assert summary.win_rate == 0.0
        assert summary.mean_score == 0.0

    def test_summarise_all_wins(self):
        judge = LLMJudge.__new__(LLMJudge)
        results = [
            JudgeResult("p", "r", "c", score=4, reason="good", latency_ms=200),
            JudgeResult("p", "r", "c", score=5, reason="great", latency_ms=150),
            JudgeResult("p", "r", "c", score=4, reason="good", latency_ms=180),
        ]
        summary = judge._summarise("exp-1", "model-1", results)
        assert summary.win_rate == 1.0
        assert summary.mean_score == pytest.approx(4.333, abs=0.01)
        assert summary.sample_count == 3

    def test_summarise_no_wins(self):
        judge = LLMJudge.__new__(LLMJudge)
        results = [
            JudgeResult("p", "r", "c", score=1, reason="bad", latency_ms=300),
            JudgeResult("p", "r", "c", score=2, reason="weak", latency_ms=250),
            JudgeResult("p", "r", "c", score=3, reason="ok", latency_ms=200),
        ]
        summary = judge._summarise("exp-1", "model-1", results)
        assert summary.win_rate == 0.0
        assert summary.mean_score == pytest.approx(2.0)

    def test_summarise_score_distribution(self):
        judge = LLMJudge.__new__(LLMJudge)
        results = [
            JudgeResult("p", "r", "c", score=1, reason="", latency_ms=100),
            JudgeResult("p", "r", "c", score=3, reason="", latency_ms=100),
            JudgeResult("p", "r", "c", score=5, reason="", latency_ms=100),
            JudgeResult("p", "r", "c", score=5, reason="", latency_ms=100),
        ]
        summary = judge._summarise("exp-1", "model-1", results)
        assert summary.score_distribution[1] == 1
        assert summary.score_distribution[3] == 1
        assert summary.score_distribution[5] == 2
        assert summary.score_distribution.get(2, 0) == 0

    def test_win_rate_boundary_score_4(self):
        """Score of exactly 4 should count as a win."""
        judge = LLMJudge.__new__(LLMJudge)
        results = [
            JudgeResult("p", "r", "c", score=4, reason="", latency_ms=100),
            JudgeResult("p", "r", "c", score=3, reason="", latency_ms=100),
        ]
        summary = judge._summarise("exp-1", "model-1", results)
        assert summary.win_rate == 0.5

    @patch("orchestrator.services.evaluator.judge.Groq")
    def test_judge_retries_on_parse_failure(self, mock_groq_cls):
        """Judge should retry up to 3 times on JSON parse failure then return score 1."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        bad_resp = MagicMock()
        bad_resp.choices[0].message.content = "not valid json at all"
        mock_client.chat.completions.create.return_value = bad_resp

        judge = LLMJudge.__new__(LLMJudge)
        judge.client = mock_client
        judge.judge_model = "llama-3.3-70b-versatile"

        score, reason = judge._judge("prompt", "reference", "candidate")

        assert score == 1
        assert reason == "parse_error"
        assert mock_client.chat.completions.create.call_count == 3

    @patch("orchestrator.services.evaluator.judge.Groq")
    def test_judge_parses_valid_response(self, mock_groq_cls):
        """Judge should correctly parse a valid JSON score response."""
        import json
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        good_resp = MagicMock()
        good_resp.choices[0].message.content = json.dumps(
            {"score": 4, "reason": "mostly equivalent"}
        )
        mock_client.chat.completions.create.return_value = good_resp

        judge = LLMJudge.__new__(LLMJudge)
        judge.client = mock_client
        judge.judge_model = "llama-3.3-70b-versatile"

        score, reason = judge._judge("prompt", "reference", "candidate")

        assert score == 4
        assert reason == "mostly equivalent"

    def test_judge_score_clamped(self):
        """Out-of-range scores should be clamped to 1-5."""
        import json
        from unittest.mock import MagicMock, patch

        with patch("orchestrator.services.evaluator.judge.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client

            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = json.dumps(
                {"score": 99, "reason": "off the charts"}
            )
            mock_client.chat.completions.create.return_value = mock_resp

            judge = LLMJudge.__new__(LLMJudge)
            judge.client = mock_client
            judge.judge_model = "llama-3.3-70b-versatile"

            score, _ = judge._judge("p", "r", "c")
            assert score == 5


# ── DeploymentManager criteria tests ─────────────────────────────────────

class TestPromotionCriteria:

    def _make_manager(self):
        from orchestrator.services.deployment.manager import DeploymentManager
        manager = DeploymentManager.__new__(DeploymentManager)
        return manager

    def _criteria(self):
        return {
            "min_accuracy": 0.85,
            "max_latency_p95_ms": 800,
            "max_cost_per_1k_tokens": 0.02,
            "min_eval_sample_size": 100,
        }

    def test_all_criteria_pass(self):
        manager = self._make_manager()
        candidate = {
            "accuracy": 0.90,
            "latency_p95_ms": 600,
            "cost_per_1k_tokens": 0.01,
            "sample_count": 150,
        }
        passed, failures = manager._check_criteria(candidate, self._criteria())
        assert passed is True
        assert failures == {}

    def test_accuracy_fails(self):
        manager = self._make_manager()
        candidate = {
            "accuracy": 0.70,          # below 0.85
            "latency_p95_ms": 600,
            "cost_per_1k_tokens": 0.01,
            "sample_count": 150,
        }
        passed, failures = manager._check_criteria(candidate, self._criteria())
        assert passed is False
        assert "accuracy" in failures

    def test_latency_fails(self):
        manager = self._make_manager()
        candidate = {
            "accuracy": 0.90,
            "latency_p95_ms": 1200,    # above 800ms
            "cost_per_1k_tokens": 0.01,
            "sample_count": 150,
        }
        passed, failures = manager._check_criteria(candidate, self._criteria())
        assert passed is False
        assert "latency" in failures

    def test_cost_fails(self):
        manager = self._make_manager()
        candidate = {
            "accuracy": 0.90,
            "latency_p95_ms": 600,
            "cost_per_1k_tokens": 0.05,  # above 0.02
            "sample_count": 150,
        }
        passed, failures = manager._check_criteria(candidate, self._criteria())
        assert passed is False
        assert "cost" in failures

    def test_sample_size_fails(self):
        manager = self._make_manager()
        candidate = {
            "accuracy": 0.90,
            "latency_p95_ms": 600,
            "cost_per_1k_tokens": 0.01,
            "sample_count": 50,          # below 100
        }
        passed, failures = manager._check_criteria(candidate, self._criteria())
        assert passed is False
        assert "sample_size" in failures

    def test_multiple_failures_reported(self):
        manager = self._make_manager()
        candidate = {
            "accuracy": 0.50,
            "latency_p95_ms": 2000,
            "cost_per_1k_tokens": 0.10,
            "sample_count": 10,
        }
        passed, failures = manager._check_criteria(candidate, self._criteria())
        assert passed is False
        assert len(failures) == 4

    def test_exact_boundary_passes(self):
        """Values exactly at the boundary should pass."""
        manager = self._make_manager()
        candidate = {
            "accuracy": 0.85,       # exactly min
            "latency_p95_ms": 800,  # exactly max
            "cost_per_1k_tokens": 0.02,  # exactly max
            "sample_count": 100,    # exactly min
        }
        passed, failures = manager._check_criteria(candidate, self._criteria())
        assert passed is True
