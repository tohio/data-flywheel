"""
Unit tests for the Celery Beat scheduled task.
Elasticsearch and MongoDB calls are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestScheduledRun:

    def _run_with_log_count(self, count: int, min_logs: int = 500) -> dict:
        """Helper — runs scheduled_run with a mocked log count and config."""
        mock_cfg = {"schedule": {"cron": "0 */6 * * *", "min_new_logs": min_logs}}

        with patch("orchestrator.workers.scheduled._count_new_logs", return_value=count), \
             patch("orchestrator.workers.scheduled._load_schedule_config",
                   return_value=mock_cfg["schedule"]), \
             patch("orchestrator.workers.scheduled.start_flywheel_run") as mock_start, \
             patch("orchestrator.workers.scheduled.MongoClient") as mock_mongo:

            mock_db = MagicMock()
            mock_mongo.return_value.__getitem__.return_value = mock_db

            from orchestrator.workers.scheduled import scheduled_run
            result = scheduled_run()
            return result, mock_start

    def test_skips_when_insufficient_logs(self):
        result, mock_start = self._run_with_log_count(count=100, min_logs=500)
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_logs"
        assert result["new_logs"] == 100
        mock_start.delay.assert_not_called()

    def test_starts_when_sufficient_logs(self):
        result, mock_start = self._run_with_log_count(count=750, min_logs=500)
        assert result["status"] == "started"
        assert "run_id" in result
        assert result["new_logs"] == 750
        mock_start.delay.assert_called_once()

    def test_skips_at_exact_boundary(self):
        """Fewer than min — not equal — should skip."""
        result, mock_start = self._run_with_log_count(count=499, min_logs=500)
        assert result["status"] == "skipped"
        mock_start.delay.assert_not_called()

    def test_starts_at_exact_minimum(self):
        """Exactly min_new_logs should trigger a run."""
        result, mock_start = self._run_with_log_count(count=500, min_logs=500)
        assert result["status"] == "started"
        mock_start.delay.assert_called_once()

    def test_run_id_is_unique(self):
        """Each scheduled run gets a unique run_id."""
        result1, _ = self._run_with_log_count(count=600)
        result2, _ = self._run_with_log_count(count=600)
        assert result1["run_id"] != result2["run_id"]

    def test_triggered_by_recorded(self):
        """Run document should record triggered_by=celery_beat."""
        with patch("orchestrator.workers.scheduled._count_new_logs", return_value=600), \
             patch("orchestrator.workers.scheduled._load_schedule_config",
                   return_value={"cron": "0 */6 * * *", "min_new_logs": 500}), \
             patch("orchestrator.workers.scheduled.start_flywheel_run"), \
             patch("orchestrator.workers.scheduled.MongoClient") as mock_mongo:

            mock_col = MagicMock()
            mock_mongo.return_value.__getitem__.return_value.flywheel_runs = mock_col

            from orchestrator.workers.scheduled import scheduled_run
            scheduled_run()

            call_args = mock_col.insert_one.call_args[0][0]
            assert call_args["triggered_by"] == "celery_beat"
            assert call_args["status"] == "pending"
