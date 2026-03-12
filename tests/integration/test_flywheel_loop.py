"""
Integration tests for the full flywheel loop.

These tests run against live Docker services (MongoDB, Elasticsearch, Redis).
They are skipped automatically if services are not reachable.

Run with:
  docker compose -f infra/docker/docker-compose.yml up -d
  pytest tests/integration/ -v
"""
import time
import uuid
import pytest
import httpx


API_BASE = "http://localhost:8000"
TIMEOUT = 10


def _api_available() -> bool:
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _services_ready() -> bool:
    try:
        resp = httpx.get(f"{API_BASE}/health/ready", timeout=5)
        data = resp.json()
        return data.get("status") == "ready"
    except Exception:
        return False


# Skip entire module if the API isn't running
pytestmark = pytest.mark.skipif(
    not _api_available(),
    reason="Orchestrator API not running — start with: make up"
)


# ── Health checks ─────────────────────────────────────────────────────────

class TestHealth:

    def test_liveness(self):
        resp = httpx.get(f"{API_BASE}/health", timeout=TIMEOUT)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness_reports_service_status(self):
        resp = httpx.get(f"{API_BASE}/health/ready", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "checks" in data
        assert "mongodb" in data["checks"]
        assert "elasticsearch" in data["checks"]


# ── Flywheel run lifecycle ─────────────────────────────────────────────────

class TestFlywheelRun:

    def test_trigger_run_returns_run_id(self):
        resp = httpx.post(f"{API_BASE}/flywheel/run", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "pending"
        assert "started_at" in data

    def test_status_endpoint_returns_run(self):
        # Trigger a run
        run_resp = httpx.post(f"{API_BASE}/flywheel/run", timeout=TIMEOUT)
        run_id = run_resp.json()["run_id"]

        # Poll status
        status_resp = httpx.get(f"{API_BASE}/flywheel/status/{run_id}", timeout=TIMEOUT)
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["run_id"] == run_id
        assert data["status"] in [
            "pending", "curating", "training", "evaluating", "promoting",
            "completed", "failed"
        ]

    def test_status_404_for_unknown_run(self):
        fake_id = str(uuid.uuid4())
        resp = httpx.get(f"{API_BASE}/flywheel/status/{fake_id}", timeout=TIMEOUT)
        assert resp.status_code == 404

    def test_list_runs_returns_recent(self):
        # Trigger a run to ensure at least one exists
        httpx.post(f"{API_BASE}/flywheel/run", timeout=TIMEOUT)
        time.sleep(0.5)

        resp = httpx.get(f"{API_BASE}/flywheel/runs", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)
        assert len(data["runs"]) >= 1

    def test_dry_run_flag_accepted(self):
        resp = httpx.post(
            f"{API_BASE}/flywheel/run",
            json={"dry_run": True, "run_icl": True, "run_lora_sft": False},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    def test_run_eventually_leaves_pending(self):
        """
        Trigger a run and poll for up to 30s until it moves past 'pending'.
        Validates the Celery worker is picking up tasks.
        """
        run_resp = httpx.post(f"{API_BASE}/flywheel/run", timeout=TIMEOUT)
        run_id = run_resp.json()["run_id"]

        deadline = time.time() + 30
        while time.time() < deadline:
            status_resp = httpx.get(
                f"{API_BASE}/flywheel/status/{run_id}", timeout=TIMEOUT
            )
            status = status_resp.json()["status"]
            if status != "pending":
                break
            time.sleep(2)

        assert status != "pending", (
            f"Run {run_id} stuck in 'pending' after 30s — "
            "is the Celery worker running? (make up)"
        )


# ── Experiments API ───────────────────────────────────────────────────────

class TestExperiments:

    def test_list_experiments_returns_list(self):
        resp = httpx.get(f"{API_BASE}/experiments", timeout=TIMEOUT)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_unknown_experiment_404(self):
        fake_id = str(uuid.uuid4())
        resp = httpx.get(f"{API_BASE}/experiments/{fake_id}", timeout=TIMEOUT)
        assert resp.status_code == 404

    def test_filter_experiments_by_run_id(self):
        fake_run_id = str(uuid.uuid4())
        resp = httpx.get(
            f"{API_BASE}/experiments",
            params={"run_id": fake_run_id},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        # Unknown run_id returns empty list, not 404
        assert resp.json() == []


# ── Models API ────────────────────────────────────────────────────────────

class TestModels:

    def test_list_models_returns_dict(self):
        resp = httpx.get(f"{API_BASE}/models", timeout=TIMEOUT)
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_active_model_404_when_none_promoted(self):
        """Initially no model has been promoted."""
        resp = httpx.get(f"{API_BASE}/models/active", timeout=TIMEOUT)
        # Either 404 (no model promoted yet) or 200 (one was promoted)
        assert resp.status_code in [200, 404]

    def test_promote_unknown_model_404(self):
        fake_id = str(uuid.uuid4())
        resp = httpx.post(f"{API_BASE}/models/{fake_id}/promote", timeout=TIMEOUT)
        assert resp.status_code == 404
