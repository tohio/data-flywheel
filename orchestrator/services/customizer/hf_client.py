"""
HuggingFaceClient
-----------------
Thin wrapper around the HuggingFace API for:
  - Uploading datasets to HF Hub (for AutoTrain / TRL jobs)
  - Submitting AutoTrain fine-tuning jobs
  - Polling job status
  - Downloading completed model artifacts

We use HF AutoTrain for serverless LoRA SFT — no local GPU required.
Jobs run on HF infrastructure and the resulting adapter is pushed
back to a private HF repo.
"""
import time
from typing import Any

import httpx

from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

HF_API_BASE = "https://huggingface.co/api"
HF_AUTOTRAIN_BASE = "https://api.autotrain.huggingface.co"


class HuggingFaceClient:

    def __init__(self):
        self.token = settings.HF_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ── Dataset upload ────────────────────────────────────────────────────

    def upload_dataset(
        self,
        dataset_id: str,
        samples: list[dict],
        repo_id: str,
    ) -> str:
        """
        Convert curated samples to JSONL and push to a HF dataset repo.
        Returns the repo_id for use in the training job.
        """
        import json
        from huggingface_hub import HfApi, CommitOperationAdd

        api = HfApi(token=self.token)

        # Ensure repo exists
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

        # Format as chat JSONL — standard SFT format
        jsonl_lines = []
        for s in samples:
            jsonl_lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": s["prompt"]},
                    {"role": "assistant", "content": s["response"]},
                ]
            }))

        content = "\n".join(jsonl_lines).encode("utf-8")

        api.commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo="train.jsonl",
                    path_or_fileobj=content,
                )
            ],
            commit_message=f"Curated dataset {dataset_id}",
        )

        logger.info("dataset_uploaded", repo_id=repo_id, samples=len(samples))
        return repo_id

    # ── AutoTrain job ─────────────────────────────────────────────────────

    def submit_lora_job(
        self,
        base_model: str,
        dataset_repo_id: str,
        output_repo_id: str,
        lora_config: dict,
    ) -> str:
        """
        Submit a LoRA SFT job to HF AutoTrain.
        Returns a job_id for polling.
        """
        payload = {
            "model": base_model,
            "task": "llm-sft",
            "backend": "spaces-a10gl",    # A10G GPU on HF Spaces
            "data": {
                "path": dataset_repo_id,
                "train_split": "train",
                "text_column": "messages",
                "chat_template": "chatml",
            },
            "params": {
                "max_seq_length": lora_config.get("max_seq_length", 2048),
                "max_steps": lora_config.get("max_steps", 500),
                "batch_size": lora_config.get("batch_size", 8),
                "lr": lora_config.get("lr", 2e-4),
                "peft": True,
                "quantization": "int4",
                "lora_r": lora_config.get("lora_r", 16),
                "lora_alpha": lora_config.get("lora_alpha", 32),
                "lora_dropout": lora_config.get("lora_dropout", 0.05),
            },
            "hub": {
                "username": self._get_username(),
                "token": self.token,
                "push_to_hub": True,
                "model_repo": output_repo_id,
                "private": True,
            },
        }

        resp = httpx.post(
            f"{HF_AUTOTRAIN_BASE}/v1/projects",
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json()["id"]
        logger.info("lora_job_submitted", job_id=job_id, base_model=base_model)
        return str(job_id)

    # ── Job polling ───────────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> dict:
        """Poll AutoTrain job status. Returns status dict."""
        resp = httpx.get(
            f"{HF_AUTOTRAIN_BASE}/v1/projects/{job_id}",
            headers=self.headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "job_id": job_id,
            "status": data.get("status", "unknown"),   # queued|running|completed|failed
            "progress": data.get("progress", 0),
            "logs": data.get("logs", ""),
        }

    def wait_for_job(self, job_id: str, poll_interval: int = 60, timeout: int = 7200) -> dict:
        """
        Block until job completes or times out.
        poll_interval and timeout are in seconds.
        """
        start = time.time()
        while True:
            status = self.get_job_status(job_id)
            logger.info("lora_job_polling",
                        job_id=job_id,
                        status=status["status"],
                        progress=status["progress"])

            if status["status"] == "completed":
                return status
            if status["status"] == "failed":
                raise RuntimeError(f"AutoTrain job {job_id} failed: {status['logs']}")
            if time.time() - start > timeout:
                raise TimeoutError(f"AutoTrain job {job_id} timed out after {timeout}s")

            time.sleep(poll_interval)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_username(self) -> str:
        resp = httpx.get(
            f"{HF_API_BASE}/whoami",
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["name"]
