"""
HuggingFaceClient
-----------------
Thin wrapper around the HuggingFace Hub API for:
  - Uploading datasets to HF Hub
  - Pushing trained adapters back to HF Hub
  - Checking if a repo exists
"""
import io
import json
import os

import httpx
from huggingface_hub import HfApi

from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

HF_API_BASE = "https://huggingface.co/api"


class HuggingFaceClient:

    def __init__(self):
        self.token = settings.HF_TOKEN
        self.api = HfApi(token=self.token)
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
        self.api.create_repo(
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

        self.api.upload_file(
            path_or_fileobj=io.BytesIO(content),
            path_in_repo="train.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Curated dataset {dataset_id}",
        )

        logger.info("dataset_uploaded", repo_id=repo_id, samples=len(samples))
        return repo_id

    # ── Adapter push ──────────────────────────────────────────────────────

    def push_adapter(self, local_path: str, repo_id: str) -> str:
        """
        Push a trained LoRA adapter directory to HF Hub.
        Returns the repo_id.
        """
        self.api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True,
        )

        self.api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="LoRA SFT adapter",
        )

        logger.info("adapter_pushed", repo_id=repo_id, local_path=local_path)
        return repo_id

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_username(self) -> str:
        resp = httpx.get(
            f"{HF_API_BASE}/whoami-v2",
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["name"]