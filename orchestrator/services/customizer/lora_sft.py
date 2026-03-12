"""
LoRASFTService
--------------
Orchestrates LoRA supervised fine-tuning experiments for each
candidate model using the curated dataset.

Two experiment types are supported:

  ICL  (in-context learning)
       No training. Evaluates the base model with a few curated
       examples stuffed into the prompt. Fast, free, zero compute.

  LoRA SFT
       Parameter-efficient fine-tuning via HF AutoTrain.
       Uploads dataset → submits job → polls until complete →
       stores adapter ref in MongoDB experiment record.

Both return an experiment_id that gets passed to the evaluator.
"""
import uuid
from datetime import datetime, timezone
from typing import Any

import yaml

from orchestrator.core.config import settings
from orchestrator.services.customizer.hf_client import HuggingFaceClient
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


def _load_configs() -> tuple[dict, dict]:
    with open(settings.FLYWHEEL_CONFIG) as f:
        flywheel_cfg = yaml.safe_load(f)
    with open(settings.MODELS_CONFIG) as f:
        models_cfg = yaml.safe_load(f)
    return flywheel_cfg, models_cfg


class LoRASFTService:

    def __init__(self):
        self.hf = HuggingFaceClient()

    def run_all_experiments(
        self,
        run_id: str,
        dataset_id: str,
        samples: list[dict],
        config: dict,
    ) -> list[str]:
        """
        Run ICL and/or LoRA SFT for every candidate model.
        Returns list of experiment_ids for the evaluation stage.
        """
        flywheel_cfg, models_cfg = _load_configs()
        candidates = models_cfg["candidates"]
        experiment_ids = []

        for candidate in candidates:
            model_id = candidate["id"]
            model_name = candidate["model"]
            experiments = candidate.get("experiments", [])

            if "icl" in experiments and config.get("run_icl", True):
                exp_id = self._run_icl_experiment(
                    run_id, model_id, model_name, dataset_id, samples
                )
                experiment_ids.append(exp_id)

            if "lora_sft" in experiments and config.get("run_lora_sft", True):
                exp_id = self._run_lora_experiment(
                    run_id, model_id, model_name, dataset_id,
                    samples, flywheel_cfg["experiments"]
                )
                experiment_ids.append(exp_id)

        logger.info("all_experiments_dispatched",
                    run_id=run_id, count=len(experiment_ids))
        return experiment_ids

    # ── ICL experiment ────────────────────────────────────────────────────

    def _run_icl_experiment(
        self,
        run_id: str,
        model_id: str,
        model_name: str,
        dataset_id: str,
        samples: list[dict],
    ) -> str:
        """
        Record an ICL experiment. No training happens here —
        the evaluator will run inference with few-shot examples.
        """
        from pymongo import MongoClient
        experiment_id = str(uuid.uuid4())

        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB]
        db.experiments.insert_one({
            "_id": experiment_id,
            "run_id": run_id,
            "model_id": model_id,
            "model_name": model_name,
            "experiment_type": "icl",
            "status": "pending_eval",
            "dataset_id": dataset_id,
            "adapter_repo_id": None,    # no adapter for ICL
            "created_at": datetime.now(timezone.utc),
            "metrics": {},
            "promoted": False,
        })
        client.close()

        logger.info("icl_experiment_registered",
                    experiment_id=experiment_id,
                    model_id=model_id)
        return experiment_id

    # ── LoRA SFT experiment ───────────────────────────────────────────────

    def _run_lora_experiment(
        self,
        run_id: str,
        model_id: str,
        model_name: str,
        dataset_id: str,
        samples: list[dict],
        exp_config: dict,
    ) -> str:
        """
        Upload dataset → submit LoRA job → poll to completion →
        record experiment with adapter ref.
        """
        from pymongo import MongoClient
        experiment_id = str(uuid.uuid4())

        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB]

        # Register experiment as "training"
        db.experiments.insert_one({
            "_id": experiment_id,
            "run_id": run_id,
            "model_id": model_id,
            "model_name": model_name,
            "experiment_type": "lora_sft",
            "status": "training",
            "dataset_id": dataset_id,
            "adapter_repo_id": None,
            "hf_job_id": None,
            "created_at": datetime.now(timezone.utc),
            "metrics": {},
            "promoted": False,
        })
        client.close()

        try:
            # 1. Upload dataset to HF Hub
            dataset_repo_id = f"data-flywheel-datasets/{run_id[:8]}"
            self.hf.upload_dataset(dataset_id, samples, dataset_repo_id)

            # 2. Submit LoRA job
            adapter_repo_id = f"data-flywheel-adapters/{model_id}-{run_id[:8]}"
            lora_config = {
                "max_seq_length": 2048,
                "max_steps": exp_config.get("lora_max_steps", 500),
                "batch_size": exp_config.get("lora_batch_size", 8),
                "lr": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            }
            job_id = self.hf.submit_lora_job(
                base_model=model_name,
                dataset_repo_id=dataset_repo_id,
                output_repo_id=adapter_repo_id,
                lora_config=lora_config,
            )

            # Update experiment with job_id
            self._update_experiment(experiment_id, {"hf_job_id": job_id})

            # 3. Poll to completion
            self.hf.wait_for_job(job_id)

            # 4. Mark as ready for evaluation
            self._update_experiment(experiment_id, {
                "status": "pending_eval",
                "adapter_repo_id": adapter_repo_id,
                "completed_at": datetime.now(timezone.utc),
            })

            logger.info("lora_experiment_completed",
                        experiment_id=experiment_id,
                        model_id=model_id,
                        adapter_repo_id=adapter_repo_id)

        except Exception as exc:
            self._update_experiment(experiment_id, {
                "status": "failed",
                "error": str(exc),
            })
            logger.error("lora_experiment_failed",
                         experiment_id=experiment_id,
                         error=str(exc))
            raise

        return experiment_id

    # ── Helpers ───────────────────────────────────────────────────────────

    def _update_experiment(self, experiment_id: str, update: dict) -> None:
        from pymongo import MongoClient
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB]
        db.experiments.update_one(
            {"_id": experiment_id},
            {"$set": {**update, "updated_at": datetime.now(timezone.utc)}}
        )
        client.close()
