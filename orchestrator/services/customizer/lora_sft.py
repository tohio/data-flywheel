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
       Parameter-efficient fine-tuning using TRL + PEFT directly
       in the worker. No external training API required.
       Trains locally, then pushes the adapter to HF Hub.

       model       → Groq model name, used for inference
       hf_model    → HF Hub model ID, used for TRL training
"""
import os
import uuid
import tempfile
from datetime import datetime, timezone

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
            model_name = candidate["model"]         # Groq name — for inference
            hf_model = candidate.get("hf_model")    # HF Hub ID — for training
            experiments = candidate.get("experiments", [])

            if "icl" in experiments and config.get("run_icl", True):
                exp_id = self._run_icl_experiment(
                    run_id, model_id, model_name, dataset_id, samples
                )
                experiment_ids.append(exp_id)

            if "lora_sft" in experiments and config.get("run_lora_sft", True):
                if not hf_model:
                    logger.warning("lora_skipped_no_hf_model",
                                   model_id=model_id,
                                   reason="hf_model not set in models.yaml")
                    continue
                exp_id = self._run_lora_experiment(
                    run_id, model_id, model_name, hf_model, dataset_id,
                    samples, flywheel_cfg.get("experiments", {})
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
            "adapter_repo_id": None,
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
        hf_model: str,
        dataset_id: str,
        samples: list[dict],
        exp_config: dict,
    ) -> str:
        """
        Fine-tune using TRL + PEFT directly in the worker.
        Uses hf_model (HF Hub ID) for training, model_name (Groq) for inference.
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
            "hf_model": hf_model,
            "experiment_type": "lora_sft",
            "status": "training",
            "dataset_id": dataset_id,
            "adapter_repo_id": None,
            "created_at": datetime.now(timezone.utc),
            "metrics": {},
            "promoted": False,
        })
        client.close()

        try:
            hf_username = self.hf._get_username()

            # 1. Upload dataset to HF Hub
            dataset_repo_id = f"{hf_username}/flywheel-dataset-{run_id[:8]}"
            self.hf.upload_dataset(dataset_id, samples, dataset_repo_id)

            # 2. Train LoRA adapter with TRL + PEFT
            adapter_dir = self._train_lora(
                hf_model=hf_model,
                dataset_repo_id=dataset_repo_id,
                exp_config=exp_config,
            )

            # 3. Push adapter to HF Hub
            adapter_repo_id = f"{hf_username}/flywheel-adapter-{model_id}-{run_id[:8]}"
            self.hf.push_adapter(adapter_dir, adapter_repo_id)

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

    # ── TRL + PEFT training ───────────────────────────────────────────────

    def _train_lora(
        self,
        hf_model: str,
        dataset_repo_id: str,
        exp_config: dict,
    ) -> str:
        """
        Fine-tune a model with LoRA SFT using TRL + PEFT.
        hf_model is the HF Hub model ID (e.g. meta-llama/Llama-3.2-1B).
        Returns path to the saved adapter directory.
        """
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig

        output_dir = tempfile.mkdtemp(prefix="flywheel-lora-")

        logger.info("lora_training_started",
                    model=hf_model,
                    dataset=dataset_repo_id,
                    output_dir=output_dir)

        # ── Determine device ──────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_4bit = device == "cuda"

        logger.info("lora_device", device=device, use_4bit=use_4bit)

        # ── Load tokenizer ────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model,
            token=settings.HF_TOKEN,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Load model ────────────────────────────────────────────────────
        model_kwargs = {
            "token": settings.HF_TOKEN,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(hf_model, **model_kwargs)

        # ── LoRA config ───────────────────────────────────────────────────
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=exp_config.get("lora_r", 16),
            lora_alpha=exp_config.get("lora_alpha", 32),
            lora_dropout=exp_config.get("lora_dropout", 0.05),
            target_modules="all-linear",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── Load dataset ──────────────────────────────────────────────────
        dataset = load_dataset(
            dataset_repo_id,
            token=settings.HF_TOKEN,
            split="train",
        )

        # ── Training config ───────────────────────────────────────────────
        # Fewer steps on CPU to keep dev runs fast
        max_steps = exp_config.get("lora_max_steps", 500) if device == "cuda" else 10

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=exp_config.get("lora_batch_size", 4),
            gradient_accumulation_steps=2,
            learning_rate=exp_config.get("lora_lr", 2e-4),
            fp16=device == "cuda",
            logging_steps=5,
            save_steps=max_steps,
            save_total_limit=1,
            remove_unused_columns=False,
        )

        # ── Train ─────────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        # ── Save adapter only ─────────────────────────────────────────────
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("lora_training_complete", output_dir=output_dir, steps=max_steps)
        return output_dir

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