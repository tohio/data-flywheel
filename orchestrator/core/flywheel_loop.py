"""
flywheel_loop.py
----------------
Orchestrates the full flywheel cycle as a Celery chain:

  curate → finetune (ICL + LoRA) → evaluate → promote

Each stage updates the run document in MongoDB so status
can be polled via GET /flywheel/status/{run_id}.
"""

from celery import chain

from orchestrator.core.celery_app import celery_app
from orchestrator.workers.curate import run_curation
from orchestrator.workers.finetune import run_finetuning
from orchestrator.workers.evaluate import run_evaluation
from orchestrator.workers.promote import run_promotion
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(name="flywheel.start", bind=True)
def start_flywheel_run(self, run_id: str, config: dict) -> None:
    """
    Entry point — builds and dispatches the full Celery task chain.

    The chain passes its result forward at each step:
      curate result → finetune → evaluate → promote
    """
    logger.info("flywheel_chain_starting", run_id=run_id, task_id=self.request.id)

    task_chain = chain(
        run_curation.s(run_id, config),
        run_finetuning.s(run_id, config),
        run_evaluation.s(run_id, config),
        run_promotion.s(run_id, config),
    )

    task_chain.apply_async()
    logger.info("flywheel_chain_dispatched", run_id=run_id)
