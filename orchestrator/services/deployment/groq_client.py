"""
GroqClient
----------
Thin wrapper around the Groq API for the deployment layer.
Handles model availability checks and inference routing —
since Groq hosts all our candidate models, "deployment"
means verifying the model is reachable and updating our
registry to point traffic at it.
"""
import time

from groq import Groq

from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)


class GroqClient:

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def is_model_available(self, model_name: str) -> bool:
        """Ping the model with a minimal request to verify it's reachable."""
        try:
            self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.warning("model_unavailable", model=model_name, error=str(e))
            return False

    def list_available_models(self) -> list[str]:
        """Return model IDs currently available on Groq."""
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error("list_models_failed", error=str(e))
            return []

    def test_inference(self, model_name: str, prompt: str) -> dict:
        """
        Run a single test inference and return response + latency.
        Used for smoke-testing a newly promoted model.
        """
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        latency_ms = (time.time() - t0) * 1000
        return {
            "response": resp.choices[0].message.content.strip(),
            "latency_ms": round(latency_ms, 2),
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
