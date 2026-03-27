"""
seed_logs.py
------------
Seeds Elasticsearch with synthetic inference logs for local testing.
Generates a realistic mix of:
  - Clean, high-quality samples (majority)
  - Short / low-quality samples (filtered out by curator)
  - Near-duplicate pairs (removed by dedup)
  - Samples with fake PII (redacted by PII filter)

Usage:
  python infra/scripts/seed_logs.py [--count 1000]
"""
import argparse
import random
import uuid
from datetime import datetime, timezone, timedelta

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

ES_HOST = "http://localhost:9200"    #os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX_LOGS", "inference_logs")

# ── Sample data pools ─────────────────────────────────────────────────────

CLEAN_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What are the key principles of transformer architectures?",
    "How does retrieval-augmented generation improve LLM accuracy?",
    "Describe the steps involved in fine-tuning a language model.",
    "What is the vanishing gradient problem and how is it addressed?",
    "Compare LoRA and full fine-tuning for adapting large language models.",
    "What metrics are used to evaluate LLM performance?",
    "How does attention mechanism work in transformer models?",
    "What is the role of tokenization in NLP pipelines?",
    "Explain the concept of model distillation.",
]

CLEAN_RESPONSES = [
    "Supervised learning uses labeled data to train models, while unsupervised learning discovers patterns in unlabeled data. Key differences include the training signal, typical use cases, and evaluation approaches.",
    "Transformer architectures rely on self-attention mechanisms, positional encodings, and feed-forward layers. The attention mechanism allows each token to attend to all others, capturing long-range dependencies efficiently.",
    "RAG improves LLM accuracy by retrieving relevant documents at inference time and including them in the context. This grounds the model's responses in factual, up-to-date information rather than relying solely on parametric knowledge.",
    "Fine-tuning involves selecting a pre-trained model, preparing domain-specific data, configuring hyperparameters like learning rate and batch size, running training with early stopping, and evaluating on a held-out validation set.",
    "The vanishing gradient problem occurs when gradients become extremely small during backpropagation, preventing early layers from learning. Solutions include residual connections, batch normalisation, and careful weight initialisation.",
]

PII_PROMPTS = [
    "My name is John Smith and my email is john.smith@example.com. Can you help me with ML?",
    "Call me at 555-123-4567. I need help understanding neural networks.",
    "I'm Sarah Johnson from New York. What is backpropagation?",
]

SHORT_RESPONSES = ["Yes.", "No.", "Maybe.", "OK.", "Sure."]

DUPLICATE_PROMPT = "What is a neural network?"
DUPLICATE_RESPONSE = "A neural network is a computational model inspired by biological neurons. It consists of layers of interconnected nodes that process and transform input data to produce outputs."


def generate_log(prompt: str, response: str, latency_ms: int = None) -> dict:
    return {
        "prompt": prompt,
        "response": response,
        "model": random.choice(["llama-3.3-70b", "llama-3.1-8b"]),
        "latency_ms": latency_ms or random.randint(150, 2000),
        "timestamp": (
            datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 48))
        ).isoformat(),
        "session_id": str(uuid.uuid4()),
    }


def seed(count: int = 1000) -> None:
    es = Elasticsearch(ES_HOST)

    # Ensure index exists
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body={
            "mappings": {
                "properties": {
                    "prompt": {"type": "text"},
                    "response": {"type": "text"},
                    "model": {"type": "keyword"},
                    "latency_ms": {"type": "integer"},
                    "timestamp": {"type": "date"},
                    "session_id": {"type": "keyword"},
                    "curated_in_run": {"type": "keyword"},
                }
            }
        })
        print(f"Created index: {ES_INDEX}")

    ops = []
    clean_count = int(count * 0.70)
    pii_count = int(count * 0.10)
    short_count = int(count * 0.10)
    dup_count = count - clean_count - pii_count - short_count

    # Clean samples
    for _ in range(clean_count):
        prompt = random.choice(CLEAN_PROMPTS)
        response = random.choice(CLEAN_RESPONSES)
        ops.append({"index": {"_index": ES_INDEX, "_id": str(uuid.uuid4())}})
        ops.append(generate_log(prompt, response))

    # PII samples
    for _ in range(pii_count):
        prompt = random.choice(PII_PROMPTS)
        response = random.choice(CLEAN_RESPONSES)
        ops.append({"index": {"_index": ES_INDEX, "_id": str(uuid.uuid4())}})
        ops.append(generate_log(prompt, response))

    # Short (low quality) samples
    for _ in range(short_count):
        prompt = random.choice(CLEAN_PROMPTS)
        response = random.choice(SHORT_RESPONSES)
        ops.append({"index": {"_index": ES_INDEX, "_id": str(uuid.uuid4())}})
        ops.append(generate_log(prompt, response))

    # Near-duplicate samples
    for _ in range(dup_count):
        ops.append({"index": {"_index": ES_INDEX, "_id": str(uuid.uuid4())}})
        # Slight variation to test near-dedup
        response = DUPLICATE_RESPONSE + random.choice(["", " It uses backpropagation.", " It can learn complex patterns."])
        ops.append(generate_log(DUPLICATE_PROMPT, response))

    es.bulk(body=ops)
    es.indices.refresh(index=ES_INDEX)

    total = es.count(index=ES_INDEX)["count"]
    print(f"Seeded {count} logs ({clean_count} clean, {pii_count} PII, "
          f"{short_count} short, {dup_count} duplicates)")
    print(f"Total docs in {ES_INDEX}: {total}")
    es.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    seed(args.count)
