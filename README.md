# data-flywheel

A self-improving AI pipeline that continuously refines models using real-world interaction data. Implements the full flywheel loop — data curation, model customisation, evaluation, guardrails, and feedback collection — feeding improvements back into the model automatically.

---

## Overview

A standard model deployment degrades over time — data drifts, user behaviour evolves, and the model's outputs become stale. A data flywheel closes that loop. As the deployed model interacts with users, it generates logs and feedback that are curated, used to fine-tune the model, evaluated against benchmarks, and redeployed — continuously and automatically.

This pipeline implements that loop end-to-end, without relying on NVIDIA NeMo infrastructure.

---

## The Flywheel Loop

```
┌─────────────────────────────────────────────┐
│           1. Data Collection                │
│   Inference logs · user feedback · signals  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           2. Data Curation                  │
│   Filter · deduplicate · remove PII/toxic   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           3. Model Customisation            │
│   SFT · LoRA · DPO on curated data          │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           4. Evaluation                     │
│   Benchmark · quality gate · regression     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           5. Deployment                     │
│   Promote checkpoint · update serving layer │
└────────────────────┬────────────────────────┘
                     │
                     ▼
              (loop repeats)
```

---

## Repository Structure

```
data-flywheel/
├── collection/          Stage 1: inference log ingestion and feedback capture
├── curator/             Stage 2: data filtering, deduplication, PII removal
├── trainer/             Stage 3: SFT / LoRA / DPO fine-tuning
├── evaluator/           Stage 4: benchmarking and quality gating
├── deployer/            Stage 5: checkpoint promotion and serving update
├── pipeline.py          Orchestrates the full loop end-to-end
├── .env.sample
├── requirements.txt
└── README.md
```

---

## Getting Started

**Prerequisites**
- Python 3.10+
- GPU recommended for training stage
- API keys (see `.env.sample`)

**Installation**

```bash
git clone https://github.com/tohio/data-flywheel.git
cd data-flywheel

python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.sample .env
# Add your API keys to .env
```

**Run the pipeline**

```bash
python pipeline.py
```

---

## Related Projects

This repo is part of a broader AI engineering portfolio:

- [slm](https://github.com/tohio/slm) — the base model this flywheel improves
- [rag-pipeline](https://github.com/tohio/rag-pipeline) — modular RAG pipeline
- [agentic-rag](https://github.com/tohio/agentic-rag) — agentic RAG with tool use and reasoning
- [multi-agent](https://github.com/tohio/multi-agent) — autonomous multi-agent investment research

---

## License

MIT
