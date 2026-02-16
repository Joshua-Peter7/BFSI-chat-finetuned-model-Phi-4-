BFSI Conversational AI Assistant (Local, Compliant, Tiered)

A privacy-first, local-only, BFSI-compliant conversational AI system designed to handle common Banking, Financial Services, and Insurance (BFSI) customer queries with deterministic responses, strict guardrails, and minimal hallucination risk.

This system is built for call-center automation, kiosks, and internal support tools, prioritizing safety, explainability, and control over raw generative creativity.

🚀 Key Objectives

Handle common BFSI queries (loans, EMI, policies, payments, accounts)

Operate fully locally using a Small Language Model (SLM)

Enforce strict safety & compliance guardrails

Prefer dataset-driven answers over free generation

Escalate responsibly when confidence is low

🏦 Supported Query Categories

Loan eligibility & application requirements

EMI details & repayment schedules

Interest rate & policy information (non-numerical)

Payment & transaction queries

Basic account & customer support requests

⚠️ The system never guesses interest rates, eligibility outcomes, or personalized financial data.

🧠 High-Level Architecture
User (Text UI / Optional Voice)
        ↓
Privacy-First Preprocessing
(PII masking, normalization, context extraction)
        ↓
Intent & Similarity Engine
(BGE-M3 Embeddings + Qdrant)
        ↓
Decision Router
(Confidence scoring + Guardrails)
        ↓
┌───────────────┬────────────────┬──────────────────┐
│ Tier 1        │ Tier 2          │ Tier 3           │
│ Dataset KB    │ Fine-tuned SLM  │ RAG / Escalation │
│ (Primary)     │ (PHI-4 + LoRA)  │ (LightRAG)       │
└───────────────┴────────────────┴──────────────────┘
        ↓
Safety & Compliance Filter
(Llama Guard 3)
        ↓
Response Formatter
(Pydantic-based, deterministic)
        ↓
Audit Logging

🧩 Three-Tier Response Strategy
Tier 1 — Dataset Knowledge Base (Primary)

Uses BGE-M3 embeddings + Qdrant

Returns exact, pre-approved responses

No text generation

Fastest and safest path

Tier 2 — Fine-Tuned SLM (PHI-4)

LoRA-fine-tuned PHI-4

Deterministic decoding (temperature = 0)

Used only when Tier-1 confidence is insufficient

Refuses guessing and unsafe outputs

Tier 3 — RAG / Escalation

Used for complex policy explanations or low confidence cases

Retrieves from structured policy documents only

Human escalation when required

🔐 Safety & Compliance Design

This system is designed with BFSI compliance as a first-class requirement.

Built-in Guardrails

PII detection & masking before any model invocation

No hallucinated numbers (interest rates, eligibility, limits)

No personalized financial decisions

Llama Guard 3 for input & output safety checks

Audit logs for every interaction

🗂️ Project Structure (Simplified View)
bfsi-conversational-ai/
├── src/
│   ├── api/            # FastAPI endpoints
│   ├── core/           # Core logic (preprocessing, routing, tiers)
│   ├── models/         # Model wrappers (PHI-4, BGE-M3, Llama Guard)
│   ├── data/           # Dataset loaders, vector store
│   ├── safety/         # Compliance & guardrails
│   └── audit/          # Audit logging
│
├── data/
│   ├── raw/            # Alpaca BFSI dataset
│   ├── models/         # Fine-tuned model artifacts
│   └── cache/          # Embedding cache
│
├── scripts/
│   ├── training/       # PHI-4 fine-tuning
│   ├── data/           # Embedding generation & Qdrant import
│   └── setup/          # Environment setup scripts
│
├── tests/              # Unit & integration tests
└── docs/               # Architecture & compliance docs

📊 Dataset Strategy

Alpaca format: instruction, input, output

Minimum 150+ curated BFSI samples

Professional, neutral, compliant tone

Used for:

Tier-1 deterministic responses

Tier-2 behavioral fine-tuning (not fact memorization)

🧪 Testing Philosophy

This project does not test “model intelligence.”

Instead, it tests:

Privacy enforcement (PII masking)

Routing correctness (Tier-1 / Tier-2 / Tier-3)

Determinism of responses

Safety & compliance behavior

This reflects real BFSI review expectations.

⚙️ Local Setup (Prototype)
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn src.api.main:app --reload


⚠️ Fine-tuning PHI-4 requires a CUDA-enabled GPU.
Training is recommended on a separate machine (Colab / Linux GPU) and adapters copied locally.

🧠 Why This Project Matters

Most conversational AI systems:

Prioritize creativity

Hallucinate confidently

Fail silently in regulated domains

This project:

Prioritizes control over creativity

Treats AI as a decision-support component

Is designed for real BFSI constraints, not demos

⚠️ Disclaimer

This project is for research and prototyping purposes only.
It does not provide financial advice and should not be deployed in production without formal compliance review.

COMMANDS TO RUN THIS PROJECT

vidia-smi
python -m venv phi4-env
phi4-env\Scripts\activate


installation of Fine tuning process

pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.6 accelerate>=0.34,<1.0 datasets>=3.4,<4.4 trl>=0.18,<0.25 peft>=0.18,<1.0 sentencepiece safetensors
pip install unsloth==2026.2.1 unsloth_zoo==2026.2.1
pip install bitsandbytes==0.49.1 xformers==0.0.34 triton-windows==3.6.0.post25

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets transformers accelerate trl peft

python -c "import torch; print(torch.cuda.is_available())"
python -c "import unsloth; print('Unsloth OK')"

python tests/unit/preprocessing/test_privacy_filter.py

python examples/test_preprocessing.py

python -c "from src.core.preprocessing import Preprocessor; p = Preprocessor(); r = p.preprocess('Call 9876543210', 'test'); print(f'✅ Works! Sanitized: {r.sanitized_text}')"

pip install transformers==4.36.0 sentence-transformers==2.2.2 torch==2.1.0 qdrant-client==1.7.0 numpy==1.24.3 scikit-learn==1.3.2

python scripts/setup/setup_intent_engine.py
python tests/unit/intent_engine/test_intent_classifier.py
python tests/unit/intent_engine/test_embedding_service.py
python examples/test_intent_engine.py



python tests/unit/router/test_guardrails.py
python tests/unit/router/test_tier_router.py
python examples/test_router.py
python examples/test_full_pipeline.py
python scripts/training/prepare_training_data.py
python scripts/training/finetune_phi4.py
py src/api/main.py
