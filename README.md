# Trustworthy RAG

An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Retrieval-Augmented Generation Systems.

## Overview

This project implements a **Trustworthy RAG Framework** with an integrated **Evaluation Agent** that assesses the reliability of RAG responses using three complementary analysis components:

- **NLI Verifier** - Factual consistency checking via Natural Language Inference (BART-MNLI)
- **Poison Detector** - Multi-signal adversarial content detection (linguistic, structural, semantic, intra/cross-document NLI)
- **Trust Index Calculator** - Weighted composite score combining factuality, consistency, and poison safety

The system produces a **Trust Score** (0-1) for every RAG response, enabling automated detection of knowledge poisoning attacks and hallucinated content.

## Research Questions

- **RQ1**: What computational strategies are effective for detecting misinformation and poisoning in retrieved contexts?
- **RQ2**: How can a composite "Trust Index" be formulated to quantify RAG response reliability?
- **RQ3**: How does the Evaluation Agent enhance security performance vs. baseline systems?

## Architecture

```
User Query
    |
    v
+-------------------+
|     RETRIEVER     |   Embedding (MiniLM-L6-v2) + FAISS vector search
+--------+----------+
         |
         v
+-------------------+
|     GENERATOR     |   Llama 3.3 70B via FARMI (OpenAI-compatible API)
+--------+----------+
         |
         v
+--------------------------------------------+
|           EVALUATION AGENT                 |
|                                            |
|  NLI Verifier    Poison Detector           |
|  (factuality)    (5 detection methods)     |
|       |                |                   |
|       v                v                   |
|     Trust Index Calculator                 |
|     T = 0.4*F + 0.35*C + 0.25*(1-P)       |
+--------------------------------------------+
         |
         v
   Answer + Trust Score + Evaluation Report
```

## Project Structure

```
src/
  retriever/              # Document retrieval (embeddings, FAISS, chunking)
  generator/              # LLM response generation (FARMI client, prompts)
  evaluation_agent/       # Core evaluation (NLI, poison detection, trust index)
  experiments/            # Experiment framework (poisoned datasets, runner)
  pipeline/               # End-to-end RAG pipeline orchestrator

tests/                    # Unit tests (78 tests, pytest)
configs/config.yaml       # All configuration parameters
figures/                  # Auto-generated thesis charts (7 figures, 300 DPI)
run_experiment.py         # Experiment CLI (main entry point)
generate_charts.py        # Thesis visualization generator
requirements.txt          # Python dependencies
```

## Setup

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set FARMI API key (required for LLM access)
# Option A: Environment variable
$env:FARMI_API_KEY = "your-api-key"
# Option B: Add to configs/config.yaml under FARMI_API_KEY
```

## Usage

### Run Full Experiment Suite

```powershell
python run_experiment.py --all
```

This runs all experiments and auto-generates charts:
- TruthfulQA main experiment (50 samples, mixed poisoning)
- Per-strategy experiments (injection, contradiction, subtle, entity swap)
- FEVER dataset experiment
- Ablation study (5 weight configurations)

### Other Experiment Options

```powershell
python run_experiment.py --quick          # Quick test (10 samples)
python run_experiment.py --samples 30     # Custom sample count
python run_experiment.py --per-strategy   # Per-strategy breakdown only
python run_experiment.py --fever          # Include FEVER dataset
python run_experiment.py --ablation       # Ablation study only
```

### Regenerate Charts Only

```powershell
python generate_charts.py
```

### Run Tests

```powershell
pytest                    # All tests (includes slow model-loading tests)
pytest -m "not slow"      # Fast tests only (~0.4s)
pytest --cov=src          # With coverage report
```

## Key Results

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| TruthfulQA (mixed) | 91% | 100% | 40% | 57.1% |
| FEVER | 78% | 33.3% | 46.7% | 38.9% |

### Per-Strategy Detection (TruthfulQA)

| Strategy | Accuracy | F1 | Detection Rate |
|----------|----------|----|----------------|
| Injection | 99% | 96.8% | 100% |
| Contradiction | 94% | 75% | 60% |
| Subtle | 89% | 42.1% | 26.7% |
| Entity Swap | 86% | 12.5% | 6.7% |

## Tech Stack

- **LLM**: Llama 3.3 70B (FARMI cluster, Tampere University)
- **NLI Model**: facebook/bart-large-mnli
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (CPU)
- **Datasets**: TruthfulQA, FEVER (HuggingFace)
- **Framework**: Python 3.10+, PyTorch, Transformers, LangChain

## Trust Index Formula

```
Trust = alpha * Factuality + beta * Consistency + gamma * (1 - PoisonProbability)

Default weights: alpha=0.4, beta=0.35, gamma=0.25
```

A non-linear dampener applies when poison probability exceeds 0.7, ensuring highly poisoned content receives significantly lower trust scores regardless of other components.

| Trust Level | Score Range | Meaning |
|-------------|-------------|---------|
| HIGH | > 0.8 | Reliable |
| MEDIUM | 0.5 - 0.8 | Verify if important |
| LOW | 0.3 - 0.5 | Likely has issues |
| VERY_LOW | < 0.3 | Do not trust |

## Author

Balkrishna Giri - Tampere University
Master's Thesis in Information Security (2026)
