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
- **RQ3**: How does the Evaluation Agent enhance security performance vs. baseline systems, and how does LLM/embedding choice affect Trust Index discriminative performance?

## Architecture

```
User Query
    |
    v
+-------------------+
|     RETRIEVER     |   Embedding (MiniLM-L6-v2 or Snowflake Arctic Embed2) + FAISS
+--------+----------+
         |
         v
+-------------------+
|     GENERATOR     |   Llama 3.3 70B or Qwen 3.5 35B via FARMI API
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
python run_experiment.py --grid           # 2x2 factorial grid (all LLM x embedding combos)
python run_experiment.py --grid --samples 50  # Full grid run (100 samples per config)
```

The interactive prompt lets you select:
- **LLM**: Llama 3.3 70B (primary) or Qwen 3.5 35B (comparison)
- **Embedding**: all-MiniLM-L6-v2 (local) or snowflake-arctic-embed2 (API)
- **K**: Retrieval depth — K=3 (faster) or K=5 (standard, default)

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

### Primary Results (Llama 3.3 70B + all-MiniLM-L6-v2, TruthfulQA, K=5)

| Dataset | Accuracy | Precision | Recall | F1 Score | vs. Baseline |
|---------|----------|-----------|--------|----------|--------------|
| TruthfulQA (mixed) | 91% | 100% | 40% | 57.1% | +7% |
| FEVER | 90% | — | 66.7% | — | +5% |

Naive always-trust baseline: **85%** (TruthfulQA), **85%** (FEVER)

### Per-Strategy Detection (TruthfulQA, 20 samples each)

| Strategy | Accuracy | F1 | Recall | Separation |
|----------|----------|----|--------|------------|
| Contradiction | 89% | 61.1% | 60% | 0.245 |
| Injection | 75% | 54.5% | 100% | 0.407 |
| Entity Swap | 84% | 28.6% | 16.7% | 0.071 |
| Subtle | 86% | 25.0% | 16.7% | 0.127 |

### 2×2 Factorial Grid (K=3, TruthfulQA, 100 samples)

| LLM | Embedding | Accuracy | Precision | Recall | F1 | Clean Trust | Separation |
|-----|-----------|----------|-----------|--------|----|-------------|------------|
| Llama 3.3 70B | all-MiniLM-L6-v2 | 91% | 100% | 40% | 57.1% | 0.830 | 0.240 |
| Llama 3.3 70B | snowflake-arctic-embed2 | 91% | 100% | 40% | 57.1% | 0.799 | 0.188 |
| Qwen 3.5 35B | all-MiniLM-L6-v2 | 71% | 25.0% | 46.7% | 32.6% | 0.633 | 0.161 |
| Qwen 3.5 35B | snowflake-arctic-embed2 | 71% | 28.1% | 60.0% | 38.3% | 0.653 | 0.199 |

**Key findings:**
- **Embedding invariance for Llama**: both embedding models yield identical detection results — Trust Index is LLM-driven, not retrieval-driven
- **Qwen generation-style problem**: verbose/hedged outputs reduce NLI entailment for clean samples → 71% accuracy, below the 85% naive baseline
- **K sensitivity null result**: K=5 produces identical detection performance to K=3 for Llama 3.3 70B — detection is bottlenecked by attack strategy difficulty, not retrieval depth

## Tech Stack

- **LLMs**: Llama 3.3 70B (primary), Qwen 3.5 35B (comparison) — FARMI cluster, Tampere University
- **NLI Model**: facebook/bart-large-mnli
- **Embeddings**: all-MiniLM-L6-v2 (384-dim), snowflake-arctic-embed2 (1024-dim)
- **Vector Store**: FAISS (CPU)
- **Datasets**: TruthfulQA, FEVER (HuggingFace)
- **Framework**: Python 3.10+, PyTorch, Transformers, LangChain

## Trust Index Formula

```
Trust = alpha * Factuality + beta * Consistency + gamma * (1 - PoisonProbability)

Default weights: alpha=0.4, beta=0.35, gamma=0.25
```

A non-linear dampener applies when poison probability exceeds 0.7:
`delta = 1 - 0.4 * (P - 0.70) / 0.30` — at P=1.0, trust is reduced by 40%.

| Trust Level | Score Range | Meaning |
|-------------|-------------|---------|
| HIGH | > 0.8 | Reliable |
| MEDIUM | 0.5 - 0.8 | Verify if important |
| LOW | 0.3 - 0.5 | Likely has issues |
| VERY_LOW | < 0.3 | Do not trust |

## Author

Balkrishna Giri - Tampere University
Master's Thesis in Information Security (2026)
