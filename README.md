# Trustworthy RAG

An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Retrieval-Augmented Generation Systems.

## Overview

This project implements a **Trustworthy RAG Framework** with an integrated **Evaluation Agent** that assesses the reliability of RAG responses using three complementary analysis components:

- **NLI Verifier** - Factual consistency checking via Natural Language Inference (BART-MNLI)
- **Poison Detector** - Multi-signal adversarial content detection (linguistic, structural, semantic, intra/cross-document NLI)
- **Trust Index Calculator** - Weighted composite score combining factuality, consistency, and poison safety

The system produces a **Trust Score** (0-1) for every RAG response, enabling automated detection of knowledge poisoning attacks and hallucinated content.

This repository is the research artifact for the ICSEA 2026 conference paper of the same title. See [`Conference_ICSEA2026/`](Conference_ICSEA2026/) for the LaTeX sources, and the [Paper](#paper) section below.

## Research Questions

- **RQ1**: What computational strategies are effective for detecting misinformation and poisoning in retrieved contexts?
- **RQ2**: How can a composite "Trust Index" be formulated to quantify RAG response reliability?
- **RQ3**: How does the Evaluation Agent enhance security performance vs. baseline systems, and how does LLM/embedding choice affect Trust Index discriminative performance?

> The ICSEA 2026 paper extends **RQ3** to a third LLM (Mistral 7B Instruct) and a secure-coding software-development-lifecycle (SDLC) setting — see the [Secure-Coding SDLC Use Case](#secure-coding-sdlc-use-case-owaspcwe) and [Three-LLM Comparison](#three-llm-comparison--threshold-independence) below.

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
|     GENERATOR     |   Llama 3.3 70B / Qwen 3.5 35B / Mistral 7B Instruct via FARMI API
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
figures/                  # Auto-generated charts (7 figures, 300 DPI)
run_experiment.py         # Experiment CLI (main entry point)
generate_charts.py        # Visualization generator
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

# Configure FARMI API access (required for LLM generation)
# Copy the template and fill in your own credentials:
#   cp .env.example .env       # then edit .env and set FARMI_API_KEY
# The .env file is .gitignored - never commit it or hardcode keys in source.
```

## Usage

### Run Full Experiment Suite

```powershell
python run_experiment.py --all
```

This runs all experiments and auto-generates charts:
- TruthfulQA main experiment (100 samples, mixed poisoning)
- Per-strategy experiments (100 samples each: injection, contradiction, entity swap, subtle)
- FEVER dataset experiment (100 samples)
- Ablation study (5 weight configurations, 60 samples each)

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
- **LLM**: Llama 3.3 70B (primary), Qwen 3.5 35B or Mistral 7B Instruct (comparison)
- **Embedding**: all-MiniLM-L6-v2 (local) or snowflake-arctic-embed2 (API)
- **K**: Retrieval depth — K=3 (faster) or K=5 (standard, default)

You can also pin the generator non-interactively via the `LLM_MODEL` environment variable
(matching `configs/config.yaml` and `run_experiment.py`'s `MODEL_DESCRIPTIONS`):

```powershell
$env:LLM_MODEL = "mistral-7b-instruct"   # or "qwen3.5:35b", "llama3.3:70b"
python run_experiment.py --all
```

### Secure-Coding SDLC Use Case

Reproduce the secure-coding assistant experiment (40 OWASP/CWE rules) from the project root.
It reuses the existing `ExperimentRunner` + `PoisonedDatasetGenerator` and writes results to
`data/experiments/seccode_*.json`:

```powershell
$env:LLM_MODEL = "llama3.3:70b"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
python Conference_ICSEA2026/run_seccode_usecase.py
# Set $env:SECCODE_N = "4" first for a quick smoke run over a few rules.
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

### Primary Results (Llama 3.3 70B + all-MiniLM-L6-v2, TruthfulQA, K=5, 100 samples)

| Dataset | Accuracy | Precision | Recall | F1 Score | vs. Baseline |
|---------|----------|-----------|--------|----------|--------------|
| TruthfulQA (mixed) | 91% | 100% | 40% | 57.1% | +7% |
| FEVER (full run) | 73% | 20% | 26.7% | 22.9% | −12% |

Naive always-trust baseline: **85%** (TruthfulQA), **85%** (FEVER).
FEVER underperforms the baseline — the Trust Index requires domain-specific calibration for short factual claims.

With 95% confidence intervals (Wilson for proportions, percentile bootstrap *B*=20,000 for F1), the primary TruthfulQA mixed result is: accuracy 91% (CI 83.8–95.2), recall 40% (CI 19.8–64.3), F1 57.1% (CI 25.0–80.0), Δ=0.225. Intervals are wide because each run has only 15 poisoned samples, so we report them to avoid overstating precision.

### Per-Strategy Detection (TruthfulQA, 100 samples each)

| Strategy | Accuracy | Precision | Recall | F1 | Separation |
|----------|----------|-----------|--------|----|------------|
| Injection | 99% | 93.8% | 100% | 96.8% | 0.498 |
| Contradiction | 92% | 88.9% | 53.3% | 66.7% | 0.311 |
| Subtle | 88% | 100% | 20.0% | 33.3% | 0.149 |
| Entity Swap | 85% | — | 0% | 0% | 0.053 |

### 2×2 Factorial Grid (K=3, TruthfulQA, 100 samples)

| LLM | Embedding | Accuracy | Precision | Recall | F1 | Clean Trust | Separation |
|-----|-----------|----------|-----------|--------|----|-------------|------------|
| Llama 3.3 70B | all-MiniLM-L6-v2 | 91% | 100% | 40% | 57.1% | 0.830 | 0.240 |
| Llama 3.3 70B | snowflake-arctic-embed2 | 91% | 100% | 40% | 57.1% | 0.799 | 0.188 |
| Qwen 3.5 35B | all-MiniLM-L6-v2 | 71% | 25.0% | 46.7% | 32.6% | 0.633 | 0.161 |
| Qwen 3.5 35B | snowflake-arctic-embed2 | 71% | 28.1% | 60.0% | 38.3% | 0.653 | 0.199 |

**Key findings:**
- **Embedding invariance for Llama**: both embedding models yield identical detection results — Trust Index is LLM-driven, not retrieval-driven
- **Qwen generation-style problem**: verbose/hedged outputs reduce NLI entailment for clean samples → 71% accuracy, **14 percentage points below** the 85% naive baseline
- **K sensitivity null result**: K=5 produces identical detection performance to K=3 for Llama 3.3 70B — trust score separation changes by only 0.015; detection is bottlenecked by attack strategy difficulty, not retrieval depth

### Secure-Coding SDLC Use Case (OWASP/CWE)

A software-engineering application of the agent: a secure-coding assistant that retrieves from a curated knowledge base of **40 rules drawn from the OWASP Top 10 and CWE** (e.g. parameterized queries for SQL injection, adaptive password hashing, TLS configuration). A developer query retrieves guidance that the Evaluation Agent screens *before* the LLM emits a recommendation. We poison 30% of the rules with the five strategies as security payloads (e.g. a spurious `CORRECTION:` directive, a swapped CWE identifier).

Detection by strategy (Llama 3.3 70B + all-MiniLM-L6-v2, K=5):

| Strategy | Acc | Prec | Recall | F1 | Δ |
|----------|-----|------|--------|----|----|
| Instruction injection | 97.5% | 85.7% | 100.0% | 92.3% | 0.542 |
| Contradiction | 85.0% | — | 0.0% | 0.0% | 0.156 |
| Subtle manipulation | 85.0% | — | 0.0% | 0.0% | 0.066 |
| Entity swap | 72.5% | 29.2% | 58.3% | 38.9% | 0.291 |
| Mixed | 86.2% | 100.0% | 8.3% | 15.4% | 0.163 |

- **Injection is blocked near-perfectly** (F1 92.3%, Δ=0.54): the agent reliably catches overt attempts to insert unsafe advice (e.g. disabling input validation) before they reach the developer.
- **Entity swap rises to 58% recall** (vs. 0% on open-domain TruthfulQA): altering a structured CWE identifier introduces cross-document inconsistencies the NLI signals can catch.
- **Contradiction and subtle manipulation still evade (0% recall)**: a quietly weakened recommendation carries no surface artifact and passes review — the dangerous, low-visibility case in a security workflow.

### Three-LLM Comparison & Threshold Independence

Beyond the τ=0.5 operating point, the Trust Index has strong threshold-independent signal across three LLMs (pooled mixed-strategy runs):

| LLM | Accuracy (τ=0.5) | ROC-AUC | Optimal τ* |
|-----|------------------|---------|------------|
| Llama 3.3 70B | 91% | 0.81 | 0.71 |
| Mistral 7B Instruct | 87% | 0.79 | 0.58 |
| Qwen 3.5 35B | 69–71% | 0.73 | 0.43 |

- **Generation style > model size**: Mistral 7B outperforms Qwen 3.5 35B despite being ~5× smaller — Qwen's hedged, verbose answers depress NLI entailment.
- **τ is an LLM-specific hyperparameter**: the three models need three different optimal thresholds (0.71 / 0.58 / 0.43). Calibrating τ on clean-only scores restores Qwen from 65.5% → 74.5% accuracy by cutting false positives.
- All three sit well above chance (ROC-AUC > 0.70), so Qwen's weak accuracy at τ=0.5 is a *thresholding* artifact, not an absence of signal.

### Stability & Overhead

- **Run-to-run variance is low**: over 5 independent repeats the mixed configuration scores 90.6 ± 0.5% accuracy and 56.1 ± 1.3% F1; injection is invariant at 99.0 ± 0.0%. Verdicts are driven by retrieved evidence, not by single-generation wording.
- **Overhead**: evaluation adds ≈14.7 s/sample (≈17× over baseline RAG), dominated by up to 20 CPU NLI passes at K=5. This suits batch/asynchronous use (e.g. a code-review/CI gate); GPU inference is projected to cut this below 2 s.

## Tech Stack

- **LLMs**: Llama 3.3 70B (primary), Qwen 3.5 35B and Mistral 7B Instruct (comparison) — FARMI cluster, Tampere University
- **NLI Model**: facebook/bart-large-mnli
- **Embeddings**: all-MiniLM-L6-v2 (384-dim), snowflake-arctic-embed2 (1024-dim)
- **Vector Store**: FAISS (CPU)
- **Datasets**: TruthfulQA, FEVER (HuggingFace), and a secure-coding KB of 40 curated OWASP Top 10 / CWE rules
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

## Contributions

- An autonomous **Evaluation Agent** that operates as defensive middleware inside the RAG inference loop
- A **five-signal poison detector** with relevance-weighted aggregation (linguistic, structural, intra-/cross-document NLI, semantic-outlier)
- A composite **Trust Index** with a non-linear dampener for high-contamination contexts
- Characterization of a **per-strategy detection hierarchy** (injection solved → entity swap undetected)
- Evidence that generation **style matters more than model size**, plus a **per-LLM threshold calibration** method
- A software-engineering **secure-coding (OWASP/CWE) SDLC use case**
- A reproducible framework, attack generator, and experiment release

## Paper

- **Conference paper** — *Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems*, ICSEA 2026. LaTeX sources in [`Conference_ICSEA2026/`](Conference_ICSEA2026/).

The compiled PDF is intentionally not included in this release; build it from the LaTeX sources.

## Authors & Contact

ICSEA 2026 conference paper authors — Tampere University (TUNI).

**Balkrishna Giri** — MSc Researcher, Tampere University
[balkrishna.giri@tuni.fi](mailto:balkrishna.giri@tuni.fi) · [balkrishna.giri07@gmail.com](mailto:balkrishna.giri07@gmail.com)

**Md Toufique Hasan** — Doctoral Researcher, GPT Lab, Tampere University
[mdtoufique.hasan@tuni.fi](mailto:mdtoufique.hasan@tuni.fi)

**Co-authors** — Faculty of Information Technology and Communication Sciences, Tampere University:
- Jussi Rasku — [jussi.rasku@tuni.fi](mailto:jussi.rasku@tuni.fi)
- Muhammad Waseem — [muhammad.waseem@tuni.fi](mailto:muhammad.waseem@tuni.fi)
- Pekka Abrahamsson — [pekka.abrahamsson@tuni.fi](mailto:pekka.abrahamsson@tuni.fi)

Developed at GPT Lab, Tampere University.
