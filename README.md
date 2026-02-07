# Trustworthy RAG

An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems.

## Project Overview

This project implements a **Trustworthy RAG Framework** that integrates an autonomous Evaluation Agent to detect knowledge poisoning attacks and misinformation in Retrieval-Augmented Generation systems.

## Research Questions

- **RQ1**: What computational strategies are effective for detecting misinformation and poisoning in retrieved contexts?
- **RQ2**: How can a composite "Trust Index" be formulated to quantify RAG response reliability?
- **RQ3**: How does the Evaluation Agent enhance security performance vs. baseline systems?

## Project Structure

```
RAG Agent/
├── data/                    # Datasets (FEVER, TruthfulQA, poisoned)
│   ├── raw/
│   ├── processed/
│   └── poisoned/
├── src/                     # Source code
│   ├── retriever/           # Document retrieval components
│   ├── evaluation_agent/    # Core evaluation agent
│   ├── generator/           # LLM response generation
│   └── pipeline/            # End-to-end RAG pipeline
├── experiments/             # Experiment scripts and results
├── tests/                   # Unit and integration tests
├── notebooks/               # Jupyter notebooks for exploration
└── configs/                 # Configuration files
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Architecture

```
User Query → Retriever → [Evaluation Agent] → Generator → Response
                              ↓
                    ┌─────────────────────┐
                    │  NLI Verifier       │
                    │  Poison Detector    │
                    │  Trust Index Calc   │
                    └─────────────────────┘
```

## Author

Balkrishna Giri - Tampere University  
Master's Thesis in Information Security (2026)
