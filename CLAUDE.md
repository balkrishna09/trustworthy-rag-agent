# CLAUDE.md - Project Guidelines for Claude

## Project Overview

**Trustworthy RAG Framework** - Master's Thesis Project at Tampere University (TUNI)

An evaluation agent for detecting misinformation and knowledge poisoning in Retrieval-Augmented Generation (RAG) systems.

- **Author**: Balkrishna Giri
- **Version**: 0.1.0

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| LLM Provider | Ollama (FARMI/local) - primary |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (flexible - can use others) |
| NLI Model | facebook/bart-large-mnli |
| Framework | LangChain |
| Experiment Tracking | MLflow |
| Testing | pytest with coverage |
| Logging | Loguru |

---

## Project Structure

```
RAG Agent/
├── src/
│   ├── retriever/          # Document retrieval & embeddings
│   ├── evaluation_agent/   # NLI verifier, poison detector, trust index
│   ├── generator/          # LLM response generation
│   └── pipeline/           # End-to-end RAG pipeline
├── configs/
│   └── config.yaml         # Main configuration
├── data/
│   ├── raw/                # Original datasets (FEVER, TruthfulQA)
│   ├── processed/          # Preprocessed data
│   └── poisoned/           # Adversarial/poisoned datasets
├── tests/                  # Test files
├── .env                    # Environment variables (local)
└── requirements.txt        # Python dependencies
```

---

## Code Standards

### Style Guidelines
- **Type hints**: Always use Python type hints for function parameters and return values
- **Docstrings**: Use Google-style docstrings for all classes and functions
- **Imports**: Group imports (stdlib, third-party, local) with blank lines between

### Example
```python
def calculate_trust_index(
    factuality_score: float,
    consistency_score: float,
    poison_probability: float
) -> float:
    """Calculate the composite trust index for a RAG response.

    Args:
        factuality_score: NLI-based factual accuracy score (0-1).
        consistency_score: Cross-reference consistency score (0-1).
        poison_probability: Estimated probability of poisoned content (0-1).

    Returns:
        Composite trust index score between 0 and 1.
    """
    alpha, beta, gamma = 0.4, 0.35, 0.25
    return alpha * factuality_score + beta * consistency_score + gamma * (1 - poison_probability)
```

---

## Git Conventions

### Commit Messages
Use **Conventional Commits** format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

**Example**: `feat: add NLI verifier for factual consistency checking`

---

## Testing

- **Framework**: pytest with coverage
- **Run tests after every code change**
- **Commands**:
  ```powershell
  # Run all tests
  pytest

  # Run with coverage
  pytest --cov=src --cov-report=html

  # Run specific test file
  pytest tests/test_retriever.py
  ```

---

## Key Commands

```powershell
# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py

# Run tests
pytest

# Download datasets
python download_datasets.py
```

---

## Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| FEVER | Factual verification evaluation | Hugging Face |
| TruthfulQA | Misinformation detection | Hugging Face |
| Poisoned (custom) | Knowledge poisoning attacks | Self-created |

---

## Configuration

Key settings in `configs/config.yaml`:

- `LLM_PROVIDER`: ollama (default)
- `LLM_MODEL`: llama3.3:70b-instruct-q4_K_M
- `EMBEDDING_MODEL`: sentence-transformers/all-MiniLM-L6-v2
- `NLI_MODEL`: facebook/bart-large-mnli
- `TOP_K_RETRIEVAL`: 5
- `TRUST_THRESHOLD`: 0.5

---

## Protected Files (Do Not Modify)

- `Master_Thesis.pdf`
- `Supervisor_Plan.pdf`

---

## Important Notes for Claude

1. **Always run tests** after making code changes
2. **Use MLflow** for tracking experiments and metrics
3. **Handle API keys securely** - never log or expose them
4. **Be mindful of memory** when working with large datasets
5. **Poisoning attack code** should be clearly documented and for research purposes only
6. **Virtual environment**: Always ensure venv is activated before running commands

---

## Research Questions (Context)

- **RQ1**: How effective is the evaluation agent at detecting misinformation?
- **RQ2**: How does factual verification improve RAG trustworthiness?
- **RQ3**: How resilient is the system against knowledge poisoning attacks?

---

## Contact

For thesis-related questions, refer to `SUPERVISOR_REQUEST_TEMPLATE.md` for supervisor communication.
