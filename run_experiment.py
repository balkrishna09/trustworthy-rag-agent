"""
Run Experiment - Trustworthy RAG Evaluation
Tests the system on TruthfulQA (and optionally FEVER) with clean and poisoned
document sets, per-strategy breakdown, baseline comparison, and ablation.

Usage:
    python run_experiment.py                # default run (50 samples, TruthfulQA)
    python run_experiment.py --quick        # quick run (10 samples)
    python run_experiment.py --per-strategy # one experiment per poisoning strategy
    python run_experiment.py --fever        # also run on FEVER
    python run_experiment.py --ablation     # ablation study (vary alpha/beta/gamma)
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load .env file so FARMI_API_KEY and other settings are available via os.environ
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.experiments import ExperimentRunner, ExperimentConfig
from src.experiments.poisoned_dataset import PoisonedDatasetGenerator, PoisonStrategy


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_truthfulqa_for_experiment(max_samples: int = 50):
    """
    Load TruthfulQA data and prepare it for RAG experiments.

    Returns:
        knowledge_base: List of document strings (correct answers as knowledge)
        questions: List of question strings
        expected_answers: List of expected correct answer strings
    """
    data_path = Path("data/raw/truthfulqa/truthfulqa.jsonl")

    if not data_path.exists():
        print(f"[ERROR] TruthfulQA not found at {data_path}")
        print("Run: python download_datasets.py")
        return None, None, None

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} TruthfulQA samples")

    good_samples = [
        s for s in samples
        if s.get('best_answer', '') and s.get('question', '')
        and len(s.get('best_answer', '')) > 10
    ]
    print(f"Filtered to {len(good_samples)} samples with good answers")

    good_samples = good_samples[:max_samples]

    knowledge_base, questions, expected_answers = [], [], []
    for s in good_samples:
        doc = f"{s['question']} {s['best_answer']}"
        knowledge_base.append(doc)
        questions.append(s['question'])
        expected_answers.append(s['best_answer'])

    print(f"Prepared {len(questions)} questions with knowledge base")
    return knowledge_base, questions, expected_answers


def load_fever_for_experiment(max_samples: int = 50):
    """
    Load FEVER data and prepare it for RAG experiments.

    Returns:
        knowledge_base, questions, expected_answers (or None x 3)
    """
    data_path = Path("data/raw/fever")
    jsonl_file = data_path / "fever.jsonl"

    if not jsonl_file.exists():
        # Try alternative name
        candidates = list(data_path.glob("*.jsonl"))
        if not candidates:
            print(f"[ERROR] FEVER not found at {data_path}")
            return None, None, None
        jsonl_file = candidates[0]

    samples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} FEVER samples")

    good_samples = []
    for s in samples:
        claim = s.get('claim', '')
        label = s.get('label', '')
        if claim and label in ('SUPPORTS', 'REFUTES') and len(claim) > 10:
            good_samples.append(s)

    good_samples = good_samples[:max_samples]

    knowledge_base, questions, expected_answers = [], [], []
    for s in good_samples:
        claim = s['claim']
        label = s['label']
        # Build richer KB documents matching TruthfulQA format (question + answer).
        # Bare claims (~10 words) are too short for NLI to find entailment,
        # causing 70% of factuality scores to fall back to 0.5 (inconclusive).
        question = f"Is the following claim true or false? {claim}"
        verdict = "supported by evidence" if label == "SUPPORTS" else "refuted by evidence"
        doc = f"{question} The claim is {verdict}. {claim}"
        knowledge_base.append(doc)
        questions.append(question)
        expected_answers.append(label)

    print(f"Prepared {len(questions)} FEVER questions")
    return knowledge_base, questions, expected_answers


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

def get_pipeline_config() -> dict:
    """Build pipeline config from environment variables (loaded from .env)."""
    return {
        'LLM_PROVIDER': os.environ.get('LLM_PROVIDER', 'openai_compatible'),
        'FARMI_API_URL': os.environ.get('FARMI_API_URL', 'https://gptlab.rd.tuni.fi/students/ollama/v1'),
        'FARMI_API_KEY': os.environ.get('FARMI_API_KEY', ''),
        'LLM_MODEL': os.environ.get('LLM_MODEL', 'llama3.3:70b'),
        'MAX_NEW_TOKENS': int(os.environ.get('MAX_NEW_TOKENS', '150')),
        'EMBEDDING_MODEL': os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        'TOP_K_RETRIEVAL': int(os.environ.get('TOP_K_RETRIEVAL', '3')),
        'NLI_MODEL': os.environ.get('NLI_MODEL', 'facebook/bart-large-mnli'),
        'TRUST_THRESHOLD': float(os.environ.get('TRUST_THRESHOLD', '0.5')),
        'POISON_THRESHOLD': float(os.environ.get('POISON_THRESHOLD', '0.7')),
        'TRUST_ALPHA': float(os.environ.get('TRUST_ALPHA', '0.4')),
        'TRUST_BETA': float(os.environ.get('TRUST_BETA', '0.35')),
        'TRUST_GAMMA': float(os.environ.get('TRUST_GAMMA', '0.25')),
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_main_experiment(
    pipeline_config: dict,
    num_samples: int = 50,
    dataset: str = "truthfulqa",
):
    """Run the main experiment on the given dataset."""
    print("\n" + "=" * 60)
    print(f"MAIN EXPERIMENT — {dataset.upper()} ({num_samples} samples)")
    print("=" * 60)

    if dataset == "truthfulqa":
        kb, questions, answers = load_truthfulqa_for_experiment(num_samples)
    elif dataset == "fever":
        kb, questions, answers = load_fever_for_experiment(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if kb is None:
        print("Failed to load data. Skipping.")
        return None

    exp_config = ExperimentConfig(
        name=f"{dataset}_poison_detection",
        description=(
            f"Evaluate poison detection on {dataset.upper()} with "
            f"{num_samples} samples, 30% poisoned (MIXED strategy)"
        ),
        num_samples=num_samples,
        poison_ratio=0.3,
        top_k=3,
    )

    runner = ExperimentRunner(pipeline_config)

    # Baseline
    gen = PoisonedDatasetGenerator()
    samples, _ = gen.create_poisoned_dataset(kb, poison_ratio=0.3, strategy=PoisonStrategy.MIXED)
    poisoned_kb = [s.poisoned_text for s in samples]

    baseline_acc = runner.run_baseline(kb, poisoned_kb, questions)

    result = runner.run_experiment(
        config=exp_config,
        knowledge_base=kb,
        questions=questions,
        expected_answers=answers,
    )
    result.baseline_accuracy = baseline_acc
    result.print_summary()

    saved = runner.save_results(result)
    print(f"Results saved to: {saved}")
    return result


def run_per_strategy_experiments(pipeline_config: dict, num_samples: int = 50):
    """Run one experiment per poisoning strategy for comparison."""
    print("\n" + "=" * 60)
    print("PER-STRATEGY EXPERIMENTS")
    print("=" * 60)

    kb, questions, answers = load_truthfulqa_for_experiment(num_samples)
    if kb is None:
        print("Failed to load data. Skipping.")
        return

    strategies = [
        PoisonStrategy.CONTRADICTION,
        PoisonStrategy.INSTRUCTION_INJECTION,
        PoisonStrategy.ENTITY_SWAP,
        PoisonStrategy.SUBTLE_MANIPULATION,
    ]

    runner = ExperimentRunner(pipeline_config)

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.value} ---")
        gen = PoisonedDatasetGenerator()
        samples, stats = gen.create_poisoned_dataset(
            kb, poison_ratio=0.3, strategy=strategy
        )
        poisoned_kb = [s.poisoned_text for s in samples]

        exp_config = ExperimentConfig(
            name=f"truthfulqa_{strategy.value}",
            description=f"Single-strategy: {strategy.value}",
            num_samples=num_samples,
            poison_ratio=0.3,
            top_k=3,
        )

        result = runner.run_experiment(
            config=exp_config,
            knowledge_base=kb,
            questions=questions,
            expected_answers=answers,
            poisoned_knowledge_base=poisoned_kb,
        )
        result.print_summary()
        runner.save_results(result)


def run_ablation_study(pipeline_config: dict, num_samples: int = 30):
    """
    Ablation: vary Trust Index weights to see how each component
    contributes to overall detection performance.
    """
    print("\n" + "=" * 60)
    print("ABLATION STUDY — Trust Index Weights")
    print("=" * 60)

    kb, questions, answers = load_truthfulqa_for_experiment(num_samples)
    if kb is None:
        print("Failed to load data. Skipping.")
        return

    weight_configs = [
        ("default",           0.40, 0.35, 0.25),
        ("factuality_heavy",  0.60, 0.20, 0.20),
        ("consistency_heavy", 0.20, 0.60, 0.20),
        ("poison_heavy",      0.20, 0.20, 0.60),
        ("equal",             0.33, 0.34, 0.33),
    ]

    runner = ExperimentRunner(pipeline_config)

    for label, alpha, beta, gamma in weight_configs:
        print(f"\n--- Weights: {label} (α={alpha}, β={beta}, γ={gamma}) ---")
        cfg = dict(pipeline_config)
        cfg['TRUST_ALPHA'] = alpha
        cfg['TRUST_BETA'] = beta
        cfg['TRUST_GAMMA'] = gamma

        ablation_runner = ExperimentRunner(cfg)

        exp_config = ExperimentConfig(
            name=f"ablation_{label}",
            description=f"Ablation: α={alpha}, β={beta}, γ={gamma}",
            num_samples=num_samples,
            poison_ratio=0.3,
            top_k=3,
            trust_alpha=alpha,
            trust_beta=beta,
            trust_gamma=gamma,
        )

        result = ablation_runner.run_experiment(
            config=exp_config,
            knowledge_base=kb,
            questions=questions,
            expected_answers=answers,
        )
        result.print_summary()
        ablation_runner.save_results(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trustworthy RAG Experiment Runner"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with 10 samples (for testing)"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of samples (default: 50)"
    )
    parser.add_argument(
        "--per-strategy", action="store_true",
        help="Run one experiment per poisoning strategy"
    )
    parser.add_argument(
        "--fever", action="store_true",
        help="Also run experiment on FEVER dataset"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study on Trust Index weights"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run everything (main + per-strategy + FEVER + ablation)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    num_samples = 10 if args.quick else args.samples
    pipeline_config = get_pipeline_config()

    print("=" * 60)
    print("TRUSTWORTHY RAG — EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Samples: {num_samples}")
    print(f"Flags: quick={args.quick}, per_strategy={args.per_strategy}, "
          f"fever={args.fever}, ablation={args.ablation}, all={args.all}")

    # --- Main experiment (TruthfulQA) ---
    run_main_experiment(pipeline_config, num_samples, "truthfulqa")

    # --- Per-strategy ---
    if args.per_strategy or args.all:
        run_per_strategy_experiments(pipeline_config, num_samples)

    # --- FEVER ---
    if args.fever or args.all:
        run_main_experiment(pipeline_config, num_samples, "fever")

    # --- Ablation ---
    if args.ablation or args.all:
        ablation_samples = min(num_samples, 30)  # Keep ablation smaller
        run_ablation_study(pipeline_config, ablation_samples)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
