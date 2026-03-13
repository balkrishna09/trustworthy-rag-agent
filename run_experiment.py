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
    python run_experiment.py --all          # run everything + generate charts
"""

import sys
import os
import json
import argparse
import requests
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
# Known model metadata
# ---------------------------------------------------------------------------

KNOWN_EMBEDDING_MODELS = {
    "snowflake-arctic-embed2:latest",
    "nomic-embed-text:latest",
    "nomic-embed-text-v2-moe:latest",
}

MODEL_DESCRIPTIONS = {
    "llama3.3:70b":                          "Llama 3.3 (70B) — thesis default, strong instruction following",
    "qwen3.5:35b":                           "Qwen 3.5 (35B) — thesis 2×2 comparison target",
    "qwen3.5:9b":                            "Qwen 3.5 (9B) — faster Qwen, good quality/speed balance",
    "qwen3.5:0.8b":                          "Qwen 3.5 (0.8B) — very small, quick smoke-tests only",
    "phi4-reasoning:14b":                    "Phi-4 Reasoning (14B) — step-by-step reasoning",
    "phi4:14b":                              "Phi-4 (14B) — Microsoft, strong factual accuracy",
    "gpt-oss:120b":                          "GPT-OSS (120B) — largest available, best quality, slowest",
    "gpt-oss:20b":                           "GPT-OSS (20B) — smaller OSS, faster than 70B",
    "deepseek-r1:8b":                        "DeepSeek-R1 (8B) — reasoning model, lightweight",
    "llama3.2:3b":                           "Llama 3.2 (3B) — lightweight, speed testing only",
    "qwen2.5-coder:32b":                     "Qwen 2.5 Coder (32B) — coding specialist, not for factual Q&A",
    "qwen3-coder-next:latest":               "Qwen3 Coder Next — latest coding model",
    "snowflake-arctic-embed2:latest":        "Snowflake Arctic Embed2 (1024-dim) — API, state-of-the-art retrieval",
    "nomic-embed-text:latest":               "Nomic Embed Text (768-dim) — API, general purpose",
    "nomic-embed-text-v2-moe:latest":        "Nomic MoE (768-dim) — API, MoE variant",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6-v2 (384-dim) — local, fast",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_api_models(api_url: str, api_key: str) -> list:
    """Fetch available model IDs from FARMI API. Returns [] on failure."""
    try:
        r = requests.get(
            f"{api_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


def select_models_interactively(pipeline_config: dict) -> dict:
    """
    Display an interactive 3-option menu (2 primaries + More...) to select
    the LLM and embedding model.  Returns an updated pipeline_config dict.
    """
    config = dict(pipeline_config)
    api_url = config.get("FARMI_API_URL", "").rstrip("/")
    api_key = config.get("FARMI_API_KEY", "")

    print("\nFetching available models from FARMI...", end="", flush=True)
    api_models = _fetch_api_models(api_url, api_key)
    print(f" {len(api_models)} models found." if api_models else " (could not reach API, using defaults)")

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         TRUSTWORTHY RAG — EXPERIMENT SETUP           ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── LLM selection ────────────────────────────────────────────────────────
    PRIMARY_LLMS = [
        ("llama3.3:70b", "Llama 3.3 (70B) — thesis default, strong instruction following [DEFAULT]"),
        ("qwen3.5:35b",  "Qwen 3.5 (35B) — thesis 2×2 comparison target"),
    ]
    current_llm = config.get("LLM_MODEL", "llama3.3:70b")
    primary_llm_ids = {m for m, _ in PRIMARY_LLMS}
    extra_llms = [
        m for m in api_models
        if m not in primary_llm_ids and m not in KNOWN_EMBEDDING_MODELS
    ]

    print()
    print("LLM Model:")
    for i, (mid, desc) in enumerate(PRIMARY_LLMS, 1):
        marker = "  <-- current" if mid == current_llm else ""
        print(f"  [{i}]  {mid:<22}  {desc}{marker}")
    print("  [3]  More models...")
    print()

    chosen_llm = current_llm
    while True:
        raw = input(f"Select LLM [1-3, Enter=keep ({current_llm})]: ").strip()
        if not raw:
            break
        if raw == "1":
            chosen_llm = PRIMARY_LLMS[0][0]
            break
        if raw == "2":
            chosen_llm = PRIMARY_LLMS[1][0]
            break
        if raw == "3":
            if not extra_llms:
                print("  (No additional models found from API)")
                break
            print()
            print("  More LLM options (fetched from FARMI):")
            for i, mid in enumerate(extra_llms, 1):
                desc = MODEL_DESCRIPTIONS.get(mid, "")
                suffix = f"  {desc}" if desc else ""
                print(f"  [{i}]  {mid:<40}{suffix}")
            print()
            sub = input(f"  Select LLM [1-{len(extra_llms)}]: ").strip()
            if sub.isdigit() and 1 <= int(sub) <= len(extra_llms):
                chosen_llm = extra_llms[int(sub) - 1]
            break
        print("  Invalid choice. Enter 1, 2, or 3.")

    config["LLM_MODEL"] = chosen_llm

    # ── Embedding selection ───────────────────────────────────────────────────
    PRIMARY_EMBEDDINGS = [
        ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6-v2 (384-dim) — local, fast [DEFAULT]", "local"),
        ("snowflake-arctic-embed2:latest",          "Snowflake Arctic (1024-dim) — API, best retrieval", "api"),
    ]
    current_emb = config.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    primary_emb_ids = {m for m, _, _ in PRIMARY_EMBEDDINGS}
    extra_embs = [m for m in KNOWN_EMBEDDING_MODELS if m not in primary_emb_ids]
    # Add any API-returned embedding models not already in the list
    for m in api_models:
        if m in KNOWN_EMBEDDING_MODELS and m not in primary_emb_ids and m not in extra_embs:
            extra_embs.append(m)

    print()
    print("Embedding Model:")
    for i, (mid, desc, _) in enumerate(PRIMARY_EMBEDDINGS, 1):
        short = mid.split("/")[-1]
        marker = "  <-- current" if mid == current_emb else ""
        print(f"  [{i}]  {short:<45}  {desc}{marker}")
    print("  [3]  More models...")
    print()

    chosen_emb = current_emb
    chosen_backend = config.get("EMBEDDING_BACKEND", "local")
    current_short = current_emb.split("/")[-1]
    while True:
        raw = input(f"Select Embedding [1-3, Enter=keep ({current_short})]: ").strip()
        if not raw:
            break
        if raw == "1":
            chosen_emb, chosen_backend = PRIMARY_EMBEDDINGS[0][0], "local"
            break
        if raw == "2":
            chosen_emb, chosen_backend = PRIMARY_EMBEDDINGS[1][0], "api"
            break
        if raw == "3":
            if not extra_embs:
                print("  (No additional embedding models found)")
                break
            print()
            print("  More embedding options:")
            for i, mid in enumerate(extra_embs, 1):
                desc = MODEL_DESCRIPTIONS.get(mid, "")
                suffix = f"  {desc}" if desc else ""
                print(f"  [{i}]  {mid:<45}{suffix}")
            print()
            sub = input(f"  Select Embedding [1-{len(extra_embs)}]: ").strip()
            if sub.isdigit() and 1 <= int(sub) <= len(extra_embs):
                chosen_emb = extra_embs[int(sub) - 1]
                chosen_backend = "api"
            break
        print("  Invalid choice. Enter 1, 2, or 3.")

    config["EMBEDDING_MODEL"] = chosen_emb
    config["EMBEDDING_BACKEND"] = chosen_backend

    # ── Retrieval Depth (K) ───────────────────────────────────────────────────
    current_k = config.get("TOP_K_RETRIEVAL", 5)
    print()
    print("Retrieval Depth (K — documents retrieved per query):")
    print(f"  [1]  K=3  — faster,  fewer NLI calls (~9 per batch)")
    print(f"  [2]  K=5  — standard, stronger cross-doc signal (~18 per batch) [DEFAULT]")
    print()
    chosen_k = current_k
    while True:
        raw = input(f"Select K [1-2, Enter=keep ({current_k})]: ").strip()
        if not raw:
            break
        if raw == "1":
            chosen_k = 3
            break
        if raw == "2":
            chosen_k = 5
            break
        print("  Invalid choice. Enter 1 or 2.")

    config["TOP_K_RETRIEVAL"] = chosen_k

    # ── Confirmation ──────────────────────────────────────────────────────────
    print()
    print("─" * 54)
    print(f"  LLM:       {chosen_llm}")
    print(f"  Embedding: {chosen_emb}  (backend: {chosen_backend})")
    print(f"  K:         {chosen_k}")
    print("─" * 54)
    if input("Proceed? [Y/n]: ").strip().lower() == "n":
        print("Aborted. Re-run to choose again.")
        sys.exit(0)

    return config


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
        'EMBEDDING_BACKEND': os.environ.get('EMBEDDING_BACKEND', 'local'),
        'TOP_K_RETRIEVAL': int(os.environ.get('TOP_K_RETRIEVAL', '5')),
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

# 2×2 factorial grid: (LLM, Embedding model, backend)
GRID_COMBINATIONS = [
    ("llama3.3:70b", "sentence-transformers/all-MiniLM-L6-v2", "local"),
    ("llama3.3:70b", "snowflake-arctic-embed2:latest",          "api"),
    ("qwen3.5:35b",  "sentence-transformers/all-MiniLM-L6-v2", "local"),
    ("qwen3.5:35b",  "snowflake-arctic-embed2:latest",          "api"),
]


def run_grid_experiments(pipeline_config: dict, num_samples: int = 50, dataset: str = "truthfulqa"):
    """
    Run all 4 combinations of the 2x2 factorial grid automatically.

    Grid: {llama3.3:70b, qwen3.5:35b} x {all-MiniLM-L6-v2, snowflake-arctic-embed2}
    Results are saved separately for each combination.
    """
    results = []
    total = len(GRID_COMBINATIONS)

    print("\n" + "=" * 60)
    print(f"2x2 GRID EXPERIMENT — {dataset.upper()} ({num_samples} samples each)")
    print("=" * 60)
    for i, (llm, emb, backend) in enumerate(GRID_COMBINATIONS, 1):
        emb_short = emb.split("/")[-1]
        print(f"  [{i}/{total}]  {llm}  +  {emb_short}")
    print("=" * 60)

    for i, (llm, emb, backend) in enumerate(GRID_COMBINATIONS, 1):
        emb_label = emb.split("/")[-1]
        print(f"\n{'─' * 60}")
        print(f"  COMBINATION {i}/{total}: {llm}  +  {emb_label}  (backend: {backend})")
        print(f"{'─' * 60}")

        cfg = dict(pipeline_config)
        cfg["LLM_MODEL"]        = llm
        cfg["EMBEDDING_MODEL"]  = emb
        cfg["EMBEDDING_BACKEND"] = backend

        result = run_main_experiment(cfg, num_samples, dataset)
        results.append((llm, emb, result))

    # Print comparison summary
    print("\n" + "=" * 60)
    print("GRID RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'LLM':<18}  {'Embedding':<35}  {'Acc':>5}  {'F1':>5}  {'Sep':>5}")
    print(f"  {'─'*18}  {'─'*35}  {'─'*5}  {'─'*5}  {'─'*5}")
    for llm, emb, result in results:
        if result is None:
            continue
        emb_label = emb.split("/")[-1][:35]
        print(
            f"  {llm:<18}  {emb_label:<35}  "
            f"{result.accuracy*100:>4.0f}%  {result.f1_score*100:>4.0f}%  {result.trust_score_separation:>5.3f}"
        )
    print("=" * 60)
    return results


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

    llm_short = pipeline_config.get("LLM_MODEL", "").replace(":", "-").split("/")[-1]
    emb_short = pipeline_config.get("EMBEDDING_MODEL", "").split("/")[-1].replace(":", "-")

    exp_config = ExperimentConfig(
        name=f"{dataset}_{llm_short}_{emb_short}_detection",
        description=(
            f"Evaluate poison detection on {dataset.upper()} with "
            f"{num_samples} samples, 30% poisoned (MIXED strategy) | "
            f"LLM={pipeline_config.get('LLM_MODEL')} | Emb={pipeline_config.get('EMBEDDING_MODEL')}"
        ),
        num_samples=num_samples,
        poison_ratio=0.3,
        top_k=pipeline_config.get('TOP_K_RETRIEVAL', 5),
        llm_model=pipeline_config.get('LLM_MODEL', 'llama3.3:70b'),
        embedding_model=pipeline_config.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
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
            top_k=pipeline_config.get('TOP_K_RETRIEVAL', 5),
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
            top_k=pipeline_config.get('TOP_K_RETRIEVAL', 5),
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
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Skip model selection prompt; use values from .env"
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Run all 4 combinations of the 2x2 grid (llama/qwen x MiniLM/Snowflake) automatically"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    num_samples = 10 if args.quick else args.samples
    pipeline_config = get_pipeline_config()

    # --grid: run all 4 combinations automatically (skip interactive menu)
    if args.grid:
        run_grid_experiments(pipeline_config, num_samples, "truthfulqa")
        print("\n" + "=" * 60)
        print("ALL GRID EXPERIMENTS COMPLETE!")
        print("=" * 60)
        return

    if not args.no_interactive:
        pipeline_config = select_models_interactively(pipeline_config)

    print("=" * 60)
    print("TRUSTWORTHY RAG — EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Samples:   {num_samples}")
    print(f"LLM:       {pipeline_config['LLM_MODEL']}")
    print(f"Embedding: {pipeline_config['EMBEDDING_MODEL']}  (backend: {pipeline_config.get('EMBEDDING_BACKEND', 'local')})")
    print(f"K:         {pipeline_config.get('TOP_K_RETRIEVAL', 5)}")
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

    # --- Auto-generate charts ---
    print("\n" + "=" * 60)
    print("GENERATING THESIS CHARTS...")
    print("=" * 60)
    try:
        from generate_charts import main as generate_charts_main
        generate_charts_main()
    except Exception as e:
        print(f"  [WARNING] Chart generation failed: {e}")
        print("  You can manually run: python generate_charts.py")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE! Charts saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
