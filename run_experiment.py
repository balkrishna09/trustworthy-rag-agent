"""
Run Experiment - Trustworthy RAG Evaluation
Tests the system on TruthfulQA with clean and poisoned document sets.
"""

import sys
import os
import json
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import ExperimentRunner, ExperimentConfig
from src.experiments.poisoned_dataset import PoisonedDatasetGenerator, PoisonStrategy


def load_truthfulqa_for_experiment(max_samples: int = 20):
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
    
    # Load data
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} TruthfulQA samples")
    
    # Filter samples that have good correct answers
    good_samples = []
    for s in samples:
        # TruthfulQA has 'best_answer' and 'correct_answers' fields
        best_answer = s.get('best_answer', '')
        correct_answers = s.get('correct_answers', [])
        question = s.get('question', '')
        
        if best_answer and question and len(best_answer) > 10:
            good_samples.append(s)
    
    print(f"Filtered to {len(good_samples)} samples with good answers")
    
    # Limit samples
    good_samples = good_samples[:max_samples]
    
    # Build knowledge base: Use correct answers as documents
    knowledge_base = []
    questions = []
    expected_answers = []
    
    for s in good_samples:
        question = s['question']
        best_answer = s['best_answer']
        category = s.get('category', 'general')
        
        # Create a "document" from the correct answer
        doc = f"{question} {best_answer}"
        knowledge_base.append(doc)
        questions.append(question)
        expected_answers.append(best_answer)
    
    print(f"Prepared {len(questions)} questions with knowledge base")
    return knowledge_base, questions, expected_answers


def main():
    print("=" * 60)
    print("TRUSTWORTHY RAG - EXPERIMENT RUNNER")
    print("=" * 60)
    
    # Pipeline configuration
    pipeline_config = {
        'LLM_PROVIDER': 'openai_compatible',
        'FARMI_API_URL': 'https://gptlab.rd.tuni.fi/students/ollama/v1',
        'FARMI_API_KEY': os.environ.get('FARMI_API_KEY', ''),  # Set via environment variable
        'LLM_MODEL': 'llama3.3:70b',
        'MAX_NEW_TOKENS': 150,
        'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
        'TOP_K_RETRIEVAL': 3,
        'NLI_MODEL': 'facebook/bart-large-mnli',
        'TRUST_THRESHOLD': 0.5,
        'POISON_THRESHOLD': 0.7,
        'TRUST_ALPHA': 0.4,
        'TRUST_BETA': 0.35,
        'TRUST_GAMMA': 0.25
    }
    
    # Experiment configuration
    # Start small (10 samples) to test, increase later
    exp_config = ExperimentConfig(
        name="truthfulqa_poison_detection",
        description="Evaluate poison detection on TruthfulQA with 30% poisoned documents",
        num_samples=10,       # Start with 10 (increase to 50+ later)
        poison_ratio=0.3,     # 30% of documents will be poisoned
        top_k=3
    )
    
    # Load data
    print("\n[1/4] Loading TruthfulQA dataset...")
    knowledge_base, questions, expected_answers = load_truthfulqa_for_experiment(
        max_samples=exp_config.num_samples
    )
    
    if knowledge_base is None:
        print("Failed to load data. Exiting.")
        return
    
    # Show some samples
    print("\nSample questions:")
    for i, (q, a) in enumerate(zip(questions[:3], expected_answers[:3])):
        print(f"  Q{i+1}: {q}")
        print(f"  A{i+1}: {a[:80]}...")
        print()
    
    # Run experiment
    print("[2/4] Initializing experiment runner...")
    runner = ExperimentRunner(pipeline_config)
    
    print("[3/4] Running experiment (this will take several minutes)...")
    print(f"       - {exp_config.num_samples} questions x 2 (clean + poisoned)")
    print(f"       - Each query involves: retrieval + generation + evaluation")
    print()
    
    result = runner.run_experiment(
        config=exp_config,
        knowledge_base=knowledge_base,
        questions=questions,
        expected_answers=expected_answers
    )
    
    # Print results
    result.print_summary()
    
    # Save results
    print("\n[4/4] Saving results...")
    saved_path = runner.save_results(result)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"\nKey findings:")
    print(f"  - Accuracy:  {result.accuracy:.2%}")
    print(f"  - Precision: {result.precision:.2%}")
    print(f"  - Recall:    {result.recall:.2%}")
    print(f"  - F1 Score:  {result.f1_score:.2%}")
    print(f"\n  Clean docs avg trust:    {result.avg_trust_clean:.3f}")
    print(f"  Poisoned docs avg trust: {result.avg_trust_poisoned:.3f}")
    print(f"  Separation:              {result.trust_score_separation:.3f}")
    print(f"\nResults saved to: {saved_path}")


if __name__ == "__main__":
    main()
