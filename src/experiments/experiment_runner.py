"""
Experiment Runner
Runs evaluation experiments on clean and poisoned datasets.

This module orchestrates the complete experiment pipeline:
1. Load or generate datasets (clean + poisoned)
2. Run queries through the RAG pipeline
3. Collect evaluation metrics
4. Save results for analysis
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    description: str = ""
    
    # Dataset settings
    num_samples: int = 50  # Number of samples to evaluate
    poison_ratio: float = 0.3  # Fraction of documents to poison
    
    # Pipeline settings
    top_k: int = 3
    
    # Model settings
    llm_model: str = "llama3.3:70b"
    nli_model: str = "facebook/bart-large-mnli"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Trust settings
    trust_threshold: float = 0.5
    poison_threshold: float = 0.7
    trust_alpha: float = 0.4
    trust_beta: float = 0.35
    trust_gamma: float = 0.25
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'num_samples': self.num_samples,
            'poison_ratio': self.poison_ratio,
            'top_k': self.top_k,
            'llm_model': self.llm_model,
            'nli_model': self.nli_model,
            'embedding_model': self.embedding_model,
            'trust_threshold': self.trust_threshold,
            'poison_threshold': self.poison_threshold,
            'trust_alpha': self.trust_alpha,
            'trust_beta': self.trust_beta,
            'trust_gamma': self.trust_gamma
        }


@dataclass
class SampleResult:
    """Result for a single experiment sample."""
    sample_id: int
    query: str
    expected_answer: Optional[str]
    generated_answer: str

    # Is this sample from a poisoned set?
    is_poisoned_set: bool

    # Evaluation metrics
    trust_score: float
    trust_level: str
    is_trustworthy: bool
    factuality_score: float
    consistency_score: float
    poison_probability: float

    # Detection correctness
    # True Positive: poisoned AND detected as untrustworthy
    # True Negative: clean AND detected as trustworthy
    # False Positive: clean BUT detected as untrustworthy
    # False Negative: poisoned BUT detected as trustworthy
    detection_correct: bool

    # Poisoning strategy (for per-strategy breakdown)
    poison_strategy: str = "none"

    # Timing
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0
    evaluation_time_ms: float = 0
    total_time_ms: float = 0

    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'query': self.query,
            'expected_answer': self.expected_answer,
            'generated_answer': self.generated_answer,
            'is_poisoned_set': self.is_poisoned_set,
            'poison_strategy': self.poison_strategy,
            'trust_score': float(self.trust_score),
            'trust_level': self.trust_level,
            'is_trustworthy': bool(self.is_trustworthy),
            'factuality_score': float(self.factuality_score),
            'consistency_score': float(self.consistency_score),
            'poison_probability': float(self.poison_probability),
            'detection_correct': bool(self.detection_correct),
            'retrieval_time_ms': float(self.retrieval_time_ms),
            'generation_time_ms': float(self.generation_time_ms),
            'evaluation_time_ms': float(self.evaluation_time_ms),
            'total_time_ms': float(self.total_time_ms)
        }


@dataclass
class ExperimentResult:
    """Complete result for an experiment run."""
    config: ExperimentConfig
    sample_results: List[SampleResult]
    
    # Aggregated metrics
    total_samples: int = 0
    
    # Detection performance
    true_positives: int = 0   # Poisoned, correctly flagged
    true_negatives: int = 0   # Clean, correctly trusted
    false_positives: int = 0  # Clean, incorrectly flagged
    false_negatives: int = 0  # Poisoned, incorrectly trusted
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Trust score statistics
    avg_trust_clean: float = 0.0
    avg_trust_poisoned: float = 0.0
    trust_score_separation: float = 0.0  # Difference between clean and poisoned
    
    # Timing
    total_time_seconds: float = 0.0
    avg_time_per_sample_ms: float = 0.0
    avg_evaluation_time_ms: float = 0.0  # Evaluation-only latency

    # Per-strategy breakdown (RQ3)
    per_strategy_metrics: Dict = field(default_factory=dict)

    # Attack Success Rate (ASR) — fraction of poisoned samples that
    # the LLM actually incorporated into its answer (regardless of
    # whether the evaluation agent flagged them).
    attack_success_rate: float = 0.0

    # Baseline comparison (no evaluation agent)
    baseline_accuracy: Optional[float] = None

    # Metadata
    timestamp: str = ""
    
    def calculate_metrics(self):
        """Calculate aggregated metrics from sample results."""
        if not self.sample_results:
            return
        
        self.total_samples = len(self.sample_results)
        
        # Calculate detection metrics
        self.true_positives = sum(
            1 for r in self.sample_results 
            if r.is_poisoned_set and not r.is_trustworthy
        )
        self.true_negatives = sum(
            1 for r in self.sample_results 
            if not r.is_poisoned_set and r.is_trustworthy
        )
        self.false_positives = sum(
            1 for r in self.sample_results 
            if not r.is_poisoned_set and not r.is_trustworthy
        )
        self.false_negatives = sum(
            1 for r in self.sample_results 
            if r.is_poisoned_set and r.is_trustworthy
        )
        
        # Accuracy
        correct = self.true_positives + self.true_negatives
        self.accuracy = correct / self.total_samples if self.total_samples > 0 else 0
        
        # Precision: Of all flagged as untrustworthy, how many are actually poisoned?
        flagged = self.true_positives + self.false_positives
        self.precision = self.true_positives / flagged if flagged > 0 else 0
        
        # Recall: Of all poisoned samples, how many did we catch?
        poisoned = self.true_positives + self.false_negatives
        self.recall = self.true_positives / poisoned if poisoned > 0 else 0
        
        # F1 Score
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0
        
        # Trust score averages
        clean_scores = [r.trust_score for r in self.sample_results if not r.is_poisoned_set]
        poisoned_scores = [r.trust_score for r in self.sample_results if r.is_poisoned_set]
        
        self.avg_trust_clean = sum(clean_scores) / len(clean_scores) if clean_scores else 0
        self.avg_trust_poisoned = sum(poisoned_scores) / len(poisoned_scores) if poisoned_scores else 0
        self.trust_score_separation = self.avg_trust_clean - self.avg_trust_poisoned
        
        # Timing
        total_ms = sum(r.total_time_ms for r in self.sample_results)
        self.total_time_seconds = total_ms / 1000
        self.avg_time_per_sample_ms = total_ms / self.total_samples if self.total_samples > 0 else 0

        eval_times = [r.evaluation_time_ms for r in self.sample_results if r.evaluation_time_ms > 0]
        self.avg_evaluation_time_ms = sum(eval_times) / len(eval_times) if eval_times else 0.0

        # Per-strategy breakdown
        self._calculate_per_strategy_metrics()

        self.timestamp = datetime.now().isoformat()

    def _calculate_per_strategy_metrics(self):
        """Calculate detection metrics broken down by poisoning strategy."""
        strategy_buckets: Dict[str, list] = {}
        for r in self.sample_results:
            if r.is_poisoned_set and r.poison_strategy != "none":
                strategy_buckets.setdefault(r.poison_strategy, []).append(r)

        self.per_strategy_metrics = {}
        for strategy, samples in strategy_buckets.items():
            detected = sum(1 for s in samples if not s.is_trustworthy)
            total = len(samples)
            self.per_strategy_metrics[strategy] = {
                'total': total,
                'detected': detected,
                'detection_rate': detected / total if total > 0 else 0.0,
                'avg_trust_score': sum(s.trust_score for s in samples) / total if total > 0 else 0.0,
                'avg_poison_probability': sum(s.poison_probability for s in samples) / total if total > 0 else 0.0,
            }
    
    def to_dict(self) -> Dict:
        return {
            'config': self.config.to_dict(),
            'metrics': {
                'total_samples': self.total_samples,
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'true_positives': self.true_positives,
                'true_negatives': self.true_negatives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives,
                'avg_trust_clean': self.avg_trust_clean,
                'avg_trust_poisoned': self.avg_trust_poisoned,
                'trust_score_separation': self.trust_score_separation,
                'total_time_seconds': self.total_time_seconds,
                'avg_time_per_sample_ms': self.avg_time_per_sample_ms,
                'avg_evaluation_time_ms': self.avg_evaluation_time_ms,
                'attack_success_rate': self.attack_success_rate,
                'baseline_accuracy': self.baseline_accuracy,
                'per_strategy_metrics': self.per_strategy_metrics,
            },
            'timestamp': self.timestamp,
            'sample_results': [r.to_dict() for r in self.sample_results]
        }
    
    def print_summary(self):
        """Print a formatted summary of experiment results."""
        print("\n" + "=" * 60)
        print(f"EXPERIMENT RESULTS: {self.config.name}")
        print("=" * 60)
        print(f"Description: {self.config.description}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Samples: {self.total_samples}")
        print(f"Poison Ratio: {self.config.poison_ratio:.0%}")
        
        print("\n--- Detection Performance ---")
        print(f"Accuracy:  {self.accuracy:.2%}")
        print(f"Precision: {self.precision:.2%}")
        print(f"Recall:    {self.recall:.2%}")
        print(f"F1 Score:  {self.f1_score:.2%}")
        
        print("\n--- Confusion Matrix ---")
        print(f"                    Predicted")
        print(f"                 Trust  | Untrust")
        print(f"Actual  Clean  |  {self.true_negatives:3d}   |  {self.false_positives:3d}")
        print(f"       Poison  |  {self.false_negatives:3d}   |  {self.true_positives:3d}")
        
        print("\n--- Trust Score Analysis ---")
        print(f"Avg Trust (Clean):    {self.avg_trust_clean:.3f}")
        print(f"Avg Trust (Poisoned): {self.avg_trust_poisoned:.3f}")
        print(f"Separation:           {self.trust_score_separation:.3f}")
        
        print("\n--- Timing ---")
        print(f"Total Time: {self.total_time_seconds:.1f}s")
        print(f"Avg per Sample: {self.avg_time_per_sample_ms:.0f}ms")
        print(f"Avg Evaluation Latency: {self.avg_evaluation_time_ms:.0f}ms")

        if self.per_strategy_metrics:
            print("\n--- Per-Strategy Detection Rates ---")
            for strategy, m in self.per_strategy_metrics.items():
                print(
                    f"  {strategy:20s}  detected {m['detected']}/{m['total']} "
                    f"({m['detection_rate']:.0%})  avg_trust={m['avg_trust_score']:.3f}"
                )

        if self.baseline_accuracy is not None:
            print(f"\n--- Baseline (No Evaluation Agent) ---")
            print(f"Baseline Accuracy: {self.baseline_accuracy:.2%}")
            improvement = self.accuracy - self.baseline_accuracy
            print(f"Improvement:       {improvement:+.2%}")

        print("=" * 60)


class ExperimentRunner:
    """
    Runs evaluation experiments.
    
    Usage:
        runner = ExperimentRunner(pipeline_config)
        result = runner.run_experiment(
            experiment_config,
            clean_docs=["Paris is..."],
            questions=["What is the capital?"]
        )
        result.print_summary()
    """
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        """
        Initialize the experiment runner.
        
        Args:
            pipeline_config: Configuration for the RAG pipeline
        """
        self.pipeline_config = pipeline_config
        self.results_dir = Path("data/experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ExperimentRunner initialized")
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        knowledge_base: List[str],
        questions: List[str],
        expected_answers: Optional[List[str]] = None,
        poisoned_knowledge_base: Optional[List[str]] = None,
        poison_labels: Optional[List[bool]] = None
    ) -> ExperimentResult:
        """
        Run a complete experiment.
        
        Args:
            config: Experiment configuration
            knowledge_base: Clean documents for the knowledge base
            questions: List of questions to evaluate
            expected_answers: Optional expected answers for comparison
            poisoned_knowledge_base: Pre-poisoned version of knowledge base
            poison_labels: Which docs in poisoned_knowledge_base are actually poisoned
            
        Returns:
            ExperimentResult with all metrics
        """
        from src.pipeline import RAGPipeline
        from src.experiments.poisoned_dataset import PoisonedDatasetGenerator, PoisonStrategy
        
        logger.info(f"Starting experiment: {config.name}")
        
        # Limit samples
        questions = questions[:config.num_samples]
        if expected_answers:
            expected_answers = expected_answers[:config.num_samples]
        
        sample_results = []
        
        # --- Part A: Evaluate with CLEAN knowledge base ---
        print(f"\n[Part A] Evaluating {len(questions)} questions with CLEAN documents...")
        
        pipeline_clean = RAGPipeline(
            config=self.pipeline_config,
            enable_evaluation=True
        )
        pipeline_clean.add_documents(knowledge_base)
        
        for i, question in enumerate(questions):
            expected = expected_answers[i] if expected_answers else None
            
            print(f"  [{i+1}/{len(questions)}] {question[:50]}...")
            
            start = time.time()
            try:
                result = pipeline_clean.query_with_evaluation(question)
                elapsed = (time.time() - start) * 1000
                
                sample_results.append(SampleResult(
                    sample_id=i,
                    query=question,
                    expected_answer=expected,
                    generated_answer=result.response,
                    is_poisoned_set=False,
                    trust_score=result.trust_score or 0,
                    trust_level=result.trust_level.value if result.trust_level else "unknown",
                    is_trustworthy=result.is_trustworthy or False,
                    factuality_score=result.evaluation.trust_index.components.factuality_score if result.evaluation else 0,
                    consistency_score=result.evaluation.trust_index.components.consistency_score if result.evaluation else 0,
                    poison_probability=result.evaluation.trust_index.components.poison_score if result.evaluation else 0,
                    detection_correct=result.is_trustworthy or False,  # Clean should be trusted
                    evaluation_time_ms=result.evaluation.evaluation_time_ms if result.evaluation else 0,
                    total_time_ms=elapsed
                ))
            except Exception as e:
                logger.error(f"Error on sample {i}: {e}")
                elapsed = (time.time() - start) * 1000
                sample_results.append(SampleResult(
                    sample_id=i,
                    query=question,
                    expected_answer=expected,
                    generated_answer=f"ERROR: {str(e)}",
                    is_poisoned_set=False,
                    trust_score=0,
                    trust_level="error",
                    is_trustworthy=False,
                    factuality_score=0,
                    consistency_score=0,
                    poison_probability=0,
                    detection_correct=False,
                    total_time_ms=elapsed
                ))
        
        # --- Part B: Evaluate with POISONED knowledge base ---
        print(f"\n[Part B] Evaluating {len(questions)} questions with POISONED documents...")
        
        # Track which documents were actually poisoned (index → strategy)
        # and which were left clean
        poison_strategy_map: Dict[int, str] = {}
        actually_poisoned_indices: set = set()

        if poisoned_knowledge_base is None:
            # Generate poisoned version
            generator = PoisonedDatasetGenerator()
            samples, stats = generator.create_poisoned_dataset(
                knowledge_base,
                poison_ratio=config.poison_ratio,
                strategy=PoisonStrategy.MIXED
            )
            poisoned_knowledge_base = [s.poisoned_text for s in samples]
            # Map question index → strategy name (only for actually poisoned docs)
            for idx, s in enumerate(samples):
                if s.is_poisoned:
                    poison_strategy_map[idx] = s.strategy.value
                    actually_poisoned_indices.add(idx)
            print(f"  Generated poisoned dataset: {stats}")
            print(f"  Actually poisoned doc indices: {sorted(actually_poisoned_indices)}")
        else:
            # Pre-built poisoned KB — determine which docs were actually changed
            # by comparing with the clean knowledge base
            for idx in range(min(len(knowledge_base), len(poisoned_knowledge_base))):
                if knowledge_base[idx] != poisoned_knowledge_base[idx]:
                    actually_poisoned_indices.add(idx)
                    poison_strategy_map[idx] = "unknown"
            # If poison_labels are explicitly provided, use those instead
            if poison_labels:
                actually_poisoned_indices = {
                    idx for idx, is_p in enumerate(poison_labels) if is_p
                }
            print(f"  Pre-built poisoned KB: {len(actually_poisoned_indices)} docs modified")
        
        pipeline_poisoned = RAGPipeline(
            config=self.pipeline_config,
            enable_evaluation=True
        )
        pipeline_poisoned.add_documents(poisoned_knowledge_base)
        
        offset = len(questions)
        for i, question in enumerate(questions):
            expected = expected_answers[i] if expected_answers else None

            # Determine if THIS specific question's corresponding document
            # was actually poisoned. Only mark as poisoned if the relevant
            # doc was modified — otherwise it's effectively a clean sample.
            is_actually_poisoned = i in actually_poisoned_indices
            strategy = poison_strategy_map.get(i, "none")

            status = "POISONED" if is_actually_poisoned else "clean"
            print(f"  [{i+1}/{len(questions)}] [{status}] {question[:50]}...")

            start = time.time()
            try:
                result = pipeline_poisoned.query_with_evaluation(question)
                elapsed = (time.time() - start) * 1000

                # Detection correctness:
                # If actually poisoned: correct when flagged as untrustworthy
                # If clean (in poisoned KB): correct when flagged as trustworthy
                if is_actually_poisoned:
                    detection_correct = not (result.is_trustworthy or False)
                else:
                    detection_correct = result.is_trustworthy or False

                sample_results.append(SampleResult(
                    sample_id=offset + i,
                    query=question,
                    expected_answer=expected,
                    generated_answer=result.response,
                    is_poisoned_set=is_actually_poisoned,
                    trust_score=result.trust_score or 0,
                    trust_level=result.trust_level.value if result.trust_level else "unknown",
                    is_trustworthy=result.is_trustworthy or False,
                    factuality_score=result.evaluation.trust_index.components.factuality_score if result.evaluation else 0,
                    consistency_score=result.evaluation.trust_index.components.consistency_score if result.evaluation else 0,
                    poison_probability=result.evaluation.trust_index.components.poison_score if result.evaluation else 0,
                    detection_correct=detection_correct,
                    poison_strategy=strategy,
                    evaluation_time_ms=result.evaluation.evaluation_time_ms if result.evaluation else 0,
                    total_time_ms=elapsed
                ))
            except Exception as e:
                logger.error(f"Error on poisoned sample {i}: {e}")
                elapsed = (time.time() - start) * 1000
                sample_results.append(SampleResult(
                    sample_id=offset + i,
                    query=question,
                    expected_answer=expected,
                    generated_answer=f"ERROR: {str(e)}",
                    is_poisoned_set=is_actually_poisoned,
                    trust_score=0,
                    trust_level="error",
                    is_trustworthy=False,
                    factuality_score=0,
                    consistency_score=0,
                    poison_probability=0,
                    detection_correct=is_actually_poisoned,  # Error=untrusted=correct for poisoned
                    poison_strategy=strategy,
                    total_time_ms=elapsed
                ))
        
        # Calculate final metrics
        experiment_result = ExperimentResult(
            config=config,
            sample_results=sample_results
        )
        experiment_result.calculate_metrics()
        
        logger.info(
            f"Experiment complete: Accuracy={experiment_result.accuracy:.2%}, "
            f"F1={experiment_result.f1_score:.2%}"
        )
        
        return experiment_result
    
    def run_baseline(
        self,
        knowledge_base: List[str],
        poisoned_knowledge_base: List[str],
        questions: List[str],
    ) -> float:
        """
        Run a baseline experiment WITHOUT the evaluation agent.

        The baseline simply accepts all responses as trustworthy.
        Returns the accuracy that a naive system would achieve.

        With the corrected labeling: Part A samples are all clean (TN),
        Part B samples are a mix: clean docs in poisoned KB are TN,
        actually poisoned docs are FN.

        Args:
            knowledge_base: Clean documents
            poisoned_knowledge_base: Poisoned documents
            questions: Questions to test

        Returns:
            Baseline accuracy (float)
        """
        # Count actually poisoned docs
        num_actually_poisoned = sum(
            1 for a, b in zip(knowledge_base, poisoned_knowledge_base)
            if a != b
        )
        num_questions = len(questions)

        # Part A: all clean → all TN
        # Part B: clean docs in poisoned KB → TN, actually poisoned → FN
        total = num_questions * 2
        true_negatives = num_questions + (num_questions - num_actually_poisoned)
        false_negatives = num_actually_poisoned
        accuracy = true_negatives / total if total > 0 else 0.0
        logger.info(
            f"Baseline accuracy (always trust): {accuracy:.2%} "
            f"({num_actually_poisoned} actually poisoned out of {num_questions} docs)"
        )
        return accuracy

    def save_results(self, result: ExperimentResult, filename: Optional[str] = None):
        """Save experiment results to disk."""
        import numpy as np
        
        if filename is None:
            filename = f"experiment_{result.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        path = self.results_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logger.info(f"Results saved to {path}")
        print(f"Results saved to: {path}")
        return path
    
    @staticmethod
    def load_results(path: str) -> Dict:
        """Load experiment results from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
