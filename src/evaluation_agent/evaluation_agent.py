"""
Evaluation Agent Module
The main orchestrator that combines NLI verification, poison detection,
and trust index calculation into a unified evaluation system.

This is the core research component of the thesis - detecting misinformation
and knowledge poisoning in RAG systems.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time

from .nli_verifier import NLIVerifier, VerificationResult
from .poison_detector import PoisonDetector, PoisonDetectionResult
from .trust_index import TrustIndexCalculator, TrustIndexResult, TrustLevel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a RAG response.
    
    This is the main output of the Evaluation Agent, containing:
    - Trust Index (the overall trustworthiness score)
    - NLI Verification details
    - Poison Detection details
    - Recommendations and warnings
    """
    # The query and response being evaluated
    query: str
    response: str
    retrieved_documents: List[str]
    
    # Core evaluation results
    trust_index: TrustIndexResult
    nli_verification: VerificationResult
    poison_detection: PoisonDetectionResult
    
    # Quick access fields
    trust_score: float
    trust_level: TrustLevel
    is_trustworthy: bool
    
    # Timing
    evaluation_time_ms: float
    
    # Summary
    summary: str
    detailed_report: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'response': self.response,
            'trust_score': self.trust_score,
            'trust_level': self.trust_level.value,
            'is_trustworthy': self.is_trustworthy,
            'trust_index': self.trust_index.to_dict(),
            'nli_summary': {
                'avg_entailment': self.nli_verification.avg_entailment,
                'is_supported': self.nli_verification.is_supported,
                'support_ratio': self.nli_verification.support_ratio
            },
            'poison_summary': {
                'probability': self.poison_detection.overall_poison_probability,
                'is_contaminated': self.poison_detection.is_contaminated,
                'suspicious_docs': self.poison_detection.num_suspicious_docs
            },
            'evaluation_time_ms': self.evaluation_time_ms,
            'summary': self.summary
        }


class EvaluationAgent:
    """
    The main Evaluation Agent that orchestrates all evaluation components.
    
    This is the central class for evaluating RAG responses. It:
    1. Verifies factual consistency using NLI
    2. Detects potential knowledge poisoning
    3. Calculates the Trust Index
    4. Generates recommendations and reports
    
    Usage:
        agent = EvaluationAgent(config)
        result = agent.evaluate(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieved_documents=["Paris is the capital of France..."]
        )
        print(f"Trust Score: {result.trust_score}")
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        nli_model: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        trust_alpha: float = 0.4,
        trust_beta: float = 0.35,
        trust_gamma: float = 0.25,
        trust_threshold: float = 0.5,
        poison_threshold: float = 0.7
    ):
        """
        Initialize the Evaluation Agent.
        
        Args:
            config: Optional config dictionary (overrides other args)
            nli_model: HuggingFace model for NLI verification
            device: 'cpu' or 'cuda'
            trust_alpha: Weight for factuality in Trust Index
            trust_beta: Weight for consistency in Trust Index
            trust_gamma: Weight for poison safety in Trust Index
            trust_threshold: Minimum Trust Index to be considered trustworthy
            poison_threshold: Threshold for flagging potential poisoning
        """
        # Load from config if provided
        if config:
            nli_model = config.get('NLI_MODEL', nli_model)
            trust_alpha = config.get('TRUST_ALPHA', trust_alpha)
            trust_beta = config.get('TRUST_BETA', trust_beta)
            trust_gamma = config.get('TRUST_GAMMA', trust_gamma)
            trust_threshold = config.get('TRUST_THRESHOLD', trust_threshold)
            poison_threshold = config.get('POISON_THRESHOLD', poison_threshold)
        
        # Initialize components
        self.nli_verifier = NLIVerifier(
            model_name=nli_model,
            device=device
        )
        
        self.poison_detector = PoisonDetector(
            poison_threshold=poison_threshold
        )
        
        self.trust_calculator = TrustIndexCalculator(
            alpha=trust_alpha,
            beta=trust_beta,
            gamma=trust_gamma,
            trust_threshold=trust_threshold
        )
        
        self.config = config or {}
        logger.info("EvaluationAgent initialized successfully")
    
    def evaluate(
        self,
        query: str,
        response: str,
        retrieved_documents: List[str],
        retrieval_scores: Optional[List[float]] = None,
        document_embeddings: Optional[Any] = None
    ) -> EvaluationResult:
        """
        Perform complete evaluation of a RAG response.
        
        Args:
            query: The user's original query
            response: The generated response to evaluate
            retrieved_documents: List of documents used for generation
            retrieval_scores: Optional similarity scores from retrieval
            document_embeddings: Optional embeddings for semantic analysis
            
        Returns:
            EvaluationResult with complete analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting evaluation for query: {query[:50]}...")
        
        # Step 1: NLI Verification
        logger.debug("Running NLI verification...")
        nli_result = self.nli_verifier.verify_answer(
            answer=response,
            documents=retrieved_documents
        )
        
        # Step 2: Poison Detection
        logger.debug("Running poison detection...")
        poison_result = self.poison_detector.detect_batch(
            documents=retrieved_documents,
            embeddings=document_embeddings
        )
        
        # Step 3: Calculate document consistency
        # (Use NLI support ratio as a proxy for consistency)
        consistency_score = self._calculate_consistency(
            nli_result, retrieved_documents
        )
        
        # Step 4: Calculate Trust Index
        logger.debug("Calculating Trust Index...")
        trust_result = self.trust_calculator.calculate(
            factuality_score=self.nli_verifier.get_factuality_score(nli_result),
            consistency_score=consistency_score,
            poison_score=self.poison_detector.get_poison_score(poison_result),
            retrieval_confidence=self._get_retrieval_confidence(retrieval_scores)
        )
        
        # Calculate timing
        evaluation_time_ms = (time.time() - start_time) * 1000
        
        # Generate summary and report
        summary = self._generate_summary(trust_result, nli_result, poison_result)
        detailed_report = self._generate_detailed_report(
            query, response, trust_result, nli_result, poison_result
        )
        
        logger.info(
            f"Evaluation complete. Trust Score: {trust_result.trust_score:.2f} "
            f"({trust_result.trust_level.value})"
        )
        
        return EvaluationResult(
            query=query,
            response=response,
            retrieved_documents=retrieved_documents,
            trust_index=trust_result,
            nli_verification=nli_result,
            poison_detection=poison_result,
            trust_score=trust_result.trust_score,
            trust_level=trust_result.trust_level,
            is_trustworthy=trust_result.is_trustworthy,
            evaluation_time_ms=evaluation_time_ms,
            summary=summary,
            detailed_report=detailed_report
        )
    
    def _calculate_consistency(
        self,
        nli_result: VerificationResult,
        documents: List[str]
    ) -> float:
        """
        Calculate document consistency score.
        
        Uses the NLI support ratio and checks for contradictions
        between documents.
        """
        if not documents:
            return 0.0
        
        # Start with support ratio
        base_consistency = nli_result.support_ratio
        
        # Penalize for high contradiction
        contradiction_penalty = nli_result.max_contradiction * 0.3
        
        # Check variance in entailment scores
        if nli_result.individual_results:
            entailment_scores = [
                r.entailment_score for r in nli_result.individual_results
            ]
            if len(entailment_scores) > 1:
                variance = self._calculate_variance(entailment_scores)
                # High variance = low consistency
                variance_penalty = variance * 0.2
            else:
                variance_penalty = 0.0
        else:
            variance_penalty = 0.0
        
        consistency = base_consistency - contradiction_penalty - variance_penalty
        return max(0.0, min(1.0, consistency))
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of a list of scores."""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance
    
    def _get_retrieval_confidence(
        self,
        retrieval_scores: Optional[List[float]]
    ) -> float:
        """Convert retrieval scores to confidence value."""
        if not retrieval_scores:
            return 1.0
        
        # Average score, clamped to [0, 1]
        avg = sum(retrieval_scores) / len(retrieval_scores)
        return max(0.0, min(1.0, avg))
    
    def _generate_summary(
        self,
        trust: TrustIndexResult,
        nli: VerificationResult,
        poison: PoisonDetectionResult
    ) -> str:
        """Generate a brief summary of evaluation results."""
        parts = []
        
        # Trust level statement
        if trust.trust_level == TrustLevel.HIGH:
            parts.append("Response is TRUSTWORTHY.")
        elif trust.trust_level == TrustLevel.MEDIUM:
            parts.append("Response has MODERATE trustworthiness.")
        elif trust.trust_level == TrustLevel.LOW:
            parts.append("Response has LOW trustworthiness.")
        else:
            parts.append("Response is NOT TRUSTWORTHY.")
        
        # Key metrics
        parts.append(
            f"Trust Score: {trust.trust_score:.0%}. "
            f"Factual support: {nli.avg_entailment:.0%}."
        )
        
        # Warnings
        if poison.is_contaminated:
            parts.append("ALERT: Potential poisoning detected!")
        if nli.max_contradiction > 0.7:
            parts.append("ALERT: Document contradictions found!")
        
        return " ".join(parts)
    
    def _generate_detailed_report(
        self,
        query: str,
        response: str,
        trust: TrustIndexResult,
        nli: VerificationResult,
        poison: PoisonDetectionResult
    ) -> str:
        """Generate a detailed evaluation report."""
        lines = [
            "=" * 60,
            "EVALUATION AGENT REPORT",
            "=" * 60,
            "",
            "QUERY:",
            f"  {query}",
            "",
            "RESPONSE:",
            f"  {response[:200]}{'...' if len(response) > 200 else ''}",
            "",
            "-" * 60,
            "TRUST INDEX",
            "-" * 60,
            f"  Overall Score: {trust.trust_score:.2f} / 1.00",
            f"  Trust Level: {trust.trust_level.value.upper()}",
            f"  Is Trustworthy: {'YES' if trust.is_trustworthy else 'NO'}",
            "",
            "  Component Scores:",
            f"    - Factuality:    {trust.components.factuality_score:.2f} "
            f"(contributes {trust.factuality_contribution:.3f})",
            f"    - Consistency:   {trust.components.consistency_score:.2f} "
            f"(contributes {trust.consistency_contribution:.3f})",
            f"    - Poison Safety: {1 - trust.components.poison_score:.2f} "
            f"(contributes {trust.poison_contribution:.3f})",
            "",
            "-" * 60,
            "NLI VERIFICATION",
            "-" * 60,
            f"  Answer Supported: {'YES' if nli.is_supported else 'NO'}",
            f"  Avg Entailment: {nli.avg_entailment:.2f}",
            f"  Avg Contradiction: {nli.avg_contradiction:.2f}",
            f"  Max Contradiction: {nli.max_contradiction:.2f}",
            f"  Support Ratio: {nli.support_ratio:.0%}",
            "",
            f"  Explanation: {nli.explanation}",
            "",
            "-" * 60,
            "POISON DETECTION",
            "-" * 60,
            f"  Contamination Detected: {'YES' if poison.is_contaminated else 'NO'}",
            f"  Overall Poison Probability: {poison.overall_poison_probability:.2f}",
            f"  Suspicious Documents: {poison.num_suspicious_docs} / {len(poison.documents)}",
            f"  High-Risk Indices: {poison.high_risk_indices or 'None'}",
            "",
            f"  Explanation: {poison.explanation}",
            "",
        ]
        
        # Add warnings
        if trust.warnings:
            lines.extend([
                "-" * 60,
                "WARNINGS",
                "-" * 60,
            ])
            for warning in trust.warnings:
                lines.append(f"  ! {warning}")
            lines.append("")
        
        # Add recommendation
        lines.extend([
            "-" * 60,
            "RECOMMENDATION",
            "-" * 60,
            f"  {trust.recommendation}",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def quick_evaluate(
        self,
        response: str,
        documents: List[str]
    ) -> Dict[str, Any]:
        """
        Quick evaluation returning just the essential metrics.
        
        Useful for batch evaluation or when you just need the numbers.
        
        Args:
            response: The generated response
            documents: Retrieved documents
            
        Returns:
            Dictionary with key metrics
        """
        result = self.evaluate(
            query="",  # Query not needed for quick eval
            response=response,
            retrieved_documents=documents
        )
        
        return {
            'trust_score': result.trust_score,
            'is_trustworthy': result.is_trustworthy,
            'factuality': result.trust_index.components.factuality_score,
            'consistency': result.trust_index.components.consistency_score,
            'poison_probability': result.trust_index.components.poison_score
        }
    
    def evaluate_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple query-response pairs.
        
        Args:
            items: List of dicts with 'query', 'response', 'documents' keys
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            logger.info(f"Evaluating item {i+1}/{total}")
            result = self.evaluate(
                query=item.get('query', ''),
                response=item.get('response', ''),
                retrieved_documents=item.get('documents', [])
            )
            results.append(result)
        
        return results
