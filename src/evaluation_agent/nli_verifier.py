"""
NLI Verifier Module
Uses Natural Language Inference to check if retrieved documents support the generated answer.

NLI classifies relationships between text pairs as:
- ENTAILMENT: The premise supports/implies the hypothesis
- CONTRADICTION: The premise contradicts the hypothesis  
- NEUTRAL: The premise neither supports nor contradicts the hypothesis
"""

import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Lazy imports to avoid loading heavy models at startup
_nli_pipeline = None
_tokenizer = None
_model = None

logger = logging.getLogger(__name__)


@dataclass
class NLIResult:
    """Result of NLI verification for a single premise-hypothesis pair."""
    premise: str
    hypothesis: str
    label: str  # ENTAILMENT, CONTRADICTION, or NEUTRAL
    scores: Dict[str, float]  # Confidence scores for each label
    
    @property
    def entailment_score(self) -> float:
        """Get the entailment probability."""
        return self.scores.get('entailment', 0.0)
    
    @property
    def contradiction_score(self) -> float:
        """Get the contradiction probability."""
        return self.scores.get('contradiction', 0.0)
    
    @property
    def neutral_score(self) -> float:
        """Get the neutral probability."""
        return self.scores.get('neutral', 0.0)


@dataclass
class VerificationResult:
    """Aggregated verification result for an answer against multiple documents."""
    answer: str
    documents: List[str]
    individual_results: List[NLIResult]
    
    # Aggregated scores
    avg_entailment: float
    avg_contradiction: float
    max_contradiction: float
    support_ratio: float  # Fraction of documents that support the answer
    
    # Overall verdict
    is_supported: bool
    confidence: float
    explanation: str


class NLIVerifier:
    """
    Verifies factual consistency between retrieved documents and generated answers
    using Natural Language Inference.
    
    How it works:
    1. Takes the generated answer (hypothesis)
    2. Compares it against each retrieved document (premise)
    3. Checks if documents ENTAIL (support) or CONTRADICT the answer
    4. Aggregates results to determine overall factual consistency
    """
    
    # Label mappings for different models
    LABEL_MAPPINGS = {
        'facebook/bart-large-mnli': {
            'ENTAILMENT': 'entailment',
            'CONTRADICTION': 'contradiction',
            'NEUTRAL': 'neutral',
            0: 'contradiction',
            1: 'neutral',
            2: 'entailment'
        },
        'microsoft/deberta-large-mnli': {
            'ENTAILMENT': 'entailment',
            'CONTRADICTION': 'contradiction',
            'NEUTRAL': 'neutral',
            0: 'contradiction',
            1: 'neutral', 
            2: 'entailment'
        }
    }
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        support_threshold: float = 0.5,
        contradiction_threshold: float = 0.7
    ):
        """
        Initialize the NLI Verifier.
        
        Args:
            model_name: HuggingFace model name for NLI
            device: 'cpu' or 'cuda' for GPU
            support_threshold: Minimum entailment score to consider supported
            contradiction_threshold: Score above which to flag contradiction
        """
        self.model_name = model_name
        self.device = device
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self._pipeline = None
        self._label_mapping = self.LABEL_MAPPINGS.get(
            model_name, 
            self.LABEL_MAPPINGS['facebook/bart-large-mnli']
        )
        
        logger.info(f"NLIVerifier initialized with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the NLI model."""
        if self._pipeline is None:
            logger.info(f"Loading NLI model: {self.model_name}")
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("NLI model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                raise
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label to lowercase standard form."""
        label_lower = label.lower()
        if 'entail' in label_lower:
            return 'entailment'
        elif 'contradict' in label_lower:
            return 'contradiction'
        else:
            return 'neutral'
    
    def verify_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Verify a single premise-hypothesis pair.
        
        Args:
            premise: The source document (what we know is true)
            hypothesis: The claim to verify (the generated answer)
            
        Returns:
            NLIResult with classification and scores
        """
        self._load_model()
        
        # Use zero-shot classification with NLI labels
        candidate_labels = ['entailment', 'neutral', 'contradiction']
        
        try:
            # Format for NLI: premise + hypothesis
            text = f"{premise}"
            result = self._pipeline(
                text,
                candidate_labels,
                hypothesis=hypothesis,
                multi_label=False
            )
            
            # Extract scores
            scores = {
                label: score 
                for label, score in zip(result['labels'], result['scores'])
            }
            
            # Determine winning label
            winning_label = result['labels'][0]
            
            return NLIResult(
                premise=premise[:200] + "..." if len(premise) > 200 else premise,
                hypothesis=hypothesis[:200] + "..." if len(hypothesis) > 200 else hypothesis,
                label=self._normalize_label(winning_label),
                scores=scores
            )
            
        except Exception as e:
            logger.error(f"Error in NLI verification: {e}")
            # Return neutral result on error
            return NLIResult(
                premise=premise[:200],
                hypothesis=hypothesis[:200],
                label='neutral',
                scores={'entailment': 0.33, 'contradiction': 0.33, 'neutral': 0.34}
            )
    
    def verify_answer(
        self,
        answer: str,
        documents: List[str],
        aggregate_method: str = "mean"
    ) -> VerificationResult:
        """
        Verify an answer against multiple retrieved documents.
        
        Args:
            answer: The generated answer to verify
            documents: List of retrieved documents
            aggregate_method: How to aggregate scores ('mean', 'max', 'voting')
            
        Returns:
            VerificationResult with aggregated metrics
        """
        if not documents:
            return VerificationResult(
                answer=answer,
                documents=[],
                individual_results=[],
                avg_entailment=0.0,
                avg_contradiction=0.0,
                max_contradiction=0.0,
                support_ratio=0.0,
                is_supported=False,
                confidence=0.0,
                explanation="No documents provided for verification"
            )
        
        # Verify against each document
        individual_results = []
        for doc in documents:
            if doc.strip():  # Skip empty documents
                result = self.verify_pair(premise=doc, hypothesis=answer)
                individual_results.append(result)
        
        if not individual_results:
            return VerificationResult(
                answer=answer,
                documents=documents,
                individual_results=[],
                avg_entailment=0.0,
                avg_contradiction=0.0,
                max_contradiction=0.0,
                support_ratio=0.0,
                is_supported=False,
                confidence=0.0,
                explanation="No valid documents for verification"
            )
        
        # Calculate aggregated metrics
        entailment_scores = [r.entailment_score for r in individual_results]
        contradiction_scores = [r.contradiction_score for r in individual_results]
        
        avg_entailment = sum(entailment_scores) / len(entailment_scores)
        avg_contradiction = sum(contradiction_scores) / len(contradiction_scores)
        max_contradiction = max(contradiction_scores)
        
        # Count supporting documents
        supporting_docs = sum(
            1 for r in individual_results 
            if r.label == 'entailment' or r.entailment_score > self.support_threshold
        )
        support_ratio = supporting_docs / len(individual_results)
        
        # Determine overall verdict
        is_supported = (
            avg_entailment > self.support_threshold and 
            max_contradiction < self.contradiction_threshold
        )
        
        # Calculate confidence
        confidence = avg_entailment * (1 - max_contradiction)
        
        # Generate explanation
        explanation = self._generate_explanation(
            individual_results, avg_entailment, max_contradiction, support_ratio
        )
        
        return VerificationResult(
            answer=answer,
            documents=documents,
            individual_results=individual_results,
            avg_entailment=avg_entailment,
            avg_contradiction=avg_contradiction,
            max_contradiction=max_contradiction,
            support_ratio=support_ratio,
            is_supported=is_supported,
            confidence=confidence,
            explanation=explanation
        )
    
    def _generate_explanation(
        self,
        results: List[NLIResult],
        avg_entailment: float,
        max_contradiction: float,
        support_ratio: float
    ) -> str:
        """Generate human-readable explanation of verification results."""
        total_docs = len(results)
        supporting = sum(1 for r in results if r.label == 'entailment')
        contradicting = sum(1 for r in results if r.label == 'contradiction')
        neutral = total_docs - supporting - contradicting
        
        parts = []
        
        # Overall assessment
        if avg_entailment > 0.7:
            parts.append("The answer is well-supported by the retrieved documents.")
        elif avg_entailment > 0.5:
            parts.append("The answer is moderately supported by the retrieved documents.")
        elif avg_entailment > 0.3:
            parts.append("The answer has weak support from the retrieved documents.")
        else:
            parts.append("The answer has little to no support from the retrieved documents.")
        
        # Document breakdown
        parts.append(
            f"Of {total_docs} documents: {supporting} support, "
            f"{contradicting} contradict, {neutral} are neutral."
        )
        
        # Contradiction warning
        if max_contradiction > self.contradiction_threshold:
            parts.append(
                "WARNING: At least one document strongly contradicts the answer."
            )
        elif contradicting > 0:
            parts.append(
                "Note: Some documents show potential contradictions."
            )
        
        return " ".join(parts)
    
    def get_factuality_score(self, verification_result: VerificationResult) -> float:
        """
        Calculate a single factuality score from verification results.
        
        This score is used as input to the Trust Index.
        
        Returns:
            Float between 0 and 1, where 1 = fully factual
        """
        if not verification_result.individual_results:
            return 0.0
        
        # Combine entailment support and lack of contradiction
        entailment_factor = verification_result.avg_entailment
        contradiction_penalty = verification_result.max_contradiction
        support_factor = verification_result.support_ratio
        
        # Weighted combination
        # High entailment + high support ratio + low contradiction = high factuality
        factuality = (
            0.4 * entailment_factor +
            0.3 * support_factor +
            0.3 * (1 - contradiction_penalty)
        )
        
        return min(max(factuality, 0.0), 1.0)  # Clamp to [0, 1]
