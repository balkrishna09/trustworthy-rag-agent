"""
Trust Index Calculator Module
Combines multiple evaluation signals into a single trustworthiness score.

The Trust Index is a composite score that considers:
1. Factuality - Does the answer align with retrieved documents?
2. Consistency - Do the retrieved documents agree with each other?
3. Poisoning Risk - Are any documents potentially adversarial?

Formula:
Trust Index = α × Factuality + β × Consistency + γ × (1 - PoisonProbability)

Where α + β + γ = 1.0
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Categorical trust levels based on Trust Index score."""
    HIGH = "high"           # > 0.8
    MEDIUM = "medium"       # 0.5 - 0.8
    LOW = "low"             # 0.3 - 0.5
    VERY_LOW = "very_low"   # < 0.3
    
    @classmethod
    def from_score(cls, score: float) -> "TrustLevel":
        """Convert numeric score to trust level."""
        if score > 0.8:
            return cls.HIGH
        elif score > 0.5:
            return cls.MEDIUM
        elif score > 0.3:
            return cls.LOW
        else:
            return cls.VERY_LOW


@dataclass
class TrustComponents:
    """Individual components that make up the Trust Index."""
    factuality_score: float  # How well answer matches documents (0-1)
    consistency_score: float  # How consistent documents are (0-1)
    poison_score: float  # Probability of poisoning (0-1, higher = worse)
    
    # Optional additional signals
    retrieval_confidence: float = 1.0  # Confidence in retrieval quality
    source_credibility: float = 1.0    # Credibility of document sources
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'factuality': self.factuality_score,
            'consistency': self.consistency_score,
            'poison_risk': self.poison_score,
            'retrieval_confidence': self.retrieval_confidence,
            'source_credibility': self.source_credibility
        }


@dataclass
class TrustIndexResult:
    """Complete Trust Index calculation result."""
    trust_score: float  # The final Trust Index (0-1)
    trust_level: TrustLevel
    components: TrustComponents
    
    # Breakdown of weighted contributions
    factuality_contribution: float
    consistency_contribution: float
    poison_contribution: float
    
    # Weights used
    weights: Dict[str, float]
    
    # Recommendations
    is_trustworthy: bool  # Above threshold?
    warnings: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            'trust_score': self.trust_score,
            'trust_level': self.trust_level.value,
            'is_trustworthy': self.is_trustworthy,
            'components': self.components.to_dict(),
            'contributions': {
                'factuality': self.factuality_contribution,
                'consistency': self.consistency_contribution,
                'poison_safety': self.poison_contribution
            },
            'weights': self.weights,
            'warnings': self.warnings,
            'recommendation': self.recommendation
        }


class TrustIndexCalculator:
    """
    Calculates the Trust Index - a composite reliability score.
    
    How it works:
    1. Takes scores from NLI Verifier (factuality)
    2. Takes scores from document consistency analysis
    3. Takes scores from Poison Detector
    4. Combines them using configurable weights
    5. Returns overall trust score with breakdown
    
    The Trust Index helps users understand how much to trust
    a RAG system's response.
    """
    
    def __init__(
        self,
        alpha: float = 0.4,    # Factuality weight
        beta: float = 0.35,    # Consistency weight
        gamma: float = 0.25,   # Poison weight (inverted)
        trust_threshold: float = 0.5
    ):
        """
        Initialize the Trust Index Calculator.
        
        Args:
            alpha: Weight for factuality score (default 0.4)
            beta: Weight for consistency score (default 0.35)
            gamma: Weight for poison safety score (default 0.25)
            trust_threshold: Minimum score to be considered trustworthy
        """
        # Validate weights sum to 1.0
        total = alpha + beta + gamma
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {total}, not 1.0. Normalizing..."
            )
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.trust_threshold = trust_threshold
        
        logger.info(
            f"TrustIndexCalculator initialized with weights: "
            f"α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}"
        )
    
    def calculate(
        self,
        factuality_score: float,
        consistency_score: float,
        poison_score: float,
        retrieval_confidence: float = 1.0,
        source_credibility: float = 1.0
    ) -> TrustIndexResult:
        """
        Calculate the Trust Index from component scores.
        
        Args:
            factuality_score: How well answer matches documents (0-1)
            consistency_score: How consistent documents are with each other (0-1)
            poison_score: Probability of poisoning (0-1, higher = worse)
            retrieval_confidence: Optional confidence in retrieval quality
            source_credibility: Optional source credibility score
            
        Returns:
            TrustIndexResult with complete breakdown
        """
        # Ensure scores are in valid range
        factuality_score = max(0.0, min(1.0, factuality_score))
        consistency_score = max(0.0, min(1.0, consistency_score))
        poison_score = max(0.0, min(1.0, poison_score))
        
        # Calculate weighted contributions
        factuality_contribution = self.alpha * factuality_score
        consistency_contribution = self.beta * consistency_score
        poison_contribution = self.gamma * (1.0 - poison_score)  # Invert poison score
        
        # Calculate base trust score
        trust_score = (
            factuality_contribution +
            consistency_contribution +
            poison_contribution
        )
        
        # Apply optional modifiers
        if retrieval_confidence < 1.0:
            trust_score *= (0.5 + 0.5 * retrieval_confidence)
        if source_credibility < 1.0:
            trust_score *= (0.5 + 0.5 * source_credibility)
        
        # Ensure final score is in range
        trust_score = max(0.0, min(1.0, trust_score))
        
        # Determine trust level and warnings
        trust_level = TrustLevel.from_score(trust_score)
        warnings = self._generate_warnings(
            factuality_score, consistency_score, poison_score
        )
        recommendation = self._generate_recommendation(
            trust_score, trust_level, warnings
        )
        
        components = TrustComponents(
            factuality_score=factuality_score,
            consistency_score=consistency_score,
            poison_score=poison_score,
            retrieval_confidence=retrieval_confidence,
            source_credibility=source_credibility
        )
        
        return TrustIndexResult(
            trust_score=trust_score,
            trust_level=trust_level,
            components=components,
            factuality_contribution=factuality_contribution,
            consistency_contribution=consistency_contribution,
            poison_contribution=poison_contribution,
            weights={
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            },
            is_trustworthy=trust_score >= self.trust_threshold,
            warnings=warnings,
            recommendation=recommendation
        )
    
    def _generate_warnings(
        self,
        factuality: float,
        consistency: float,
        poison: float
    ) -> List[str]:
        """Generate warnings based on component scores."""
        warnings = []
        
        # Factuality warnings
        if factuality < 0.3:
            warnings.append(
                "CRITICAL: Answer poorly supported by retrieved documents"
            )
        elif factuality < 0.5:
            warnings.append(
                "WARNING: Answer has weak support from documents"
            )
        
        # Consistency warnings
        if consistency < 0.3:
            warnings.append(
                "CRITICAL: Retrieved documents are highly inconsistent"
            )
        elif consistency < 0.5:
            warnings.append(
                "WARNING: Some inconsistency detected among documents"
            )
        
        # Poison warnings
        if poison > 0.7:
            warnings.append(
                "CRITICAL: High probability of poisoned/adversarial content"
            )
        elif poison > 0.5:
            warnings.append(
                "WARNING: Some documents show suspicious patterns"
            )
        
        return warnings
    
    def _generate_recommendation(
        self,
        score: float,
        level: TrustLevel,
        warnings: List[str]
    ) -> str:
        """Generate a human-readable recommendation."""
        if level == TrustLevel.HIGH:
            return (
                "This response appears trustworthy. The answer is well-supported "
                "by consistent, clean documents."
            )
        elif level == TrustLevel.MEDIUM:
            if warnings:
                return (
                    "This response is moderately trustworthy but has some concerns. "
                    "Consider verifying key claims from additional sources."
                )
            else:
                return (
                    "This response is moderately trustworthy. "
                    "The evidence is reasonable but not conclusive."
                )
        elif level == TrustLevel.LOW:
            return (
                "This response has low trustworthiness. "
                "Key claims may not be well-supported or documents may be inconsistent. "
                "Verify information independently before relying on it."
            )
        else:  # VERY_LOW
            return (
                "WARNING: This response is NOT trustworthy. "
                "Do not rely on this information. "
                "Retrieved documents may be poisoned or answer contradicts sources."
            )
    
    def calculate_from_results(
        self,
        nli_verification_result,
        poison_detection_result,
        retrieval_scores: Optional[List[float]] = None
    ) -> TrustIndexResult:
        """
        Calculate Trust Index directly from evaluation module results.
        
        This is a convenience method that extracts scores from the
        NLI Verifier and Poison Detector results.
        
        Args:
            nli_verification_result: Result from NLIVerifier.verify_answer()
            poison_detection_result: Result from PoisonDetector.detect_batch()
            retrieval_scores: Optional list of retrieval similarity scores
            
        Returns:
            TrustIndexResult
        """
        # Extract factuality from NLI verification
        factuality_score = nli_verification_result.avg_entailment
        
        # Use support ratio as consistency proxy
        consistency_score = nli_verification_result.support_ratio
        
        # Extract poison probability
        poison_score = poison_detection_result.overall_poison_probability
        
        # Calculate retrieval confidence from retrieval scores
        retrieval_confidence = 1.0
        if retrieval_scores:
            avg_score = sum(retrieval_scores) / len(retrieval_scores)
            retrieval_confidence = min(avg_score, 1.0)
        
        return self.calculate(
            factuality_score=factuality_score,
            consistency_score=consistency_score,
            poison_score=poison_score,
            retrieval_confidence=retrieval_confidence
        )
    
    def explain_score(self, result: TrustIndexResult) -> str:
        """Generate a detailed explanation of the Trust Index score."""
        lines = [
            f"=== Trust Index Analysis ===",
            f"",
            f"Overall Score: {result.trust_score:.2f} ({result.trust_level.value.upper()})",
            f"Trustworthy: {'Yes' if result.is_trustworthy else 'No'}",
            f"",
            f"--- Component Breakdown ---",
            f"Factuality:  {result.components.factuality_score:.2f} "
            f"(weight: {self.alpha:.0%}) -> contributes {result.factuality_contribution:.3f}",
            f"Consistency: {result.components.consistency_score:.2f} "
            f"(weight: {self.beta:.0%}) -> contributes {result.consistency_contribution:.3f}",
            f"Poison Risk: {result.components.poison_score:.2f} "
            f"(weight: {self.gamma:.0%}) -> safety contributes {result.poison_contribution:.3f}",
            f"",
        ]
        
        if result.warnings:
            lines.append("--- Warnings ---")
            for warning in result.warnings:
                lines.append(f"  ! {warning}")
            lines.append("")
        
        lines.append("--- Recommendation ---")
        lines.append(result.recommendation)
        
        return "\n".join(lines)
