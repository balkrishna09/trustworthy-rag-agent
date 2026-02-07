"""
Poison Detector Module
Detects adversarial or poisoned content in retrieved documents.

Knowledge poisoning attacks work by injecting false information into a RAG system's
knowledge base. This module aims to detect such poisoned documents using:
1. Semantic anomaly detection (outliers compared to other documents)
2. Linguistic pattern analysis (suspicious phrases, contradictions)
3. Source credibility signals (if available)
4. Cross-document consistency checking
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoisonSignal:
    """Individual signal indicating potential poisoning."""
    signal_type: str  # 'semantic', 'linguistic', 'consistency', 'credibility'
    severity: float  # 0.0 to 1.0
    description: str
    evidence: str  # The text that triggered this signal


@dataclass 
class DocumentPoisonResult:
    """Poison detection result for a single document."""
    document: str
    document_index: int
    is_suspicious: bool
    poison_probability: float  # 0.0 to 1.0
    signals: List[PoisonSignal]
    explanation: str


@dataclass
class PoisonDetectionResult:
    """Aggregated poison detection result for a set of documents."""
    documents: List[str]
    individual_results: List[DocumentPoisonResult]
    
    # Aggregated metrics
    overall_poison_probability: float
    num_suspicious_docs: int
    high_risk_indices: List[int]  # Indices of highly suspicious documents
    
    # Overall assessment
    is_contaminated: bool
    confidence: float
    explanation: str


class PoisonDetector:
    """
    Detects potential knowledge poisoning in retrieved documents.
    
    How it works:
    1. Analyzes each document for suspicious linguistic patterns
    2. Checks semantic consistency across documents
    3. Identifies outliers that don't fit with other documents
    4. Combines signals to estimate poisoning probability
    
    This is a heuristic-based detector. In production, you might train
    a classifier on examples of poisoned vs. clean documents.
    """
    
    # Suspicious phrase patterns that might indicate poisoning
    SUSPICIOUS_PATTERNS = [
        # Contradiction indicators
        (r'\bactually\b.*\bnot\b', 0.3, "Contradiction pattern: 'actually...not'"),
        (r'\bcontrary to\b', 0.3, "Contradiction indicator"),
        (r'\bdespite what.*(?:say|claim|believe)', 0.4, "Counter-claim pattern"),
        (r'\bignore\b.*\bprevious\b', 0.5, "Instruction to ignore context"),
        
        # Instruction injection
        (r'\bforget\b.*\b(?:everything|all|above)\b', 0.7, "Instruction injection attempt"),
        (r'\bignore\b.*\b(?:instructions?|above|previous)\b', 0.7, "Instruction override attempt"),
        (r'\binstead\b.*\b(?:say|respond|answer)\b', 0.6, "Response manipulation attempt"),
        (r'\byou must\b.*\b(?:say|respond|believe)\b', 0.5, "Forced response pattern"),
        
        # Claim manipulation
        (r'\bthe (?:real|actual|true) (?:answer|fact|truth)\b', 0.4, "Claim override pattern"),
        (r'\beveryone knows\b', 0.2, "Appeal to common knowledge"),
        (r'\bit is (?:a )?(?:well-known|established|proven) fact\b', 0.2, "Unverified fact claim"),
        
        # Urgency/pressure tactics
        (r'\bIMPORTANT:\s', 0.3, "Urgency marker"),
        (r'\bNOTE:\s', 0.2, "Attention-grabbing marker"),
        (r'\bWARNING:\s', 0.2, "Warning marker in content"),
        
        # Data format anomalies
        (r'(?:system|user|assistant):\s', 0.6, "Chat format injection"),
        (r'</?(?:system|user|assistant|prompt)>', 0.7, "XML tag injection"),
    ]
    
    # Phrases that often appear in legitimate educational/factual content
    BENIGN_INDICATORS = [
        r'\baccording to\b',
        r'\bresearch (?:shows|suggests|indicates)\b',
        r'\bstudies (?:show|suggest|indicate)\b',
        r'\b(?:scientists|researchers|experts) (?:have found|believe)\b',
    ]
    
    def __init__(
        self,
        poison_threshold: float = 0.7,
        embedding_model = None,
        use_semantic_analysis: bool = True
    ):
        """
        Initialize the Poison Detector.
        
        Args:
            poison_threshold: Probability above which to flag as poisoned
            embedding_model: Optional embedding model for semantic analysis
            use_semantic_analysis: Whether to use embedding-based outlier detection
        """
        self.poison_threshold = poison_threshold
        self.embedding_model = embedding_model
        self.use_semantic_analysis = use_semantic_analysis and embedding_model is not None
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), severity, desc)
            for pattern, severity, desc in self.SUSPICIOUS_PATTERNS
        ]
        self.compiled_benign = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.BENIGN_INDICATORS
        ]
        
        logger.info(f"PoisonDetector initialized (threshold: {poison_threshold})")
    
    def _analyze_linguistic_patterns(self, document: str) -> List[PoisonSignal]:
        """Analyze document for suspicious linguistic patterns."""
        signals = []
        
        for pattern, severity, description in self.compiled_patterns:
            matches = pattern.findall(document)
            if matches:
                for match in matches[:3]:  # Limit to first 3 matches
                    signals.append(PoisonSignal(
                        signal_type='linguistic',
                        severity=severity,
                        description=description,
                        evidence=match if isinstance(match, str) else str(match)
                    ))
        
        # Check for benign indicators that reduce suspicion
        benign_count = sum(
            1 for pattern in self.compiled_benign 
            if pattern.search(document)
        )
        
        # If document has many benign indicators, reduce severity of signals
        if benign_count >= 2 and signals:
            for signal in signals:
                signal.severity *= 0.7  # Reduce severity by 30%
        
        return signals
    
    def _analyze_structural_anomalies(self, document: str) -> List[PoisonSignal]:
        """Check for structural anomalies in the document."""
        signals = []
        
        # Check for unusual character distributions
        if document:
            # High ratio of uppercase
            uppercase_ratio = sum(1 for c in document if c.isupper()) / len(document)
            if uppercase_ratio > 0.3:
                signals.append(PoisonSignal(
                    signal_type='structural',
                    severity=0.3,
                    description="Unusually high uppercase ratio",
                    evidence=f"{uppercase_ratio:.1%} uppercase characters"
                ))
            
            # Check for repeated phrases (copy-paste attacks)
            words = document.lower().split()
            if len(words) > 10:
                word_counts = Counter(words)
                max_repeat = max(word_counts.values())
                if max_repeat > len(words) * 0.15:  # Word appears >15% of time
                    most_common = word_counts.most_common(1)[0]
                    signals.append(PoisonSignal(
                        signal_type='structural',
                        severity=0.4,
                        description="Suspicious word repetition",
                        evidence=f"'{most_common[0]}' appears {most_common[1]} times"
                    ))
            
            # Check for very long "words" (possible encoding attacks)
            long_tokens = [w for w in words if len(w) > 30]
            if long_tokens:
                signals.append(PoisonSignal(
                    signal_type='structural',
                    severity=0.5,
                    description="Unusually long tokens detected",
                    evidence=f"{len(long_tokens)} tokens over 30 characters"
                ))
        
        return signals
    
    def _analyze_semantic_consistency(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None
    ) -> List[Tuple[int, float]]:
        """
        Analyze semantic consistency across documents.
        Returns list of (doc_index, outlier_score) for each document.
        """
        if not self.use_semantic_analysis or embeddings is None:
            return [(i, 0.0) for i in range(len(documents))]
        
        if len(documents) < 3:
            return [(i, 0.0) for i in range(len(documents))]
        
        try:
            # Calculate centroid of all document embeddings
            centroid = np.mean(embeddings, axis=0)
            
            # Calculate distance of each document from centroid
            distances = []
            for i, emb in enumerate(embeddings):
                dist = np.linalg.norm(emb - centroid)
                distances.append(dist)
            
            # Convert to outlier scores (0-1 range)
            if max(distances) > 0:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                
                outlier_scores = []
                for i, dist in enumerate(distances):
                    # Z-score based outlier detection
                    if std_dist > 0:
                        z_score = (dist - mean_dist) / std_dist
                        # Convert z-score to 0-1 probability
                        outlier_prob = min(max(z_score / 3, 0), 1)  # 3 std devs = 1.0
                    else:
                        outlier_prob = 0.0
                    outlier_scores.append((i, outlier_prob))
                
                return outlier_scores
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
        
        return [(i, 0.0) for i in range(len(documents))]
    
    def _check_cross_document_consistency(
        self,
        documents: List[str]
    ) -> List[PoisonSignal]:
        """
        Check for inconsistencies between documents.
        Documents that contradict each other might indicate poisoning.
        """
        signals = []
        
        if len(documents) < 2:
            return signals
        
        # Extract key claims/entities from documents (simplified)
        # In a real implementation, you'd use NER and relation extraction
        
        # Look for direct contradictions using simple heuristics
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], start=i+1):
                # Check for "X is Y" vs "X is not Y" patterns
                doc1_lower = doc1.lower()
                doc2_lower = doc2.lower()
                
                # Very simplified contradiction detection
                if (' is not ' in doc1_lower and ' is ' in doc2_lower) or \
                   (' is ' in doc1_lower and ' is not ' in doc2_lower):
                    # Check if they're talking about the same thing
                    words1 = set(doc1_lower.split())
                    words2 = set(doc2_lower.split())
                    overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                    
                    if overlap > 0.3:  # Significant word overlap
                        signals.append(PoisonSignal(
                            signal_type='consistency',
                            severity=0.4,
                            description=f"Potential contradiction between documents {i} and {j}",
                            evidence="Documents may contain conflicting claims"
                        ))
        
        return signals
    
    def detect_document(
        self,
        document: str,
        document_index: int = 0,
        context_documents: Optional[List[str]] = None
    ) -> DocumentPoisonResult:
        """
        Detect potential poisoning in a single document.
        
        Args:
            document: The document to analyze
            document_index: Index of this document in the collection
            context_documents: Other documents for cross-checking
            
        Returns:
            DocumentPoisonResult with analysis
        """
        signals = []
        
        # 1. Linguistic pattern analysis
        linguistic_signals = self._analyze_linguistic_patterns(document)
        signals.extend(linguistic_signals)
        
        # 2. Structural anomaly detection
        structural_signals = self._analyze_structural_anomalies(document)
        signals.extend(structural_signals)
        
        # Calculate poison probability
        if signals:
            # Combine signal severities (with diminishing returns)
            severities = sorted([s.severity for s in signals], reverse=True)
            combined_severity = severities[0]
            for i, sev in enumerate(severities[1:], start=1):
                combined_severity += sev * (0.5 ** i)  # Diminishing weight
            
            poison_probability = min(combined_severity, 1.0)
        else:
            poison_probability = 0.0
        
        is_suspicious = poison_probability > self.poison_threshold
        
        # Generate explanation
        if signals:
            top_signals = sorted(signals, key=lambda s: s.severity, reverse=True)[:3]
            explanations = [f"- {s.description}" for s in top_signals]
            explanation = "Suspicious patterns detected:\n" + "\n".join(explanations)
        else:
            explanation = "No suspicious patterns detected."
        
        return DocumentPoisonResult(
            document=document[:200] + "..." if len(document) > 200 else document,
            document_index=document_index,
            is_suspicious=is_suspicious,
            poison_probability=poison_probability,
            signals=signals,
            explanation=explanation
        )
    
    def detect_batch(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None
    ) -> PoisonDetectionResult:
        """
        Detect potential poisoning across a batch of documents.
        
        Args:
            documents: List of documents to analyze
            embeddings: Optional document embeddings for semantic analysis
            
        Returns:
            PoisonDetectionResult with aggregated analysis
        """
        if not documents:
            return PoisonDetectionResult(
                documents=[],
                individual_results=[],
                overall_poison_probability=0.0,
                num_suspicious_docs=0,
                high_risk_indices=[],
                is_contaminated=False,
                confidence=1.0,
                explanation="No documents provided for analysis."
            )
        
        # Analyze each document individually
        individual_results = []
        for i, doc in enumerate(documents):
            result = self.detect_document(doc, i, documents)
            individual_results.append(result)
        
        # Add semantic outlier analysis
        if embeddings is not None:
            outlier_scores = self._analyze_semantic_consistency(documents, embeddings)
            for (idx, outlier_score), result in zip(outlier_scores, individual_results):
                if outlier_score > 0.5:
                    result.signals.append(PoisonSignal(
                        signal_type='semantic',
                        severity=outlier_score * 0.6,  # Scale down
                        description="Semantic outlier compared to other documents",
                        evidence=f"Outlier score: {outlier_score:.2f}"
                    ))
                    # Recalculate poison probability
                    result.poison_probability = min(
                        result.poison_probability + outlier_score * 0.3,
                        1.0
                    )
                    result.is_suspicious = result.poison_probability > self.poison_threshold
        
        # Cross-document consistency check
        consistency_signals = self._check_cross_document_consistency(documents)
        if consistency_signals:
            # Add consistency signals to all documents (they apply globally)
            for result in individual_results:
                result.signals.extend(consistency_signals)
        
        # Calculate aggregated metrics
        poison_probs = [r.poison_probability for r in individual_results]
        suspicious_docs = [r for r in individual_results if r.is_suspicious]
        high_risk_indices = [
            r.document_index for r in individual_results 
            if r.poison_probability > 0.8
        ]
        
        # Overall poison probability (max of individual + small boost for multiple)
        overall_probability = max(poison_probs) if poison_probs else 0.0
        if len(suspicious_docs) > 1:
            overall_probability = min(
                overall_probability + 0.1 * (len(suspicious_docs) - 1),
                1.0
            )
        
        is_contaminated = overall_probability > self.poison_threshold
        confidence = 1.0 - (0.1 * len([p for p in poison_probs if 0.3 < p < 0.7]))
        
        # Generate explanation
        if is_contaminated:
            explanation = (
                f"POTENTIAL POISONING DETECTED. "
                f"{len(suspicious_docs)} of {len(documents)} documents flagged. "
                f"High-risk document indices: {high_risk_indices or 'None'}"
            )
        elif suspicious_docs:
            explanation = (
                f"Some documents show suspicious patterns. "
                f"{len(suspicious_docs)} documents warrant further review."
            )
        else:
            explanation = "No significant poisoning indicators detected."
        
        return PoisonDetectionResult(
            documents=documents,
            individual_results=individual_results,
            overall_poison_probability=overall_probability,
            num_suspicious_docs=len(suspicious_docs),
            high_risk_indices=high_risk_indices,
            is_contaminated=is_contaminated,
            confidence=confidence,
            explanation=explanation
        )
    
    def get_poison_score(self, detection_result: PoisonDetectionResult) -> float:
        """
        Get a single poison score from detection results.
        
        This score is used as input to the Trust Index.
        Higher score = more likely poisoned = LESS trustworthy
        
        Returns:
            Float between 0 and 1, where 0 = clean, 1 = poisoned
        """
        return detection_result.overall_poison_probability
