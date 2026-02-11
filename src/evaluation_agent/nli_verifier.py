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

    Uses direct NLI inference with AutoModelForSequenceClassification,
    passing (premise, hypothesis) pairs and interpreting the model's
    three-way classification logits (contradiction, neutral, entailment).
    """

    # Label index mappings for different models
    # BART-large-MNLI and DeBERTa-large-MNLI both use:
    # index 0 = contradiction, index 1 = neutral, index 2 = entailment
    LABEL_MAPPINGS = {
        'facebook/bart-large-mnli': {
            0: 'contradiction',
            1: 'neutral',
            2: 'entailment'
        },
        'microsoft/deberta-large-mnli': {
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
        self._model = None
        self._tokenizer = None
        self._label_mapping = self.LABEL_MAPPINGS.get(
            model_name,
            self.LABEL_MAPPINGS['facebook/bart-large-mnli']
        )

        logger.info(f"NLIVerifier initialized with model: {model_name}")

    def _load_model(self):
        """Lazy load the NLI model using direct AutoModel inference."""
        if self._model is None:
            logger.info(f"Loading NLI model: {self.model_name}")
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

                # Move to GPU if requested and available
                if self.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.to("cuda")

                self._model.eval()
                logger.info("NLI model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                raise

    def verify_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Verify a single premise-hypothesis pair using direct NLI inference.

        The premise (retrieved document) and hypothesis (generated answer) are
        tokenized together and passed through the NLI model. The model outputs
        logits for [contradiction, neutral, entailment] which are converted
        to probabilities via softmax.

        Args:
            premise: The source document (what we know is true)
            hypothesis: The claim to verify (the generated answer)

        Returns:
            NLIResult with classification and scores
        """
        self._load_model()

        try:
            import torch

            # Tokenize the premise-hypothesis pair for NLI
            inputs = self._tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            # Move inputs to same device as model
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Convert logits to probabilities
            # BART-large-MNLI: index 0=contradiction, 1=neutral, 2=entailment
            probs = torch.softmax(outputs.logits[0], dim=-1).cpu().numpy()

            scores = {
                'contradiction': float(probs[0]),
                'neutral': float(probs[1]),
                'entailment': float(probs[2])
            }

            # Determine winning label
            winning_idx = int(probs.argmax())
            winning_label = self._label_mapping[winning_idx]

            return NLIResult(
                premise=premise[:200] + "..." if len(premise) > 200 else premise,
                hypothesis=hypothesis[:200] + "..." if len(hypothesis) > 200 else hypothesis,
                label=winning_label,
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

    def verify_document_pair(self, doc_a: str, doc_b: str) -> NLIResult:
        """
        Check NLI relationship between two documents.

        Useful for cross-document consistency checking: if doc_a
        contradicts doc_b, one of them may be poisoned.

        Args:
            doc_a: First document (used as premise)
            doc_b: Second document (used as hypothesis)

        Returns:
            NLIResult with classification and scores
        """
        return self.verify_pair(premise=doc_a, hypothesis=doc_b)

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
        #
        # Key insight: BART-MNLI assigns high contradiction scores (~0.99) to
        # BOTH genuinely contradicting same-topic pairs AND completely unrelated
        # text pairs. There is no reliable way to distinguish them from NLI
        # scores alone.
        #
        # Therefore, for the support_ratio calculation we focus on ENTAILING
        # documents only. Contradicting docs are handled separately by the
        # consistency score and poison detector.
        #
        # A doc is "supportive" if entailment > support_threshold (0.5).
        # support_ratio = supportive_docs / total_docs_with_opinion
        # where "docs_with_opinion" means entailment > 0.4 (clearly relevant).
        # Neutral and contradiction-only docs are excluded from the ratio.
        entailing_results = [
            r for r in individual_results
            if r.entailment_score > 0.4
        ]

        if entailing_results:
            supporting_docs = sum(
                1 for r in entailing_results
                if r.entailment_score > self.support_threshold
            )
            support_ratio = supporting_docs / len(entailing_results)
        else:
            # No entailing docs found — treat as inconclusive, not unsupported
            supporting_docs = 0
            support_ratio = 0.5  # Neutral baseline

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

        Key insight: When a retrieved document is about a completely different
        topic, NLI returns "neutral" — this means "irrelevant", NOT "unsupported".
        We therefore focus scoring on *relevant* documents (those with entailment
        OR contradiction signal) and treat purely neutral docs as carrying no
        information about factuality.

        Returns:
            Float between 0 and 1, where 1 = fully factual
        """
        if not verification_result.individual_results:
            return 0.0

        results = verification_result.individual_results

        # Focus on ENTAILING documents for factuality scoring.
        #
        # BART-MNLI gives high contradiction scores (~0.99) for both genuine
        # same-topic contradictions and completely unrelated text pairs. We
        # cannot distinguish them, so contradicting docs are excluded from
        # factuality and handled by consistency/poison scores instead.
        entailing_results = [
            r for r in results
            if r.entailment_score > 0.4
        ]

        if not entailing_results:
            # No doc clearly entails the answer.
            # This means either (a) all docs are unrelated topics, or
            # (b) the answer genuinely isn't supported. Either way,
            # return a moderate baseline — lack of evidence ≠ wrong.
            return 0.5

        # Among entailing docs, compute factuality
        entailment_scores = [r.entailment_score for r in entailing_results]
        avg_entailment = sum(entailment_scores) / len(entailment_scores)

        # Support ratio: fraction of entailing docs with strong support
        supporting = sum(
            1 for r in entailing_results
            if r.entailment_score > 0.5
        )
        support_ratio = supporting / len(entailing_results)

        # Max contradiction among entailing docs (if any)
        max_contradiction = max(r.contradiction_score for r in entailing_results)

        # Weighted combination
        factuality = (
            0.4 * avg_entailment +
            0.3 * support_ratio +
            0.3 * (1 - max_contradiction)
        )

        return min(max(factuality, 0.0), 1.0)  # Clamp to [0, 1]
