"""
Tests for the NLI Verifier.

Tests marked with @pytest.mark.slow require loading the real
facebook/bart-large-mnli model (~1.6 GB). Run them explicitly:
    pytest -m slow
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation_agent.nli_verifier import (
    NLIVerifier,
    NLIResult,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Unit tests (no model needed — test logic & data structures)
# ---------------------------------------------------------------------------

class TestNLIResultProperties:
    """Tests for the NLIResult dataclass."""

    def test_entailment_score(self):
        r = NLIResult(
            premise="p", hypothesis="h", label="entailment",
            scores={"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05},
        )
        assert r.entailment_score == pytest.approx(0.9)

    def test_contradiction_score(self):
        r = NLIResult(
            premise="p", hypothesis="h", label="contradiction",
            scores={"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05},
        )
        assert r.contradiction_score == pytest.approx(0.9)

    def test_neutral_score(self):
        r = NLIResult(
            premise="p", hypothesis="h", label="neutral",
            scores={"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8},
        )
        assert r.neutral_score == pytest.approx(0.8)

    def test_missing_key_returns_zero(self):
        r = NLIResult(premise="p", hypothesis="h", label="neutral", scores={})
        assert r.entailment_score == 0.0
        assert r.contradiction_score == 0.0


class TestVerificationResultStructure:
    """Tests for the VerificationResult dataclass."""

    def test_empty_documents(self):
        verifier = NLIVerifier()
        result = verifier.verify_answer("Some answer", [])
        assert result.is_supported is False
        assert result.support_ratio == 0.0
        assert result.avg_entailment == 0.0

    def test_blank_documents_skipped(self):
        verifier = NLIVerifier()
        # All blank docs → no valid results → inconclusive baseline
        result = verifier.verify_answer("Answer", ["", "   ", ""])
        # With no valid individual results, support_ratio defaults to 0.0
        assert result.support_ratio == 0.0


class TestFactualityScore:
    """Tests for get_factuality_score."""

    def test_perfect_entailment(self):
        verifier = NLIVerifier()
        # Manually build a VerificationResult with a relevant, supporting doc
        vr = VerificationResult(
            answer="answer",
            documents=["doc"],
            individual_results=[
                NLIResult("doc", "answer", "entailment",
                          {"entailment": 1.0, "contradiction": 0.0, "neutral": 0.0})
            ],
            avg_entailment=1.0,
            avg_contradiction=0.0,
            max_contradiction=0.0,
            support_ratio=1.0,
            is_supported=True,
            confidence=1.0,
            explanation="",
        )
        score = verifier.get_factuality_score(vr)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_full_contradiction_no_entailment(self):
        """Doc with pure contradiction (ent=0) is treated as inconclusive.

        BART-MNLI gives high contradiction for both genuine same-topic
        contradictions AND completely unrelated text. With ent=0 we cannot
        tell which case this is, so factuality returns the 0.5 baseline.
        Actual contradictions are caught by the consistency score instead.
        """
        verifier = NLIVerifier()
        vr = VerificationResult(
            answer="answer",
            documents=["doc"],
            individual_results=[
                NLIResult("doc", "answer", "contradiction",
                          {"entailment": 0.0, "contradiction": 1.0, "neutral": 0.0})
            ],
            avg_entailment=0.0,
            avg_contradiction=1.0,
            max_contradiction=1.0,
            support_ratio=0.0,
            is_supported=False,
            confidence=0.0,
            explanation="",
        )
        score = verifier.get_factuality_score(vr)
        # No entailing doc found, so returns 0.5 (inconclusive)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_contradiction_with_some_entailment(self):
        """Doc that contradicts but has some entailment signal scores low."""
        verifier = NLIVerifier()
        vr = VerificationResult(
            answer="answer",
            documents=["doc"],
            individual_results=[
                # A doc that mostly contradicts but has slight entailment
                # (this happens with genuine same-topic contradiction)
                NLIResult("doc", "answer", "contradiction",
                          {"entailment": 0.45, "contradiction": 0.50, "neutral": 0.05})
            ],
            avg_entailment=0.45,
            avg_contradiction=0.50,
            max_contradiction=0.50,
            support_ratio=0.0,
            is_supported=False,
            confidence=0.0,
            explanation="",
        )
        score = verifier.get_factuality_score(vr)
        # entailment > 0.4, so doc is counted. entailment < 0.5 so not supporting.
        # factuality = 0.4*0.45 + 0.3*0.0 + 0.3*(1-0.50) = 0.18+0+0.15 = 0.33
        assert score < 0.5

    def test_empty_results(self):
        verifier = NLIVerifier()
        vr = VerificationResult(
            answer="answer", documents=[], individual_results=[],
            avg_entailment=0.0, avg_contradiction=0.0, max_contradiction=0.0,
            support_ratio=0.0, is_supported=False, confidence=0.0,
            explanation="",
        )
        assert verifier.get_factuality_score(vr) == 0.0

    def test_all_neutral_gives_moderate_score(self):
        """When all docs are irrelevant (neutral), factuality should be 0.5, not 0.0."""
        verifier = NLIVerifier()
        # Simulate 3 docs about completely different topics → all neutral
        neutral_results = [
            NLIResult("doc1", "answer", "neutral",
                      {"entailment": 0.2, "contradiction": 0.1, "neutral": 0.7}),
            NLIResult("doc2", "answer", "neutral",
                      {"entailment": 0.15, "contradiction": 0.15, "neutral": 0.7}),
            NLIResult("doc3", "answer", "neutral",
                      {"entailment": 0.25, "contradiction": 0.05, "neutral": 0.7}),
        ]
        vr = VerificationResult(
            answer="answer", documents=["d1", "d2", "d3"],
            individual_results=neutral_results,
            avg_entailment=0.2, avg_contradiction=0.1, max_contradiction=0.15,
            support_ratio=0.5, is_supported=False, confidence=0.0,
            explanation="",
        )
        score = verifier.get_factuality_score(vr)
        # No doc is relevant → should return 0.5 (neutral baseline)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_mixed_relevant_and_irrelevant_docs(self):
        """1 supporting doc + 4 neutral docs should give a high factuality score."""
        verifier = NLIVerifier()
        results = [
            # Doc about the right topic → entails the answer
            NLIResult("doc1", "answer", "entailment",
                      {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10}),
            # 4 docs about different topics → neutral
            NLIResult("doc2", "answer", "neutral",
                      {"entailment": 0.15, "contradiction": 0.1, "neutral": 0.75}),
            NLIResult("doc3", "answer", "neutral",
                      {"entailment": 0.2, "contradiction": 0.1, "neutral": 0.7}),
            NLIResult("doc4", "answer", "neutral",
                      {"entailment": 0.1, "contradiction": 0.15, "neutral": 0.75}),
            NLIResult("doc5", "answer", "neutral",
                      {"entailment": 0.2, "contradiction": 0.05, "neutral": 0.75}),
        ]
        vr = VerificationResult(
            answer="answer", documents=["d1", "d2", "d3", "d4", "d5"],
            individual_results=results,
            avg_entailment=0.3, avg_contradiction=0.09, max_contradiction=0.15,
            support_ratio=0.5, is_supported=False, confidence=0.0,
            explanation="",
        )
        score = verifier.get_factuality_score(vr)
        # Only doc1 is relevant (entailment=0.85 > 0.4), and it supports
        # So: avg_entailment=0.85, max_contradiction=0.05, support_ratio=1.0
        # factuality = 0.4*0.85 + 0.3*1.0 + 0.3*(1-0.05) = 0.34+0.30+0.285 = 0.925
        assert score > 0.7


# ---------------------------------------------------------------------------
# Integration tests (load real model — slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestNLIVerifierWithModel:
    """Integration tests that load the real NLI model."""

    @pytest.fixture(autouse=True, scope="class")
    def verifier(self):
        """Load model once per test class."""
        TestNLIVerifierWithModel._verifier = NLIVerifier(
            model_name="facebook/bart-large-mnli", device="cpu"
        )

    def _v(self):
        return TestNLIVerifierWithModel._verifier

    def test_entailment_pair(self):
        result = self._v().verify_pair(
            premise="Paris is the capital of France.",
            hypothesis="The capital of France is Paris.",
        )
        assert result.label == "entailment"
        assert result.entailment_score > 0.5

    def test_contradiction_pair(self):
        result = self._v().verify_pair(
            premise="Paris is the capital of France.",
            hypothesis="Berlin is the capital of France.",
        )
        assert result.contradiction_score > 0.3

    def test_neutral_pair(self):
        result = self._v().verify_pair(
            premise="Paris is the capital of France.",
            hypothesis="The weather in Tokyo is humid.",
        )
        assert result.neutral_score > 0.3

    def test_verify_answer_supported(self):
        docs = [
            "Paris is the capital of France.",
            "France's capital city is Paris.",
        ]
        result = self._v().verify_answer(
            "The capital of France is Paris.", docs
        )
        assert result.avg_entailment > 0.4
        assert result.support_ratio > 0.5

    def test_verify_answer_contradicted(self):
        docs = [
            "Berlin is the capital of Germany.",
            "Tokyo is the capital of Japan.",
        ]
        result = self._v().verify_answer(
            "Paris is the capital of Germany.", docs
        )
        assert result.avg_entailment < 0.5

    def test_document_pair_consistency(self):
        result = self._v().verify_document_pair(
            "Paris is the capital of France.",
            "The capital of France is Paris.",
        )
        assert result.entailment_score > 0.3

    def test_document_pair_contradiction(self):
        result = self._v().verify_document_pair(
            "Paris is the capital of France.",
            "Berlin is the capital of France.",
        )
        assert result.contradiction_score > 0.3

    def test_clean_docs_high_factuality(self):
        """Clean docs + correct answer should produce high factuality."""
        docs = [
            "Paris is the capital of France. It is known for the Eiffel Tower.",
            "France is a country in Western Europe. Its capital is Paris.",
        ]
        vr = self._v().verify_answer("The capital of France is Paris.", docs)
        score = self._v().get_factuality_score(vr)
        assert score > 0.5
