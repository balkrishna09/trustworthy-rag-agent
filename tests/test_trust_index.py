"""
Tests for the Trust Index Calculator.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation_agent.trust_index import (
    TrustIndexCalculator,
    TrustLevel,
    TrustComponents,
    TrustIndexResult,
)


class TestTrustLevel:
    """Tests for TrustLevel enum classification."""

    def test_high_trust(self):
        assert TrustLevel.from_score(0.85) == TrustLevel.HIGH
        assert TrustLevel.from_score(0.99) == TrustLevel.HIGH

    def test_medium_trust(self):
        assert TrustLevel.from_score(0.6) == TrustLevel.MEDIUM
        assert TrustLevel.from_score(0.79) == TrustLevel.MEDIUM

    def test_low_trust(self):
        assert TrustLevel.from_score(0.35) == TrustLevel.LOW
        assert TrustLevel.from_score(0.49) == TrustLevel.LOW

    def test_very_low_trust(self):
        assert TrustLevel.from_score(0.1) == TrustLevel.VERY_LOW
        assert TrustLevel.from_score(0.0) == TrustLevel.VERY_LOW

    def test_boundary_values(self):
        # Exact boundaries
        assert TrustLevel.from_score(0.8) == TrustLevel.MEDIUM  # > 0.8 for HIGH
        assert TrustLevel.from_score(0.5) == TrustLevel.LOW     # > 0.5 for MEDIUM
        assert TrustLevel.from_score(0.3) == TrustLevel.VERY_LOW  # > 0.3 for LOW


class TestTrustIndexCalculator:
    """Tests for the TrustIndexCalculator."""

    def test_perfect_scores_yield_high_trust(self):
        calc = TrustIndexCalculator()
        result = calc.calculate(
            factuality_score=1.0,
            consistency_score=1.0,
            poison_score=0.0,
        )
        assert result.trust_score == pytest.approx(1.0, abs=0.01)
        assert result.trust_level == TrustLevel.HIGH
        assert result.is_trustworthy is True

    def test_zero_scores_yield_very_low_trust(self):
        calc = TrustIndexCalculator()
        result = calc.calculate(
            factuality_score=0.0,
            consistency_score=0.0,
            poison_score=1.0,
        )
        assert result.trust_score == pytest.approx(0.0, abs=0.01)
        assert result.trust_level == TrustLevel.VERY_LOW
        assert result.is_trustworthy is False

    def test_default_weights_sum_to_one(self):
        calc = TrustIndexCalculator()
        total = calc.alpha + calc.beta + calc.gamma
        assert total == pytest.approx(1.0, abs=0.01)

    def test_weight_normalization(self):
        """Non-normalized weights should be auto-normalized."""
        calc = TrustIndexCalculator(alpha=2.0, beta=2.0, gamma=1.0)
        total = calc.alpha + calc.beta + calc.gamma
        assert total == pytest.approx(1.0, abs=0.01)

    def test_poison_reduces_trust(self):
        calc = TrustIndexCalculator()
        clean = calc.calculate(1.0, 1.0, 0.0)
        poisoned = calc.calculate(1.0, 1.0, 0.8)
        assert poisoned.trust_score < clean.trust_score

    def test_threshold_trustworthy(self):
        calc = TrustIndexCalculator(trust_threshold=0.5)
        above = calc.calculate(0.8, 0.8, 0.0)
        assert above.is_trustworthy is True

    def test_threshold_untrustworthy(self):
        calc = TrustIndexCalculator(trust_threshold=0.5)
        below = calc.calculate(0.2, 0.2, 0.8)
        assert below.is_trustworthy is False

    def test_score_clamped_to_unit_range(self):
        calc = TrustIndexCalculator()
        result = calc.calculate(1.5, 1.5, -0.5)
        assert 0.0 <= result.trust_score <= 1.0

    def test_retrieval_confidence_modifier(self):
        calc = TrustIndexCalculator()
        full_conf = calc.calculate(0.8, 0.8, 0.0, retrieval_confidence=1.0)
        low_conf = calc.calculate(0.8, 0.8, 0.0, retrieval_confidence=0.3)
        assert low_conf.trust_score < full_conf.trust_score

    def test_warnings_generated_for_low_factuality(self):
        calc = TrustIndexCalculator()
        result = calc.calculate(0.2, 0.8, 0.0)
        assert any("support" in w.lower() or "factual" in w.lower()
                    for w in result.warnings)

    def test_warnings_generated_for_high_poison(self):
        calc = TrustIndexCalculator()
        result = calc.calculate(0.8, 0.8, 0.8)
        assert any("poison" in w.lower() or "adversarial" in w.lower()
                    for w in result.warnings)

    def test_to_dict_keys(self):
        calc = TrustIndexCalculator()
        result = calc.calculate(0.7, 0.7, 0.1)
        d = result.to_dict()
        assert "trust_score" in d
        assert "trust_level" in d
        assert "components" in d
        assert "contributions" in d
        assert "warnings" in d

    def test_explain_score_returns_string(self):
        calc = TrustIndexCalculator()
        result = calc.calculate(0.7, 0.7, 0.1)
        explanation = calc.explain_score(result)
        assert isinstance(explanation, str)
        assert "Trust Index" in explanation


class TestPoisonDampener:
    """Tests for the non-linear poison dampening mechanism."""

    def test_no_dampening_below_threshold(self):
        """Poison score <= 0.7 should not trigger dampening."""
        calc = TrustIndexCalculator()
        result_low = calc.calculate(0.9, 0.9, 0.5)
        result_at_threshold = calc.calculate(0.9, 0.9, 0.7)
        # At exactly 0.7 the dampener is 1.0 (no change)
        assert result_at_threshold.trust_score == pytest.approx(
            0.4 * 0.9 + 0.35 * 0.9 + 0.25 * 0.3, abs=0.01
        )
        # Below threshold — no dampening at all
        assert result_low.trust_score == pytest.approx(
            0.4 * 0.9 + 0.35 * 0.9 + 0.25 * 0.5, abs=0.01
        )

    def test_dampening_at_max_poison(self):
        """Poison=1.0 should apply 0.6 multiplier (max dampening)."""
        calc = TrustIndexCalculator()
        result = calc.calculate(1.0, 1.0, 1.0)
        base = 0.4 * 1.0 + 0.35 * 1.0 + 0.25 * 0.0  # = 0.75
        expected = base * 0.6  # = 0.45
        assert result.trust_score == pytest.approx(expected, abs=0.01)

    def test_dampening_pushes_high_factuality_below_threshold(self):
        """Very high poison should push trust below 0.5 even with good factuality."""
        calc = TrustIndexCalculator()
        result = calc.calculate(
            factuality_score=0.95, consistency_score=0.9, poison_score=1.0
        )
        # base = 0.4*0.95 + 0.35*0.9 + 0.25*0 = 0.38+0.315+0 = 0.695
        # dampener at poison=1.0 is 0.6
        # final = 0.695 * 0.6 = 0.417
        assert result.trust_score < 0.5
        assert not result.is_trustworthy

    def test_clean_with_moderate_poison_stays_trusted(self):
        """Clean sample with poison=0.7 should remain above threshold."""
        calc = TrustIndexCalculator()
        result = calc.calculate(
            factuality_score=0.99, consistency_score=1.0, poison_score=0.7
        )
        # No dampening at 0.7, base = 0.4*0.99+0.35*1.0+0.25*0.3 = 0.821
        assert result.trust_score > 0.5
        assert result.is_trustworthy
