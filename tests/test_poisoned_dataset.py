"""
Tests for the Poisoned Dataset Generator.
"""

import pytest
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.poisoned_dataset import (
    PoisonedDatasetGenerator,
    PoisonStrategy,
    PoisonedSample,
)


@pytest.fixture
def generator():
    return PoisonedDatasetGenerator(seed=42)


@pytest.fixture
def sample_docs():
    return [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "Berlin is the capital of Germany. The Brandenburg Gate is a famous monument.",
        "Tokyo is the capital of Japan. It is one of the largest cities in the world.",
        "London is the capital of the United Kingdom. Big Ben is a famous landmark.",
        "Rome is the capital of Italy. The Colosseum is an ancient amphitheater.",
    ]


class TestPoisonStrategies:
    """Test individual poisoning strategies."""

    def test_contradiction_adds_text(self, generator, sample_docs):
        result = generator.poison_text_contradiction(sample_docs[0])
        assert len(result) > len(sample_docs[0])
        # Should contain contradiction template language
        assert any(
            phrase in result.lower()
            for phrase in ["contrary", "actually", "misconception", "believe", "confirms"]
        )

    def test_injection_adds_override(self, generator, sample_docs):
        result = generator.poison_text_injection(sample_docs[0])
        assert len(result) > len(sample_docs[0])
        assert any(
            phrase in result.upper()
            for phrase in ["IMPORTANT", "NOTE", "CORRECTION", "OVERRIDE", "SYSTEM"]
        )

    def test_entity_swap_changes_entity(self, generator):
        text = "Paris is the capital of France."
        result = generator.poison_text_entity_swap(text)
        # Paris should be replaced with something else
        assert "Paris" not in result or "France" not in result

    def test_entity_swap_no_match_still_modifies(self, generator):
        """Text without known entities should still be modified via fallback."""
        text = "This is a sentence about nothing specific."
        result = generator.poison_text_entity_swap(text)
        assert result != text

    def test_entity_swap_number_fallback(self, generator):
        """Entity swap should modify numbers when no named entities found."""
        text = "The event occurred in 1776 and involved 13 colonies."
        result = generator.poison_text_entity_swap(text)
        assert result != text

    def test_subtle_manipulation(self, generator, sample_docs):
        result = generator.poison_text_subtle(sample_docs[0])
        # Should be different from original
        assert result != sample_docs[0]

    def test_poison_document_returns_sample(self, generator, sample_docs):
        sample = generator.poison_document(
            sample_docs[0], PoisonStrategy.CONTRADICTION
        )
        assert isinstance(sample, PoisonedSample)
        assert sample.is_poisoned is True
        assert sample.original_text == sample_docs[0]
        assert sample.poisoned_text != sample_docs[0]

    def test_mixed_strategy_picks_random(self, generator, sample_docs):
        """Mixed should resolve to one of the 4 concrete strategies."""
        sample = generator.poison_document(
            sample_docs[0], PoisonStrategy.MIXED
        )
        assert sample.strategy in [
            PoisonStrategy.CONTRADICTION,
            PoisonStrategy.INSTRUCTION_INJECTION,
            PoisonStrategy.ENTITY_SWAP,
            PoisonStrategy.SUBTLE_MANIPULATION,
        ]


class TestDatasetGeneration:
    """Test full dataset creation."""

    def test_poison_ratio_respected(self, generator, sample_docs):
        samples, stats = generator.create_poisoned_dataset(
            sample_docs, poison_ratio=0.4
        )
        assert len(samples) == len(sample_docs)
        poisoned_count = sum(1 for s in samples if s.is_poisoned)
        assert poisoned_count == 2  # 40% of 5 = 2

    def test_clean_samples_unchanged(self, generator, sample_docs):
        samples, _ = generator.create_poisoned_dataset(
            sample_docs, poison_ratio=0.2
        )
        clean_samples = [s for s in samples if not s.is_poisoned]
        for s in clean_samples:
            assert s.original_text == s.poisoned_text

    def test_stats_correct(self, generator, sample_docs):
        _, stats = generator.create_poisoned_dataset(
            sample_docs, poison_ratio=0.6
        )
        assert stats["total_documents"] == 5
        assert stats["poisoned_count"] == 3
        assert stats["clean_count"] == 2

    def test_reproducibility(self, sample_docs):
        gen1 = PoisonedDatasetGenerator(seed=123)
        gen2 = PoisonedDatasetGenerator(seed=123)
        s1, _ = gen1.create_poisoned_dataset(sample_docs, poison_ratio=0.4)
        s2, _ = gen2.create_poisoned_dataset(sample_docs, poison_ratio=0.4)
        for a, b in zip(s1, s2):
            assert a.poisoned_text == b.poisoned_text

    def test_zero_poison_ratio(self, generator, sample_docs):
        samples, stats = generator.create_poisoned_dataset(
            sample_docs, poison_ratio=0.0
        )
        assert stats["poisoned_count"] == 0
        assert all(not s.is_poisoned for s in samples)

    def test_full_poison_ratio(self, generator, sample_docs):
        samples, stats = generator.create_poisoned_dataset(
            sample_docs, poison_ratio=1.0
        )
        assert stats["poisoned_count"] == 5
        assert all(s.is_poisoned for s in samples)


class TestDatasetIO:
    """Test save/load round-trip."""

    def test_save_and_load(self, generator, sample_docs):
        samples, stats = generator.create_poisoned_dataset(
            sample_docs, poison_ratio=0.4
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            tmp_path = f.name

        try:
            generator.save_dataset(samples, tmp_path, stats)
            loaded_samples, loaded_stats = PoisonedDatasetGenerator.load_dataset(
                tmp_path
            )
            assert len(loaded_samples) == len(samples)
            for orig, loaded in zip(samples, loaded_samples):
                assert orig.original_text == loaded.original_text
                assert orig.poisoned_text == loaded.poisoned_text
                assert orig.is_poisoned == loaded.is_poisoned
        finally:
            os.unlink(tmp_path)
