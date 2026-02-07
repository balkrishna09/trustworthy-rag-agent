"""
Experiments Module
==================

Framework for running experiments to evaluate the Trustworthy RAG system.

Components:
- PoisonedDatasetGenerator: Creates poisoned versions of clean datasets
- ExperimentRunner: Runs evaluation experiments
- ResultsAnalyzer: Analyzes and visualizes experiment results
"""

from .poisoned_dataset import PoisonedDatasetGenerator, PoisonStrategy
from .experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult

__all__ = [
    'PoisonedDatasetGenerator',
    'PoisonStrategy',
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
]
