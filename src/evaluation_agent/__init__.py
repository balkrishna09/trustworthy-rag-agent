"""
Evaluation Agent Module
=======================

The core research component for detecting misinformation and knowledge
poisoning in RAG systems.

Components:
- NLIVerifier: Verifies factual consistency using Natural Language Inference
- PoisonDetector: Detects adversarial/poisoned content in documents
- TrustIndexCalculator: Combines signals into a composite trust score
- EvaluationAgent: Main orchestrator that combines all components

Usage:
    from src.evaluation_agent import EvaluationAgent
    
    agent = EvaluationAgent(config)
    result = agent.evaluate(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        retrieved_documents=["Paris is the capital of France..."]
    )
    
    print(f"Trust Score: {result.trust_score}")
    print(f"Trustworthy: {result.is_trustworthy}")
    print(result.detailed_report)
"""

from .nli_verifier import (
    NLIVerifier,
    NLIResult,
    VerificationResult
)

from .poison_detector import (
    PoisonDetector,
    PoisonSignal,
    DocumentPoisonResult,
    PoisonDetectionResult
)

from .trust_index import (
    TrustIndexCalculator,
    TrustLevel,
    TrustComponents,
    TrustIndexResult
)

from .evaluation_agent import (
    EvaluationAgent,
    EvaluationResult
)

__all__ = [
    # Main class
    'EvaluationAgent',
    'EvaluationResult',
    
    # NLI Verifier
    'NLIVerifier',
    'NLIResult',
    'VerificationResult',
    
    # Poison Detector
    'PoisonDetector',
    'PoisonSignal',
    'DocumentPoisonResult',
    'PoisonDetectionResult',
    
    # Trust Index
    'TrustIndexCalculator',
    'TrustLevel',
    'TrustComponents',
    'TrustIndexResult',
]
