# Pipeline Module
"""
End-to-end RAG pipeline integrating retriever, evaluation agent, and generator.

Components:
- RAGPipeline: Main pipeline connecting all components
- RAGResponse: Response dataclass with metadata
"""

from .rag_pipeline import RAGPipeline, RAGResponse

__all__ = [
    'RAGPipeline',
    'RAGResponse'
]
