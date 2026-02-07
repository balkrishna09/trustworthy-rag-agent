# Retriever Module
"""
Document retrieval components including embeddings and vector store.

Components:
- EmbeddingGenerator: Generates embeddings using sentence-transformers
- DocumentLoader: Loads documents from various sources
- TextChunker: Splits documents into chunks
- FAISSVectorStore: FAISS-based vector storage and search
- Retriever: Main orchestrator combining all components
"""

from .embeddings import EmbeddingGenerator
from .document_processor import Document, TextChunker, DocumentLoader
from .vector_store import FAISSVectorStore
from .retriever import Retriever

__all__ = [
    'EmbeddingGenerator',
    'Document',
    'TextChunker',
    'DocumentLoader',
    'FAISSVectorStore',
    'Retriever'
]
