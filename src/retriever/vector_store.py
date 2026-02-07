"""
Vector Store for Trustworthy RAG
FAISS-based vector storage and similarity search.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle
from loguru import logger

from .document_processor import Document


class FAISSVectorStore:
    """
    FAISS-based vector store for document retrieval.
    
    Attributes:
        dimension: Embedding dimension
        index: FAISS index
        documents: List of stored documents
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.documents: List[Document] = []
        
        # Create FAISS index
        if index_type == "flat":
            # Exact search (L2 distance)
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "flat_ip":
            # Exact search (Inner Product / Cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        logger.info(f"Initialized FAISS index: type={index_type}, dimension={dimension}")
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of Document objects
            embeddings: Numpy array of embeddings (num_docs x dimension)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Normalize embeddings for cosine similarity (if using IP index)
        if self.index_type == "flat_ip":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings.astype(np.float32))
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity (if using IP index)
        if self.index_type == "flat_ip":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), k
        )
        
        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.documents):
                # Convert L2 distance to similarity score (higher is better)
                if self.index_type == "flat":
                    score = 1 / (1 + dist)  # Convert L2 to similarity
                else:
                    score = float(dist)  # IP is already similarity
                results.append((self.documents[idx], score))
        
        return results
    
    def save(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump({
                'documents': self.documents,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved vector store to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FAISSVectorStore':
        """
        Load a vector store from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded FAISSVectorStore
        """
        path = Path(path)
        
        # Load metadata and documents
        with open(path / "documents.pkl", "rb") as f:
            data = pickle.load(f)
        
        # Create instance
        store = cls(
            dimension=data['dimension'],
            index_type=data['index_type']
        )
        store.documents = data['documents']
        
        # Load FAISS index
        store.index = faiss.read_index(str(path / "index.faiss"))
        
        logger.info(f"Loaded vector store from {path}: {len(store.documents)} documents")
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'num_documents': len(self.documents),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'index_size': self.index.ntotal
        }
    
    def clear(self):
        """Clear all documents from the store."""
        self.documents = []
        self.index.reset()
        logger.info("Cleared vector store")


# Example usage
if __name__ == "__main__":
    # Test the vector store
    dimension = 384  # MiniLM embedding size
    store = FAISSVectorStore(dimension=dimension)
    
    # Create dummy documents and embeddings
    docs = [
        Document("Paris is the capital of France", {"id": 1}),
        Document("London is the capital of the UK", {"id": 2}),
        Document("Tokyo is the capital of Japan", {"id": 3}),
    ]
    
    # Random embeddings for testing
    embeddings = np.random.randn(3, dimension).astype(np.float32)
    
    # Add to store
    store.add_documents(docs, embeddings)
    
    # Search
    query_embedding = np.random.randn(dimension).astype(np.float32)
    results = store.search(query_embedding, k=2)
    
    print("Search results:")
    for doc, score in results:
        print(f"  Score: {score:.4f} - {doc.content[:50]}...")
    
    # Get stats
    print(f"\nStats: {store.get_stats()}")
