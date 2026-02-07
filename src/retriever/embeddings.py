"""
Embedding Generator for Trustworthy RAG
Generates embeddings for documents and queries using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from loguru import logger


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers models.
    
    Attributes:
        model_name: Name of the sentence-transformer model
        model: Loaded SentenceTransformer model
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
    
    def embed_documents(self, documents: List[str], batch_size: int = 32, 
                        show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of document texts to embed
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (num_docs x embedding_dim)
        """
        logger.info(f"Embedding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Numpy array of query embedding (1 x embedding_dim)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        return embedding
    
    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dimension


# Example usage
if __name__ == "__main__":
    # Test the embedding generator
    embedder = EmbeddingGenerator()
    
    # Test document embedding
    docs = [
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "The Eiffel Tower is located in Paris."
    ]
    doc_embeddings = embedder.embed_documents(docs)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    
    # Test query embedding
    query = "What is the capital of France?"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Test similarity (cosine similarity)
    from numpy.linalg import norm
    similarities = np.dot(doc_embeddings, query_embedding) / (
        norm(doc_embeddings, axis=1) * norm(query_embedding)
    )
    print(f"Similarities: {similarities}")
    print(f"Most similar document: {docs[np.argmax(similarities)]}")
