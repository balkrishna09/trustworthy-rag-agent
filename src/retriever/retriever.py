"""
Main Retriever Class for Trustworthy RAG
Orchestrates document processing, embedding, and retrieval.
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import yaml
from loguru import logger

from .document_processor import Document, TextChunker, DocumentLoader
from .embeddings import EmbeddingGenerator
from .vector_store import FAISSVectorStore


class Retriever:
    """
    Main retriever class that combines embedding generation and vector search.
    
    This class orchestrates:
    1. Document loading and preprocessing
    2. Text chunking
    3. Embedding generation
    4. Vector storage and retrieval
    
    Attributes:
        config: Configuration dictionary
        embedder: EmbeddingGenerator instance
        chunker: TextChunker instance
        vector_store: FAISSVectorStore instance
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the retriever.
        
        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (overrides config_path)
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
                'CHUNK_SIZE': 512,
                'CHUNK_OVERLAP': 50,
                'TOP_K_RETRIEVAL': 5,
                'FAISS_INDEX_PATH': 'data/processed/faiss_index'
            }
        
        # Initialize components
        self.embedder = EmbeddingGenerator(
            model_name=self.config.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        
        self.chunker = TextChunker(
            chunk_size=self.config.get('CHUNK_SIZE', 512),
            chunk_overlap=self.config.get('CHUNK_OVERLAP', 50)
        )
        
        self.vector_store = FAISSVectorStore(
            dimension=self.embedder.get_dimension()
        )
        
        self.top_k = self.config.get('TOP_K_RETRIEVAL', 5)
        self.index_path = self.config.get('FAISS_INDEX_PATH', 'data/processed/faiss_index')
        
        logger.info("Retriever initialized")
    
    def add_documents(self, documents: List[Document], chunk: bool = True):
        """
        Add documents to the retriever.
        
        Args:
            documents: List of Document objects
            chunk: Whether to chunk documents before adding
        """
        # Chunk documents if requested
        if chunk:
            docs_to_embed = self.chunker.split_documents(documents)
        else:
            docs_to_embed = documents
        
        # Generate embeddings
        texts = [doc.content for doc in docs_to_embed]
        embeddings = self.embedder.embed_documents(texts)
        
        # Add to vector store
        self.vector_store.add_documents(docs_to_embed, embeddings)
        
        logger.info(f"Added {len(docs_to_embed)} document chunks to retriever")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None,
                  chunk: bool = True):
        """
        Add text strings to the retriever.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            chunk: Whether to chunk texts before adding
        """
        metadatas = metadatas or [{} for _ in texts]
        documents = [
            Document(content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        self.add_documents(documents, chunk=chunk)
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (default: self.top_k)
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        k = k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def retrieve_texts(self, query: str, k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant document texts for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of document content strings
        """
        results = self.retrieve(query, k)
        return [doc.content for doc, _ in results]
    
    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve documents with detailed information.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with query, documents, and scores
        """
        results = self.retrieve(query, k)
        
        return {
            'query': query,
            'results': [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
        }
    
    def save(self, path: Optional[str] = None):
        """
        Save the retriever's vector store to disk.
        
        Args:
            path: Path to save to (default: self.index_path)
        """
        path = path or self.index_path
        self.vector_store.save(path)
        logger.info(f"Saved retriever to {path}")
    
    def load(self, path: Optional[str] = None):
        """
        Load a saved vector store.
        
        Args:
            path: Path to load from (default: self.index_path)
        """
        path = path or self.index_path
        self.vector_store = FAISSVectorStore.load(path)
        logger.info(f"Loaded retriever from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        return {
            'embedding_model': self.config.get('EMBEDDING_MODEL'),
            'embedding_dimension': self.embedder.get_dimension(),
            'chunk_size': self.config.get('CHUNK_SIZE'),
            'chunk_overlap': self.config.get('CHUNK_OVERLAP'),
            'top_k': self.top_k,
            **self.vector_store.get_stats()
        }
    
    def clear(self):
        """Clear all documents from the retriever."""
        self.vector_store.clear()
        logger.info("Cleared retriever")
    
    # Convenience methods for loading different data sources
    
    def load_text_files(self, directory: str, pattern: str = "*.txt"):
        """Load and index text files from a directory."""
        documents = DocumentLoader.load_text_files(directory, pattern)
        self.add_documents(documents)
    
    def load_fever_data(self, file_path: str):
        """Load and index FEVER dataset."""
        documents = DocumentLoader.load_fever_dataset(file_path)
        self.add_documents(documents, chunk=False)  # Claims are already short
    
    def load_truthfulqa_data(self, file_path: str):
        """Load and index TruthfulQA dataset."""
        documents = DocumentLoader.load_truthfulqa_dataset(file_path)
        self.add_documents(documents, chunk=False)  # Questions are already short


# Example usage
if __name__ == "__main__":
    # Test the retriever
    retriever = Retriever()
    
    # Add some test documents
    test_docs = [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "London is the capital of the United Kingdom. Big Ben is a famous landmark.",
        "Tokyo is the capital of Japan. It is one of the largest cities in the world.",
        "Berlin is the capital of Germany. The Brandenburg Gate is a famous monument.",
        "Rome is the capital of Italy. The Colosseum is an ancient amphitheater."
    ]
    
    retriever.add_texts(test_docs)
    
    # Test retrieval
    query = "What is the capital of France?"
    results = retriever.retrieve_with_scores(query)
    
    print(f"\nQuery: {results['query']}")
    print("\nResults:")
    for i, result in enumerate(results['results']):
        print(f"  {i+1}. Score: {result['score']:.4f}")
        print(f"     Content: {result['content'][:80]}...")
    
    # Print stats
    print(f"\nRetriever stats: {retriever.get_stats()}")
