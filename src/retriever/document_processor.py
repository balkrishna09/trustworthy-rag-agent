"""
Document Processor for Trustworthy RAG
Handles document loading, chunking, and preprocessing.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from loguru import logger


class Document:
    """
    Represents a document with content and metadata.
    
    Attributes:
        content: The text content of the document
        metadata: Dictionary of metadata (source, page, etc.)
    """
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}
    
    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


class TextChunker:
    """
    Splits documents into smaller chunks for embedding.
    
    Attributes:
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Overlap between consecutive chunks
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of this chunk
            end = start + self.chunk_size
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation
                for boundary in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_boundary = text.rfind(boundary, start, end)
                    if last_boundary != -1:
                        end = last_boundary + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= chunks[-1] if chunks else 0:
                start = end  # Avoid infinite loop
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks, preserving metadata.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata['chunk_index'] = i
                metadata['total_chunks'] = len(chunks)
                chunked_docs.append(Document(content=chunk, metadata=metadata))
        
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs


class DocumentLoader:
    """
    Loads documents from various sources.
    """
    
    @staticmethod
    def load_text_file(file_path: str) -> Document:
        """Load a single text file."""
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            metadata={'source': str(path), 'filename': path.name}
        )
    
    @staticmethod
    def load_text_files(directory: str, pattern: str = "*.txt") -> List[Document]:
        """Load all text files from a directory."""
        path = Path(directory)
        documents = []
        
        for file_path in path.glob(pattern):
            try:
                doc = DocumentLoader.load_text_file(str(file_path))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    @staticmethod
    def load_jsonl(file_path: str, content_field: str = "content", 
                   metadata_fields: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            content_field: Field name containing the text content
            metadata_fields: List of fields to include in metadata
            
        Returns:
            List of Document objects
        """
        documents = []
        metadata_fields = metadata_fields or []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    content = data.get(content_field, "")
                    
                    if not content:
                        continue
                    
                    metadata = {field: data.get(field) for field in metadata_fields}
                    metadata['source'] = file_path
                    
                    documents.append(Document(content=content, metadata=metadata))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON line: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    
    @staticmethod
    def load_fever_dataset(file_path: str) -> List[Document]:
        """
        Load FEVER dataset for fact verification.
        
        Args:
            file_path: Path to FEVER JSONL file
            
        Returns:
            List of Document objects with claims
        """
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    claim = data.get("claim", "")
                    
                    if not claim:
                        continue
                    
                    metadata = {
                        'id': data.get('id'),
                        'label': data.get('label'),
                        'verifiable': data.get('verifiable'),
                        'evidence': data.get('evidence'),
                        'source': 'fever'
                    }
                    
                    documents.append(Document(content=claim, metadata=metadata))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing FEVER line: {e}")
        
        logger.info(f"Loaded {len(documents)} claims from FEVER dataset")
        return documents
    
    @staticmethod
    def load_truthfulqa_dataset(file_path: str) -> List[Document]:
        """
        Load TruthfulQA dataset for misinformation detection.
        
        Args:
            file_path: Path to TruthfulQA JSONL file
            
        Returns:
            List of Document objects with questions
        """
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question = data.get("question", "")
                    
                    if not question:
                        continue
                    
                    metadata = {
                        'type': data.get('type'),
                        'category': data.get('category'),
                        'best_answer': data.get('best_answer'),
                        'correct_answers': data.get('correct_answers'),
                        'incorrect_answers': data.get('incorrect_answers'),
                        'source': 'truthfulqa'
                    }
                    
                    documents.append(Document(content=question, metadata=metadata))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing TruthfulQA line: {e}")
        
        logger.info(f"Loaded {len(documents)} questions from TruthfulQA dataset")
        return documents


# Example usage
if __name__ == "__main__":
    # Test the chunker
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    test_text = """Paris is the capital of France. It is known for the Eiffel Tower. 
    The city has a rich history dating back centuries. Many tourists visit Paris each year. 
    The Louvre Museum is one of the most famous attractions."""
    
    chunks = chunker.split_text(test_text)
    print("Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  {i}: {chunk[:50]}...")
    
    # Test document loading
    doc = Document(content=test_text, metadata={'source': 'test'})
    chunked_docs = chunker.split_documents([doc])
    print(f"\nChunked documents: {len(chunked_docs)}")
