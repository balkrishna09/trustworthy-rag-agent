"""
RAG Pipeline for Trustworthy RAG
Connects retriever, evaluation agent, and generator into a complete system.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from loguru import logger

# Load .env if available (makes FARMI_API_KEY etc. accessible via os.environ)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from src.retriever import Retriever, Document
from src.generator import Generator
from src.evaluation_agent import EvaluationAgent, EvaluationResult, TrustLevel


@dataclass
class RAGResponse:
    """
    Response from the RAG pipeline.
    
    Attributes:
        query: Original user query
        response: Generated response text
        retrieved_docs: List of retrieved documents
        scores: Retrieval scores for each document
        metadata: Additional metadata
        evaluation: Optional evaluation result from the Evaluation Agent
    """
    query: str
    response: str
    retrieved_docs: List[str]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation: Optional[EvaluationResult] = None
    
    @property
    def trust_score(self) -> Optional[float]:
        """Get the trust score if evaluation was performed."""
        return self.evaluation.trust_score if self.evaluation else None
    
    @property
    def is_trustworthy(self) -> Optional[bool]:
        """Check if response is trustworthy based on evaluation."""
        return self.evaluation.is_trustworthy if self.evaluation else None
    
    @property
    def trust_level(self) -> Optional[TrustLevel]:
        """Get trust level if evaluation was performed."""
        return self.evaluation.trust_level if self.evaluation else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'query': self.query,
            'response': self.response,
            'retrieved_docs': self.retrieved_docs,
            'scores': self.scores,
            'metadata': self.metadata
        }
        if self.evaluation:
            result['evaluation'] = self.evaluation.to_dict()
        return result


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) Pipeline.
    
    This pipeline:
    1. Takes a user query
    2. Retrieves relevant documents using the Retriever
    3. Generates a response using the Generator
    4. Evaluates the response using the Evaluation Agent
    
    Attributes:
        retriever: Document retrieval component
        generator: Response generation component
        evaluation_agent: Evaluation agent for trustworthiness assessment
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        evaluation_agent: Optional[EvaluationAgent] = None,
        enable_evaluation: bool = True
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (overrides config_path)
            retriever: Pre-configured retriever (overrides config)
            generator: Pre-configured generator (overrides config)
            evaluation_agent: Pre-configured evaluation agent
            enable_evaluation: Whether to enable evaluation by default
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config path
            default_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
            if default_path.exists():
                with open(default_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}
        
        # Initialize components
        self.retriever = retriever or Retriever(config=self.config)
        self.generator = generator or Generator(config=self.config)
        
        # Initialize evaluation agent (lazy load to avoid loading NLI model immediately)
        self._evaluation_agent = evaluation_agent
        self._evaluation_enabled = enable_evaluation
        
        # Configuration
        self.top_k = self.config.get('TOP_K_RETRIEVAL', 5)
        
        logger.info("RAG Pipeline initialized")
    
    @property
    def evaluation_agent(self) -> EvaluationAgent:
        """Lazy load the evaluation agent."""
        if self._evaluation_agent is None:
            logger.info("Initializing Evaluation Agent...")
            self._evaluation_agent = EvaluationAgent(config=self.config)
        return self._evaluation_agent
    
    def enable_evaluation(self):
        """Enable evaluation for queries."""
        self._evaluation_enabled = True
        logger.info("Evaluation enabled")
    
    def disable_evaluation(self):
        """Disable evaluation for queries (faster)."""
        self._evaluation_enabled = False
        logger.info("Evaluation disabled")
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        prompt_style: str = "default",
        **kwargs
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            prompt_style: Style of prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            RAGResponse with the generated answer and metadata
        """
        k = k or self.top_k
        
        logger.info(f"Processing query: {question[:50]}...")
        
        # Step 1: Retrieve relevant documents
        retrieved = self.retriever.retrieve(question, k=k)
        
        if not retrieved:
            logger.warning("No documents retrieved")
            return RAGResponse(
                query=question,
                response="I couldn't find any relevant information to answer your question.",
                retrieved_docs=[],
                scores=[],
                metadata={'error': 'no_documents_retrieved'}
            )
        
        # Extract documents and scores
        docs = [doc.content for doc, _ in retrieved]
        scores = [score for _, score in retrieved]
        
        logger.debug(f"Retrieved {len(docs)} documents")
        
        # Step 2: Generate response
        response = self.generator.generate(
            question=question,
            context_docs=docs,
            prompt_style=prompt_style,
            **kwargs
        )
        
        logger.info(f"Generated response: {response[:100]}...")
        
        return RAGResponse(
            query=question,
            response=response,
            retrieved_docs=docs,
            scores=scores,
            metadata={
                'k': k,
                'prompt_style': prompt_style,
                'model': self.config.get('LLM_MODEL')
            }
        )
    
    def query_with_evaluation(
        self,
        question: str,
        k: Optional[int] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Process a query with full evaluation.
        
        This method runs the complete pipeline including:
        1. Retrieval
        2. Generation
        3. Evaluation (NLI verification, poison detection, trust index)
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            RAGResponse with evaluation results included
        """
        k = k or self.top_k
        
        logger.info(f"Processing query with evaluation: {question[:50]}...")

        # Step 1: Retrieve relevant documents WITH embeddings
        # Embeddings are used by the poison detector for semantic outlier analysis
        retrieved, doc_embeddings = self.retriever.retrieve_with_embeddings(question, k=k)

        if not retrieved:
            logger.warning("No documents retrieved")
            return RAGResponse(
                query=question,
                response="I couldn't find any relevant information to answer your question.",
                retrieved_docs=[],
                scores=[],
                metadata={'error': 'no_documents_retrieved'},
                evaluation=None
            )

        # Extract documents and scores
        docs = [doc.content for doc, _ in retrieved]
        scores = [score for _, score in retrieved]

        logger.debug(f"Retrieved {len(docs)} documents")

        # Step 2: Generate response
        response = self.generator.generate(
            question=question,
            context_docs=docs,
            **kwargs
        )

        logger.debug(f"Generated response: {response[:100]}...")

        # Step 3: Evaluate the response (with embeddings for semantic analysis)
        logger.info("Running evaluation...")
        evaluation_result = self.evaluation_agent.evaluate(
            query=question,
            response=response,
            retrieved_documents=docs,
            retrieval_scores=scores,
            document_embeddings=doc_embeddings if len(doc_embeddings) > 0 else None
        )
        
        logger.info(
            f"Evaluation complete: Trust Score = {evaluation_result.trust_score:.2f} "
            f"({evaluation_result.trust_level.value})"
        )
        
        return RAGResponse(
            query=question,
            response=response,
            retrieved_docs=docs,
            scores=scores,
            metadata={
                'k': k,
                'model': self.config.get('LLM_MODEL'),
                'evaluated': True,
                'evaluation_time_ms': evaluation_result.evaluation_time_ms
            },
            evaluation=evaluation_result
        )
    
    def evaluate_response(
        self,
        response: str,
        documents: List[str],
        query: str = ""
    ) -> EvaluationResult:
        """
        Evaluate a response independently.
        
        Useful for evaluating responses that were generated elsewhere.
        
        Args:
            response: The response text to evaluate
            documents: The documents used to generate the response
            query: Optional original query
            
        Returns:
            EvaluationResult with trust index and details
        """
        return self.evaluation_agent.evaluate(
            query=query,
            response=response,
            retrieved_documents=documents
        )
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the retriever's knowledge base.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        self.retriever.add_texts(documents, metadatas)
        logger.info(f"Added {len(documents)} documents to pipeline")
    
    def add_document_objects(self, documents: List[Document]):
        """Add Document objects to the retriever."""
        self.retriever.add_documents(documents)
    
    def load_knowledge_base(self, directory: str, pattern: str = "*.txt"):
        """Load documents from a directory as the knowledge base."""
        self.retriever.load_text_files(directory, pattern)
    
    def load_fever_data(self, file_path: str):
        """Load FEVER dataset."""
        self.retriever.load_fever_data(file_path)
    
    def load_truthfulqa_data(self, file_path: str):
        """Load TruthfulQA dataset."""
        self.retriever.load_truthfulqa_data(file_path)
    
    def save(self, path: Optional[str] = None):
        """Save the retriever's index."""
        self.retriever.save(path)
    
    def load(self, path: Optional[str] = None):
        """Load a saved retriever index."""
        self.retriever.load(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        return {
            'retriever': self.retriever.get_stats(),
            'generator': {
                'model': self.config.get('LLM_MODEL'),
                'provider': self.config.get('LLM_PROVIDER')
            },
            'evaluation': {
                'enabled': self._evaluation_enabled,
                'initialized': self._evaluation_agent is not None,
                'nli_model': self.config.get('NLI_MODEL', 'facebook/bart-large-mnli'),
                'trust_threshold': self.config.get('TRUST_THRESHOLD', 0.5)
            },
            'config': {
                'top_k': self.top_k
            }
        }
    
    def test_connection(self) -> bool:
        """Test if all components are working."""
        return self.generator.test_connection()
    
    def print_evaluation_report(self, result: RAGResponse):
        """Print a formatted evaluation report for a RAG response."""
        if not result.evaluation:
            print("No evaluation available for this response.")
            return
        
        print(result.evaluation.detailed_report)


# Example usage
if __name__ == "__main__":
    # Test the pipeline
    config = {
        'LLM_PROVIDER': 'openai_compatible',
        'FARMI_API_URL': 'https://gptlab.rd.tuni.fi/students/ollama/v1',
        'FARMI_API_KEY': '',  # Set your API key
        'LLM_MODEL': 'llama3.3:70b',
        'MAX_NEW_TOKENS': 200,
        'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
        'TOP_K_RETRIEVAL': 3,
        'NLI_MODEL': 'facebook/bart-large-mnli',
        'TRUST_THRESHOLD': 0.5,
        'POISON_THRESHOLD': 0.7,
        'TRUST_ALPHA': 0.4,
        'TRUST_BETA': 0.35,
        'TRUST_GAMMA': 0.25
    }
    
    # Initialize pipeline (evaluation disabled by default for basic query)
    pipeline = RAGPipeline(config=config, enable_evaluation=False)
    
    # Add some test documents
    documents = [
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "London is the capital of the United Kingdom. Big Ben is a famous landmark.",
        "Tokyo is the capital of Japan. It is one of the largest cities in the world.",
        "Berlin is the capital of Germany. The Brandenburg Gate is a famous monument.",
        "Rome is the capital of Italy. The Colosseum is an ancient amphitheater."
    ]
    
    print("Adding documents...")
    pipeline.add_documents(documents)
    
    # Test basic query (no evaluation)
    print("\n" + "="*60)
    print("TEST 1: Basic Query (No Evaluation)")
    print("="*60)
    result = pipeline.query("What is the capital of France and what is it famous for?")
    
    print(f"\nQuery: {result.query}")
    print(f"\nRetrieved Documents:")
    for i, (doc, score) in enumerate(zip(result.retrieved_docs, result.scores)):
        print(f"  {i+1}. [{score:.4f}] {doc[:60]}...")
    print(f"\nResponse: {result.response}")
    
    # Test query with evaluation
    print("\n" + "="*60)
    print("TEST 2: Query with Evaluation")
    print("="*60)
    print("(This will load the NLI model - may take a moment...)")
    
    result_with_eval = pipeline.query_with_evaluation(
        "What is the capital of Germany?"
    )
    
    print(f"\nQuery: {result_with_eval.query}")
    print(f"Response: {result_with_eval.response}")
    print(f"\nTrust Score: {result_with_eval.trust_score:.2f}")
    print(f"Trust Level: {result_with_eval.trust_level.value}")
    print(f"Is Trustworthy: {result_with_eval.is_trustworthy}")
    
    # Print full evaluation report
    print("\n" + "="*60)
    print("FULL EVALUATION REPORT")
    print("="*60)
    pipeline.print_evaluation_report(result_with_eval)
    
    # Print stats
    print(f"\nPipeline Stats: {pipeline.get_stats()}")
