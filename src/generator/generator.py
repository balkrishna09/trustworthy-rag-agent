"""
Generator Module for Trustworthy RAG
Handles response generation using LLM with retrieved context.
"""

from typing import List, Dict, Any, Optional, Tuple
import yaml
from loguru import logger

from .llm_client import BaseLLMClient, FARMIClient, create_llm_client
from .prompts import (
    create_rag_prompt,
    create_fact_check_prompt,
    format_context,
    RAG_PROMPT,
    RAG_PROMPT_CONCISE,
    RAG_PROMPT_DETAILED
)


class Generator:
    """
    Response generator for the RAG system.
    
    Combines retrieved documents with user queries to generate
    contextually grounded responses using an LLM.
    
    Attributes:
        llm_client: LLM client for generation
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        llm_client: Optional[BaseLLMClient] = None
    ):
        """
        Initialize the generator.
        
        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (overrides config_path)
            llm_client: Pre-configured LLM client (overrides config)
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
                'LLM_PROVIDER': 'openai_compatible',
                'FARMI_API_URL': 'https://gptlab.rd.tuni.fi/students/ollama/v1',
                'FARMI_API_KEY': '',
                'LLM_MODEL': 'llama3.3:70b',
                'MAX_NEW_TOKENS': 512,
                'TEMPERATURE': 0.7
            }
        
        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = create_llm_client(self.config)
        
        self.max_tokens = self.config.get('MAX_NEW_TOKENS', 512)
        self.temperature = self.config.get('TEMPERATURE', 0.7)
        
        logger.info("Generator initialized")
    
    def generate(
        self,
        question: str,
        context_docs: List[str],
        prompt_style: str = "default",
        **kwargs
    ) -> str:
        """
        Generate a response based on the question and context.
        
        Args:
            question: User's question
            context_docs: List of relevant document texts
            prompt_style: Style of prompt ("default", "concise", "detailed")
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Create the prompt
        prompt = create_rag_prompt(question, context_docs, prompt_style)
        
        logger.debug(f"Generating response for: {question[:50]}...")
        
        # Generate response
        response = self.llm_client.generate(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature']}
        )
        
        logger.debug(f"Generated response: {response[:100]}...")
        return response
    
    def generate_with_metadata(
        self,
        question: str,
        context_docs: List[str],
        prompt_style: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response with additional metadata.
        
        Args:
            question: User's question
            context_docs: List of relevant document texts
            prompt_style: Style of prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing response and metadata
        """
        prompt = create_rag_prompt(question, context_docs, prompt_style)
        
        response = self.llm_client.generate(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature)
        )
        
        return {
            'question': question,
            'context': context_docs,
            'prompt': prompt,
            'response': response,
            'model': self.config.get('LLM_MODEL'),
            'prompt_style': prompt_style
        }
    
    def generate_chat(
        self,
        question: str,
        context_docs: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using chat format.
        
        Args:
            question: User's question
            context_docs: List of relevant document texts
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the context doesn't contain enough information, say so."""
        
        # Format context
        context = format_context(context_docs)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        # Generate using chat endpoint
        response = self.llm_client.chat(
            messages,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature)
        )
        
        return response
    
    def verify_fact(self, claim: str, evidence: str, **kwargs) -> Dict[str, Any]:
        """
        Verify a claim against evidence using the LLM.
        
        Args:
            claim: The claim to verify
            evidence: Evidence to check against
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with verdict and explanation
        """
        prompt = create_fact_check_prompt(claim, evidence)
        
        response = self.llm_client.generate(
            prompt,
            max_tokens=kwargs.get('max_tokens', 100),
            temperature=kwargs.get('temperature', 0.3)  # Lower temperature for consistency
        )
        
        # Parse verdict
        response_upper = response.upper()
        if "SUPPORTED" in response_upper:
            verdict = "SUPPORTED"
        elif "REFUTED" in response_upper:
            verdict = "REFUTED"
        else:
            verdict = "NOT ENOUGH INFO"
        
        return {
            'claim': claim,
            'evidence': evidence,
            'verdict': verdict,
            'raw_response': response
        }
    
    def generate_raw(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from a raw prompt (no formatting).
        
        Args:
            prompt: Raw prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        return self.llm_client.generate(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature)
        )
    
    def test_connection(self) -> bool:
        """Test if the LLM connection is working."""
        if hasattr(self.llm_client, 'test_connection'):
            return self.llm_client.test_connection()
        
        try:
            response = self.generate_raw("Say 'OK'", max_tokens=10)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Test the generator
    config = {
        'LLM_PROVIDER': 'openai_compatible',
        'FARMI_API_URL': 'https://gptlab.rd.tuni.fi/students/ollama/v1',
        'FARMI_API_KEY': '',  # Set your API key
        'LLM_MODEL': 'llama3.3:70b',
        'MAX_NEW_TOKENS': 200,
        'TEMPERATURE': 0.7
    }
    
    generator = Generator(config=config)
    
    # Test connection
    print("Testing connection...")
    if generator.test_connection():
        print("Connection successful!\n")
        
        # Test RAG generation
        context = [
            "Paris is the capital of France.",
            "The Eiffel Tower is a famous landmark in Paris.",
            "France is located in Western Europe."
        ]
        question = "What is the capital of France and what is it famous for?"
        
        print(f"Question: {question}")
        print(f"Context: {context}")
        print("\nGenerating response...")
        
        response = generator.generate(question, context)
        print(f"\nResponse: {response}")
    else:
        print("Connection failed!")
