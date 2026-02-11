"""
LLM Client for Trustworthy RAG
Handles communication with FARMI/Ollama/OpenAI APIs.
"""

import os
import requests
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger

# Load .env if available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response for chat messages."""
        pass


class FARMIClient(BaseLLMClient):
    """
    Client for FARMI GPTLab API (OpenAI-compatible).
    
    Attributes:
        api_url: FARMI API endpoint URL
        api_key: Authentication key
        model: Model name to use
    """
    
    def __init__(
        self,
        api_url: str = "https://gptlab.rd.tuni.fi/students/ollama/v1",
        api_key: str = "",
        model: str = "llama3.3:70b",
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        """
        Initialize the FARMI client.
        
        Args:
            api_url: FARMI API endpoint
            api_key: API authentication key
            model: Model name
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"FARMI client initialized: model={model}, url={api_url}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the completions endpoint.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            Generated text response
        """
        url = f"{self.api_url}/completions"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        # Add optional parameters
        if "stop" in kwargs:
            data["stop"] = kwargs["stop"]
        if "top_p" in kwargs:
            data["top_p"] = kwargs["top_p"]
        
        logger.debug(f"Sending request to FARMI: {prompt[:50]}...")
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["text"]
            
            # Log usage
            usage = result.get("usage", {})
            logger.debug(f"Tokens used: {usage.get('total_tokens', 'N/A')}")
            
            return generated_text.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FARMI API error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using the chat completions endpoint.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        url = f"{self.api_url}/chat/completions"
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        logger.debug(f"Sending chat request to FARMI...")
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            return generated_text.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FARMI chat API error: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test if the API connection is working."""
        try:
            response = self.generate("Say 'OK' if you can hear me.", max_tokens=10)
            logger.info(f"FARMI connection test successful: {response[:50]}")
            return True
        except Exception as e:
            logger.error(f"FARMI connection test failed: {e}")
            return False


class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI API.
    
    Can be used as fallback or alternative to FARMI.
    """
    
    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.max_tokens = max_tokens
            self.temperature = temperature
            logger.info(f"OpenAI client initialized: model={model}")
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI completions."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using OpenAI chat completions."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature)
        )
        return response.choices[0].message.content.strip()


def create_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    """
    Factory function to create the appropriate LLM client.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM client instance
    """
    provider = config.get("LLM_PROVIDER", "openai_compatible")
    
    if provider == "openai_compatible" or provider == "farmi":
        return FARMIClient(
            api_url=config.get("FARMI_API_URL", os.environ.get("FARMI_API_URL", "https://gptlab.rd.tuni.fi/students/ollama/v1")),
            api_key=config.get("FARMI_API_KEY", "") or os.environ.get("FARMI_API_KEY", ""),
            model=config.get("LLM_MODEL", os.environ.get("LLM_MODEL", "llama3.3:70b")),
            max_tokens=config.get("MAX_NEW_TOKENS", 512),
            temperature=config.get("TEMPERATURE", 0.7)
        )
    elif provider == "openai":
        return OpenAIClient(
            api_key=config.get("OPENAI_API_KEY", ""),
            model=config.get("LLM_MODEL", "gpt-3.5-turbo"),
            max_tokens=config.get("MAX_NEW_TOKENS", 512),
            temperature=config.get("TEMPERATURE", 0.7)
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Example usage
if __name__ == "__main__":
    # Test FARMI client
    import os
    client = FARMIClient(
        api_url="https://gptlab.rd.tuni.fi/students/ollama/v1",
        api_key=os.environ.get("FARMI_API_KEY", ""),
        model="llama3.3:70b"
    )
    
    # Test connection
    if client.test_connection():
        # Test generation
        response = client.generate(
            "What is the capital of France? Answer in one sentence.",
            max_tokens=50
        )
        print(f"Response: {response}")
