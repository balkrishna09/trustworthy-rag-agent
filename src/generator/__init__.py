# Generator Module
"""
LLM response generation components.

Components:
- FARMIClient: Client for FARMI GPTLab API
- OpenAIClient: Client for OpenAI API (fallback)
- Generator: Main generator class for RAG responses
- Prompt templates for various tasks
"""

from .llm_client import FARMIClient, OpenAIClient, BaseLLMClient, create_llm_client
from .generator import Generator
from .prompts import (
    create_rag_prompt,
    create_fact_check_prompt,
    format_context,
    RAG_PROMPT,
    RAG_PROMPT_CONCISE,
    RAG_PROMPT_DETAILED,
    FACT_CHECK_PROMPT
)

__all__ = [
    'FARMIClient',
    'OpenAIClient',
    'BaseLLMClient',
    'create_llm_client',
    'Generator',
    'create_rag_prompt',
    'create_fact_check_prompt',
    'format_context',
    'RAG_PROMPT',
    'RAG_PROMPT_CONCISE',
    'RAG_PROMPT_DETAILED',
    'FACT_CHECK_PROMPT'
]
