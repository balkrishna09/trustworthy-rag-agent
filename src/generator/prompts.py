"""
Prompt Templates for Trustworthy RAG
Defines prompts for RAG generation and evaluation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A template for generating prompts."""
    template: str
    input_variables: List[str]
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)


# =============================================================================
# RAG Generation Prompts
# =============================================================================

RAG_PROMPT = PromptTemplate(
    template="""You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

RAG_PROMPT_CONCISE = PromptTemplate(
    template="""Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer (be concise):""",
    input_variables=["context", "question"]
)

RAG_PROMPT_DETAILED = PromptTemplate(
    template="""You are a knowledgeable assistant. Based on the provided context, give a detailed and accurate answer to the question.

Context:
{context}

Question: {question}

Instructions:
1. Only use information from the context
2. If the context is insufficient, clearly state what information is missing
3. Cite specific parts of the context when possible

Detailed Answer:""",
    input_variables=["context", "question"]
)


# =============================================================================
# Fact Verification Prompts
# =============================================================================

FACT_CHECK_PROMPT = PromptTemplate(
    template="""You are a fact-checking assistant. Determine if the claim is supported by the evidence.

Evidence: {evidence}

Claim: {claim}

Based on the evidence, is this claim:
- SUPPORTED: The evidence confirms the claim
- REFUTED: The evidence contradicts the claim
- NOT ENOUGH INFO: The evidence neither confirms nor contradicts the claim

Your verdict (SUPPORTED/REFUTED/NOT ENOUGH INFO):""",
    input_variables=["evidence", "claim"]
)


# =============================================================================
# Trustworthiness Assessment Prompts
# =============================================================================

TRUST_ASSESSMENT_PROMPT = PromptTemplate(
    template="""Analyze the following response for potential issues:

Question: {question}
Context Used: {context}
Generated Response: {response}

Evaluate:
1. Does the response accurately reflect the context?
2. Are there any unsupported claims?
3. Is there any potentially misleading information?

Assessment:""",
    input_variables=["question", "context", "response"]
)


# =============================================================================
# Helper Functions
# =============================================================================

def format_context(documents: List[str], separator: str = "\n\n") -> str:
    """
    Format a list of documents into a context string.
    
    Args:
        documents: List of document texts
        separator: Separator between documents
        
    Returns:
        Formatted context string
    """
    return separator.join([f"[{i+1}] {doc}" for i, doc in enumerate(documents)])


def format_context_with_metadata(
    documents: List[Dict[str, Any]],
    content_key: str = "content",
    metadata_keys: Optional[List[str]] = None
) -> str:
    """
    Format documents with their metadata.
    
    Args:
        documents: List of document dicts with content and metadata
        content_key: Key for document content
        metadata_keys: Keys to include from metadata
        
    Returns:
        Formatted context string
    """
    formatted = []
    metadata_keys = metadata_keys or []
    
    for i, doc in enumerate(documents):
        content = doc.get(content_key, "")
        meta_str = ""
        
        if metadata_keys:
            meta_parts = [f"{k}: {doc.get(k, 'N/A')}" for k in metadata_keys if k in doc]
            if meta_parts:
                meta_str = f" ({', '.join(meta_parts)})"
        
        formatted.append(f"[{i+1}]{meta_str} {content}")
    
    return "\n\n".join(formatted)


def create_rag_prompt(
    question: str,
    context_docs: List[str],
    prompt_style: str = "default"
) -> str:
    """
    Create a RAG prompt with the given question and context.
    
    Args:
        question: User's question
        context_docs: List of relevant document texts
        prompt_style: Style of prompt ("default", "concise", "detailed")
        
    Returns:
        Formatted prompt string
    """
    context = format_context(context_docs)
    
    if prompt_style == "concise":
        return RAG_PROMPT_CONCISE.format(context=context, question=question)
    elif prompt_style == "detailed":
        return RAG_PROMPT_DETAILED.format(context=context, question=question)
    else:
        return RAG_PROMPT.format(context=context, question=question)


def create_fact_check_prompt(claim: str, evidence: str) -> str:
    """
    Create a fact-checking prompt.
    
    Args:
        claim: The claim to verify
        evidence: The evidence to check against
        
    Returns:
        Formatted prompt string
    """
    return FACT_CHECK_PROMPT.format(claim=claim, evidence=evidence)


# Example usage
if __name__ == "__main__":
    # Test RAG prompt
    context_docs = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris."
    ]
    question = "What is the capital of France?"
    
    prompt = create_rag_prompt(question, context_docs)
    print("RAG Prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Test fact check prompt
    claim = "Paris is the capital of Germany."
    evidence = "Paris is the capital of France. Berlin is the capital of Germany."
    
    prompt = create_fact_check_prompt(claim, evidence)
    print("Fact Check Prompt:")
    print(prompt)
