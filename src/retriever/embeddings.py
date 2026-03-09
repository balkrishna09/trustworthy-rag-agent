"""
Embedding Generator for Trustworthy RAG

Supports two backends:
  - Local (sentence-transformers): default, runs on CPU/GPU, no API needed.
  - Ollama API: calls the FARMI GPT Lab API, enabling models like
    snowflake-arctic-embed2 that are not available as local packages.

Backend selection is controlled by the EMBEDDING_BACKEND environment variable:
  EMBEDDING_BACKEND=local   -> EmbeddingGenerator (sentence-transformers)
  EMBEDDING_BACKEND=api     -> OllamaEmbeddingGenerator (FARMI API)

Use create_embedding_generator(config) to get the correct instance automatically.
"""

import os
import requests
import numpy as np
from typing import List, Union, Optional
from loguru import logger
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Local backend (sentence-transformers) — unchanged from original
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """
    Generates embeddings using local sentence-transformers models.

    Runs entirely on CPU (or GPU if available). No API key required.
    Default model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional).

    Attributes:
        model_name: Name of the sentence-transformer model.
        model: Loaded SentenceTransformer instance.
        embedding_dimension: Output vector dimensionality.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the local embedding generator.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
        """
        self.model_name = model_name
        logger.info(f"[Local] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"[Local] Embedding dimension: {self.embedding_dimension}")

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (num_docs, embedding_dim).
        """
        logger.info(f"[Local] Embedding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.info(f"[Local] Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.

        Args:
            query: Query text.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        return self.model.encode(query, convert_to_numpy=True)

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one or more texts (convenience wrapper).

        Args:
            texts: Single string or list of strings.

        Returns:
            numpy array of embeddings.
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)

    def get_dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self.embedding_dimension


# ---------------------------------------------------------------------------
# API backend (Ollama / FARMI GPT Lab)
# ---------------------------------------------------------------------------

class OllamaEmbeddingGenerator:
    """
    Generates embeddings via the FARMI GPT Lab Ollama-compatible API.

    Supports any embedding model hosted on the FARMI cluster, including:
      - snowflake-arctic-embed2:latest  (1024-dimensional, state-of-the-art retrieval)
      - nomic-embed-text:latest         (768-dimensional)
      - nomic-embed-text-v2-moe:latest  (768-dimensional, MoE variant)

    The API follows the OpenAI embeddings format:
      POST /v1/embeddings  {"model": "<name>", "input": "<text>"}

    Attributes:
        model_name: Ollama model identifier (e.g. "snowflake-arctic-embed2:latest").
        api_url: Base URL for the FARMI API.
        api_key: Bearer token for authentication.
        embedding_dimension: Auto-detected on first call.
    """

    def __init__(
        self,
        model_name: str = "snowflake-arctic-embed2:latest",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the API-based embedding generator.

        Args:
            model_name: Ollama model identifier.
            api_url: FARMI API base URL. Falls back to FARMI_API_URL env var.
            api_key: API bearer token. Falls back to FARMI_API_KEY env var.
        """
        self.model_name = model_name
        self.api_url = (api_url or os.environ.get("FARMI_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("FARMI_API_KEY", "")

        if not self.api_url:
            raise ValueError(
                "FARMI_API_URL not set. Add it to .env or pass api_url explicitly."
            )
        if not self.api_key:
            raise ValueError(
                "FARMI_API_KEY not set. Add it to .env or pass api_key explicitly."
            )

        self._endpoint = f"{self.api_url}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Auto-detect dimension with a probe call
        logger.info(f"[API] Probing embedding model: {model_name}")
        probe = self._call_api("probe")
        self.embedding_dimension = len(probe)
        logger.info(f"[API] Model: {model_name} | Dimension: {self.embedding_dimension}")

    # ------------------------------------------------------------------
    # Public interface (same as EmbeddingGenerator)
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents via the API.

        Documents are sent one at a time (Ollama API does not guarantee
        batched input support across all models). Progress is logged every
        10 documents when show_progress=True.

        Args:
            documents: List of document texts.
            batch_size: Ignored (kept for interface compatibility).
            show_progress: Whether to log progress.

        Returns:
            numpy array of shape (num_docs, embedding_dim).
        """
        logger.info(f"[API] Embedding {len(documents)} documents via {self.model_name}...")
        vectors = []
        for i, doc in enumerate(documents):
            vec = self._call_api(doc)
            vectors.append(vec)
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"[API] Embedded {i + 1}/{len(documents)} documents")

        embeddings = np.array(vectors, dtype=np.float32)
        logger.info(f"[API] Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string via the API.

        Args:
            query: Query text.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        return np.array(self._call_api(query), dtype=np.float32)

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one or more texts (convenience wrapper).

        Args:
            texts: Single string or list of strings.

        Returns:
            numpy array of embeddings.
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)

    def get_dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self.embedding_dimension

    # ------------------------------------------------------------------
    # Internal API call
    # ------------------------------------------------------------------

    def _call_api(self, text: str) -> List[float]:
        """
        Call the FARMI embeddings API for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            RuntimeError: If the API call fails.
        """
        payload = {"model": self.model_name, "input": text}
        try:
            response = requests.post(
                self._endpoint,
                headers=self._headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # OpenAI-compatible response: data[0]["embedding"]
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]

            # Ollama native response: data["embedding"]
            if "embedding" in data:
                return data["embedding"]

            raise RuntimeError(f"Unexpected API response format: {list(data.keys())}")

        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"API timeout embedding text (len={len(text)}). "
                "Check FARMI cluster availability."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"API HTTP error: {e} | Response: {response.text[:200]}")
        except Exception as e:
            raise RuntimeError(f"API embedding failed: {e}")


# ---------------------------------------------------------------------------
# Factory function — use this everywhere instead of instantiating directly
# ---------------------------------------------------------------------------

def create_embedding_generator(
    config: Optional[dict] = None,
) -> Union[EmbeddingGenerator, OllamaEmbeddingGenerator]:
    """
    Factory that returns the appropriate embedding generator based on config.

    Selection logic (in priority order):
      1. config["EMBEDDING_BACKEND"] == "api"  -> OllamaEmbeddingGenerator
      2. env EMBEDDING_BACKEND == "api"         -> OllamaEmbeddingGenerator
      3. Anything else                          -> EmbeddingGenerator (local)

    The model name is read from config["EMBEDDING_MODEL"] or the
    EMBEDDING_MODEL environment variable.

    Args:
        config: Optional config dictionary (from .env / config.yaml).

    Returns:
        An embedding generator instance with the standard interface.

    Example::

        # Local (default):
        gen = create_embedding_generator({"EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"})

        # API (snowflake):
        gen = create_embedding_generator({
            "EMBEDDING_BACKEND": "api",
            "EMBEDDING_MODEL": "snowflake-arctic-embed2:latest",
        })
    """
    config = config or {}

    backend = config.get(
        "EMBEDDING_BACKEND",
        os.environ.get("EMBEDDING_BACKEND", "local"),
    ).lower()

    model_name = config.get(
        "EMBEDDING_MODEL",
        os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    if backend == "api":
        logger.info(f"Embedding backend: API ({model_name})")
        return OllamaEmbeddingGenerator(
            model_name=model_name,
            api_url=config.get("FARMI_API_URL") or os.environ.get("FARMI_API_URL"),
            api_key=config.get("FARMI_API_KEY") or os.environ.get("FARMI_API_KEY"),
        )
    else:
        logger.info(f"Embedding backend: local ({model_name})")
        return EmbeddingGenerator(model_name=model_name)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "local"

    if mode == "api":
        from dotenv import load_dotenv
        load_dotenv()
        gen = create_embedding_generator({
            "EMBEDDING_BACKEND": "api",
            "EMBEDDING_MODEL": "snowflake-arctic-embed2:latest",
        })
    else:
        gen = create_embedding_generator()

    docs = [
        "Paris is the capital of France.",
        "London is the capital of the United Kingdom.",
        "The Eiffel Tower is located in Paris.",
    ]
    embs = gen.embed_documents(docs)
    print(f"Document embeddings shape: {embs.shape}")

    q_emb = gen.embed_query("What is the capital of France?")
    print(f"Query embedding shape: {q_emb.shape}")

    from numpy.linalg import norm
    sims = np.dot(embs, q_emb) / (norm(embs, axis=1) * norm(q_emb))
    print(f"Similarities: {sims.round(4)}")
    print(f"Most similar: {docs[np.argmax(sims)]}")
