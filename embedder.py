"""
embedder.py

Minimal embedding helper that loads GEMINI_API_KEY (or GOOGLE_API_KEY) from environment
and returns embeddings for given text using Google GenAI SDK.
"""

from __future__ import annotations

import os
from typing import List, Sequence

# Load .env if available (no-op if python-dotenv is not installed)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
except Exception as exc:
    raise ImportError(
        "The 'google-genai' package is required. Install with: pip install google-genai"
    ) from exc


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback to check GOOGLE_API_KEY as well
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not found in environment."
        )
    return genai.Client(api_key=api_key)


_client = None  # lazily initialized


def _client_singleton() -> genai.Client:
    global _client
    if _client is None:
        _client = _get_client()
    return _client


def get_embedding(text: str, model: str = "models/gemini-embedding-001") -> List[float]:
    """Return embedding vector for a single text.

    - Normalizes newlines.
    - Raises ValueError for empty input.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        raise ValueError("text is empty after trimming")
    client = _client_singleton()
    # output_dimensionality=768 is default for text-embedding-004, but good to be explicit if needed.
    # Weaviate collections must match this dimension.
    resp = client.models.embed_content(model=model, contents=cleaned)
    return resp.embeddings[0].values


def get_embeddings(texts: Sequence[str], model: str = "models/gemini-embedding-001") -> List[List[float]]:
    """Return embeddings for a sequence of texts."""
    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts must be a list or tuple of strings")
    cleaned_list: List[str] = []
    for item in texts:
        if not isinstance(item, str):
            raise ValueError("each item in texts must be a string")
        cleaned = item.replace("\n", " ").strip()
        if not cleaned:
            raise ValueError("one of the texts is empty after trimming")
        cleaned_list.append(cleaned)
    client = _client_singleton()
    # Batch embedding
    # The SDK supports batching via embed_content with list of contents? 
    # Actually client.models.embed_content 'contents' arg can be list of strings.
    # Let's verify SDK signature or assume 'contents' takes str or List[str].
    # Based on google-genai docs, it takes 'contents'.
    resp = client.models.embed_content(model=model, contents=cleaned_list)
    
    # resp.embeddings is a list of ContentEmbedding objects
    return [e.values for e in resp.embeddings]


if __name__ == "__main__":
    sample = "Hello world. This is an embedding test."
    vec = get_embedding(sample)
    print(f"Embedding length: {len(vec)}")
    print(vec[:8], "...")


