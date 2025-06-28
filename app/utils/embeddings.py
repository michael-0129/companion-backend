"""
Embeddings utilities for semantic search and memory.
Uses the local intfloat/e5-small-v2 model via sentence-transformers for privacy, speed, and quality.
"""
from sentence_transformers import SentenceTransformer
from threading import Lock
from typing import List
import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "intfloat/e5-small-v2"
_model = None
_model_lock = Lock()

def get_embedding_model() -> SentenceTransformer:
    """
    Loads and returns the singleton SentenceTransformer model instance.
    Thread-safe and lazy-loaded.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info(f"Loading embedding model: {_MODEL_NAME}")
                _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_query(text: str) -> List[float]:
    """
    Generate an embedding for a query string using E5 prefixing and normalization.
    Args:
        text: The query text to embed.
    Returns:
        A list of floats representing the embedding vector.
    Raises:
        ValueError: If the input text is empty.
    """
    model = get_embedding_model()
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty query text.")
    return model.encode(f"query: {text}", normalize_embeddings=True).tolist()

def embed_passage(text: str) -> List[float]:
    """
    Generate an embedding for a passage/document string using E5 prefixing and normalization.
    Args:
        text: The passage or document text to embed.
    Returns:
        A list of floats representing the embedding vector.
    Raises:
        ValueError: If the input text is empty.
    """
    model = get_embedding_model()
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty passage text.")
    return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

def embed_queries(texts: List[str]) -> List[List[float]]:
    """
    Batch embed queries. Returns a list of embedding vectors.
    Args:
        texts: List of query strings.
    Returns:
        List of embedding vectors (one per input).
    Raises:
        ValueError: If no valid queries are provided.
    """
    model = get_embedding_model()
    texts = [f"query: {t.strip()}" for t in texts if t.strip()]
    if not texts:
        raise ValueError("No valid queries to embed.")
    return model.encode(texts, normalize_embeddings=True).tolist()

def embed_passages(texts: List[str]) -> List[List[float]]:
    """
    Batch embed passages. Returns a list of embedding vectors.
    Args:
        texts: List of passage/document strings.
    Returns:
        List of embedding vectors (one per input).
    Raises:
        ValueError: If no valid passages are provided.
    """
    model = get_embedding_model()
    texts = [f"passage: {t.strip()}" for t in texts if t.strip()]
    if not texts:
        raise ValueError("No valid passages to embed.")
    return model.encode(texts, normalize_embeddings=True).tolist()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
 