"""
Semantic Cache Service.

Caches responses based on semantic similarity of queries.
Uses embeddings + vector similarity search to find cached responses
for semantically similar queries.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..config import get_settings


@dataclass
class CachedResponse:
    """A cached response entry."""
    query: str
    response: str
    model_used: str
    created_at: datetime
    similarity: float = 1.0


class SemanticCache:
    """
    Semantic caching using embeddings for similarity matching.
    
    For MVP, this uses an in-memory store with simple embeddings.
    In production, this would use Redis + ChromaDB/Qdrant.
    """
    
    def __init__(self, similarity_threshold: float = 0.92):
        self.settings = get_settings()
        self.threshold = similarity_threshold
        self.ttl_hours = self.settings.cache_ttl_hours
        
        # In-memory stores (replace with Redis + vector DB in production)
        self._cache: dict = {}  # cache_key -> CachedResponse
        self._embeddings: List[Tuple[str, List[float]]] = []  # (cache_key, embedding)
        
        # Lazy load embedding model
        self._model = None
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to simple hash-based matching if sentence-transformers not available
                self._model = "hash_fallback"
        return self._model
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model = self._get_model()
        
        if model == "hash_fallback":
            # Simple fallback: use character n-gram frequencies
            # This won't be as good but allows testing without sentence-transformers
            from collections import Counter
            ngrams = [text[i:i+3].lower() for i in range(len(text)-2)]
            counts = Counter(ngrams)
            # Create a simple 100-dim vector
            vector = [0.0] * 100
            for ngram, count in counts.items():
                idx = hash(ngram) % 100
                vector[idx] += count
            # Normalize
            norm = sum(v*v for v in vector) ** 0.5
            if norm > 0:
                vector = [v/norm for v in vector]
            return vector
        else:
            return model.encode(text).tolist()
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(x*x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a unique cache key for a query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def _is_expired(self, cached: CachedResponse) -> bool:
        """Check if a cached entry has expired."""
        expiry = cached.created_at + timedelta(hours=self.ttl_hours)
        return datetime.utcnow() > expiry
    
    async def get(self, query: str) -> Optional[CachedResponse]:
        """
        Try to find a cached response for a semantically similar query.
        
        Returns CachedResponse if found (with similarity score), None otherwise.
        """
        if not self._embeddings:
            return None
        
        query_embedding = self._generate_embedding(query)
        
        best_match: Optional[Tuple[str, float]] = None
        best_similarity = 0.0
        
        # Find most similar cached query
        for cache_key, cached_embedding in self._embeddings:
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (cache_key, similarity)
        
        # Check if similarity meets threshold
        if best_match and best_similarity >= self.threshold:
            cache_key, similarity = best_match
            cached = self._cache.get(cache_key)
            
            if cached and not self._is_expired(cached):
                # Return with similarity score
                return CachedResponse(
                    query=cached.query,
                    response=cached.response,
                    model_used=cached.model_used,
                    created_at=cached.created_at,
                    similarity=similarity,
                )
            elif cached:
                # Expired - remove from cache
                await self.invalidate(cache_key)
        
        return None
    
    async def set(
        self,
        query: str,
        response: str,
        model_used: str,
    ) -> str:
        """
        Cache a response for a query.
        
        Returns the cache key.
        """
        cache_key = self._generate_cache_key(query)
        embedding = self._generate_embedding(query)
        
        cached = CachedResponse(
            query=query,
            response=response,
            model_used=model_used,
            created_at=datetime.utcnow(),
        )
        
        self._cache[cache_key] = cached
        self._embeddings.append((cache_key, embedding))
        
        return cache_key
    
    async def invalidate(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry."""
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._embeddings = [
                (k, e) for k, e in self._embeddings if k != cache_key
            ]
            return True
        return False
    
    async def clear(self) -> int:
        """Clear all cache entries. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        self._embeddings.clear()
        return count
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        expired_count = sum(
            1 for cached in self._cache.values()
            if self._is_expired(cached)
        )
        
        return {
            "total_entries": len(self._cache),
            "active_entries": len(self._cache) - expired_count,
            "expired_entries": expired_count,
            "similarity_threshold": self.threshold,
            "ttl_hours": self.ttl_hours,
        }


# Global cache instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance."""
    global _semantic_cache
    if _semantic_cache is None:
        settings = get_settings()
        _semantic_cache = SemanticCache(
            similarity_threshold=settings.cache_similarity_threshold
        )
    return _semantic_cache
