"""
Pydantic models for request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class QueryComplexity(str, Enum):
    """Query complexity levels for routing decisions."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ChatRequest(BaseModel):
    """Incoming chat request from client."""
    message: str = Field(..., min_length=1, max_length=32000)
    conversation_id: Optional[str] = None
    force_model: Optional[str] = None  # Override routing
    skip_cache: bool = False
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is the capital of France?",
                    "conversation_id": "conv_123",
                    "force_model": None,
                    "skip_cache": False
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response returned to client."""
    response: str
    model_used: str
    complexity: QueryComplexity
    was_cached: bool
    cache_similarity: Optional[float] = None
    input_tokens: int
    output_tokens: int
    actual_cost: float
    baseline_cost: float  # Cost if GPT-4 was used
    latency_ms: int
    request_id: str


class ClassificationResult(BaseModel):
    """Result from query complexity classifier."""
    complexity: QueryComplexity
    confidence: float
    features: dict
    recommended_model: str


class CacheEntry(BaseModel):
    """Cached response entry."""
    query: str
    response: str
    model_used: str
    created_at: datetime
    hit_count: int = 0


class CostAnalytics(BaseModel):
    """Cost analytics for a time period."""
    period_start: datetime
    period_end: datetime
    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    actual_cost: float
    baseline_cost: float
    savings: float
    savings_percentage: float
    requests_by_model: dict
    requests_by_complexity: dict


class DailyStats(BaseModel):
    """Daily statistics summary."""
    date: str
    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    actual_cost: float
    baseline_cost: float
    savings: float
    savings_percentage: float
    avg_latency_ms: float


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    redis: str
    timestamp: datetime


class ModelInfo(BaseModel):
    """Information about available models."""
    name: str
    provider: str
    complexity_level: QueryComplexity
    input_cost_per_1k: float
    output_cost_per_1k: float
    available: bool
