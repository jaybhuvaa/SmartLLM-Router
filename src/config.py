"""
Configuration management for SmartLLM Router.
Uses pydantic-settings for type-safe environment variable handling.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Keys (optional - not needed for Ollama)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/smartllm"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Caching
    cache_similarity_threshold: float = 0.92
    cache_ttl_hours: int = 24
    
    # Model Configuration - Default to Ollama (FREE!)
    default_simple_model: str = "ollama/llama3.2"
    default_medium_model: str = "ollama/llama3.2"
    default_complex_model: str = "ollama/llama3.2"
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    
    # Model Pricing (per 1K tokens)
    # Ollama models are FREE - $0 cost!
    pricing: dict = {
        # Paid models (for reference/comparison)
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        # FREE Ollama models
        "ollama/llama3.2": {"input": 0.0, "output": 0.0},
        "ollama/llama3": {"input": 0.0, "output": 0.0},
        "ollama/mistral": {"input": 0.0, "output": 0.0},
        "ollama/phi3": {"input": 0.0, "output": 0.0},
        "ollama/tinyllama": {"input": 0.0, "output": 0.0},
        "ollama/codellama": {"input": 0.0, "output": 0.0},
        "mock": {"input": 0.0, "output": 0.0},
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
