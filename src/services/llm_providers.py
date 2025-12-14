"""
LLM Provider integrations.

Handles communication with different LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (Local models like LLaMA, Mistral) - FREE!
"""

import asyncio
import time
import httpx
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..config import get_settings


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for GPT models."""
    
    def __init__(self, model: str = "gpt-4"):
        self.settings = get_settings()
        self.model = model
        self.api_key = self.settings.openai_api_key
        self.base_url = "https://api.openai.com/v1"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception:
            return False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider for Claude models."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        self.settings = get_settings()
        self.model = model
        self.api_key = self.settings.anthropic_api_key
        self.base_url = "https://api.anthropic.com/v1"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        start_time = time.time()
        
        request_body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return LLMResponse(
            content=data["content"][0]["text"],
            model=self.model,
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def is_available(self) -> bool:
        if not self.api_key:
            return False
        return True


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local models - FREE!"""
    
    def __init__(self, model: str = "llama3.2"):
        self.settings = get_settings()
        # Strip "ollama/" prefix if present
        self.model = model.replace("ollama/", "")
        self.base_url = self.settings.ollama_base_url
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        start_time = time.time()
        
        # Use chat endpoint for better responses
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            try:
                # Try chat endpoint first (better for conversation)
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                    timeout=300.0,  # 5 minutes timeout for slow machines
                )
                response.raise_for_status()
                data = response.json()
                
                content = data.get("message", {}).get("content", "")
                input_tokens = data.get("prompt_eval_count", len(prompt.split()))
                output_tokens = data.get("eval_count", len(content.split()))
                
            except httpx.TimeoutException:
                raise Exception(f"Ollama timeout - model '{self.model}' took too long to respond. Try a shorter query or simpler model.")
            except httpx.HTTPStatusError as e:
                raise Exception(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                # Try fallback to generate endpoint
                try:
                    full_prompt = prompt
                    if system_prompt:
                        full_prompt = f"{system_prompt}\n\n{prompt}"
                    
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": full_prompt,
                            "stream": False,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": temperature,
                            },
                        },
                        timeout=300.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    content = data.get("response", "")
                    input_tokens = data.get("prompt_eval_count", len(prompt.split()))
                    output_tokens = data.get("eval_count", len(content.split()))
                except Exception as fallback_error:
                    raise Exception(f"Ollama error with model '{self.model}': {str(fallback_error)}")
                output_tokens = data.get("eval_count", len(content.split()))
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return LLMResponse(
            content=content,
            model=f"ollama/{self.model}",
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"].split(":")[0] for m in data.get("models", [])]
                    # Check if our model is available
                    return self.model in models or any(self.model in m for m in models)
                return False
        except Exception:
            return False


class MockProvider(BaseLLMProvider):
    """Mock provider for testing without any API calls."""
    
    def __init__(self, model: str = "mock"):
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Simulate some latency
        await asyncio.sleep(0.1)
        
        # Generate a mock response based on the query
        if "python" in prompt.lower():
            response = "Python is a high-level, interpreted programming language known for its simple syntax and readability. It's widely used in web development, data science, AI, and automation."
        elif "api" in prompt.lower() or "rest" in prompt.lower():
            response = "REST APIs use HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources. They are stateless, scalable, and widely used for web services."
        elif "design" in prompt.lower() or "system" in prompt.lower():
            response = "When designing distributed systems, consider: 1) Scalability through horizontal scaling, 2) Reliability via replication, 3) Consistency vs Availability tradeoffs (CAP theorem), 4) Caching strategies, 5) Load balancing."
        else:
            response = f"This is a mock response for: {prompt[:100]}... In production, this would be answered by a real LLM."
        
        return LLMResponse(
            content=response,
            model="mock",
            input_tokens=len(prompt.split()),
            output_tokens=len(response.split()),
            latency_ms=100,
        )
    
    async def is_available(self) -> bool:
        return True


def get_provider(model: str) -> BaseLLMProvider:
    """
    Factory function to get the appropriate provider for a model.
    
    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-sonnet", "ollama/llama3.2")
    
    Returns:
        Appropriate LLM provider instance
    """
    model_lower = model.lower()
    
    # Check for Ollama models first (most common for free usage)
    if model_lower.startswith("ollama/") or model_lower in ["llama3.2", "llama3", "llama2", "mistral", "phi3", "tinyllama", "codellama"]:
        return OllamaProvider(model=model)
    elif model_lower.startswith("gpt"):
        return OpenAIProvider(model=model)
    elif model_lower.startswith("claude"):
        return AnthropicProvider(model=model)
    elif model_lower == "mock":
        return MockProvider()
    else:
        # Default to Ollama for unknown models (free!)
        return OllamaProvider(model=model)