"""
Chat Router API Endpoint.

Main endpoint for processing chat requests through the SmartLLM Router.
"""

import time
import uuid
from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    ChatRequest,
    ChatResponse,
    QueryComplexity,
    ClassificationResult,
)
from ..services.complexity_classifier import classify_query_complexity
from ..services.llm_providers import get_provider, LLMResponse
from ..services.semantic_cache import get_semantic_cache
from ..services.cost_tracker import get_cost_tracker
from ..config import get_settings

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request through the SmartLLM Router.
    
    The router will:
    1. Check semantic cache for similar queries
    2. Classify query complexity if cache miss
    3. Route to appropriate model
    4. Track costs and return response
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    settings = get_settings()
    cache = get_semantic_cache()
    cost_tracker = get_cost_tracker()
    
    # Step 1: Check semantic cache (unless skip_cache is set)
    cached_response = None
    cache_similarity = None
    
    if not request.skip_cache:
        cached_response = await cache.get(request.message)
        if cached_response:
            cache_similarity = cached_response.similarity
    
    if cached_response:
        # Cache HIT - return cached response
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Still classify to know complexity for logging
        classification = classify_query_complexity(request.message)
        
        # Log the request (with $0 cost since cached)
        cost_tracker.log_request(
            query_text=request.message,
            query_complexity=classification.complexity.value,
            model_used=cached_response.model_used,
            was_cached=True,
            input_tokens=0,  # No tokens consumed
            output_tokens=0,
            latency_ms=latency_ms,
            cache_similarity=cache_similarity,
        )
        
        # Calculate what it would have cost
        from ..utils.token_counter import count_tokens, estimate_cost
        input_tokens = count_tokens(request.message)
        output_tokens = count_tokens(cached_response.response)
        baseline_cost = estimate_cost(input_tokens, output_tokens, "gpt-4", settings.pricing)
        
        return ChatResponse(
            response=cached_response.response,
            model_used=cached_response.model_used,
            complexity=classification.complexity,
            was_cached=True,
            cache_similarity=cache_similarity,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            actual_cost=0.0,
            baseline_cost=baseline_cost,
            latency_ms=latency_ms,
            request_id=request_id,
        )
    
    # Step 2: Cache MISS - classify query complexity
    classification = classify_query_complexity(request.message)
    
    # Step 3: Determine which model to use
    if request.force_model:
        model_to_use = request.force_model
    else:
        model_to_use = classification.recommended_model
    
    # Step 4: Get response from LLM
    try:
        provider = get_provider(model_to_use)
        
        # Check if provider is available
        if not await provider.is_available():
            # Fallback to a different model
            if model_to_use != settings.default_medium_model:
                model_to_use = settings.default_medium_model
                provider = get_provider(model_to_use)
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model_to_use} is not available"
                )
        
        llm_response = await provider.generate(
            prompt=request.message,
            system_prompt="You are a helpful AI assistant. Be concise and accurate.",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )
    
    # Step 5: Cache the response for future similar queries
    await cache.set(
        query=request.message,
        response=llm_response.content,
        model_used=llm_response.model,
    )
    
    # Step 6: Calculate costs
    from ..utils.token_counter import estimate_cost
    
    actual_cost = estimate_cost(
        llm_response.input_tokens,
        llm_response.output_tokens,
        llm_response.model,
        settings.pricing
    )
    
    baseline_cost = estimate_cost(
        llm_response.input_tokens,
        llm_response.output_tokens,
        "gpt-4",
        settings.pricing
    )
    
    latency_ms = llm_response.latency_ms
    
    # Step 7: Log the request
    cost_tracker.log_request(
        query_text=request.message,
        query_complexity=classification.complexity.value,
        model_used=llm_response.model,
        was_cached=False,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        latency_ms=latency_ms,
        cache_similarity=None,
    )
    
    return ChatResponse(
        response=llm_response.content,
        model_used=llm_response.model,
        complexity=classification.complexity,
        was_cached=False,
        cache_similarity=None,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        latency_ms=latency_ms,
        request_id=request_id,
    )


@router.post("/classify", response_model=ClassificationResult)
async def classify_query(request: ChatRequest) -> ClassificationResult:
    """
    Classify a query's complexity without generating a response.
    
    Useful for testing and understanding the routing logic.
    """
    return classify_query_complexity(request.message)


@router.get("/models")
async def list_models():
    """List available models and their configurations."""
    settings = get_settings()
    
    models = []
    for model, pricing in settings.pricing.items():
        complexity = "complex" if "gpt-4" in model or "opus" in model else \
                    "simple" if "ollama" in model else "medium"
        
        models.append({
            "name": model,
            "complexity_level": complexity,
            "input_cost_per_1k": pricing["input"],
            "output_cost_per_1k": pricing["output"],
        })
    
    return {"models": models}
