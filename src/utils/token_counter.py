"""
Token counting utilities for different model providers.
"""

from typing import Tuple
import tiktoken


# Cache encodings for performance
_encodings = {}


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the appropriate tiktoken encoding for a model."""
    if model in _encodings:
        return _encodings[model]
    
    try:
        if model.startswith("gpt-4"):
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif model.startswith("gpt-3.5"):
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Default to cl100k_base for other models (Claude, etc.)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        _encodings[model] = encoding
        return encoding
    except Exception:
        # Fallback to cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")
        _encodings[model] = encoding
        return encoding


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model name to use for encoding
        
    Returns:
        Number of tokens
    """
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_message_tokens(messages: list, model: str = "gpt-4") -> int:
    """
    Count tokens for a list of chat messages.
    
    Accounts for the overhead of message formatting.
    """
    encoding = get_encoding(model)
    
    # Token overhead per message (role, content separators)
    tokens_per_message = 4
    
    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(encoding.encode(str(value)))
    
    # Every reply is primed with <|start|>assistant<|message|>
    total += 3
    
    return total


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    pricing: dict
) -> float:
    """
    Calculate the cost for a request.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        pricing: Dict of model pricing {model: {input: price, output: price}}
        
    Returns:
        Cost in dollars
    """
    if model not in pricing:
        # Default to GPT-4 pricing if unknown
        model_pricing = pricing.get("gpt-4", {"input": 0.03, "output": 0.06})
    else:
        model_pricing = pricing[model]
    
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    return input_cost + output_cost


def calculate_savings(
    input_tokens: int,
    output_tokens: int,
    actual_model: str,
    baseline_model: str,
    pricing: dict
) -> Tuple[float, float, float]:
    """
    Calculate cost savings compared to baseline model.
    
    Returns:
        Tuple of (actual_cost, baseline_cost, savings)
    """
    actual_cost = estimate_cost(input_tokens, output_tokens, actual_model, pricing)
    baseline_cost = estimate_cost(input_tokens, output_tokens, baseline_model, pricing)
    savings = baseline_cost - actual_cost
    
    return actual_cost, baseline_cost, savings
