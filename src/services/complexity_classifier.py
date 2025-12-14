"""
Query Complexity Classifier.

Analyzes incoming queries and determines their complexity level
to route them to the optimal model.
"""

import re
from typing import Dict, Set
from ..models.schemas import QueryComplexity, ClassificationResult
from ..config import get_settings


# Technical terms that indicate complex queries
TECHNICAL_TERMS: Set[str] = {
    # Programming
    "algorithm", "recursion", "polymorphism", "inheritance", "encapsulation",
    "middleware", "microservices", "kubernetes", "docker", "api", "rest",
    "graphql", "database", "sql", "nosql", "cache", "redis", "async",
    "concurrent", "thread", "process", "memory", "garbage collection",
    "optimization", "complexity", "big-o", "data structure", "binary tree",
    "hash map", "linked list", "queue", "stack", "heap", "graph",
    
    # ML/AI
    "machine learning", "neural network", "deep learning", "transformer",
    "attention mechanism", "gradient descent", "backpropagation", "embedding",
    "fine-tuning", "llm", "gpt", "bert", "tokenization", "vector",
    
    # System Design
    "scalability", "load balancer", "sharding", "replication", "consistency",
    "availability", "partition tolerance", "cap theorem", "distributed",
    "consensus", "raft", "paxos", "eventual consistency", "latency",
    "throughput", "fault tolerance", "failover", "redundancy",
    
    # Security
    "encryption", "authentication", "authorization", "oauth", "jwt",
    "ssl", "tls", "vulnerability", "injection", "xss", "csrf",
}

# System design keywords - these are STRONG indicators of complex queries
SYSTEM_DESIGN_TERMS: Set[str] = {
    "design", "architect", "scale", "million", "billion", "distributed",
    "high availability", "fault tolerant", "load balance", "microservice",
    "tradeoff", "trade-off", "system design", "architecture",
    "requests per second", "rps", "qps", "queries per second",
    "database schema", "api design", "caching strategy", "message queue",
}

# Reasoning indicators that suggest complex analysis
REASONING_PATTERNS = [
    r'\bwhy\b',
    r'\bhow\s+(?:does|do|can|could|would|should)\b',
    r'\bexplain\b',
    r'\banalyze\b',
    r'\bcompare\b',
    r'\bcontrast\b',
    r'\bevaluate\b',
    r'\bdesign\b',
    r'\barchitect\b',
    r'\bimplement\b',
    r'\boptimize\b',
    r'\bdebug\b',
    r'\btroubleshoot\b',
    r'\bwhat\s+(?:are|is)\s+the\s+(?:best|optimal|most\s+efficient)\b',
    r'\btradeoff\b',
    r'\btrade-off\b',
]

# Multi-step task indicators
MULTI_STEP_PATTERNS = [
    r'\bfirst\b.*\bthen\b',
    r'\bstep\s*(?:by\s*step|1|one)\b',
    r'\bafter\s+that\b',
    r'\bfinally\b',
    r'\bnext\b',
    r'\bfollow(?:ing|ed)\s+by\b',
    r'\band\s+then\b',
]

# Code indicators
CODE_PATTERNS = [
    r'```',
    r'\bdef\s+\w+\s*\(',
    r'\bclass\s+\w+',
    r'\bfunction\s+\w+',
    r'\bconst\s+\w+\s*=',
    r'\blet\s+\w+\s*=',
    r'\bvar\s+\w+\s*=',
    r'\bimport\s+',
    r'\bfrom\s+\w+\s+import\b',
    r'\breturn\s+',
    r'<[a-zA-Z][^>]*>',  # HTML/XML tags
    r'\{\s*"?\w+"?\s*:',  # JSON-like
]


def count_technical_terms(query: str) -> int:
    """Count the number of technical terms in the query."""
    query_lower = query.lower()
    count = 0
    for term in TECHNICAL_TERMS:
        if term in query_lower:
            count += 1
    return count


def count_system_design_terms(query: str) -> int:
    """Count system design specific terms - strong complexity indicator."""
    query_lower = query.lower()
    count = 0
    for term in SYSTEM_DESIGN_TERMS:
        if term in query_lower:
            count += 1
    return count


def has_pattern_match(query: str, patterns: list) -> bool:
    """Check if query matches any of the given patterns."""
    for pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def count_pattern_matches(query: str, patterns: list) -> int:
    """Count how many patterns match in the query."""
    count = 0
    for pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            count += 1
    return count


def classify_query_complexity(query: str) -> ClassificationResult:
    """
    Classify the complexity of a query.
    
    Returns a ClassificationResult with:
    - complexity: simple, medium, or complex
    - confidence: 0.0 to 1.0
    - features: dict of extracted features
    - recommended_model: the model to use
    """
    settings = get_settings()
    
    # Extract features
    words = query.split()
    word_count = len(words)
    char_count = len(query)
    
    features = {
        "word_count": word_count,
        "char_count": char_count,
        "has_code": has_pattern_match(query, CODE_PATTERNS),
        "reasoning_indicators": count_pattern_matches(query, REASONING_PATTERNS),
        "technical_term_count": count_technical_terms(query),
        "system_design_terms": count_system_design_terms(query),
        "is_multi_step": has_pattern_match(query, MULTI_STEP_PATTERNS),
        "has_question_mark": "?" in query,
        "sentence_count": len(re.findall(r'[.!?]+', query)) + 1,
    }
    
    # Calculate complexity score
    score = 0
    
    # Length-based scoring
    if word_count > 100:
        score += 2
    elif word_count > 30:
        score += 1
    
    # Code presence is a strong signal
    if features["has_code"]:
        score += 2
    
    # Reasoning complexity
    if features["reasoning_indicators"] >= 3:
        score += 2
    elif features["reasoning_indicators"] >= 1:
        score += 1
    
    # Technical depth
    if features["technical_term_count"] >= 5:
        score += 2
    elif features["technical_term_count"] >= 2:
        score += 1
    
    # SYSTEM DESIGN - Strong indicator for complex (NEW!)
    if features["system_design_terms"] >= 3:
        score += 3  # Big boost for system design questions
    elif features["system_design_terms"] >= 1:
        score += 2
    
    # Multi-step tasks
    if features["is_multi_step"]:
        score += 1
    
    # Multiple sentences often indicate complex requests
    if features["sentence_count"] >= 3:
        score += 1
    
    # Store score in features for debugging
    features["total_score"] = score
    
    # Determine complexity level (ADJUSTED THRESHOLDS)
    if score >= 4:  # Lowered from 5 to 4
        complexity = QueryComplexity.COMPLEX
        recommended_model = settings.default_complex_model
        confidence = min(0.95, 0.7 + (score - 4) * 0.05)
    elif score >= 2:
        complexity = QueryComplexity.MEDIUM
        recommended_model = settings.default_medium_model
        confidence = 0.75 + (score - 2) * 0.05
    else:
        complexity = QueryComplexity.SIMPLE
        recommended_model = settings.default_simple_model
        confidence = 0.85 + (2 - score) * 0.05
    
    # Cap confidence at 0.95
    confidence = min(0.95, confidence)
    
    return ClassificationResult(
        complexity=complexity,
        confidence=confidence,
        features={**features, "total_score": score},
        recommended_model=recommended_model,
    )


# Example usage and testing
if __name__ == "__main__":
    test_queries = [
        # Simple queries
        "What is Python?",
        "Convert 100 USD to EUR",
        "What's 15% of 230?",
        
        # Medium queries
        "Explain the difference between REST and GraphQL APIs",
        "Write a function to reverse a string in Python",
        "What are the pros and cons of using Redis vs Memcached?",
        
        # Complex queries
        "Design a distributed cache system for a social media platform that handles 10 million requests per second with low latency and high availability",
        """Analyze this code for security vulnerabilities and suggest fixes:
        ```python
        def login(username, password):
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            return db.execute(query)
        ```""",
        "Explain how transformer attention mechanisms work, including multi-head attention, and compare them to RNN-based sequence models in terms of parallelization and long-range dependencies",
    ]
    
    print("Query Complexity Classification Tests\n" + "=" * 50)
    for query in test_queries:
        result = classify_query_complexity(query)
        print(f"\nQuery: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"  Complexity: {result.complexity.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Model: {result.recommended_model}")
        print(f"  Score: {result.features['total_score']}")