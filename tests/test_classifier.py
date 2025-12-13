"""
Tests for the Query Complexity Classifier.
"""

import pytest
from src.services.complexity_classifier import (
    classify_query_complexity,
    count_technical_terms,
    has_pattern_match,
    CODE_PATTERNS,
    REASONING_PATTERNS,
)
from src.models.schemas import QueryComplexity


class TestComplexityClassifier:
    """Test suite for query complexity classification."""
    
    def test_simple_queries(self):
        """Simple factual queries should be classified as simple."""
        simple_queries = [
            "What is Python?",
            "Convert 100 USD to EUR",
            "What's 15% of 230?",
            "Hello!",
            "What time is it?",
            "Who is the president?",
        ]
        
        for query in simple_queries:
            result = classify_query_complexity(query)
            assert result.complexity == QueryComplexity.SIMPLE, \
                f"Expected SIMPLE for: {query}, got {result.complexity}"
    
    def test_medium_queries(self):
        """Moderately complex queries should be classified as medium."""
        medium_queries = [
            "Explain the difference between REST and GraphQL APIs",
            "What are the pros and cons of using Redis?",
            "How does Python's garbage collection work?",
            "Write a function to reverse a string",
        ]
        
        for query in medium_queries:
            result = classify_query_complexity(query)
            assert result.complexity in [QueryComplexity.MEDIUM, QueryComplexity.COMPLEX], \
                f"Expected MEDIUM/COMPLEX for: {query}, got {result.complexity}"
    
    def test_complex_queries(self):
        """Complex queries with multiple indicators should be classified as complex."""
        complex_queries = [
            "Design a distributed cache system for a social media platform that handles 10 million requests per second with low latency and high availability",
            """Analyze this code for security vulnerabilities:
            ```python
            def login(username, password):
                query = f"SELECT * FROM users WHERE username='{username}'"
                return db.execute(query)
            ```""",
            "Explain how transformer attention mechanisms work, including multi-head attention, and compare them to RNN-based models in terms of parallelization",
        ]
        
        for query in complex_queries:
            result = classify_query_complexity(query)
            assert result.complexity == QueryComplexity.COMPLEX, \
                f"Expected COMPLEX for: {query[:50]}..., got {result.complexity}"
    
    def test_code_detection(self):
        """Queries with code should be detected."""
        code_queries = [
            "```python\nprint('hello')\n```",
            "def my_function():",
            "class MyClass:",
            "function test() {}",
        ]
        
        for query in code_queries:
            assert has_pattern_match(query, CODE_PATTERNS), \
                f"Code not detected in: {query}"
    
    def test_reasoning_detection(self):
        """Queries with reasoning indicators should be detected."""
        reasoning_queries = [
            "Why does Python use indentation?",
            "How does memory allocation work?",
            "Explain the concept of recursion",
            "Compare REST and GraphQL",
            "Analyze this algorithm",
        ]
        
        for query in reasoning_queries:
            assert has_pattern_match(query, REASONING_PATTERNS), \
                f"Reasoning not detected in: {query}"
    
    def test_technical_terms(self):
        """Technical terms should be counted correctly."""
        query = "Explain kubernetes microservices architecture with docker containers"
        count = count_technical_terms(query)
        assert count >= 3, f"Expected at least 3 technical terms, got {count}"
    
    def test_confidence_scores(self):
        """Confidence scores should be reasonable."""
        queries = [
            "What is Python?",
            "Design a distributed system",
        ]
        
        for query in queries:
            result = classify_query_complexity(query)
            assert 0.0 <= result.confidence <= 1.0, \
                f"Confidence {result.confidence} out of range for: {query}"
    
    def test_classification_result_structure(self):
        """Classification result should have all required fields."""
        result = classify_query_complexity("Test query")
        
        assert hasattr(result, 'complexity')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'features')
        assert hasattr(result, 'recommended_model')
        
        assert isinstance(result.features, dict)
        assert 'total_score' in result.features


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_query(self):
        """Empty query should be classified as simple."""
        result = classify_query_complexity("")
        assert result.complexity == QueryComplexity.SIMPLE
    
    def test_very_long_query(self):
        """Very long queries should be handled."""
        long_query = "What is " + "Python " * 1000
        result = classify_query_complexity(long_query)
        # Should not raise an exception
        assert result.complexity in QueryComplexity
    
    def test_special_characters(self):
        """Queries with special characters should be handled."""
        queries = [
            "What about @decorators?",
            "How to use $variables in bash?",
            "What's the difference between == and ===?",
        ]
        
        for query in queries:
            result = classify_query_complexity(query)
            assert result.complexity in QueryComplexity
    
    def test_unicode_query(self):
        """Unicode characters should be handled."""
        query = "What is café? 你好 мир"
        result = classify_query_complexity(query)
        assert result.complexity in QueryComplexity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
