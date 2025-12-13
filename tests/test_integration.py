"""
Integration tests for the SmartLLM Router API.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


class TestRootEndpoints:
    """Test root and health endpoints."""
    
    def test_root(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert data["name"] == "SmartLLM Router"
        assert "version" in data
    
    def test_health(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestChatEndpoints:
    """Test chat-related endpoints."""
    
    def test_classify_simple_query(self, client):
        """Simple query should be classified correctly."""
        response = client.post(
            "/api/v1/classify",
            json={"message": "What is Python?"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["complexity"] == "simple"
        assert "confidence" in data
        assert "recommended_model" in data
    
    def test_classify_complex_query(self, client):
        """Complex query should be classified correctly."""
        response = client.post(
            "/api/v1/classify",
            json={
                "message": "Design a distributed cache system for a social media platform with high availability and explain the tradeoffs between consistency and partition tolerance"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["complexity"] == "complex"
    
    def test_list_models(self, client):
        """Models endpoint should list available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0
        
        # Each model should have required fields
        for model in data["models"]:
            assert "name" in model
            assert "input_cost_per_1k" in model
            assert "output_cost_per_1k" in model
    
    def test_chat_validation_error(self, client):
        """Chat endpoint should validate request."""
        # Empty message should fail
        response = client.post(
            "/api/v1/chat",
            json={"message": ""}
        )
        assert response.status_code == 422  # Validation error


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""
    
    def test_summary(self, client):
        """Summary endpoint should return stats."""
        response = client.get("/api/v1/analytics/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "total_savings" in data
        assert "cache_hit_rate" in data
    
    def test_costs(self, client):
        """Costs endpoint should return analytics."""
        response = client.get("/api/v1/analytics/costs")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "actual_cost" in data
        assert "baseline_cost" in data
    
    def test_daily_stats(self, client):
        """Daily stats endpoint should return array."""
        response = client.get("/api/v1/analytics/daily?days=7")
        assert response.status_code == 200
        
        data = response.json()
        assert "stats" in data
        assert isinstance(data["stats"], list)
    
    def test_cache_stats(self, client):
        """Cache stats endpoint should return cache info."""
        response = client.get("/api/v1/analytics/cache")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_entries" in data
        assert "similarity_threshold" in data
    
    def test_savings_report(self, client):
        """Savings report should include resume metrics."""
        response = client.get("/api/v1/analytics/savings-report?days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert "cost_metrics" in data
        assert "cache_metrics" in data
        assert "resume_metrics" in data
        
        # Resume metrics should be formatted strings
        resume = data["resume_metrics"]
        assert "%" in resume["cost_reduction"]
        assert "%" in resume["cache_hit_rate"]


class TestOpenAPISpec:
    """Test OpenAPI documentation."""
    
    def test_openapi_json(self, client):
        """OpenAPI spec should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "SmartLLM Router"
    
    def test_docs_page(self, client):
        """Swagger docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
