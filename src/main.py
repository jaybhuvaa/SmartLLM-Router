"""
SmartLLM Router - Main Application Entry Point.

An intelligent LLM cost optimization middleware that:
- Routes queries to optimal models based on complexity
- Caches responses semantically to avoid redundant API calls
- Tracks costs and provides analytics
"""

from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import chat_router, analytics_router
from .models.schemas import HealthCheck


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 SmartLLM Router starting up...")
    settings = get_settings()
    print(f"   Debug mode: {settings.debug}")
    print(f"   Simple model: {settings.default_simple_model}")
    print(f"   Medium model: {settings.default_medium_model}")
    print(f"   Complex model: {settings.default_complex_model}")
    
    yield
    
    # Shutdown
    print("👋 SmartLLM Router shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SmartLLM Router",
    description="""
    ## Intelligent LLM Cost Optimizer
    
    SmartLLM Router is a production-grade middleware that intelligently routes 
    LLM requests to optimal models based on query complexity, implements semantic 
    caching, and provides cost analytics.
    
    ### Features
    - **Smart Routing**: Automatically routes queries to the most cost-effective model
    - **Semantic Caching**: Caches responses for semantically similar queries
    - **Cost Analytics**: Real-time tracking of costs and savings
    
    ### Endpoints
    - `/api/v1/chat` - Main chat endpoint
    - `/api/v1/classify` - Query complexity classification
    - `/api/v1/analytics/*` - Cost and performance analytics
    """,
    version="0.1.0",
    lifespan=lifespan,
)


# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(chat_router)
app.include_router(analytics_router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SmartLLM Router",
        "version": "0.1.0",
        "description": "Intelligent LLM Cost Optimizer",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns status of the API and its dependencies.
    """
    # In production, these would actually check connections
    return HealthCheck(
        status="healthy",
        version="0.1.0",
        database="connected",  # Would check PostgreSQL
        redis="connected",     # Would check Redis
        timestamp=datetime.utcnow(),
    )


# Run with: uvicorn src.main:app --reload
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
