"""
Analytics Router.

Endpoints for cost analytics and reporting.
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Query

from ..services.cost_tracker import get_cost_tracker
from ..services.semantic_cache import get_semantic_cache
from ..models.schemas import CostAnalytics, DailyStats

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/summary")
async def get_summary():
    """
    Get a quick summary of all-time statistics.
    
    Returns total requests, savings, cache hit rate, and average latency.
    """
    cost_tracker = get_cost_tracker()
    return cost_tracker.get_summary()


@router.get("/costs", response_model=CostAnalytics)
async def get_cost_analytics(
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    days: int = Query(30, description="Number of days to look back if no dates provided"),
):
    """
    Get detailed cost analytics for a time period.
    
    Includes total requests, cache hits, actual vs baseline costs,
    and breakdowns by model and complexity.
    """
    cost_tracker = get_cost_tracker()
    
    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=days)
    if end_date is None:
        end_date = datetime.utcnow()
    
    return cost_tracker.get_analytics(start_date, end_date)


@router.get("/daily")
async def get_daily_stats(
    days: int = Query(7, ge=1, le=90, description="Number of days of history"),
):
    """
    Get daily statistics for the specified number of days.
    
    Returns an array of daily stats including requests, costs, and savings.
    """
    cost_tracker = get_cost_tracker()
    return {"stats": cost_tracker.get_daily_stats(days)}


@router.get("/cache")
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns cache size, hit rate configuration, and TTL settings.
    """
    cache = get_semantic_cache()
    return cache.get_stats()


@router.delete("/cache")
async def clear_cache():
    """
    Clear all cached responses.
    
    Use with caution - this will remove all cached responses.
    """
    cache = get_semantic_cache()
    cleared = await cache.clear()
    return {"message": f"Cleared {cleared} cache entries"}


@router.get("/savings-report")
async def get_savings_report(
    days: int = Query(30, ge=1, le=365, description="Number of days for report"),
):
    """
    Generate a comprehensive savings report.
    
    Ideal for stakeholder presentations and resume metrics.
    """
    cost_tracker = get_cost_tracker()
    cache = get_semantic_cache()
    
    analytics = cost_tracker.get_analytics(
        start_date=datetime.utcnow() - timedelta(days=days)
    )
    cache_stats = cache.get_stats()
    
    # Calculate per-request averages
    avg_actual = analytics.actual_cost / analytics.total_requests if analytics.total_requests > 0 else 0
    avg_baseline = analytics.baseline_cost / analytics.total_requests if analytics.total_requests > 0 else 0
    
    return {
        "report_period_days": days,
        "total_requests": analytics.total_requests,
        "cost_metrics": {
            "actual_total": round(analytics.actual_cost, 4),
            "baseline_total": round(analytics.baseline_cost, 4),
            "total_savings": round(analytics.savings, 4),
            "savings_percentage": round(analytics.savings_percentage, 2),
            "avg_cost_per_request": round(avg_actual, 6),
            "avg_baseline_per_request": round(avg_baseline, 6),
        },
        "cache_metrics": {
            "hit_rate": analytics.cache_hit_rate,
            "total_hits": analytics.cache_hits,
            "active_entries": cache_stats["active_entries"],
        },
        "routing_breakdown": {
            "by_model": analytics.requests_by_model,
            "by_complexity": analytics.requests_by_complexity,
        },
        "resume_metrics": {
            "cost_reduction": f"{analytics.savings_percentage:.0f}%",
            "cache_hit_rate": f"{analytics.cache_hit_rate:.0f}%",
            "requests_processed": f"{analytics.total_requests:,}",
            "dollars_saved": f"${analytics.savings:.2f}",
        },
    }
