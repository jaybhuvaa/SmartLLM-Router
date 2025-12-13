"""
Cost tracking and analytics service.

Tracks all requests, calculates costs, and provides analytics.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from ..config import get_settings
from ..models.schemas import CostAnalytics, DailyStats
from ..utils.token_counter import estimate_cost


@dataclass
class RequestLog:
    """Log entry for a single request."""
    id: str
    timestamp: datetime
    query_text: str
    query_complexity: str
    model_used: str
    was_cached: bool
    input_tokens: int
    output_tokens: int
    actual_cost: float
    baseline_cost: float
    latency_ms: int
    cache_similarity: Optional[float] = None


class CostTracker:
    """
    In-memory cost tracker for MVP.
    
    In production, this would be backed by PostgreSQL.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logs: List[RequestLog] = []
        self.baseline_model = "gpt-4"  # Always compare against GPT-4
    
    def log_request(
        self,
        query_text: str,
        query_complexity: str,
        model_used: str,
        was_cached: bool,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        cache_similarity: Optional[float] = None,
    ) -> RequestLog:
        """Log a request and calculate costs."""
        
        # Calculate actual cost
        actual_cost = estimate_cost(
            input_tokens, output_tokens, model_used, self.settings.pricing
        )
        
        # Calculate what it would have cost with GPT-4
        baseline_cost = estimate_cost(
            input_tokens, output_tokens, self.baseline_model, self.settings.pricing
        )
        
        # If cached, actual cost is $0 but we still track baseline
        if was_cached:
            actual_cost = 0.0
        
        log = RequestLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            query_text=query_text[:500],  # Truncate for storage
            query_complexity=query_complexity,
            model_used=model_used,
            was_cached=was_cached,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            actual_cost=actual_cost,
            baseline_cost=baseline_cost,
            latency_ms=latency_ms,
            cache_similarity=cache_similarity,
        )
        
        self.logs.append(log)
        return log
    
    def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> CostAnalytics:
        """Get cost analytics for a time period."""
        
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
        
        # Filter logs by date range
        filtered_logs = [
            log for log in self.logs
            if start_date <= log.timestamp <= end_date
        ]
        
        if not filtered_logs:
            return CostAnalytics(
                period_start=start_date,
                period_end=end_date,
                total_requests=0,
                cache_hits=0,
                cache_hit_rate=0.0,
                actual_cost=0.0,
                baseline_cost=0.0,
                savings=0.0,
                savings_percentage=0.0,
                requests_by_model={},
                requests_by_complexity={},
            )
        
        total_requests = len(filtered_logs)
        cache_hits = sum(1 for log in filtered_logs if log.was_cached)
        actual_cost = sum(log.actual_cost for log in filtered_logs)
        baseline_cost = sum(log.baseline_cost for log in filtered_logs)
        
        # Count by model
        requests_by_model: Dict[str, int] = {}
        for log in filtered_logs:
            requests_by_model[log.model_used] = requests_by_model.get(log.model_used, 0) + 1
        
        # Count by complexity
        requests_by_complexity: Dict[str, int] = {}
        for log in filtered_logs:
            requests_by_complexity[log.query_complexity] = requests_by_complexity.get(log.query_complexity, 0) + 1
        
        savings = baseline_cost - actual_cost
        savings_percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return CostAnalytics(
            period_start=start_date,
            period_end=end_date,
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_hit_rate=round(cache_hit_rate, 2),
            actual_cost=round(actual_cost, 6),
            baseline_cost=round(baseline_cost, 6),
            savings=round(savings, 6),
            savings_percentage=round(savings_percentage, 2),
            requests_by_model=requests_by_model,
            requests_by_complexity=requests_by_complexity,
        )
    
    def get_daily_stats(self, days: int = 7) -> List[DailyStats]:
        """Get daily statistics for the last N days."""
        
        stats = []
        end_date = datetime.utcnow().replace(hour=23, minute=59, second=59)
        
        for i in range(days):
            day_end = end_date - timedelta(days=i)
            day_start = day_end.replace(hour=0, minute=0, second=0)
            
            day_logs = [
                log for log in self.logs
                if day_start <= log.timestamp <= day_end
            ]
            
            if day_logs:
                total = len(day_logs)
                hits = sum(1 for log in day_logs if log.was_cached)
                actual = sum(log.actual_cost for log in day_logs)
                baseline = sum(log.baseline_cost for log in day_logs)
                avg_latency = sum(log.latency_ms for log in day_logs) / total
                
                stats.append(DailyStats(
                    date=day_start.strftime("%Y-%m-%d"),
                    total_requests=total,
                    cache_hits=hits,
                    cache_hit_rate=round(hits / total * 100, 2) if total > 0 else 0.0,
                    actual_cost=round(actual, 6),
                    baseline_cost=round(baseline, 6),
                    savings=round(baseline - actual, 6),
                    savings_percentage=round((baseline - actual) / baseline * 100, 2) if baseline > 0 else 0.0,
                    avg_latency_ms=round(avg_latency, 2),
                ))
            else:
                stats.append(DailyStats(
                    date=day_start.strftime("%Y-%m-%d"),
                    total_requests=0,
                    cache_hits=0,
                    cache_hit_rate=0.0,
                    actual_cost=0.0,
                    baseline_cost=0.0,
                    savings=0.0,
                    savings_percentage=0.0,
                    avg_latency_ms=0.0,
                ))
        
        return list(reversed(stats))  # Oldest first
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary of all-time stats."""
        
        if not self.logs:
            return {
                "total_requests": 0,
                "total_savings": 0.0,
                "cache_hit_rate": 0.0,
                "avg_latency_ms": 0.0,
            }
        
        total = len(self.logs)
        hits = sum(1 for log in self.logs if log.was_cached)
        savings = sum(log.baseline_cost - log.actual_cost for log in self.logs)
        avg_latency = sum(log.latency_ms for log in self.logs) / total
        
        return {
            "total_requests": total,
            "total_savings": round(savings, 4),
            "cache_hit_rate": round(hits / total * 100, 2),
            "avg_latency_ms": round(avg_latency, 2),
        }


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
