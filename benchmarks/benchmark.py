#!/usr/bin/env python3
"""
Benchmark script for SmartLLM Router.

Run this to generate realistic metrics for your resume!

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --requests 100
"""

import argparse
import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import httpx

# Test queries categorized by expected complexity
TEST_QUERIES = {
    "simple": [
        "What is Python?",
        "Convert 100 USD to EUR",
        "What's 15% of 230?",
        "What is the capital of France?",
        "How many days are in a year?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?",
        "Define machine learning in one sentence",
        "What year did World War 2 end?",
    ],
    "medium": [
        "Explain the difference between REST and GraphQL APIs",
        "Write a function to reverse a string in Python",
        "What are the pros and cons of using Redis vs Memcached?",
        "How does Python's garbage collection work?",
        "Explain the concept of polymorphism in OOP",
        "What is the difference between SQL and NoSQL databases?",
        "How do you handle exceptions in Python?",
        "Explain the MVC architecture pattern",
        "What are environment variables and why are they useful?",
        "Describe the difference between threads and processes",
    ],
    "complex": [
        "Design a distributed cache system for a social media platform that handles 10 million requests per second with low latency and high availability",
        "Explain how transformer attention mechanisms work, including multi-head attention, and compare them to RNN-based sequence models",
        "Design a URL shortening service like bit.ly that can handle 100M daily users. Include database schema, API design, and scaling considerations",
        "Analyze the trade-offs between eventual consistency and strong consistency in distributed systems, with examples of when to use each",
        "Explain how to implement a rate limiter using the token bucket algorithm, including handling distributed environments",
        "Design a recommendation system for an e-commerce platform. Include data collection, model architecture, and real-time serving considerations",
        "How would you debug a memory leak in a production Python application? Walk through your systematic approach",
        "Design a real-time collaborative text editor like Google Docs. Address conflict resolution and synchronization",
    ],
}


class BenchmarkRunner:
    """Runs benchmark tests against the SmartLLM Router."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
    
    async def run_single_request(
        self,
        query: str,
        expected_complexity: str,
        skip_cache: bool = False
    ) -> Dict[str, Any]:
        """Run a single request and record results."""
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat",
                    json={
                        "message": query,
                        "skip_cache": skip_cache,
                    },
                    timeout=60.0,
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "expected_complexity": expected_complexity,
                        "actual_complexity": data.get("complexity"),
                        "model_used": data.get("model_used"),
                        "was_cached": data.get("was_cached"),
                        "actual_cost": data.get("actual_cost"),
                        "baseline_cost": data.get("baseline_cost"),
                        "latency_ms": data.get("latency_ms"),
                        "total_time": elapsed,
                    }
                else:
                    return {
                        "success": False,
                        "query": query[:50],
                        "error": f"HTTP {response.status_code}",
                        "total_time": elapsed,
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "query": query[:50],
                    "error": str(e),
                    "total_time": time.time() - start_time,
                }
    
    async def run_benchmark(
        self,
        num_requests: int = 50,
        include_cache_test: bool = True
    ) -> Dict[str, Any]:
        """Run full benchmark suite."""
        print(f"\n🚀 Starting SmartLLM Router Benchmark")
        print(f"   Requests per complexity level: {num_requests // 3}")
        print(f"   Cache testing: {'enabled' if include_cache_test else 'disabled'}")
        print("-" * 50)
        
        all_queries = []
        
        # Distribute requests across complexity levels
        per_level = num_requests // 3
        
        for complexity, queries in TEST_QUERIES.items():
            for i in range(per_level):
                query = queries[i % len(queries)]
                all_queries.append((query, complexity))
        
        # Run initial requests (skip cache)
        print("\n📤 Running initial requests (no cache)...")
        for query, complexity in all_queries:
            result = await self.run_single_request(query, complexity, skip_cache=True)
            self.results.append(result)
            status = "✅" if result["success"] else "❌"
            print(f"   {status} [{complexity}] {result.get('query', 'error')}")
        
        # Run again to test caching
        if include_cache_test:
            print("\n📥 Running repeat requests (testing cache)...")
            for query, complexity in all_queries[:len(all_queries)//2]:
                result = await self.run_single_request(query, complexity, skip_cache=False)
                self.results.append(result)
                cached = "🎯 CACHE HIT" if result.get("was_cached") else "💨 MISS"
                print(f"   {cached} [{complexity}] {result.get('query', 'error')}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate benchmark report."""
        successful = [r for r in self.results if r.get("success")]
        
        if not successful:
            return {"error": "No successful requests"}
        
        # Calculate metrics
        total_actual_cost = sum(r.get("actual_cost", 0) for r in successful)
        total_baseline_cost = sum(r.get("baseline_cost", 0) for r in successful)
        
        cache_hits = sum(1 for r in successful if r.get("was_cached"))
        cache_hit_rate = cache_hits / len(successful) * 100
        
        savings = total_baseline_cost - total_actual_cost
        savings_pct = (savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
        
        # Latency analysis
        cached_latencies = [r["latency_ms"] for r in successful if r.get("was_cached")]
        uncached_latencies = [r["latency_ms"] for r in successful if not r.get("was_cached")]
        
        avg_cached = sum(cached_latencies) / len(cached_latencies) if cached_latencies else 0
        avg_uncached = sum(uncached_latencies) / len(uncached_latencies) if uncached_latencies else 0
        
        # Routing accuracy
        correct_routing = sum(
            1 for r in successful 
            if r.get("expected_complexity") == r.get("actual_complexity")
        )
        routing_accuracy = correct_routing / len(successful) * 100
        
        # Model distribution
        model_counts = {}
        for r in successful:
            model = r.get("model_used", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": len(self.results),
            "successful_requests": len(successful),
            "failed_requests": len(self.results) - len(successful),
            
            "cost_metrics": {
                "total_actual_cost": round(total_actual_cost, 6),
                "total_baseline_cost": round(total_baseline_cost, 6),
                "total_savings": round(savings, 6),
                "savings_percentage": round(savings_pct, 2),
            },
            
            "cache_metrics": {
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hit_rate, 2),
            },
            
            "latency_metrics": {
                "avg_cached_latency_ms": round(avg_cached, 2),
                "avg_uncached_latency_ms": round(avg_uncached, 2),
                "latency_reduction_pct": round(
                    (avg_uncached - avg_cached) / avg_uncached * 100, 2
                ) if avg_uncached > 0 else 0,
            },
            
            "routing_metrics": {
                "routing_accuracy": round(routing_accuracy, 2),
                "model_distribution": model_counts,
            },
            
            "resume_metrics": {
                "cost_reduction": f"{savings_pct:.0f}%",
                "cache_hit_rate": f"{cache_hit_rate:.0f}%",
                "latency_improvement": f"{((avg_uncached - avg_cached) / avg_uncached * 100):.0f}%" if avg_uncached > 0 else "N/A",
                "requests_processed": f"{len(successful):,}",
            }
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted benchmark report."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK REPORT")
        print("=" * 60)
        
        print(f"\n📈 Overview")
        print(f"   Total Requests: {report['total_requests']}")
        print(f"   Successful: {report['successful_requests']}")
        print(f"   Failed: {report['failed_requests']}")
        
        cost = report["cost_metrics"]
        print(f"\n💰 Cost Metrics")
        print(f"   Actual Cost: ${cost['total_actual_cost']:.4f}")
        print(f"   Baseline (GPT-4): ${cost['total_baseline_cost']:.4f}")
        print(f"   Savings: ${cost['total_savings']:.4f} ({cost['savings_percentage']}%)")
        
        cache = report["cache_metrics"]
        print(f"\n🎯 Cache Metrics")
        print(f"   Cache Hits: {cache['cache_hits']}")
        print(f"   Hit Rate: {cache['cache_hit_rate']}%")
        
        latency = report["latency_metrics"]
        print(f"\n⚡ Latency Metrics")
        print(f"   Cached Avg: {latency['avg_cached_latency_ms']}ms")
        print(f"   Uncached Avg: {latency['avg_uncached_latency_ms']}ms")
        print(f"   Improvement: {latency['latency_reduction_pct']}%")
        
        routing = report["routing_metrics"]
        print(f"\n🎯 Routing Metrics")
        print(f"   Accuracy: {routing['routing_accuracy']}%")
        print(f"   Model Distribution: {routing['model_distribution']}")
        
        resume = report["resume_metrics"]
        print(f"\n📝 RESUME METRICS (copy these!)")
        print(f"   ✅ Cost Reduction: {resume['cost_reduction']}")
        print(f"   ✅ Cache Hit Rate: {resume['cache_hit_rate']}")
        print(f"   ✅ Latency Improvement: {resume['latency_improvement']}")
        print(f"   ✅ Requests Processed: {resume['requests_processed']}")
        
        print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark SmartLLM Router")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=30, help="Number of requests")
    parser.add_argument("--no-cache-test", action="store_true", help="Skip cache testing")
    parser.add_argument("--output", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(base_url=args.url)
    
    try:
        report = await runner.run_benchmark(
            num_requests=args.requests,
            include_cache_test=not args.no_cache_test,
        )
        
        runner.print_report(report)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved to {args.output}")
            
    except httpx.ConnectError:
        print("\n❌ Error: Could not connect to the API.")
        print("   Make sure the server is running: uvicorn src.main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
