"""
Metrics: Prometheus-compatible + in-memory tracking with A/B test support.
"""
import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, using in-memory metrics only")


class MetricsCollector:
    """Unified metrics with A/B test group tracking."""

    def __init__(self):
        self._cache_hits = 0
        self._cache_misses = 0
        self._false_positives = 0
        self._total_queries = 0
        self._total_cost_saved = 0.0
        self._total_cost_spent = 0.0
        self._latencies_cache = []
        self._latencies_llm = []
        self._quality_scores = []
        self._threshold_selections = {}
        # A/B test tracking
        self._ab_groups = defaultdict(lambda: {
            "hits": 0, "misses": 0, "false_positives": 0,
            "cost_saved": 0.0, "cost_spent": 0.0,
            "latencies_cache": [], "latencies_llm": [],
        })
        # Per-domain tracking
        self._domain_stats = defaultdict(lambda: {
            "hits": 0, "misses": 0, "false_positives": 0,
        })

        if PROMETHEUS_AVAILABLE:
            self.prom_cache_hits = Counter("cache_hits_total", "Total cache hits")
            self.prom_cache_misses = Counter("cache_misses_total", "Total cache misses")
            self.prom_false_positives = Counter("false_positives_total", "False positive cache hits")
            self.prom_latency = Histogram(
                "query_latency_seconds", "Query latency",
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
            )
            self.prom_cost_saved = Counter("cost_saved_usd", "USD saved by cache")
            self.prom_cost_spent = Counter("cost_spent_usd", "USD spent on LLM")
            self.prom_hit_rate = Gauge("cache_hit_rate", "Current hit rate")
            self.prom_threshold = Gauge("selected_threshold", "Last threshold", ["domain"])

    def record_cache_hit(self, latency_s: float, similarity: float, domain: str,
                         threshold: float, estimated_cost_saved: float, ab_group: str = "experiment"):
        self._cache_hits += 1
        self._total_queries += 1
        self._total_cost_saved += estimated_cost_saved
        self._latencies_cache.append(latency_s)
        self._threshold_selections[domain] = threshold
        self._domain_stats[domain]["hits"] += 1
        # A/B
        g = self._ab_groups[ab_group]
        g["hits"] += 1
        g["cost_saved"] += estimated_cost_saved
        g["latencies_cache"].append(latency_s)

        if PROMETHEUS_AVAILABLE:
            self.prom_cache_hits.inc()
            self.prom_latency.observe(latency_s)
            self.prom_cost_saved.inc(estimated_cost_saved)
            self.prom_hit_rate.set(self.hit_rate)
            self.prom_threshold.labels(domain=domain).set(threshold)

    def record_cache_miss(self, latency_s: float, cost_usd: float, domain: str,
                          threshold: float, ab_group: str = "experiment"):
        self._cache_misses += 1
        self._total_queries += 1
        self._total_cost_spent += cost_usd
        self._latencies_llm.append(latency_s)
        self._threshold_selections[domain] = threshold
        self._domain_stats[domain]["misses"] += 1
        # A/B
        g = self._ab_groups[ab_group]
        g["misses"] += 1
        g["cost_spent"] += cost_usd
        g["latencies_llm"].append(latency_s)

        if PROMETHEUS_AVAILABLE:
            self.prom_cache_misses.inc()
            self.prom_latency.observe(latency_s)
            self.prom_cost_spent.inc(cost_usd)
            self.prom_hit_rate.set(self.hit_rate)

    def record_false_positive(self, ab_group: str = "experiment"):
        self._false_positives += 1
        self._ab_groups[ab_group]["false_positives"] += 1
        if PROMETHEUS_AVAILABLE:
            self.prom_false_positives.inc()

    def record_quality_score(self, score: float):
        self._quality_scores.append(score)

    @property
    def hit_rate(self) -> float:
        return self._cache_hits / max(self._total_queries, 1)

    @property
    def precision(self) -> float:
        if self._cache_hits == 0:
            return 0.0
        return (self._cache_hits - self._false_positives) / self._cache_hits

    def get_summary(self) -> dict:
        cache_lats = self._latencies_cache or [0]
        llm_lats = self._latencies_llm or [0]
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "false_positives": self._false_positives,
            "hit_rate": round(self.hit_rate * 100, 2),
            "precision": round(self.precision * 100, 2),
            "cost_saved_usd": round(self._total_cost_saved, 6),
            "cost_spent_usd": round(self._total_cost_spent, 6),
            "latency_cache_p50_ms": round(float(np.percentile(cache_lats, 50)) * 1000, 1),
            "latency_cache_p95_ms": round(float(np.percentile(cache_lats, 95)) * 1000, 1),
            "latency_llm_p50_ms": round(float(np.percentile(llm_lats, 50)) * 1000, 1),
            "latency_llm_p95_ms": round(float(np.percentile(llm_lats, 95)) * 1000, 1),
            "avg_quality_score": round(float(np.mean(self._quality_scores)), 3) if self._quality_scores else None,
        }

    def get_ab_summary(self) -> dict:
        """Per-group A/B test results."""
        result = {}
        for group, g in self._ab_groups.items():
            total = g["hits"] + g["misses"]
            hit_rate = g["hits"] / max(total, 1) * 100
            precision = (g["hits"] - g["false_positives"]) / max(g["hits"], 1) * 100
            avg_cache_lat = float(np.mean(g["latencies_cache"])) * 1000 if g["latencies_cache"] else 0
            avg_llm_lat = float(np.mean(g["latencies_llm"])) * 1000 if g["latencies_llm"] else 0
            result[group] = {
                "total_queries": total,
                "hit_rate_%": round(hit_rate, 1),
                "precision_%": round(precision, 1),
                "cost_saved_$": round(g["cost_saved"], 6),
                "cost_spent_$": round(g["cost_spent"], 6),
                "false_positives": g["false_positives"],
                "avg_cache_latency_ms": round(avg_cache_lat, 1),
                "avg_llm_latency_ms": round(avg_llm_lat, 1),
            }
        return result

    def get_domain_summary(self) -> dict:
        result = {}
        for domain, d in self._domain_stats.items():
            total = d["hits"] + d["misses"]
            result[domain] = {
                "total": total,
                "hit_rate_%": round(d["hits"] / max(total, 1) * 100, 1),
                "false_positives": d["false_positives"],
            }
        return result

    def reset(self):
        self._cache_hits = 0
        self._cache_misses = 0
        self._false_positives = 0
        self._total_queries = 0
        self._total_cost_saved = 0.0
        self._total_cost_spent = 0.0
        self._latencies_cache = []
        self._latencies_llm = []
        self._quality_scores = []
        self._ab_groups.clear()
        self._domain_stats.clear()


metrics = MetricsCollector()
