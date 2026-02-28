"""
Comprehensive Benchmark Suite.

Compares our Adaptive MAB system against:
1. No Cache (baseline cost)
2. Exact Match (hash dedup)
3. GPTCache (multi-stage, static threshold)
4. MeanCache (FL-style, fixed threshold)
5. vLLM Prefix Cache (token prefix matching)
6. SCALM (cluster-based)
7. MinCache (three-tier hybrid)
8. Static thresholds (0.80, 0.85, 0.90)
9. Ours: Adaptive MAB

Usage:
    python -m evaluation.benchmark --dataset synthetic --n 300
    python -m evaluation.benchmark --dataset sharegpt --n 500
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings import EmbeddingService
from app.mab import ContextualMAB
from app.quality import QualityChecker
from evaluation.dataset_loader import QueryPair, load_dataset, DatasetStats
from evaluation.baselines import (
    CacheBaseline, GPTCacheBaseline, MeanCacheBaseline,
    VLLMPrefixCacheBaseline, SCALMBaseline, MinCacheBaseline,
    get_all_baselines,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LLM_LATENCY_MS = 800.0  # simulated
CACHE_LATENCY_MS = 5.0
COST_PER_QUERY = 0.0001


@dataclass
class BenchmarkResult:
    name: str
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_cost: float = 0.0
    latencies: list = field(default_factory=list)

    @property
    def hit_rate(self): return self.cache_hits / max(self.total_queries, 1)
    @property
    def precision(self): return self.true_positives / max(self.cache_hits, 1)
    @property
    def recall(self): return self.true_positives / max(self.true_positives + self.false_negatives, 1)
    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)
    @property
    def avg_latency(self): return np.mean(self.latencies) if self.latencies else 0
    @property
    def p95_latency(self): return np.percentile(self.latencies, 95) if self.latencies else 0
    @property
    def cost_savings_pct(self):
        no_cache_cost = self.total_queries * COST_PER_QUERY
        return (1 - self.total_cost / max(no_cache_cost, 1e-9)) * 100

    def summary(self) -> dict:
        return {
            "method": self.name,
            "queries": self.total_queries,
            "hit_rate_%": round(self.hit_rate * 100, 1),
            "precision_%": round(self.precision * 100, 1),
            "recall_%": round(self.recall * 100, 1),
            "f1_%": round(self.f1 * 100, 1),
            "avg_lat_ms": round(self.avg_latency, 1),
            "p95_lat_ms": round(self.p95_latency, 1),
            "cost_$": round(self.total_cost, 4),
            "savings_%": round(self.cost_savings_pct, 1),
            "false_pos": self.false_positives,
        }


def _is_correct_hit(qp: QueryPair, matched_query: str, all_originals: dict) -> bool:
    """Check if a cache hit is correct (matched the right original)."""
    if qp.paraphrase_of:
        # This query IS a paraphrase — check if it matched the right original
        if matched_query == qp.paraphrase_of:
            return True
        # Check if the matched query is another paraphrase of the same original
        if qp.paraphrase_of in all_originals:
            siblings = all_originals[qp.paraphrase_of]
            if matched_query in siblings:
                return True
        return False
    if qp.is_near_miss:
        return False  # near-miss hits are always false positives
    return True  # unique query — assume ok


def _build_original_map(dataset: List[QueryPair]) -> dict:
    """Map original → list of paraphrases for correctness checking."""
    originals = {}
    for qp in dataset:
        if qp.paraphrase_of:
            if qp.paraphrase_of not in originals:
                originals[qp.paraphrase_of] = set()
            originals[qp.paraphrase_of].add(qp.query)
    return originals


# ─── Benchmark Runners ───────────────────────────────────────────────────

def run_no_cache(dataset: List[QueryPair]) -> BenchmarkResult:
    r = BenchmarkResult(name="No Cache")
    for qp in dataset:
        r.total_queries += 1
        r.cache_misses += 1
        r.latencies.append(LLM_LATENCY_MS)
        r.total_cost += COST_PER_QUERY
    return r


def run_exact_match(dataset: List[QueryPair]) -> BenchmarkResult:
    r = BenchmarkResult(name="Exact Match")
    seen = {}
    for qp in dataset:
        r.total_queries += 1
        key = qp.query.lower().strip()
        if key in seen:
            r.cache_hits += 1
            r.true_positives += 1
            r.latencies.append(CACHE_LATENCY_MS)
        else:
            seen[key] = True
            r.cache_misses += 1
            r.latencies.append(LLM_LATENCY_MS)
            r.total_cost += COST_PER_QUERY
            if qp.paraphrase_of:
                r.false_negatives += 1
    return r


def run_baseline(baseline: CacheBaseline, dataset: List[QueryPair],
                 embedder: EmbeddingService) -> BenchmarkResult:
    """Run any CacheBaseline implementation."""
    baseline.reset()
    r = BenchmarkResult(name=baseline.name)
    originals = _build_original_map(dataset)

    for qp in dataset:
        r.total_queries += 1
        emb = embedder.encode(qp.query)

        result = baseline.lookup(qp.query, emb)

        if result.hit:
            r.cache_hits += 1
            r.latencies.append(CACHE_LATENCY_MS + result.latency_ms)
            # Can't check response correctness without stored query,
            # so use is_near_miss as proxy
            if qp.is_near_miss:
                r.false_positives += 1
            elif qp.paraphrase_of:
                r.true_positives += 1
            # else: unique query hit — ambiguous
        else:
            r.cache_misses += 1
            r.latencies.append(LLM_LATENCY_MS)
            r.total_cost += COST_PER_QUERY
            baseline.store(qp.query, f"[response for {qp.query}]", emb)
            if qp.paraphrase_of:
                r.false_negatives += 1

    return r


def run_static_threshold(dataset: List[QueryPair], embedder: EmbeddingService,
                         threshold: float) -> BenchmarkResult:
    """Static threshold semantic cache."""
    r = BenchmarkResult(name=f"Static τ={threshold}")
    cache_store = []
    originals = _build_original_map(dataset)

    for qp in dataset:
        r.total_queries += 1
        emb = embedder.encode(qp.query)

        best_sim = 0.0
        best_query = None
        for sq, se in cache_store:
            sim = float(np.dot(emb, se))
            if sim > best_sim:
                best_sim = sim
                best_query = sq

        if best_sim >= threshold and best_query:
            r.cache_hits += 1
            r.latencies.append(CACHE_LATENCY_MS)
            if _is_correct_hit(qp, best_query, originals):
                r.true_positives += 1
            else:
                r.false_positives += 1
        else:
            r.cache_misses += 1
            r.latencies.append(LLM_LATENCY_MS)
            r.total_cost += COST_PER_QUERY
            cache_store.append((qp.query, emb))
            if qp.paraphrase_of:
                orig_cached = any(sq == qp.paraphrase_of for sq, _ in cache_store[:-1])
                if orig_cached:
                    r.false_negatives += 1

    return r


def run_adaptive_mab(dataset: List[QueryPair], embedder: EmbeddingService,
                     use_quality_gate: bool = True) -> BenchmarkResult:
    """Our method: Adaptive MAB with quality gate."""
    r = BenchmarkResult(name="Adaptive MAB (Ours)")
    mab = ContextualMAB()
    quality_checker = QualityChecker() if use_quality_gate else None
    cache_store = []
    originals = _build_original_map(dataset)

    for qp in dataset:
        r.total_queries += 1
        emb = embedder.encode(qp.query)
        threshold, arm_idx, domain, length_bin = mab.select_threshold(qp.query, emb)

        best_sim = 0.0
        best_query = None
        best_response = None
        for sq, se, sr in cache_store:
            sim = float(np.dot(emb, se))
            if sim > best_sim:
                best_sim = sim
                best_query = sq
                best_response = sr

        hit = False
        if best_sim >= threshold and best_query:
            # Quality gate
            if quality_checker and best_response:
                ok, conf, reason = quality_checker.check(qp.query, best_query, best_response)
                hit = ok
            else:
                hit = True

        if hit:
            r.cache_hits += 1
            r.latencies.append(CACHE_LATENCY_MS)
            correct = _is_correct_hit(qp, best_query, originals)
            if correct:
                r.true_positives += 1
                mab.update(domain, length_bin, arm_idx, "good_hit", similarity=best_sim)
            else:
                r.false_positives += 1
                mab.update(domain, length_bin, arm_idx, "bad_hit", similarity=best_sim)
        else:
            r.cache_misses += 1
            r.latencies.append(LLM_LATENCY_MS)
            r.total_cost += COST_PER_QUERY
            resp = f"[response for {qp.query}]"
            cache_store.append((qp.query, emb, resp))
            if qp.paraphrase_of:
                orig_cached = any(sq == qp.paraphrase_of for sq, _, _ in cache_store[:-1])
                if orig_cached:
                    r.false_negatives += 1
                    mab.update(domain, length_bin, arm_idx, "miss", similarity=best_sim)

    return r


# ─── Main ────────────────────────────────────────────────────────────────

def run_benchmark(dataset_name: str = "synthetic", n: int = 300):
    logger.info("=" * 70)
    logger.info("SEMANTIC CACHE BENCHMARK SUITE")
    logger.info("=" * 70)

    dataset, stats = load_dataset(dataset_name, n=n)
    logger.info(f"Dataset: {dataset_name} | {stats.total} queries")

    embedder = EmbeddingService()
    results = []

    # Run all methods
    methods = [
        ("No Cache", lambda: run_no_cache(dataset)),
        ("Exact Match", lambda: run_exact_match(dataset)),
    ]

    # Competitive baselines
    for baseline in get_all_baselines():
        methods.append((baseline.name, lambda b=baseline: run_baseline(b, dataset, embedder)))

    # Static thresholds
    for t in [0.80, 0.85, 0.90]:
        methods.append((f"Static τ={t}", lambda t=t: run_static_threshold(dataset, embedder, t)))

    # Our method
    methods.append(("Adaptive MAB (Ours)", lambda: run_adaptive_mab(dataset, embedder)))

    for name, run_fn in methods:
        logger.info(f"Running: {name}...")
        start = time.time()
        result = run_fn()
        elapsed = time.time() - start
        results.append(result)
        logger.info(f"  Done in {elapsed:.1f}s | {json.dumps(result.summary())}")

    # Print comparison table
    print("\n" + "=" * 130)
    header = (f"{'Method':<28} {'Hit%':>6} {'Prec%':>6} {'Rec%':>6} {'F1%':>6} "
              f"{'AvgLat':>8} {'P95Lat':>8} {'Cost$':>8} {'Save%':>7} {'FP':>4}")
    print(header)
    print("-" * 130)
    for r in results:
        s = r.summary()
        print(f"{s['method']:<28} {s['hit_rate_%']:>5.1f}% {s['precision_%']:>5.1f}% "
              f"{s['recall_%']:>5.1f}% {s['f1_%']:>5.1f}% {s['avg_lat_ms']:>7.1f} "
              f"{s['p95_lat_ms']:>7.1f} ${s['cost_$']:>6.4f} {s['savings_%']:>6.1f}% "
              f"{s['false_pos']:>4}")
    print("=" * 130)

    # Save results
    output = {
        "dataset": dataset_name,
        "n_queries": stats.total,
        "results": [r.summary() for r in results],
    }
    outfile = f"benchmark_{dataset_name}_{n}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {outfile}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "sharegpt", "msmarco"])
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    run_benchmark(args.dataset, args.n)
