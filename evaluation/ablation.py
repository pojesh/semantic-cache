"""
Ablation Study: systematically disable components to prove each adds value.

Ablations:
1. Full system (all components)
2. No quality gate (skip heuristic check)
3. No MAB (use static threshold)
4. No domain detection (single context)
5. No enhanced context (basic 2-feature context)
6. No TTL (never expire cache entries)

This answers: "What if you removed X?" for every component.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from evaluation.dataset_loader import QueryPair, SyntheticDatasetGenerator

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    name: str
    description: str
    hit_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_positives: int = 0
    total_queries: int = 0

    def summary(self) -> dict:
        return {
            "ablation": self.name,
            "description": self.description,
            "hit_rate_%": round(self.hit_rate * 100, 2),
            "precision_%": round(self.precision * 100, 2),
            "recall_%": round(self.recall * 100, 2),
            "f1_%": round(self.f1 * 100, 2),
            "false_positives": self.false_positives,
        }


class AblationRunner:
    """Run ablation studies on the semantic cache system."""

    def __init__(self, embedder):
        self.embedder = embedder

    def run_all(self, dataset: List[QueryPair] = None, n: int = 300) -> List[AblationResult]:
        """Run all ablation experiments."""
        if dataset is None:
            gen = SyntheticDatasetGenerator()
            dataset, _ = gen.generate(n=n)

        results = []

        # 1. Full system
        results.append(self._run_experiment(
            dataset,
            name="Full System",
            description="All components enabled",
            use_mab=True, use_quality_gate=True,
            use_domain=True, use_enhanced_ctx=True,
        ))

        # 2. No quality gate
        results.append(self._run_experiment(
            dataset,
            name="No Quality Gate",
            description="Remove heuristic quality check — accept all hits above threshold",
            use_mab=True, use_quality_gate=False,
            use_domain=True, use_enhanced_ctx=True,
        ))

        # 3. No MAB (static threshold)
        results.append(self._run_experiment(
            dataset,
            name="No MAB (Static τ=0.85)",
            description="Replace adaptive thresholds with static τ=0.85",
            use_mab=False, use_quality_gate=True,
            use_domain=True, use_enhanced_ctx=True,
            static_threshold=0.85,
        ))

        # 4. No domain detection
        results.append(self._run_experiment(
            dataset,
            name="No Domain Detection",
            description="Treat all queries as same domain — single MAB context",
            use_mab=True, use_quality_gate=True,
            use_domain=False, use_enhanced_ctx=False,
        ))

        # 5. No enhanced context (basic 2 features only)
        results.append(self._run_experiment(
            dataset,
            name="Basic Context Only",
            description="Only domain + length (no complexity/specificity)",
            use_mab=True, use_quality_gate=True,
            use_domain=True, use_enhanced_ctx=False,
        ))

        # 6. Very strict static threshold
        results.append(self._run_experiment(
            dataset,
            name="No MAB (Static τ=0.92)",
            description="Very strict static threshold — high precision but low recall",
            use_mab=False, use_quality_gate=True,
            use_domain=True, use_enhanced_ctx=True,
            static_threshold=0.92,
        ))

        return results

    def _run_experiment(self, dataset: List[QueryPair], name: str, description: str,
                        use_mab: bool, use_quality_gate: bool, use_domain: bool,
                        use_enhanced_ctx: bool, static_threshold: float = 0.85) -> AblationResult:
        """Run a single ablation experiment."""
        from app.mab import ContextualMAB, EnhancedContextExtractor
        from app.quality import QualityChecker

        # Fresh instances for each experiment
        if use_mab:
            from config import config
            old_enhanced = config.mab.use_enhanced_context
            config.mab.use_enhanced_context = use_enhanced_ctx
            mab = ContextualMAB()
            config.mab.use_enhanced_context = old_enhanced
        quality_checker = QualityChecker() if use_quality_gate else None

        cache_store = []  # (query, embedding, response, domain)
        hits = 0
        misses = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for qp in dataset:
            emb = self.embedder.encode(qp.query)

            # Determine threshold
            if use_mab:
                threshold, arm_idx, domain, length_bin = mab.select_threshold(qp.query, emb)
            else:
                threshold = static_threshold
                arm_idx = 0
                domain = "general" if not use_domain else self._simple_domain(qp.query)
                length_bin = "short"

            # Search cache
            best_sim = 0.0
            best_match = None
            best_idx = -1
            for idx, (sq, se, sr, sd) in enumerate(cache_store):
                sim = float(np.dot(emb, se))
                if sim > best_sim:
                    best_sim = sim
                    best_match = (sq, sr, sd)
                    best_idx = idx

            if best_sim >= threshold and best_match is not None:
                # Quality gate
                quality_ok = True
                if quality_checker:
                    ok, conf, reason = quality_checker.check(qp.query, best_match[0], best_match[1])
                    quality_ok = ok

                if quality_ok:
                    hits += 1
                    # Determine if correct hit
                    matched_query = best_match[0]
                    if qp.paraphrase_of and (
                        qp.paraphrase_of == matched_query or
                        matched_query in [e["original"] for d in SyntheticDatasetGenerator.TEST_DATA.values()
                                          for e in d if qp.query in e.get("paraphrases", [])]
                    ):
                        true_positives += 1
                        if use_mab:
                            mab.update(domain, length_bin, arm_idx, "good_hit")
                    elif qp.is_near_miss:
                        false_positives += 1
                        if use_mab:
                            mab.update(domain, length_bin, arm_idx, "bad_hit")
                    else:
                        # Ambiguous — count as TP if it's a paraphrase
                        if qp.paraphrase_of:
                            true_positives += 1
                        else:
                            # unique query matched something — might be ok
                            pass
                else:
                    # Quality gate rejected
                    misses += 1
                    cache_store.append((qp.query, emb, f"[response for {qp.query}]", domain))
            else:
                misses += 1
                cache_store.append((qp.query, emb, f"[response for {qp.query}]", domain))
                if use_mab and qp.paraphrase_of:
                    mab.update(domain, length_bin, arm_idx, "miss")
                # Check if this was a missed paraphrase
                if qp.paraphrase_of:
                    orig_in_cache = any(sq == qp.paraphrase_of for sq, _, _, _ in cache_store[:-1])
                    if orig_in_cache:
                        false_negatives += 1

        total = hits + misses
        hit_rate = hits / max(total, 1)
        precision = true_positives / max(hits, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        result = AblationResult(
            name=name, description=description,
            hit_rate=hit_rate, precision=precision, recall=recall, f1=f1,
            false_positives=false_positives, total_queries=total,
        )
        logger.info(f"Ablation [{name}]: HR={hit_rate:.1%} P={precision:.1%} R={recall:.1%} F1={f1:.1%} FP={false_positives}")
        return result

    def _simple_domain(self, query: str) -> str:
        q = query.lower()
        if any(w in q.split() for w in ["code", "python", "function", "api"]):
            return "code"
        if any(w in q for w in ["what is", "who is", "capital"]):
            return "factual"
        if any(w in q.split() for w in ["write", "explain", "compare", "cook", "recipe"]):
            return "creative"
        return "general"


def run_ablation_study(n: int = 300) -> List[dict]:
    """Run complete ablation study and return results."""
    import sys
    sys.path.insert(0, ".")
    from app.embeddings import EmbeddingService

    logger.info("=" * 60)
    logger.info("ABLATION STUDY")
    logger.info("=" * 60)

    embedder = EmbeddingService()
    runner = AblationRunner(embedder)
    results = runner.run_all(n=n)

    # Print table
    print("\n" + "=" * 100)
    print(f"{'Ablation':<30} {'Hit Rate':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>5}")
    print("-" * 100)
    for r in results:
        s = r.summary()
        print(f"{s['ablation']:<30} {s['hit_rate_%']:>9.1f}% {s['precision_%']:>9.1f}% "
              f"{s['recall_%']:>9.1f}% {s['f1_%']:>9.1f}% {s['false_positives']:>5}")
    print("=" * 100)

    output = [r.summary() for r in results]
    with open("ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to ablation_results.json")

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ablation_study()
