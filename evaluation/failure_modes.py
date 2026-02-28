"""
Failure Mode Analysis: systematically test edge cases and document system breakpoints.

Tests:
1. Negation sensitivity: "do X" vs "don't do X"
2. Entity swaps: same structure, different entities
3. Language mixing: multilingual queries
4. Adversarial: intentionally confusing queries
5. Temporal: time-sensitive queries
6. Length extremes: very short / very long
7. Ambiguity: vague queries
8. Out-of-distribution: domains not in training

This answers: "When does the system break? What are its limits?"
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FailureCase:
    category: str
    query_a: str
    query_b: str
    should_match: bool
    actual_match: bool = False
    similarity: float = 0.0
    is_failure: bool = False
    explanation: str = ""


class FailureModeAnalyzer:
    """Test system against known failure modes."""

    # (query_a, query_b, should_match, category, explanation)
    FAILURE_TESTS: List[Tuple[str, str, bool, str, str]] = [
        # ── Negation Sensitivity ──
        ("How to delete a file in Python",
         "How to NOT delete a file in Python",
         False, "negation",
         "Negation flips intent entirely; high embedding similarity is dangerous"),
        ("Enable dark mode in VS Code",
         "Disable dark mode in VS Code",
         False, "negation",
         "Enable vs disable — opposite actions"),
        ("How to start a Docker container",
         "How to stop a Docker container",
         False, "negation",
         "Start vs stop are antonyms with similar embeddings"),
        ("Allow incoming connections on port 80",
         "Block incoming connections on port 80",
         False, "negation",
         "Security-critical: allow vs block"),

        # ── Entity Swaps ──
        ("Convert Celsius to Fahrenheit",
         "Convert Fahrenheit to Celsius",
         False, "entity_swap",
         "Same words, reversed direction — different formula needed"),
        ("How to migrate from MySQL to PostgreSQL",
         "How to migrate from PostgreSQL to MySQL",
         False, "entity_swap",
         "Migration direction changes the entire procedure"),
        ("Compare Python and JavaScript",
         "Compare JavaScript and Python",
         True, "entity_swap",
         "Order shouldn't matter for comparisons"),
        ("Translate English to French",
         "Translate French to English",
         False, "entity_swap",
         "Translation direction matters"),

        # ── Specificity Traps ──
        ("Sort a list in Python",
         "Sort a list of dictionaries by a specific key in Python",
         False, "specificity",
         "Generic vs specific — completely different code"),
        ("How to connect to a database",
         "How to connect to PostgreSQL 15 on AWS RDS with SSL",
         False, "specificity",
         "Vague vs precise — different answers"),
        ("What is machine learning",
         "What is machine learning with focus on transformer architectures",
         False, "specificity",
         "General definition vs specific subfield"),

        # ── Temporal Sensitivity ──
        ("Who is the president of the United States",
         "Who was the president of the United States in 2020",
         False, "temporal",
         "Current vs historical — different answers"),
        ("Latest version of Python",
         "Latest version of Python 2",
         False, "temporal",
         "Python 2 is EOL, Python 3 evolves — very different answers"),
        ("Best practices for React",
         "Best practices for React in 2024",
         True, "temporal",
         "If cached recently, still valid"),

        # ── Length Extremes ──
        ("AI", "Explain artificial intelligence",
         True, "length",
         "Ultra-short query should match expanded form"),
        ("ML", "Machine learning",
         True, "length",
         "Abbreviation should match full form"),
        ("Sort", "Sort a list of integers in descending order using Python's built-in sorted function with a custom key",
         False, "length",
         "One-word query is too ambiguous to match specific request"),

        # ── Ambiguity ──
        ("Python", "Python programming language tutorial",
         False, "ambiguity",
         "Python could mean the snake, Monty Python, or the language"),
        ("How to handle exceptions", "How to handle exceptions in Python",
         False, "ambiguity",
         "Language-agnostic vs Python-specific"),
        ("Apple", "Apple Inc stock price",
         False, "ambiguity",
         "Fruit vs company"),

        # ── Multilingual ──
        ("How to reverse a string in Python",
         "Comment inverser une chaîne en Python",
         True, "multilingual",
         "Same question in French — MPNet may not handle this well"),
        ("What is artificial intelligence",
         "¿Qué es la inteligencia artificial?",
         True, "multilingual",
         "Same question in Spanish"),

        # ── Adversarial ──
        ("How to hack a website",
         "How to secure a website from hackers",
         False, "adversarial",
         "Opposite intent despite similar vocabulary"),
        ("Write malware in Python",
         "Write security scanning tool in Python",
         False, "adversarial",
         "Malicious vs defensive — critical distinction"),

        # ── Cross-Domain Confusion ──
        ("What is a tree",
         "What is a binary tree",
         False, "cross_domain",
         "Biology vs computer science"),
        ("Define class",
         "Define a class in Python",
         False, "cross_domain",
         "Social class vs OOP class"),
        ("How does inheritance work",
         "How does inheritance work in Java",
         False, "cross_domain",
         "Legal inheritance vs OOP inheritance"),
    ]

    def __init__(self, embedder, quality_checker=None):
        self.embedder = embedder
        self.quality_checker = quality_checker

    def run_analysis(self, threshold: float = 0.85) -> List[FailureCase]:
        """Run all failure mode tests and report results."""
        results = []

        for query_a, query_b, should_match, category, explanation in self.FAILURE_TESTS:
            emb_a = self.embedder.encode(query_a)
            emb_b = self.embedder.encode(query_b)
            similarity = float(np.dot(emb_a, emb_b))

            actual_match = similarity >= threshold
            is_failure = actual_match != should_match

            # Also check quality gate if available
            quality_gate_caught = False
            if self.quality_checker and is_failure and actual_match and not should_match:
                ok, conf, reason = self.quality_checker.check(
                    query_b, query_a, f"[response for {query_a}]"
                )
                if not ok:
                    quality_gate_caught = True
                    is_failure = False  # quality gate saved us

            case = FailureCase(
                category=category,
                query_a=query_a,
                query_b=query_b,
                should_match=should_match,
                actual_match=actual_match,
                similarity=round(similarity, 4),
                is_failure=is_failure,
                explanation=explanation + (
                    " [CAUGHT by quality gate]" if quality_gate_caught else ""
                ),
            )
            results.append(case)

        return results

    def report(self, threshold: float = 0.85) -> dict:
        """Generate comprehensive failure mode report."""
        results = self.run_analysis(threshold)

        # Group by category
        by_category = {}
        for r in results:
            if r.category not in by_category:
                by_category[r.category] = {"total": 0, "failures": 0, "cases": []}
            by_category[r.category]["total"] += 1
            if r.is_failure:
                by_category[r.category]["failures"] += 1
            by_category[r.category]["cases"].append({
                "query_a": r.query_a,
                "query_b": r.query_b,
                "should_match": r.should_match,
                "actual_match": r.actual_match,
                "similarity": r.similarity,
                "FAILURE": r.is_failure,
                "explanation": r.explanation,
            })

        total_tests = len(results)
        total_failures = sum(1 for r in results if r.is_failure)
        false_pos = sum(1 for r in results if r.is_failure and r.actual_match and not r.should_match)
        false_neg = sum(1 for r in results if r.is_failure and not r.actual_match and r.should_match)

        summary = {
            "threshold": threshold,
            "total_tests": total_tests,
            "total_failures": total_failures,
            "accuracy_%": round((total_tests - total_failures) / total_tests * 100, 1),
            "false_positives": false_pos,
            "false_negatives": false_neg,
            "categories": {
                cat: {
                    "accuracy_%": round((d["total"] - d["failures"]) / d["total"] * 100, 1),
                    "failures": d["failures"],
                    "total": d["total"],
                }
                for cat, d in by_category.items()
            },
            "details": by_category,
        }

        return summary

    def report_multi_threshold(self, thresholds: list = None) -> dict:
        """Run failure mode analysis at multiple thresholds to find sweet spot."""
        if thresholds is None:
            thresholds = [0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]

        comparison = []
        for t in thresholds:
            results = self.run_analysis(t)
            total = len(results)
            failures = sum(1 for r in results if r.is_failure)
            fps = sum(1 for r in results if r.is_failure and r.actual_match and not r.should_match)
            fns = sum(1 for r in results if r.is_failure and not r.actual_match and r.should_match)
            comparison.append({
                "threshold": t,
                "accuracy_%": round((total - failures) / total * 100, 1),
                "failures": failures,
                "false_positives": fps,
                "false_negatives": fns,
            })

        return {"threshold_comparison": comparison}


def run_failure_analysis():
    """CLI entry point."""
    import sys
    sys.path.insert(0, ".")
    from app.embeddings import EmbeddingService
    from app.quality import QualityChecker

    logging.basicConfig(level=logging.INFO)

    embedder = EmbeddingService()
    quality_checker = QualityChecker()
    analyzer = FailureModeAnalyzer(embedder, quality_checker)

    # Single threshold report
    report = analyzer.report(threshold=0.85)
    print(f"\n{'='*80}")
    print(f"FAILURE MODE ANALYSIS (τ={report['threshold']})")
    print(f"{'='*80}")
    print(f"Overall: {report['accuracy_%']}% accuracy ({report['total_failures']}/{report['total_tests']} failures)")
    print(f"  False Positives: {report['false_positives']} (wrong matches)")
    print(f"  False Negatives: {report['false_negatives']} (missed matches)")
    print()

    for cat, data in report["categories"].items():
        status = "✅" if data["failures"] == 0 else "❌"
        print(f"  {status} {cat}: {data['accuracy_%']}% ({data['failures']} failures / {data['total']} tests)")

    # Multi-threshold comparison
    print(f"\n{'='*80}")
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    multi = analyzer.report_multi_threshold()
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Failures':>10} {'FP':>5} {'FN':>5}")
    print("-" * 45)
    for row in multi["threshold_comparison"]:
        print(f"{row['threshold']:>10.2f} {row['accuracy_%']:>9.1f}% {row['failures']:>10} "
              f"{row['false_positives']:>5} {row['false_negatives']:>5}")

    # Save
    with open("failure_analysis.json", "w") as f:
        json.dump({"single": report, "multi": multi}, f, indent=2)
    print(f"\nResults saved to failure_analysis.json")


if __name__ == "__main__":
    run_failure_analysis()
