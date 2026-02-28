"""
Dataset Loader for evaluation.

Supports:
1. ShareGPT: Real chatbot conversations (downloaded from HuggingFace)
2. LMSYS-Chat-1M: Real user-LLM interactions
3. MS MARCO: Question-answer pairs with known relevance
4. Synthetic: Built-in paraphrase test set (no download needed)

Each dataset outputs standardized QueryPair objects for the benchmark suite.
"""
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("evaluation/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QueryPair:
    """A test query with metadata for evaluation."""
    query: str
    paraphrase_of: Optional[str] = None    # if this is a paraphrase, what's the original
    expected_response: Optional[str] = None  # ground truth response
    domain: str = "general"
    is_near_miss: bool = False              # similar embedding but DIFFERENT answer needed
    difficulty: str = "normal"               # "easy" | "normal" | "hard"


@dataclass
class DatasetStats:
    total: int = 0
    unique_queries: int = 0
    paraphrases: int = 0
    near_misses: int = 0
    domains: dict = field(default_factory=dict)


# ─── ShareGPT Loader ────────────────────────────────────────────────────

class ShareGPTLoader:
    """
    Load real conversations from ShareGPT dataset.
    Download: huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    """

    def load(self, filepath: str = None, n_samples: int = 1000) -> List[QueryPair]:
        """Load ShareGPT and extract first-turn user queries."""
        if filepath is None:
            filepath = str(DATA_DIR / "sharegpt.json")

        if not os.path.exists(filepath):
            logger.warning(
                f"ShareGPT not found at {filepath}. "
                f"Download from HuggingFace or use: "
                f"python -c \"from datasets import load_dataset; "
                f"d=load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered'); "
                f"d['train'].to_json('{filepath}')\""
            )
            return []

        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ShareGPT: {e}")
            return []

        pairs = []
        for conv in data[:n_samples * 2]:
            conversations = conv.get("conversations", [])
            if len(conversations) >= 2:
                user_msg = conversations[0].get("value", "").strip()
                assistant_msg = conversations[1].get("value", "").strip()
                if 10 < len(user_msg) < 500 and len(assistant_msg) > 10:
                    pairs.append(QueryPair(
                        query=user_msg,
                        expected_response=assistant_msg[:1000],
                        domain=self._detect_domain(user_msg),
                    ))
            if len(pairs) >= n_samples:
                break

        logger.info(f"Loaded {len(pairs)} queries from ShareGPT")
        return pairs

    def _detect_domain(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["code", "python", "javascript", "function", "error", "bug"]):
            return "code"
        if any(w in q for w in ["what is", "who is", "when", "where", "define"]):
            return "factual"
        if any(w in q for w in ["write", "create", "story", "poem", "essay"]):
            return "creative"
        return "general"


# ─── MS MARCO Loader ────────────────────────────────────────────────────

class MSMARCOLoader:
    """
    Load MS MARCO question-answer pairs.
    Good for evaluating factual query caching.
    """

    def load(self, filepath: str = None, n_samples: int = 1000) -> List[QueryPair]:
        if filepath is None:
            filepath = str(DATA_DIR / "msmarco.json")

        if not os.path.exists(filepath):
            logger.warning(
                f"MS MARCO not found at {filepath}. "
                f"Download from: msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz"
            )
            return []

        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception:
            # Try TSV format
            pairs = []
            try:
                with open(filepath) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            pairs.append(QueryPair(
                                query=parts[1] if len(parts) > 1 else parts[0],
                                domain="factual",
                            ))
                            if len(pairs) >= n_samples:
                                break
                return pairs
            except Exception as e:
                logger.error(f"Failed to load MS MARCO: {e}")
                return []

        return [QueryPair(query=item.get("query", ""), domain="factual")
                for item in data[:n_samples]]


# ─── Comprehensive Synthetic Dataset ─────────────────────────────────────

class SyntheticDatasetGenerator:
    """
    Rich synthetic dataset with known ground truth.
    Includes paraphrases, near-misses, and difficulty levels.
    No external downloads needed — works out of the box.
    """

    # (original, [paraphrases], [near_misses_that_should_NOT_match])
    TEST_DATA = {
        "code": [
            {
                "original": "How to reverse a string in Python",
                "paraphrases": [
                    "Give me Python code to reverse a string",
                    "Python string reversal method",
                    "Reverse a string using Python",
                    "What's the best way to reverse a string in Python",
                ],
                "near_misses": [
                    "How to reverse a list in Python",           # different data structure
                    "How to reverse a string in JavaScript",     # different language
                    "How to sort a string in Python",            # different operation
                ],
            },
            {
                "original": "Write a function to sort a list in Python",
                "paraphrases": [
                    "Python code for sorting a list",
                    "How do I sort a list in Python",
                    "Sorting algorithm for Python lists",
                ],
                "near_misses": [
                    "Write a function to sort a dictionary in Python",
                    "Write a function to sort a list in Java",
                ],
            },
            {
                "original": "How to read a CSV file in pandas",
                "paraphrases": [
                    "Read CSV with pandas Python",
                    "Loading a CSV file using pandas",
                    "pandas read_csv example",
                ],
                "near_misses": [
                    "How to write a CSV file in pandas",
                    "How to read an Excel file in pandas",
                    "How to read a JSON file in Python",
                ],
            },
            {
                "original": "Explain list comprehension in Python",
                "paraphrases": [
                    "What is list comprehension in Python",
                    "Python list comprehension tutorial",
                    "How does list comprehension work in Python",
                ],
                "near_misses": [
                    "Explain dictionary comprehension in Python",
                    "Explain generators in Python",
                ],
            },
            {
                "original": "How to handle exceptions in Python",
                "paraphrases": [
                    "Python try except error handling",
                    "Exception handling in Python",
                    "How to catch errors in Python",
                ],
                "near_misses": [
                    "How to handle exceptions in Java",
                    "How to raise exceptions in Python",
                ],
            },
            {
                "original": "What is the difference between a list and tuple in Python",
                "paraphrases": [
                    "List vs tuple in Python",
                    "Compare Python list and tuple",
                    "When to use list or tuple in Python",
                ],
                "near_misses": [
                    "What is the difference between list and set in Python",
                    "What is the difference between list and array in Python",
                ],
            },
            {
                "original": "How to create a REST API with FastAPI",
                "paraphrases": [
                    "FastAPI REST API tutorial",
                    "Build a REST API using FastAPI",
                    "Getting started with FastAPI",
                ],
                "near_misses": [
                    "How to create a REST API with Flask",
                    "How to create a REST API with Django",
                ],
            },
            {
                "original": "Explain decorators in Python",
                "paraphrases": [
                    "What are Python decorators",
                    "How do decorators work in Python",
                    "Python decorator tutorial",
                ],
                "near_misses": [
                    "Explain decorators in TypeScript",
                    "Explain context managers in Python",
                ],
            },
        ],
        "factual": [
            {
                "original": "What is the capital of France",
                "paraphrases": [
                    "Capital city of France",
                    "France capital name",
                    "Tell me the capital of France",
                    "Which city is the capital of France",
                ],
                "near_misses": [
                    "What is the population of France",
                    "What is the capital of Germany",
                    "What is the largest city in France",
                ],
            },
            {
                "original": "Who invented the telephone",
                "paraphrases": [
                    "Inventor of the telephone",
                    "Who created the telephone",
                    "Who made the first telephone",
                ],
                "near_misses": [
                    "Who invented the television",
                    "Who invented the internet",
                    "When was the telephone invented",
                ],
            },
            {
                "original": "What is photosynthesis",
                "paraphrases": [
                    "Explain photosynthesis",
                    "How does photosynthesis work",
                    "Define photosynthesis process",
                ],
                "near_misses": [
                    "What is cellular respiration",
                    "What is chemosynthesis",
                ],
            },
            {
                "original": "How many planets are in the solar system",
                "paraphrases": [
                    "Number of planets in our solar system",
                    "How many planets orbit the sun",
                ],
                "near_misses": [
                    "How many moons does Jupiter have",
                    "How many galaxies are there",
                ],
            },
            {
                "original": "What is machine learning",
                "paraphrases": [
                    "Define machine learning",
                    "Explain machine learning",
                    "What does machine learning mean",
                    "Introduction to machine learning",
                ],
                "near_misses": [
                    "What is deep learning",
                    "What is reinforcement learning",
                    "What is artificial intelligence",
                ],
            },
        ],
        "math": [
            {
                "original": "Explain the Pythagorean theorem",
                "paraphrases": [
                    "What is the Pythagorean theorem",
                    "Pythagorean theorem explanation",
                    "How does the Pythagorean theorem work",
                ],
                "near_misses": [
                    "Prove the Pythagorean theorem",  # different task: explain vs prove
                    "What is the law of cosines",
                ],
            },
            {
                "original": "How to calculate the area of a circle",
                "paraphrases": [
                    "Circle area formula",
                    "Area of a circle calculation",
                    "What is the formula for circle area",
                ],
                "near_misses": [
                    "How to calculate the circumference of a circle",
                    "How to calculate the area of a triangle",
                ],
            },
            {
                "original": "Explain Bayes theorem",
                "paraphrases": [
                    "What is Bayes theorem",
                    "Bayes rule explanation",
                    "How does Bayesian probability work",
                ],
                "near_misses": [
                    "Explain the central limit theorem",
                    "Explain conditional probability",
                ],
            },
        ],
        "creative": [
            {
                "original": "How to cook chicken biryani",
                "paraphrases": [
                    "Chicken biryani recipe",
                    "Recipe for making chicken biryani",
                    "How to make biryani with chicken",
                ],
                "near_misses": [
                    "How to cook vegetable biryani",
                    "How to cook chicken tikka",
                    "How to cook chicken fried rice",
                ],
            },
            {
                "original": "Compare and contrast TCP and UDP",
                "paraphrases": [
                    "Differences between TCP and UDP",
                    "TCP vs UDP comparison",
                    "How are TCP and UDP different",
                ],
                "near_misses": [
                    "Compare HTTP and HTTPS",
                    "Compare TCP and SCTP",
                ],
            },
            {
                "original": "Compare SQL and NoSQL databases",
                "paraphrases": [
                    "SQL vs NoSQL differences",
                    "When to use SQL vs NoSQL",
                    "Comparison of SQL and NoSQL",
                ],
                "near_misses": [
                    "Compare MySQL and PostgreSQL",
                    "Compare MongoDB and Redis",
                ],
            },
        ],
    }

    def generate(self, n: int = 500, seed: int = 42) -> Tuple[List[QueryPair], DatasetStats]:
        """Generate full test dataset with known ground truth."""
        rng = random.Random(seed)
        dataset = []
        stats = DatasetStats()

        for domain, entries in self.TEST_DATA.items():
            for entry in entries:
                # Add original
                dataset.append(QueryPair(
                    query=entry["original"],
                    domain=domain,
                    difficulty="normal",
                ))
                stats.unique_queries += 1

                # Add paraphrases (should be CACHE HITS)
                for para in entry["paraphrases"]:
                    dataset.append(QueryPair(
                        query=para,
                        paraphrase_of=entry["original"],
                        domain=domain,
                        difficulty="normal",
                    ))
                    stats.paraphrases += 1

                # Add near-misses (should be CACHE MISSES — tests precision)
                for nm in entry.get("near_misses", []):
                    dataset.append(QueryPair(
                        query=nm,
                        paraphrase_of=None,
                        domain=domain,
                        is_near_miss=True,
                        difficulty="hard",
                    ))
                    stats.near_misses += 1

            stats.domains[domain] = sum(
                1 for e in entries
                for _ in [e["original"]] + e["paraphrases"] + e.get("near_misses", [])
            )

        rng.shuffle(dataset)
        if len(dataset) > n:
            dataset = dataset[:n]

        stats.total = len(dataset)
        logger.info(
            f"Generated dataset: {stats.total} queries, "
            f"{stats.unique_queries} unique, {stats.paraphrases} paraphrases, "
            f"{stats.near_misses} near-misses"
        )
        return dataset, stats


def load_dataset(name: str = "synthetic", n: int = 500, **kwargs) -> Tuple[List[QueryPair], DatasetStats]:
    """Unified dataset loading interface."""
    if name == "synthetic":
        gen = SyntheticDatasetGenerator()
        return gen.generate(n=n)
    elif name == "sharegpt":
        loader = ShareGPTLoader()
        pairs = loader.load(n_samples=n, **kwargs)
        stats = DatasetStats(total=len(pairs), unique_queries=len(pairs))
        return pairs, stats
    elif name == "msmarco":
        loader = MSMARCOLoader()
        pairs = loader.load(n_samples=n, **kwargs)
        stats = DatasetStats(total=len(pairs), unique_queries=len(pairs))
        return pairs, stats
    else:
        raise ValueError(f"Unknown dataset: {name}. Options: synthetic, sharegpt, msmarco")
