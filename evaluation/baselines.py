"""
Competitive Baselines for evaluation.

Implements simplified but faithful versions of existing caching approaches:
1. GPTCache: multi-stage (exact → fuzzy → embedding) with static threshold
2. MeanCache-style: federated-friendly with fixed per-user thresholds
3. vLLM Prefix Cache: token-level prefix matching
4. SCALM-style: cluster-based semantic matching
5. MinCache-style: three-tier hierarchy with MinHash

Each baseline implements the same interface for fair comparison.

References:
- GPTCache: arxiv.org/abs/2309.05534
- MeanCache: arxiv.org/abs/2403.02694
- CacheBlend: arxiv.org/abs/2405.16444
- vLLM: arxiv.org/abs/2401.08771
- SCALM: IWQoS 2024
- MinCache: FGCS 2025
"""
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Standardized result from any baseline."""
    hit: bool
    response: Optional[str]
    similarity: float
    latency_ms: float
    method: str
    details: str = ""


class CacheBaseline(ABC):
    """Interface all baselines must implement."""

    @abstractmethod
    def lookup(self, query: str, embedding: np.ndarray) -> BaselineResult:
        pass

    @abstractmethod
    def store(self, query: str, response: str, embedding: np.ndarray):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ─── 1. GPTCache Baseline ────────────────────────────────────────────────

class GPTCacheBaseline(CacheBaseline):
    """
    Faithful GPTCache implementation: three-stage lookup pipeline.
    Stage 1: Exact string match (hash)
    Stage 2: Normalized fuzzy match (lowercase, strip punct)
    Stage 3: Embedding similarity with static threshold

    GPTCache's known weakness: static threshold can't adapt across domains.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.exact_cache: Dict[str, str] = {}
        self.fuzzy_cache: Dict[str, str] = {}
        self.embedding_cache: List[Tuple[str, np.ndarray, str]] = []

    @property
    def name(self) -> str:
        return f"GPTCache (τ={self.threshold})"

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def lookup(self, query: str, embedding: np.ndarray) -> BaselineResult:
        start = time.time()

        # Stage 1: Exact match
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.exact_cache:
            lat = (time.time() - start) * 1000
            return BaselineResult(hit=True, response=self.exact_cache[key],
                                  similarity=1.0, latency_ms=lat,
                                  method=self.name, details="exact_match")

        # Stage 2: Fuzzy match
        normalized = self._normalize(query)
        if normalized in self.fuzzy_cache:
            lat = (time.time() - start) * 1000
            return BaselineResult(hit=True, response=self.fuzzy_cache[normalized],
                                  similarity=0.99, latency_ms=lat,
                                  method=self.name, details="fuzzy_match")

        # Stage 3: Embedding similarity
        best_sim = 0.0
        best_response = None
        for _, stored_emb, stored_resp in self.embedding_cache:
            sim = float(np.dot(embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_response = stored_resp

        lat = (time.time() - start) * 1000
        if best_sim >= self.threshold and best_response:
            return BaselineResult(hit=True, response=best_response,
                                  similarity=best_sim, latency_ms=lat,
                                  method=self.name, details=f"embedding_sim={best_sim:.3f}")

        return BaselineResult(hit=False, response=None, similarity=best_sim,
                              latency_ms=lat, method=self.name)

    def store(self, query: str, response: str, embedding: np.ndarray):
        key = hashlib.md5(query.encode()).hexdigest()
        self.exact_cache[key] = response
        self.fuzzy_cache[self._normalize(query)] = response
        self.embedding_cache.append((query, embedding, response))

    def reset(self):
        self.exact_cache.clear()
        self.fuzzy_cache.clear()
        self.embedding_cache.clear()


# ─── 2. MeanCache Baseline ──────────────────────────────────────────────

class MeanCacheBaseline(CacheBaseline):
    """
    MeanCache (Gill et al., 2025): User-centric semantic caching.
    Uses per-user local caches with fixed similarity threshold.
    FL-trained embedding (we skip FL, use same embeddings for fairness).
    Key gap: no adaptive thresholding.
    """

    def __init__(self, threshold: float = 0.85, n_users: int = 3):
        self.threshold = threshold
        self.n_users = n_users
        # Simulate per-user caches
        self.user_caches: Dict[int, List[Tuple[np.ndarray, str]]] = {
            i: [] for i in range(n_users)
        }
        self._query_count = 0

    @property
    def name(self) -> str:
        return f"MeanCache (τ={self.threshold})"

    def _assign_user(self) -> int:
        self._query_count += 1
        return self._query_count % self.n_users

    def lookup(self, query: str, embedding: np.ndarray) -> BaselineResult:
        start = time.time()
        user_id = self._assign_user()
        user_cache = self.user_caches[user_id]

        best_sim = 0.0
        best_response = None
        for stored_emb, stored_resp in user_cache:
            sim = float(np.dot(embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_response = stored_resp

        lat = (time.time() - start) * 1000
        if best_sim >= self.threshold and best_response:
            return BaselineResult(hit=True, response=best_response,
                                  similarity=best_sim, latency_ms=lat,
                                  method=self.name, details=f"user={user_id}")

        return BaselineResult(hit=False, response=None, similarity=best_sim,
                              latency_ms=lat, method=self.name)

    def store(self, query: str, response: str, embedding: np.ndarray):
        user_id = self._query_count % self.n_users
        self.user_caches[user_id].append((embedding, response))

    def reset(self):
        self.user_caches = {i: [] for i in range(self.n_users)}
        self._query_count = 0


# ─── 3. vLLM Prefix Cache Baseline ──────────────────────────────────────

class VLLMPrefixCacheBaseline(CacheBaseline):
    """
    Simulates vLLM's automatic prefix caching.
    Works at token level: finds longest common prefix between new query
    and cached queries. Only reuses KV cache for the matching prefix.

    This is fundamentally different from semantic caching — it's token-exact,
    so "reverse a string" and "reverse a string in python" share a prefix
    but "reverse a string" and "how to reverse a string" don't share much.
    """

    def __init__(self, min_prefix_ratio: float = 0.6):
        self.min_prefix_ratio = min_prefix_ratio
        self.cached_sequences: List[Tuple[List[str], str]] = []

    @property
    def name(self) -> str:
        return "vLLM Prefix Cache"

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _longest_common_prefix(self, a: List[str], b: List[str]) -> int:
        length = 0
        for t1, t2 in zip(a, b):
            if t1 == t2:
                length += 1
            else:
                break
        return length

    def lookup(self, query: str, embedding: np.ndarray) -> BaselineResult:
        start = time.time()
        tokens = self._tokenize(query)

        best_prefix_len = 0
        best_response = None
        best_total_len = 0

        for cached_tokens, cached_resp in self.cached_sequences:
            prefix_len = self._longest_common_prefix(tokens, cached_tokens)
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_response = cached_resp
                best_total_len = len(cached_tokens)

        lat = (time.time() - start) * 1000
        ratio = best_prefix_len / max(len(tokens), 1)

        if ratio >= self.min_prefix_ratio and best_response:
            return BaselineResult(hit=True, response=best_response,
                                  similarity=ratio, latency_ms=lat,
                                  method=self.name,
                                  details=f"prefix_ratio={ratio:.2f}")

        return BaselineResult(hit=False, response=None, similarity=ratio,
                              latency_ms=lat, method=self.name)

    def store(self, query: str, response: str, embedding: np.ndarray):
        tokens = self._tokenize(query)
        self.cached_sequences.append((tokens, response))

    def reset(self):
        self.cached_sequences.clear()


# ─── 4. SCALM Baseline ──────────────────────────────────────────────────

class SCALMBaseline(CacheBaseline):
    """
    SCALM (Li et al., 2024): Semantic Caching for Automated Chat Services.
    Uses vector quantization + semantic clustering.
    Simplified: cluster embeddings and match within cluster first.
    """

    def __init__(self, threshold: float = 0.85, n_clusters: int = 8):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.clusters: Dict[int, List[Tuple[np.ndarray, str, str]]] = {
            i: [] for i in range(n_clusters)
        }

    @property
    def name(self) -> str:
        return f"SCALM (τ={self.threshold})"

    def _assign_cluster(self, embedding: np.ndarray) -> int:
        # Simple hash-based cluster assignment (real SCALM uses k-means)
        hash_val = int(hashlib.md5(embedding.tobytes()[:32]).hexdigest()[:8], 16)
        return hash_val % self.n_clusters

    def lookup(self, query: str, embedding: np.ndarray) -> BaselineResult:
        start = time.time()
        cluster_id = self._assign_cluster(embedding)
        cluster = self.clusters[cluster_id]

        best_sim = 0.0
        best_response = None
        for stored_emb, stored_query, stored_resp in cluster:
            sim = float(np.dot(embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_response = stored_resp

        lat = (time.time() - start) * 1000
        if best_sim >= self.threshold and best_response:
            return BaselineResult(hit=True, response=best_response,
                                  similarity=best_sim, latency_ms=lat,
                                  method=self.name, details=f"cluster={cluster_id}")

        return BaselineResult(hit=False, response=None, similarity=best_sim,
                              latency_ms=lat, method=self.name)

    def store(self, query: str, response: str, embedding: np.ndarray):
        cluster_id = self._assign_cluster(embedding)
        self.clusters[cluster_id].append((embedding, query, response))

    def reset(self):
        self.clusters = {i: [] for i in range(self.n_clusters)}


# ─── 5. MinCache Baseline ───────────────────────────────────────────────

class MinCacheBaseline(CacheBaseline):
    """
    MinCache (Haqiq et al., 2025): Three-tier hybrid cache.
    Tier 1: Exact match
    Tier 2: MinHash resemblance (approximate n-gram similarity)
    Tier 3: Embedding semantic match

    Known gap: MinHash misses nuanced semantics.
    """

    def __init__(self, threshold: float = 0.85, minhash_threshold: float = 0.5,
                 num_hashes: int = 128):
        self.threshold = threshold
        self.minhash_threshold = minhash_threshold
        self.num_hashes = num_hashes
        self.exact_cache: Dict[str, str] = {}
        self.minhash_cache: List[Tuple[np.ndarray, str, str]] = []  # (minhash, query, response)
        self.embedding_cache: List[Tuple[np.ndarray, str]] = []

    @property
    def name(self) -> str:
        return f"MinCache (τ={self.threshold})"

    def _compute_minhash(self, text: str) -> np.ndarray:
        """Simplified MinHash: hash character n-grams."""
        text = text.lower().strip()
        ngrams = set()
        for i in range(len(text) - 2):
            ngrams.add(text[i:i+3])
        if not ngrams:
            return np.zeros(self.num_hashes)

        signatures = np.full(self.num_hashes, np.inf)
        for ngram in ngrams:
            for i in range(self.num_hashes):
                h = hash(f"{i}_{ngram}") % (2**32)
                signatures[i] = min(signatures[i], h)
        return signatures

    def _minhash_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a == b))

    def lookup(self, query: str, embedding: np.ndarray) -> BaselineResult:
        start = time.time()

        # Tier 1: Exact
        key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        if key in self.exact_cache:
            lat = (time.time() - start) * 1000
            return BaselineResult(hit=True, response=self.exact_cache[key],
                                  similarity=1.0, latency_ms=lat,
                                  method=self.name, details="exact")

        # Tier 2: MinHash
        query_mh = self._compute_minhash(query)
        for stored_mh, stored_q, stored_resp in self.minhash_cache:
            mh_sim = self._minhash_similarity(query_mh, stored_mh)
            if mh_sim >= self.minhash_threshold:
                lat = (time.time() - start) * 1000
                return BaselineResult(hit=True, response=stored_resp,
                                      similarity=mh_sim, latency_ms=lat,
                                      method=self.name, details=f"minhash={mh_sim:.3f}")

        # Tier 3: Embedding
        best_sim = 0.0
        best_response = None
        for stored_emb, stored_resp in self.embedding_cache:
            sim = float(np.dot(embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_response = stored_resp

        lat = (time.time() - start) * 1000
        if best_sim >= self.threshold and best_response:
            return BaselineResult(hit=True, response=best_response,
                                  similarity=best_sim, latency_ms=lat,
                                  method=self.name, details=f"embedding={best_sim:.3f}")

        return BaselineResult(hit=False, response=None, similarity=best_sim,
                              latency_ms=lat, method=self.name)

    def store(self, query: str, response: str, embedding: np.ndarray):
        key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        self.exact_cache[key] = response
        self.minhash_cache.append((self._compute_minhash(query), query, response))
        self.embedding_cache.append((embedding, response))

    def reset(self):
        self.exact_cache.clear()
        self.minhash_cache.clear()
        self.embedding_cache.clear()


def get_all_baselines() -> List[CacheBaseline]:
    """Returns all baseline implementations for benchmarking."""
    return [
        GPTCacheBaseline(threshold=0.85),
        MeanCacheBaseline(threshold=0.85),
        VLLMPrefixCacheBaseline(min_prefix_ratio=0.6),
        SCALMBaseline(threshold=0.85),
        MinCacheBaseline(threshold=0.85),
    ]
