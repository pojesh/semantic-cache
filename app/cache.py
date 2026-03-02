"""
Redis Vector Cache: stores query-response pairs with HNSW vector indexing.

Uses Redis Stack's native vector search (no external libraries needed beyond redis-py).
Each cache entry stores: query text, response text, embedding vector, domain, timestamp, hit count.
"""
import json
import logging
import time
import hashlib
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import redis
try:
    from redis.commands.search.field import TextField, NumericField, TagField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError:
    try:
        from redis.commands.search.field import TextField, NumericField, TagField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType
        from redis.commands.search.query import Query
    except ImportError as e:
        raise ImportError(
            "redis[search] is required. Install with: pip install 'redis[hiredis]>=5.0.0'\n"
            f"Original error: {e}"
        )

from config import config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query: str
    response: str
    domain: str
    similarity: float = 0.0
    timestamp: float = 0.0
    hit_count: int = 0
    cache_key: str = ""


class VectorCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            decode_responses=False,  # need bytes for vectors
        )
        self.index_name = config.redis.index_name
        self.prefix = config.redis.prefix
        self.dim = config.embedding.dimension
        self._ensure_index()

    def _ensure_index(self):
        """Create Redis search index if it doesn't exist."""
        try:
            self.redis_client.ft(self.index_name).info()
            logger.info(f"Redis index '{self.index_name}' already exists")
        except redis.ResponseError:
            logger.info(f"Creating Redis index '{self.index_name}'")
            schema = (
                TextField("query"),
                TextField("response"),
                TagField("domain"),
                NumericField("timestamp"),
                NumericField("hit_count"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dim,
                        "DISTANCE_METRIC": config.redis.distance_metric,
                        "M": config.redis.hnsw_m,
                        "EF_CONSTRUCTION": config.redis.hnsw_ef_construction,
                    },
                ),
            )
            definition = IndexDefinition(
                prefix=[f"{self.prefix}:"],
                index_type=IndexType.HASH,
            )
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition,
            )
            logger.info("Redis HNSW index created successfully")

    def search(self, embedding: np.ndarray, top_k: int = 3) -> List[CacheEntry]:
        """
        Search for similar cached queries using HNSW.
        Returns top_k results sorted by similarity (highest first).
        """
        query_blob = embedding.astype(np.float32).tobytes()

        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("query", "response", "domain", "timestamp", "hit_count", "score")
            .paging(0, top_k)
            .dialect(2)
        )
        q.timeout(5000)

        try:
            params = {"vec": query_blob}
            # Set EF_RUNTIME for this query
            self.redis_client.ft(self.index_name).config_set(
                "DEFAULT_DIALECT", 2
            )
            results = self.redis_client.ft(self.index_name).search(q, query_params=params)
        except redis.ResponseError as e:
            logger.error(f"Redis search error: {e}")
            return []

        entries = []
        for doc in results.docs:
            # Redis COSINE returns distance (0=identical, 2=opposite)
            # Convert to similarity: sim = 1 - distance
            distance = float(doc.score)
            similarity = 1.0 - distance

            entries.append(CacheEntry(
                query=doc.query.decode() if isinstance(doc.query, bytes) else doc.query,
                response=doc.response.decode() if isinstance(doc.response, bytes) else doc.response,
                domain=doc.domain.decode() if isinstance(doc.domain, bytes) else doc.domain,
                similarity=similarity,
                timestamp=float(doc.timestamp),
                hit_count=int(doc.hit_count),
                cache_key=doc.id,
            ))

        return entries

    def store(self, query: str, response: str, embedding: np.ndarray, domain: str) -> str:
        """Store a new query-response pair in the cache."""
        # Use full content hash as key to avoid duplicates
        key_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"{self.prefix}:{key_hash}"

        mapping = {
            "query": query,
            "response": response,
            "domain": domain,
            "timestamp": time.time(),
            "hit_count": 0,
            "embedding": embedding.astype(np.float32).tobytes(),
        }

        self.redis_client.hset(cache_key, mapping=mapping)

        # Set TTL based on domain
        ttl = config.cache.ttl_by_domain.get(domain, config.cache.ttl_by_domain["general"])
        self.redis_client.expire(cache_key, ttl)

        logger.debug(f"Stored cache entry: key={cache_key}, domain={domain}, ttl={ttl}s")

        # Periodic eviction check (every 100 inserts)
        self._insert_count = getattr(self, '_insert_count', 0) + 1
        if self._insert_count % 100 == 0:
            self._enforce_max_entries()

        return cache_key

    def _enforce_max_entries(self):
        """Evict oldest cache entries when above max_entries limit."""
        try:
            keys = self.redis_client.keys(f"{self.prefix}:*")
            if len(keys) <= config.cache.max_entries:
                return

            # Collect timestamps for all entries
            entries = []
            for key in keys:
                ts = self.redis_client.hget(key, "timestamp")
                if ts is not None:
                    entries.append((key, float(ts)))

            # Sort by timestamp ascending (oldest first)
            entries.sort(key=lambda x: x[1])

            # Delete oldest entries until we're under the limit
            num_to_delete = len(entries) - config.cache.max_entries
            if num_to_delete > 0:
                keys_to_delete = [e[0] for e in entries[:num_to_delete]]
                self.redis_client.delete(*keys_to_delete)
                logger.info(f"Evicted {num_to_delete} oldest cache entries (was {len(entries)}, max={config.cache.max_entries})")
        except Exception as e:
            logger.warning(f"Cache eviction check failed: {e}")

    def increment_hit(self, cache_key: str):
        """Increment hit count for a cache entry."""
        try:
            self.redis_client.hincrby(cache_key, "hit_count", 1)
        except redis.RedisError:
            pass

    def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            info = self.redis_client.ft(self.index_name).info()
            # FT.INFO returns different formats across redis-py versions
            # Try attribute access first, then dict, then count keys manually
            num_docs = 0
            if hasattr(info, "num_docs"):
                num_docs = int(info.num_docs)
            elif isinstance(info, dict):
                num_docs = int(info.get("num_docs", 0))
            else:
                # Fallback: count keys directly
                keys = self.redis_client.keys(f"{self.prefix}:*")
                num_docs = len(keys)
            return {
                "total_entries": num_docs,
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats via FT.INFO: {e}")
            # Fallback: count keys directly
            try:
                keys = self.redis_client.keys(f"{self.prefix}:*")
                return {"total_entries": len(keys)}
            except Exception:
                return {"total_entries": 0}

    def flush(self):
        """Clear all cache entries (for testing)."""
        keys = self.redis_client.keys(f"{self.prefix}:*")
        if keys:
            self.redis_client.delete(*keys)
        logger.info(f"Flushed {len(keys)} cache entries")

    def is_connected(self) -> bool:
        try:
            self.redis_client.ping()
            return True
        except redis.ConnectionError:
            return False
