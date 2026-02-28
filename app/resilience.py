"""
Production Resilience Layer: circuit breaker, request dedup, cache warming.

Circuit Breaker: After N consecutive LLM failures, stop calling it and return
degraded responses. Auto-recovery after timeout.

Request Deduplication: If the same query arrives within a short window
(thundering herd), only one LLM call is made.

Cache Warming: Pre-populate cache with common queries on startup.
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Callable, Dict, Optional

from config import config

logger = logging.getLogger(__name__)


# ─── Circuit Breaker ─────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # LLM is down, reject calls
    HALF_OPEN = "half_open" # trying one call to see if recovered


class CircuitBreaker:
    """
    Protects against LLM outages.
    CLOSED → normal. After `failure_threshold` consecutive failures → OPEN.
    OPEN → reject all calls, return fallback. After `recovery_timeout` → HALF_OPEN.
    HALF_OPEN → try one call. Success → CLOSED. Failure → OPEN.
    """

    def __init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.failure_threshold = config.resilience.failure_threshold
        self.recovery_timeout = config.resilience.recovery_timeout_s
        self._stats = {"trips": 0, "recoveries": 0, "rejected": 0}

    def can_execute(self) -> bool:
        if not config.resilience.circuit_breaker_enabled:
            return True

        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker: OPEN → HALF_OPEN (trying recovery)")
                self.state = CircuitState.HALF_OPEN
                return True
            self._stats["rejected"] += 1
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True  # allow one probe
        return False

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker: HALF_OPEN → CLOSED (recovered)")
            self._stats["recoveries"] += 1
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker: HALF_OPEN → OPEN (still failing)")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker: CLOSED → OPEN after {self.failure_count} consecutive failures"
            )
            self.state = CircuitState.OPEN
            self._stats["trips"] += 1

    def get_stats(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            **self._stats,
        }


# ─── Request Deduplication ───────────────────────────────────────────────

class RequestDeduplicator:
    """
    Collapse identical concurrent requests into one LLM call.
    Prevents thundering herd when many users ask the same thing.
    """

    def __init__(self):
        self._pending: Dict[str, asyncio.Future] = {}
        self._window = config.resilience.dedup_window_s
        self._stats = {"deduped": 0, "unique": 0}

    def _query_key(self, query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def execute_or_wait(self, query: str, llm_func: Callable) -> any:
        """
        If this query is already being processed, wait for the result.
        Otherwise, execute and share the result.
        """
        if not config.resilience.dedup_enabled:
            return await llm_func()

        key = self._query_key(query)

        if key in self._pending:
            self._stats["deduped"] += 1
            logger.debug(f"Dedup: waiting for in-flight query '{query[:40]}...'")
            return await self._pending[key]

        # Create a future for this query
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[key] = future
        self._stats["unique"] += 1

        try:
            result = await llm_func()
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            # Remove after a short window to catch near-simultaneous requests
            await asyncio.sleep(self._window)
            self._pending.pop(key, None)

    def get_stats(self) -> dict:
        return {
            "pending_requests": len(self._pending),
            **self._stats,
        }


# ─── Cache Warmer ────────────────────────────────────────────────────────

class CacheWarmer:
    """
    Pre-populate cache with common queries on startup.
    Queries come from a JSON file or hardcoded common patterns.
    """

    DEFAULT_WARMUP_QUERIES = [
        # Code
        "How to reverse a string in Python",
        "How to sort a list in Python",
        "What is a decorator in Python",
        "How to read a file in Python",
        "Explain list comprehension in Python",
        "How to handle exceptions in Python",
        "What is the difference between a list and tuple",
        "How to use async await in Python",
        # Factual
        "What is machine learning",
        "What is artificial intelligence",
        "Explain neural networks",
        "What is deep learning",
        "What is natural language processing",
        # General
        "How to write a good resume",
        "How to prepare for a job interview",
        "What is the best programming language to learn",
    ]

    def __init__(self):
        self.queries = self._load_queries()

    def _load_queries(self) -> list:
        filepath = config.resilience.warmup_queries_file
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load warmup queries: {e}")
        return self.DEFAULT_WARMUP_QUERIES

    async def warmup(self, chat_func: Callable, batch_size: int = 5):
        """Run warmup queries through the system."""
        if not config.resilience.warmup_enabled:
            return

        logger.info(f"Cache warming: processing {len(self.queries)} queries...")
        total = 0
        for i in range(0, len(self.queries), batch_size):
            batch = self.queries[i:i + batch_size]
            tasks = []
            for q in batch:
                # Import here to avoid circular deps
                from pydantic import BaseModel
                class WarmupReq:
                    query = q
                    force_llm = False
                    system_prompt = "You are a helpful assistant. Answer concisely."
                tasks.append(chat_func(WarmupReq()))
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                total += len(batch)
            except Exception as e:
                logger.warning(f"Warmup batch failed: {e}")

        logger.info(f"Cache warming complete: {total}/{len(self.queries)} queries processed")
