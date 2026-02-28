"""
Semantic Cache API Server (Enhanced).

Core lifecycle:
1. User sends query → /chat
2. Embed query with MPNet
3. A/B routing: MAB (experiment) vs static threshold (control)
4. MAB selects similarity threshold based on rich context
5. Search Redis HNSW for cached similar queries
6. If hit + quality check passes → return cached response (fast, free)
7. If miss → circuit breaker check → LLM call (with dedup) → cache
8. Sampled quality verification feeds back into MAB

Production features: circuit breaker, request dedup, A/B testing.
"""
import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from app.embeddings import EmbeddingService
from app.cache import VectorCache, CacheEntry
from app.mab import ContextualMAB
from app.llm import LLMProvider
from app.quality import QualityChecker, build_judge_prompt
from app.metrics import metrics
from app.resilience import CircuitBreaker, RequestDeduplicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Global services
embedder: Optional[EmbeddingService] = None
cache: Optional[VectorCache] = None
mab: Optional[ContextualMAB] = None
llm: Optional[LLMProvider] = None
quality_checker: Optional[QualityChecker] = None
circuit_breaker: Optional[CircuitBreaker] = None
deduplicator: Optional[RequestDeduplicator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, cache, mab, llm, quality_checker, circuit_breaker, deduplicator
    logger.info("Initializing services...")
    embedder = EmbeddingService()
    cache = VectorCache()
    mab = ContextualMAB()
    llm = LLMProvider()
    quality_checker = QualityChecker()
    circuit_breaker = CircuitBreaker()
    deduplicator = RequestDeduplicator()
    logger.info("All services initialized")
    yield
    mab._save_state()
    logger.info("Shutting down...")


app = FastAPI(
    title="Semantic Cache for LLM Serving",
    description="Adaptive semantic caching with MAB threshold selection, quality verification, and production resilience",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    force_llm: bool = False
    system_prompt: str = "You are a helpful assistant. Answer concisely."


class ChatResponse(BaseModel):
    response: str
    source: str  # "cache" or "llm"
    latency_ms: float
    similarity: Optional[float] = None
    threshold_used: Optional[float] = None
    domain: Optional[str] = None
    cost_usd: float = 0.0
    quality_confidence: Optional[float] = None
    cached_query: Optional[str] = None
    ab_group: Optional[str] = None  # "experiment" or "control"


class FeedbackRequest(BaseModel):
    query: str
    was_helpful: bool


# ─── A/B Test Routing ────────────────────────────────────────────────────

def _select_ab_group() -> str:
    """Route request to experiment (MAB) or control (static threshold)."""
    if not config.ab_test.enabled:
        return "experiment"
    return "experiment" if random.random() < config.ab_test.experiment_traffic_pct else "control"


# ─── Main Chat Endpoint ──────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # Step 1: Embed the query
    embedding = embedder.encode(query)

    # Step 2: A/B test routing
    ab_group = _select_ab_group()

    # Step 3: Select threshold
    if ab_group == "experiment":
        threshold, arm_idx, domain, length_bin = mab.select_threshold(query, embedding)
    else:
        threshold = config.ab_test.control_threshold
        arm_idx = -1
        ctx = mab.context_extractor.extract(query, embedding)
        domain = ctx["domain"]
        length_bin = ctx["length_bin"]

    # Step 4: Search cache (unless forced LLM)
    if not req.force_llm:
        results = cache.search(embedding, top_k=3)

        if results and results[0].similarity >= threshold:
            best = results[0]

            # Step 5: Quality gate
            is_ok, confidence, reason = quality_checker.check(query, best.query, best.response)

            if is_ok:
                latency_s = time.time() - start
                estimated_tokens = len(query.split()) * 2 + 150
                estimated_cost = (estimated_tokens / 1_000_000) * (
                    config.llm.cost_per_1m_input_tokens + config.llm.cost_per_1m_output_tokens
                )

                metrics.record_cache_hit(latency_s, best.similarity, domain, threshold, estimated_cost, ab_group)
                cache.increment_hit(best.cache_key)

                if ab_group == "experiment":
                    mab.update(domain, length_bin, arm_idx, "good_hit", similarity=best.similarity)

                # Sampled async quality verification
                if random.random() < config.cache.quality_sample_rate:
                    asyncio.create_task(
                        _async_quality_verify(query, best.query, best.response, domain, length_bin, arm_idx, ab_group)
                    )

                logger.info(
                    f"CACHE HIT [{ab_group}] | sim={best.similarity:.3f} τ={threshold:.2f} "
                    f"domain={domain} lat={latency_s*1000:.1f}ms | '{best.query[:60]}...'"
                )

                return ChatResponse(
                    response=best.response, source="cache",
                    latency_ms=round(latency_s * 1000, 1),
                    similarity=round(best.similarity, 4),
                    threshold_used=threshold, domain=domain,
                    cost_usd=0.0, quality_confidence=round(confidence, 3),
                    cached_query=best.query, ab_group=ab_group,
                )
            else:
                logger.info(f"QUALITY GATE REJECTED | sim={best.similarity:.3f} reason={reason}")
                if ab_group == "experiment":
                    mab.update(domain, length_bin, arm_idx, "bad_hit", similarity=best.similarity)
                metrics.record_false_positive(ab_group)

    # Step 6: Circuit breaker check
    if not circuit_breaker.can_execute():
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable (circuit breaker open). Try again shortly."
        )

    # Step 7: LLM call with deduplication
    try:
        async def _call_llm():
            return await llm.generate(query, system_prompt=req.system_prompt)

        llm_response = await deduplicator.execute_or_wait(query, _call_llm)
        circuit_breaker.record_success()
    except Exception as e:
        circuit_breaker.record_failure()
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    latency_s = time.time() - start

    # Step 8: Cache the new response
    cache.store(query, llm_response.text, embedding, domain)

    # Step 9: Update MAB
    if ab_group == "experiment":
        mab.update(domain, length_bin, arm_idx, "miss")

    metrics.record_cache_miss(latency_s, llm_response.cost_usd, domain, threshold, ab_group)

    logger.info(
        f"CACHE MISS [{ab_group}] | τ={threshold:.2f} domain={domain} "
        f"lat={latency_s*1000:.1f}ms cost=${llm_response.cost_usd:.6f} "
        f"tokens={llm_response.input_tokens+llm_response.output_tokens}"
    )

    return ChatResponse(
        response=llm_response.text, source="llm",
        latency_ms=round(latency_s * 1000, 1),
        threshold_used=threshold, domain=domain,
        cost_usd=llm_response.cost_usd, ab_group=ab_group,
    )


async def _async_quality_verify(query, cached_query, cached_response,
                                 domain, length_bin, arm_idx, ab_group):
    """Background quality verification using LLM-as-judge."""
    try:
        judge_prompt = build_judge_prompt(query, cached_query, cached_response)
        judge_resp = await llm.generate(judge_prompt, system_prompt="You are a quality evaluator. Be concise.")
        verdict = judge_resp.text.strip().upper()
        if "BAD" in verdict:
            logger.warning(f"QUALITY VERIFY: BAD — '{query[:50]}...'")
            if ab_group == "experiment" and arm_idx >= 0:
                mab.update(domain, length_bin, arm_idx, "bad_hit")
            metrics.record_false_positive(ab_group)
            metrics.record_quality_score(0.0)
        else:
            metrics.record_quality_score(1.0)
    except Exception as e:
        logger.warning(f"Quality verification failed: {e}")


# ─── Feedback Endpoint ───────────────────────────────────────────────────

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    return {"status": "recorded", "helpful": req.was_helpful}


# ─── Monitoring Endpoints ────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    return {
        "metrics": metrics.get_summary(),
        "cache": cache.get_stats(),
        "mab_thresholds": mab.get_recommended_thresholds(),
        "llm_costs": llm.get_cost_stats(),
        "resilience": {
            "circuit_breaker": circuit_breaker.get_stats(),
            "deduplicator": deduplicator.get_stats(),
        },
        "ab_test": metrics.get_ab_summary() if config.ab_test.enabled else None,
    }


@app.get("/stats/mab")
async def get_mab_stats():
    return mab.get_stats()


@app.get("/stats/mab/decisions")
async def get_mab_decisions(last_n: int = 100):
    """Recent MAB decisions for visualization."""
    return mab.get_decision_log(last_n)


@app.get("/stats/mab/learning")
async def get_mab_learning():
    """Threshold evolution curves for dashboard."""
    return mab.get_learning_curves()


@app.get("/stats/mab/regret")
async def get_mab_regret():
    """Cumulative regret analysis."""
    return mab.get_regret_analysis()


@app.get("/health")
async def health():
    redis_ok = cache.is_connected()
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "embedding_model": config.embedding.model_name,
        "llm_provider": config.llm.provider,
        "circuit_breaker": circuit_breaker.state.value if circuit_breaker else "unknown",
        "ab_test": "enabled" if config.ab_test.enabled else "disabled",
    }


@app.post("/cache/flush")
async def flush_cache():
    cache.flush()
    metrics.reset()
    return {"status": "flushed"}


@app.get("/metrics")
async def prometheus_metrics():
    from fastapi.responses import Response
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "prometheus_client not installed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
