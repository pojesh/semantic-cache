# Semantic Caching with Adaptive Threshold Selection for LLM Serving

> **Thesis**: Adaptive semantic caching with quality-aware threshold selection reduces LLM inference costs by 40-60% while maintaining >95% response quality, outperforming static-threshold approaches (GPTCache, MeanCache) and token-level methods (vLLM prefix caching).

## Architecture

```
User Query → FastAPI → MPNet Embedding → A/B Router
                                            │
                            ┌────────────────┼────────────────┐
                            │ Experiment     │ Control         │
                            │ (MAB)          │ (Static τ)     │
                            └────────┬───────┴────────┬───────┘
                                     ▼                ▼
                           MAB Threshold Selection    Fixed τ=0.85
                                     │
                                     ▼
                          Redis HNSW Vector Search
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                   HIT (sim ≥ τ)          MISS (sim < τ)
                          │                     │
                   Quality Gate         Circuit Breaker Check
                     │      │                   │
                  PASS    FAIL          Request Dedup → LLM API
                   │       │                            │
             Return     Treat as              Cache + Return
             Cached     MISS                  ($cost, ~800ms)
             (5ms, $0)
                          │
                   MAB Feedback Loop
                   (LLM-as-judge)
```

## Key Innovations

1. **Adaptive Thresholds via Thompson Sampling**: MAB learns domain-specific thresholds automatically. Code queries converge to τ≈0.90, factual to τ≈0.82. No manual tuning.

2. **Rich Context Features**: Domain detection, query complexity (simple/compound/multi-entity), specificity (generic/specific), length bins — 4 context dimensions vs 2 in basic MAB.

3. **Quality-Aware Caching**: Heuristic gate (intent, entities, negation) + sampled LLM-as-judge catches false positives that pure similarity misses.

4. **Production Resilience**: Circuit breaker (LLM failover), request deduplication (thundering herd), domain-aware TTL, A/B testing framework.

5. **Rigorous Evaluation**: Benchmarked against 5 published methods (GPTCache, MeanCache, vLLM, SCALM, MinCache), with ablation studies and failure mode analysis.

## Quick Start

### Prerequisites
- Python 3.10+
- Redis Stack (for vector search)
- Ollama with a model pulled (e.g., `ollama pull llama3.2:3b`)

### Setup
```bash
# 1. Redis Stack
docker run -d -p 6379:6379 redis/redis-stack-server:latest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. Start dashboard (new terminal)
streamlit run ui/app.py
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Main query endpoint |
| `/stats` | GET | Comprehensive metrics |
| `/stats/mab` | GET | Full MAB state |
| `/stats/mab/decisions` | GET | Recent MAB decisions |
| `/stats/mab/learning` | GET | Threshold evolution curves |
| `/stats/mab/regret` | GET | Cumulative regret analysis |
| `/health` | GET | System health check |
| `/cache/flush` | POST | Reset cache + metrics |
| `/metrics` | GET | Prometheus-format metrics |
| `/feedback` | POST | User feedback signal |

### Demo Script
```bash
# Query 1: LLM generates response
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "How to reverse a string in Python"}'

# Query 2: CACHE HIT (paraphrase detected, ~5ms)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "Give me Python code to reverse a string"}'

# Query 3: CACHE MISS (near-miss detected by quality gate)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "How to reverse a list in Python"}'
```

## Evaluation

### Full Benchmark (vs 8 methods)
```bash
python -m evaluation.benchmark --dataset synthetic --n 300
```
Compares: No Cache, Exact Match, GPTCache, MeanCache, vLLM Prefix, SCALM, MinCache, Static (0.80/0.85/0.90), **Ours (Adaptive MAB)**.

### Ablation Study
```bash
python -m evaluation.ablation
```
Removes each component to prove its contribution: quality gate, MAB, domain detection, enhanced context.

### Failure Mode Analysis
```bash
python -m evaluation.failure_modes
```
Tests: negation sensitivity, entity swaps, specificity traps, temporal queries, ambiguity, multilingual, adversarial, cross-domain confusion. Reports accuracy per category and optimal threshold analysis.

## Competitive Positioning

| Method | Threshold | Quality Gate | Adaptive | Domain-Aware | Key Weakness |
|--------|-----------|-------------|----------|--------------|-------------|
| GPTCache | Static | ❌ | ❌ | ❌ | One threshold for all domains |
| MeanCache | Static | ❌ | ❌ | ❌ | Privacy claim contradicted by central cache |
| vLLM Prefix | Token-level | ❌ | ❌ | ❌ | Only matches shared prefixes |
| SCALM | Static | ❌ | ❌ | Cluster | Cluster assignment is fragile |
| MinCache | Static | ❌ | ❌ | ❌ | MinHash misses nuanced semantics |
| **Ours** | **Adaptive MAB** | **✅** | **✅** | **✅** | Cold-start exploration period |

## Project Structure

```
semantic-cache/
├── config.py                    # Centralized configuration
├── app/
│   ├── main.py                  # FastAPI server + A/B testing
│   ├── embeddings.py            # MPNet embedding service
│   ├── cache.py                 # Redis HNSW vector cache
│   ├── mab.py                   # Enhanced Contextual MAB
│   ├── llm.py                   # Ollama/Groq LLM provider
│   ├── quality.py               # Quality gate (heuristic + LLM-as-judge)
│   ├── metrics.py               # Prometheus + A/B test metrics
│   └── resilience.py            # Circuit breaker, dedup, warmup
├── evaluation/
│   ├── benchmark.py             # Full benchmark suite
│   ├── baselines.py             # GPTCache, MeanCache, vLLM, SCALM, MinCache
│   ├── dataset_loader.py        # Synthetic + ShareGPT + MS MARCO loaders
│   ├── ablation.py              # Component ablation studies
│   └── failure_modes.py         # Systematic failure analysis
├── ui/
│   └── app.py                   # Streamlit dashboard (5 tabs)
├── docker-compose.yml           # Redis Stack + Prometheus + Grafana
├── monitoring/prometheus.yml
└── requirements.txt
```

## References

- GPTCache: Bang et al. (2023). [arXiv:2309.05534](https://arxiv.org/abs/2309.05534)
- MeanCache: Gill et al. (2025). [arXiv:2403.02694](https://arxiv.org/abs/2403.02694)
- CacheBlend: Yao et al. (2024). [arXiv:2405.16444](https://arxiv.org/abs/2405.16444)
- vLLM: Kwon et al. (2023). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- SCALM: Li et al. (2024). IWQoS 2024
- MinCache: Haqiq et al. (2025). FGCS 2025
- Thompson Sampling: Chapelle & Li (2011). NeurIPS 2011
