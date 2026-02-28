"""
Centralized configuration for the Semantic Cache system.
All tunables in one place — no magic numbers scattered in code.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    dimension: int = 768
    device: str = "cuda"
    batch_size: int = 32


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", 6379))
    index_name: str = "semantic_cache"
    prefix: str = "cache"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_runtime: int = 50
    distance_metric: str = "COSINE"


@dataclass
class MABConfig:
    arms: List[float] = field(default_factory=lambda: [0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95])
    alpha_init: float = 2.0
    beta_init: float = 1.0
    length_bins: List[str] = field(default_factory=lambda: ["short", "medium", "long"])
    domains: dict = field(default_factory=lambda: {
        "code": ["code", "python", "java", "function", "debug", "error", "script", "api", "sql", "html", "css",
                 "javascript", "react", "class", "import", "variable", "loop", "array", "git", "docker",
                 "algorithm", "compile", "syntax", "library", "module", "package", "npm", "pip"],
        "math": ["calculate", "equation", "integral", "derivative", "probability", "statistics", "formula",
                 "algebra", "geometry", "matrix", "vector", "theorem", "proof", "factorial", "logarithm"],
        "factual": ["who is", "what is", "when did", "where is", "how many", "capital of", "capital",
                    "president", "population", "founded", "invented", "born", "died", "define",
                    "meaning of", "definition", "history of", "discovered"],
        "creative": ["write", "story", "poem", "essay", "summarize", "explain", "describe", "compare",
                     "analyze", "suggest", "recommend", "generate", "create", "cook", "recipe",
                     "make", "prepare", "bake", "ingredients", "how to cook", "how to make",
                     "how to prepare", "how to bake", "design", "plan", "draft", "compose"],
    })
    default_domain: str = "general"
    use_enhanced_context: bool = True


@dataclass
class CacheConfig:
    ttl_by_domain: dict = field(default_factory=lambda: {
        "factual": 3600,
        "code": 86400 * 7,
        "math": 86400 * 30,
        "creative": 86400 * 3,
        "general": 86400,
    })
    max_entries: int = 50000
    quality_sample_rate: float = 0.1


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "llama-3.1-8b-instant"
    cost_per_1m_input_tokens: float = 0.06
    cost_per_1m_output_tokens: float = 0.06
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass
class ResilienceConfig:
    """Production hardening: circuit breaker, dedup, warmup."""
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    dedup_enabled: bool = True
    dedup_window_s: float = 2.0
    warmup_enabled: bool = False
    warmup_queries_file: str = "warmup_queries.json"


@dataclass
class ABTestConfig:
    """A/B testing: compare MAB vs static threshold in production."""
    enabled: bool = False
    experiment_traffic_pct: float = 0.5
    control_threshold: float = 0.85


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    mab: MABConfig = field(default_factory=MABConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


config = Config()
