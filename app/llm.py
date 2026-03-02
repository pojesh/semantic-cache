"""
LLM Provider: abstraction layer for LLM API calls.
Supports Ollama (free local) and Groq (fast cloud).
Tracks token usage and cost.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from config import config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    provider: str


@dataclass
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_latency_ms: float = 0.0

    def record(self, resp: LLMResponse):
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cost_usd += resp.cost_usd
        self.total_calls += 1
        self.total_latency_ms += resp.latency_ms

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_calls": self.total_calls,
            "avg_latency_ms": round(self.total_latency_ms / max(self.total_calls, 1), 1),
        }


class LLMProvider:
    def __init__(self):
        self.provider = config.llm.provider
        self.cost_tracker = CostTracker()
        # Shared async client — reuses connections across requests
        self._async_client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"LLM provider initialized: {self.provider}")

    async def close(self):
        """Close the shared HTTP client on shutdown."""
        await self._async_client.aclose()

    async def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Generate a response from the LLM."""
        if self.provider == "ollama":
            return await self._ollama_generate(prompt, system_prompt)
        elif self.provider == "groq":
            return await self._groq_generate(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _ollama_generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Call Ollama local LLM."""
        start = time.time()

        payload = {
            "model": config.llm.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.llm.temperature,
                "num_predict": config.llm.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = await self._async_client.post(
                f"{config.llm.ollama_base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

        latency_ms = (time.time() - start) * 1000
        text = data.get("response", "")

        # Ollama provides token counts
        input_tokens = data.get("prompt_eval_count", len(prompt.split()) * 2)
        output_tokens = data.get("eval_count", len(text.split()) * 2)

        cost = self._calculate_cost(input_tokens, output_tokens)

        result = LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            provider="ollama",
        )
        self.cost_tracker.record(result)
        return result

    async def _groq_generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Call Groq cloud API."""
        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config.llm.groq_model,
            "messages": messages,
            "max_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
        }

        try:
            resp = await self._async_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.llm.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            raise

        latency_ms = (time.time() - start) * 1000
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        cost = self._calculate_cost(input_tokens, output_tokens)

        result = LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            provider="groq",
        )
        self.cost_tracker.record(result)
        return result

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * config.llm.cost_per_1m_input_tokens
        output_cost = (output_tokens / 1_000_000) * config.llm.cost_per_1m_output_tokens
        return input_cost + output_cost

    def get_cost_stats(self) -> dict:
        return self.cost_tracker.to_dict()

