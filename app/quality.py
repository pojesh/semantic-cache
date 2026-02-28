"""
Quality Verification: validates whether a cached response actually answers the new query.

This is the KEY INNOVATION that fixes the false-positive problem in existing semantic caches.
Instead of blindly trusting similarity scores, we verify cache hit quality.

Two modes:
1. Heuristic check (fast, free): basic sanity checks
2. LLM-as-judge (slower, costs tokens): sample-based verification using the LLM itself
"""
import logging
import re
from typing import Optional, Tuple

from config import config

logger = logging.getLogger(__name__)


class QualityChecker:
    """Lightweight heuristic quality checks for cache hits."""

    def check(self, original_query: str, cached_query: str, cached_response: str) -> Tuple[bool, float, str]:
        """
        Heuristic quality check. Returns (is_acceptable, confidence, reason).
        """
        issues = []
        confidence = 1.0

        # 1. Check if queries have different intent indicators
        if self._different_intent(original_query, cached_query):
            confidence -= 0.4
            issues.append("different_intent")

        # 2. Check if queries mention different specific entities
        orig_entities = self._extract_specifics(original_query)
        cached_entities = self._extract_specifics(cached_query)
        if orig_entities and cached_entities and not orig_entities.intersection(cached_entities):
            confidence -= 0.5
            issues.append("different_entities")

        # 3. Check response isn't truly empty (but short answers like "Paris." are valid)
        stripped = cached_response.strip()
        if len(stripped) == 0:
            confidence -= 0.8
            issues.append("empty_response")
        elif len(stripped) < 3:
            confidence -= 0.3
            issues.append("very_short_response")

        # 4. Check for negation mismatch
        if self._negation_mismatch(original_query, cached_query):
            confidence -= 0.6
            issues.append("negation_mismatch")

        is_acceptable = confidence > 0.5
        reason = ", ".join(issues) if issues else "passed"

        return is_acceptable, confidence, reason

    def _different_intent(self, q1: str, q2: str) -> bool:
        """Check if two queries have fundamentally different intents."""
        intent_words = {
            "how": "procedural", "why": "causal", "what": "factual",
            "when": "temporal", "where": "spatial", "who": "entity",
            "compare": "comparative", "list": "enumerative",
            "explain": "explanatory", "fix": "troubleshooting",
            "create": "generative", "delete": "destructive",
        }
        q1_lower = q1.lower().split()
        q2_lower = q2.lower().split()

        intent1 = None
        intent2 = None
        for word, intent in intent_words.items():
            if word in q1_lower[:5]:
                intent1 = intent
            if word in q2_lower[:5]:
                intent2 = intent

        if intent1 and intent2 and intent1 != intent2:
            return True
        return False

    def _extract_specifics(self, query: str) -> set:
        """Extract specific terms (numbers, quoted strings, technical terms)."""
        specifics = set()
        # Numbers
        specifics.update(re.findall(r'\b\d+\b', query))
        # Quoted strings
        specifics.update(re.findall(r'"([^"]+)"', query))
        specifics.update(re.findall(r"'([^']+)'", query))
        # Programming-specific: variable names, function calls
        specifics.update(re.findall(r'\b[a-z_]+\([^)]*\)', query.lower()))
        return specifics

    def _negation_mismatch(self, q1: str, q2: str) -> bool:
        """Check if one query negates what the other asks."""
        neg_words = {"not", "don't", "doesn't", "without", "never", "no", "isn't", "aren't", "won't"}
        q1_has_neg = bool(neg_words.intersection(q1.lower().split()))
        q2_has_neg = bool(neg_words.intersection(q2.lower().split()))
        return q1_has_neg != q2_has_neg


def build_judge_prompt(original_query: str, cached_query: str, cached_response: str) -> str:
    """
    Build a prompt for LLM-as-judge quality verification.
    Used for sampled verification of cache hits.
    """
    return f"""You are evaluating whether a cached response is appropriate for a new query.

ORIGINAL CACHED QUERY: {cached_query}
CACHED RESPONSE: {cached_response[:500]}

NEW QUERY: {original_query}

Is this cached response a good answer for the new query?
Consider: Does it answer what's actually being asked? Are there any factual mismatches?

Respond with ONLY one of:
- GOOD: The cached response adequately answers the new query
- BAD: The cached response does NOT adequately answer the new query

Your verdict:"""
