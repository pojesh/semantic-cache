"""
Enhanced Contextual Multi-Armed Bandit for adaptive threshold selection.

Context features (enhanced):
  - domain: code, math, factual, creative, general
  - length_bin: short, medium, long
  - complexity: simple, compound, multi_entity (NEW)
  - specificity: generic, specific (NEW)

Thompson Sampling with Beta distributions.
Full decision logging for real-time visualization.
"""
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ArmState:
    alpha: float
    beta: float

    def sample(self) -> float:
        return float(np.random.beta(self.alpha, self.beta))

    def expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Bayesian credible interval."""
        from scipy.stats import beta as beta_dist
        low = beta_dist.ppf((1 - level) / 2, self.alpha, self.beta)
        high = beta_dist.ppf((1 + level) / 2, self.alpha, self.beta)
        return (round(float(low), 4), round(float(high), 4))

    def total_observations(self) -> int:
        return int(self.alpha + self.beta - 2)  # subtract priors

    def to_dict(self):
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d):
        return cls(alpha=d["alpha"], beta=d["beta"])


@dataclass
class DecisionRecord:
    """Single MAB decision, stored for analysis and visualization."""
    timestamp: float
    query_preview: str
    context_key: str
    domain: str
    length_bin: str
    complexity: str
    specificity: str
    arm_index: int
    threshold: float
    reward: str = ""  # filled after outcome
    similarity: float = 0.0
    alpha_after: float = 0.0
    beta_after: float = 0.0


class EnhancedContextExtractor:
    """Rich context extraction for better MAB decisions."""

    def __init__(self):
        self.domains = config.mab.domains

    def extract(self, query: str, embedding: Optional[np.ndarray] = None) -> dict:
        """Returns full context dict."""
        return {
            "domain": self._detect_domain(query),
            "length_bin": self._length_bin(query),
            "complexity": self._query_complexity(query),
            "specificity": self._query_specificity(query),
        }

    def extract_simple(self, query: str) -> Tuple[str, str]:
        """Backward-compatible: returns (domain, length_bin)."""
        return self._detect_domain(query), self._length_bin(query)

    def _detect_domain(self, query: str) -> str:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scores = {}
        for domain, keywords in self.domains.items():
            score = 0
            for kw in keywords:
                if " " in kw:
                    if kw in query_lower:
                        score += 2  # multi-word matches are higher confidence
                else:
                    if kw in query_words:
                        score += 1
            if score > 0:
                scores[domain] = score
        if scores:
            return max(scores, key=scores.get)
        return config.mab.default_domain

    def _length_bin(self, query: str) -> str:
        word_count = len(query.split())
        if word_count <= 8:
            return "short"
        elif word_count <= 25:
            return "medium"
        return "long"

    def _query_complexity(self, query: str) -> str:
        """Detect query complexity — compound queries need stricter matching."""
        q_lower = query.lower()
        # Multi-hop: multiple questions or "and then"
        if q_lower.count("?") > 1 or " and " in q_lower and "?" in q_lower:
            return "compound"
        # Queries with specific numbers/versions/dates
        if len(re.findall(r'\b\d+\.?\d*\b', query)) >= 2:
            return "multi_entity"
        # Queries with comparisons
        if any(w in q_lower for w in ["vs", "versus", "compare", "difference between", "or"]):
            return "compound"
        return "simple"

    def _query_specificity(self, query: str) -> str:
        """Generic queries can tolerate looser thresholds; specific ones can't."""
        specificity_markers = 0
        # Quoted terms
        specificity_markers += len(re.findall(r'"[^"]+"', query))
        specificity_markers += len(re.findall(r"'[^']+'", query))
        # Specific versions or identifiers
        specificity_markers += len(re.findall(r'\b\d+\.\d+', query))
        # Proper nouns (capitalized words not at sentence start)
        words = query.split()
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and w.isalpha():
                specificity_markers += 1
        # Technical identifiers
        specificity_markers += len(re.findall(r'\b[a-z]+_[a-z]+\b', query))

        return "specific" if specificity_markers >= 2 else "generic"


class ContextualMAB:
    """
    Enhanced Contextual Thompson Sampling with rich features and decision logging.
    """

    def __init__(self):
        self.arms = config.mab.arms
        self.use_enhanced = config.mab.use_enhanced_context
        self.context_extractor = EnhancedContextExtractor()
        self._state: Dict[str, List[ArmState]] = defaultdict(self._init_arms)
        self._state_file = "mab_state.json"
        self._load_state()
        # Decision log — ring buffer for visualization
        self._decision_log: List[DecisionRecord] = []
        self._max_log_size = 2000

    def _init_arms(self) -> List[ArmState]:
        arms = []
        for threshold in self.arms:
            if 0.82 <= threshold <= 0.90:
                alpha = config.mab.alpha_init * 3
            elif threshold >= 0.92:
                alpha = config.mab.alpha_init * 2
            else:
                alpha = config.mab.alpha_init
            arms.append(ArmState(alpha=alpha, beta=config.mab.beta_init))
        return arms

    def _context_key(self, ctx: dict) -> str:
        if self.use_enhanced:
            return f"{ctx['domain']}:{ctx['length_bin']}:{ctx['complexity']}:{ctx['specificity']}"
        return f"{ctx['domain']}:{ctx['length_bin']}"

    def select_threshold(self, query: str, embedding: np.ndarray = None) -> Tuple[float, int, str, str]:
        """
        Select threshold via Thompson Sampling.
        Returns: (threshold, arm_index, domain, length_bin) — backward compatible.
        Also logs the decision for visualization.
        """
        ctx = self.context_extractor.extract(query, embedding)
        ctx_key = self._context_key(ctx)
        arm_states = self._state[ctx_key]

        # Thompson Sampling
        samples = [arm.sample() for arm in arm_states]
        best_idx = int(np.argmax(samples))
        threshold = self.arms[best_idx]

        # Log decision
        record = DecisionRecord(
            timestamp=time.time(),
            query_preview=query[:80],
            context_key=ctx_key,
            domain=ctx["domain"],
            length_bin=ctx["length_bin"],
            complexity=ctx.get("complexity", "simple"),
            specificity=ctx.get("specificity", "generic"),
            arm_index=best_idx,
            threshold=threshold,
        )
        self._decision_log.append(record)
        if len(self._decision_log) > self._max_log_size:
            self._decision_log = self._decision_log[-self._max_log_size:]

        logger.debug(
            f"MAB: ctx={ctx_key} | τ={threshold:.2f} (arm {best_idx}) | "
            f"samples=[{', '.join(f'{s:.3f}' for s in samples)}]"
        )

        return threshold, best_idx, ctx["domain"], ctx["length_bin"]

    def update(self, domain: str, length_bin: str, arm_index: int, reward: str,
               similarity: float = 0.0, complexity: str = "simple", specificity: str = "generic"):
        """Update arm distribution. Supports both simple and enhanced context."""
        if self.use_enhanced:
            ctx = {"domain": domain, "length_bin": length_bin,
                   "complexity": complexity, "specificity": specificity}
        else:
            ctx = {"domain": domain, "length_bin": length_bin,
                   "complexity": "simple", "specificity": "generic"}

        ctx_key = self._context_key(ctx)
        arm = self._state[ctx_key][arm_index]

        if reward == "good_hit":
            arm.alpha += 1.0
        elif reward == "bad_hit":
            arm.beta += 2.0
        elif reward == "miss":
            arm.beta += 0.3

        # Update last decision record
        if self._decision_log:
            last = self._decision_log[-1]
            if last.arm_index == arm_index and last.context_key == ctx_key:
                last.reward = reward
                last.similarity = similarity
                last.alpha_after = arm.alpha
                last.beta_after = arm.beta

        if len(self._decision_log) % 50 == 0:
            self._save_state()

    def get_stats(self) -> dict:
        stats = {}
        for ctx_key, arms in self._state.items():
            stats[ctx_key] = {
                f"τ={self.arms[i]:.2f}": {
                    "α": round(arm.alpha, 1),
                    "β": round(arm.beta, 1),
                    "E[reward]": round(arm.expected_value(), 3),
                    "observations": arm.total_observations(),
                }
                for i, arm in enumerate(arms)
            }
        return stats

    def get_recommended_thresholds(self) -> dict:
        result = {}
        for ctx_key, arms in self._state.items():
            best_idx = max(range(len(arms)), key=lambda i: arms[i].expected_value())
            result[ctx_key] = {
                "threshold": self.arms[best_idx],
                "confidence": round(arms[best_idx].expected_value(), 3),
                "observations": arms[best_idx].total_observations(),
            }
        return result

    def get_decision_log(self, last_n: int = 100) -> List[dict]:
        """Return recent decisions for visualization."""
        records = self._decision_log[-last_n:]
        return [
            {
                "timestamp": r.timestamp,
                "query": r.query_preview,
                "context": r.context_key,
                "domain": r.domain,
                "complexity": r.complexity,
                "threshold": r.threshold,
                "reward": r.reward,
                "similarity": r.similarity,
            }
            for r in records
        ]

    def get_learning_curves(self) -> dict:
        """Compute per-context threshold evolution over time for charting."""
        curves = defaultdict(list)
        # Walk through decision log, compute running expected value
        running_state: Dict[str, Dict[int, ArmState]] = {}

        for record in self._decision_log:
            ctx = record.context_key
            if ctx not in running_state:
                running_state[ctx] = {i: ArmState(alpha=config.mab.alpha_init, beta=config.mab.beta_init)
                                      for i in range(len(self.arms))}
            arm = running_state[ctx][record.arm_index]
            if record.reward == "good_hit":
                arm.alpha += 1.0
            elif record.reward == "bad_hit":
                arm.beta += 2.0
            elif record.reward == "miss":
                arm.beta += 0.3

            # Find current best arm for this context
            best_idx = max(range(len(self.arms)),
                          key=lambda i: running_state[ctx][i].expected_value())
            curves[ctx].append({
                "timestamp": record.timestamp,
                "threshold": self.arms[best_idx],
                "reward": record.reward,
            })

        return dict(curves)

    def get_regret_analysis(self) -> dict:
        """Cumulative regret: how much cost we wasted due to suboptimal decisions."""
        cumulative_regret = 0.0
        regret_over_time = []
        for r in self._decision_log:
            if r.reward == "bad_hit":
                cumulative_regret += 1.0  # false positive = wasted user trust
            elif r.reward == "miss" and r.similarity > 0.80:
                cumulative_regret += 0.5  # potentially avoidable miss
            regret_over_time.append({
                "timestamp": r.timestamp,
                "cumulative_regret": round(cumulative_regret, 2),
            })
        return {
            "total_regret": round(cumulative_regret, 2),
            "decisions": len(self._decision_log),
            "timeline": regret_over_time[-100:],  # last 100 for charting
        }

    def _save_state(self):
        try:
            data = {}
            for ctx_key, arms in self._state.items():
                data[ctx_key] = [arm.to_dict() for arm in arms]
            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save MAB state: {e}")

    def _load_state(self):
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                for ctx_key, arms_data in data.items():
                    self._state[ctx_key] = [ArmState.from_dict(a) for a in arms_data]
                logger.info(f"Loaded MAB state with {len(data)} contexts")
            except Exception as e:
                logger.warning(f"Failed to load MAB state: {e}")
