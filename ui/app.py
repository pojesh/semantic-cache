"""
Enhanced Semantic Cache Dashboard.

Tabs:
1. Chat — interactive query testing with cache indicators
2. Live Metrics — hit rate, precision, cost, latency
3. MAB Learning — real-time threshold adaptation visualization
4. A/B Testing — experiment vs control comparison
5. Evaluation — run benchmarks, ablations, failure analysis
"""
import json
import time
import requests
import streamlit as st
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Semantic Cache Dashboard",
    page_icon="⚡",
    layout="wide",
)


def api_get(path: str, default=None):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        return r.json() if r.ok else default
    except Exception:
        return default


def api_post(path: str, data: dict = None, default=None):
    try:
        r = requests.post(f"{API_URL}{path}", json=data, timeout=60)
        return r.json() if r.ok else default
    except Exception:
        return default


# ─── Header ──────────────────────────────────────────────────────────────

st.title("⚡ Semantic Cache for LLM Serving")
st.caption("Adaptive threshold selection via Contextual Multi-Armed Bandit")

health = api_get("/health", {})
status = health.get("status", "unknown")
redis = health.get("redis", "unknown")
llm_provider = health.get("llm_provider", "unknown")
cb_state = health.get("circuit_breaker", "unknown")
ab_status = health.get("ab_test", "disabled")

cols = st.columns(5)
cols[0].metric("Status", f"{'✅' if status == 'ok' else '⚠️'} {status}")
cols[1].metric("Redis", redis)
cols[2].metric("LLM", llm_provider)
cols[3].metric("Circuit Breaker", cb_state)
cols[4].metric("A/B Test", ab_status)

# ─── Tabs ────────────────────────────────────────────────────────────────

tab_chat, tab_metrics, tab_mab, tab_ab, tab_eval = st.tabs([
    "💬 Chat", "📊 Metrics", "🧠 MAB Learning", "🔬 A/B Testing", "📋 Evaluation"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Chat
# ═══════════════════════════════════════════════════════════════════════════

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "meta" in msg:
                m = msg["meta"]
                if m.get("source") == "cache":
                    st.success(
                        f"⚡ CACHE HIT | Similarity: {m.get('similarity', 0):.4f} | "
                        f"Threshold: {m.get('threshold', 0):.2f} | "
                        f"Latency: {m.get('latency', 0):.1f}ms | "
                        f"Domain: {m.get('domain', '?')}"
                    )
                    if m.get("cached_query"):
                        st.caption(f'Matched: "{m["cached_query"]}"')
                else:
                    st.info(
                        f"🤖 LLM GENERATED | Latency: {m.get('latency', 0):.1f}ms | "
                        f"Cost: ${m.get('cost', 0):.6f} | Domain: {m.get('domain', '?')}"
                    )

    user_input = st.chat_input("Ask anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        resp = api_post("/chat", {"query": user_input})
        if resp:
            meta = {
                "source": resp.get("source"),
                "similarity": resp.get("similarity"),
                "threshold": resp.get("threshold_used"),
                "latency": resp.get("latency_ms"),
                "cost": resp.get("cost_usd"),
                "domain": resp.get("domain"),
                "cached_query": resp.get("cached_query"),
                "ab_group": resp.get("ab_group"),
            }
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.get("response", "Error"),
                "meta": meta,
            })
            st.rerun()

    # Quick test pairs
    st.divider()
    st.subheader("🧪 Quick Test Pairs")
    st.caption("Click a pair to test cache behavior")

    test_pairs = [
        ("How to reverse a string in Python", "Give me Python code to reverse a string"),
        ("What is the capital of France", "France's capital city"),
        ("How to cook chicken biryani", "Chicken biryani recipe"),
        ("Explain list comprehension in Python", "What is list comprehension in Python"),
        ("How to reverse a string in Python", "How to reverse a list in Python"),  # near-miss
    ]
    for idx, (q1, q2) in enumerate(test_pairs):
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"1️⃣ {q1[:45]}...", key=f"q1_{idx}"):
                st.session_state.messages.append({"role": "user", "content": q1})
                resp = requests.post(f"{API_URL}/chat", json={"query": q1}, timeout=60).json()
                st.session_state.messages.append({
                    "role": "assistant", "content": resp.get("response", ""),
                    "meta": {"source": resp.get("source"), "latency": resp.get("latency_ms"),
                             "domain": resp.get("domain"), "cost": resp.get("cost_usd", 0),
                             "similarity": resp.get("similarity"), "threshold": resp.get("threshold_used"),
                             "cached_query": resp.get("cached_query")},
                })
                st.rerun()
        with c2:
            if st.button(f"2️⃣ {q2[:45]}...", key=f"q2_{idx}"):
                st.session_state.messages.append({"role": "user", "content": q2})
                resp = requests.post(f"{API_URL}/chat", json={"query": q2}, timeout=60).json()
                st.session_state.messages.append({
                    "role": "assistant", "content": resp.get("response", ""),
                    "meta": {"source": resp.get("source"), "latency": resp.get("latency_ms"),
                             "domain": resp.get("domain"), "cost": resp.get("cost_usd", 0),
                             "similarity": resp.get("similarity"), "threshold": resp.get("threshold_used"),
                             "cached_query": resp.get("cached_query")},
                })
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Live Metrics
# ═══════════════════════════════════════════════════════════════════════════

with tab_metrics:
    stats = api_get("/stats", {})
    m = stats.get("metrics", {})

    st.subheader("📊 Live Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Hit Rate", f"{m.get('hit_rate', 0):.1f}%")
    c2.metric("Precision", f"{m.get('precision', 0):.1f}%")
    c3.metric("Queries", m.get("total_queries", 0))
    c4.metric("💰 Saved", f"${m.get('cost_saved_usd', 0):.4f}")
    c5.metric("💸 Spent", f"${m.get('cost_spent_usd', 0):.4f}")
    c6.metric("False Pos.", m.get("false_positives", 0))

    # Latency comparison chart
    st.subheader("⏱ Latency Comparison")
    lat_data = {
        "Type": ["Cache (P50)", "Cache (P95)", "LLM (P50)", "LLM (P95)"],
        "Latency (ms)": [
            m.get("latency_cache_p50_ms", 0), m.get("latency_cache_p95_ms", 0),
            m.get("latency_llm_p50_ms", 0), m.get("latency_llm_p95_ms", 0),
        ],
    }
    st.bar_chart(pd.DataFrame(lat_data).set_index("Type"))

    # Cache & LLM stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🗄 Cache")
        cache_info = stats.get("cache", {})
        st.metric("Entries", cache_info.get("total_entries", 0))

    with col2:
        st.subheader("🧠 MAB Thresholds")
        thresholds = stats.get("mab_thresholds", {})
        for ctx, info in thresholds.items():
            st.write(f"**{ctx}**: τ={info.get('threshold', '?'):.2f} "
                     f"(conf={info.get('confidence', 0):.3f}, "
                     f"obs={info.get('observations', 0)})")

    with col3:
        st.subheader("💵 LLM Usage")
        st.json(stats.get("llm_costs", {}))

    # Resilience stats
    resilience = stats.get("resilience", {})
    if resilience:
        st.subheader("🛡️ Resilience")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.write("**Circuit Breaker**")
            st.json(resilience.get("circuit_breaker", {}))
        with rc2:
            st.write("**Request Deduplication**")
            st.json(resilience.get("deduplicator", {}))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: MAB Learning Visualization
# ═══════════════════════════════════════════════════════════════════════════

with tab_mab:
    st.subheader("🧠 MAB Online Learning Visualization")

    # Decision log
    decisions = api_get("/stats/mab/decisions?last_n=200", [])
    if decisions:
        df = pd.DataFrame(decisions)
        df["time"] = pd.to_datetime(df["timestamp"], unit="s")

        # Threshold over time
        st.subheader("📈 Threshold Selections Over Time")
        if "threshold" in df.columns:
            chart_data = df[["time", "threshold", "domain"]].copy()
            st.scatter_chart(chart_data, x="time", y="threshold", color="domain")

        # Reward distribution
        st.subheader("🎯 Reward Distribution")
        if "reward" in df.columns and len(df[df["reward"] != ""]) > 0:
            reward_counts = df[df["reward"] != ""]["reward"].value_counts()
            st.bar_chart(reward_counts)

        # Per-domain threshold evolution
        st.subheader("🗺️ Per-Domain Threshold Evolution")
        for domain in df["domain"].unique():
            domain_df = df[df["domain"] == domain]
            if len(domain_df) > 1:
                st.write(f"**{domain}** ({len(domain_df)} decisions)")
                st.line_chart(domain_df.set_index("time")["threshold"])
    else:
        st.info("No MAB decisions yet. Send some queries first!")

    # Regret analysis
    st.subheader("📉 Cumulative Regret Analysis")
    regret = api_get("/stats/mab/regret", {})
    if regret and regret.get("timeline"):
        regret_df = pd.DataFrame(regret["timeline"])
        regret_df["time"] = pd.to_datetime(regret_df["timestamp"], unit="s")
        st.line_chart(regret_df.set_index("time")["cumulative_regret"])
        st.metric("Total Regret", regret.get("total_regret", 0))
    else:
        st.info("Regret data builds as you use the system.")

    # Full MAB state
    st.subheader("📋 Full MAB State")
    mab_state = api_get("/stats/mab", {})
    if mab_state:
        for ctx_key, arms in mab_state.items():
            with st.expander(f"Context: {ctx_key}"):
                arm_data = []
                for arm_name, arm_info in arms.items():
                    arm_data.append({
                        "Arm": arm_name,
                        "E[reward]": arm_info.get("E[reward]", 0),
                        "α": arm_info.get("α", 0),
                        "β": arm_info.get("β", 0),
                        "Observations": arm_info.get("observations", 0),
                    })
                if arm_data:
                    adf = pd.DataFrame(arm_data)
                    st.dataframe(adf, use_container_width=True)
                    st.bar_chart(adf.set_index("Arm")["E[reward]"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: A/B Testing
# ═══════════════════════════════════════════════════════════════════════════

with tab_ab:
    st.subheader("🔬 A/B Test Results: MAB (Experiment) vs Static Threshold (Control)")

    stats = api_get("/stats", {})
    ab_data = stats.get("ab_test")

    if ab_data:
        exp = ab_data.get("experiment", {})
        ctrl = ab_data.get("control", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧪 Experiment (Adaptive MAB)")
            st.metric("Queries", exp.get("total_queries", 0))
            st.metric("Hit Rate", f"{exp.get('hit_rate_%', 0):.1f}%")
            st.metric("Precision", f"{exp.get('precision_%', 0):.1f}%")
            st.metric("Cost Saved", f"${exp.get('cost_saved_$', 0):.4f}")
            st.metric("False Positives", exp.get("false_positives", 0))

        with col2:
            st.markdown("### 🎛️ Control (Static τ=0.85)")
            st.metric("Queries", ctrl.get("total_queries", 0))
            st.metric("Hit Rate", f"{ctrl.get('hit_rate_%', 0):.1f}%")
            st.metric("Precision", f"{ctrl.get('precision_%', 0):.1f}%")
            st.metric("Cost Saved", f"${ctrl.get('cost_saved_$', 0):.4f}")
            st.metric("False Positives", ctrl.get("false_positives", 0))

        # Comparison chart
        compare_df = pd.DataFrame({
            "Metric": ["Hit Rate %", "Precision %", "Cost Saved $"],
            "Experiment (MAB)": [exp.get("hit_rate_%", 0), exp.get("precision_%", 0), exp.get("cost_saved_$", 0) * 10000],
            "Control (Static)": [ctrl.get("hit_rate_%", 0), ctrl.get("precision_%", 0), ctrl.get("cost_saved_$", 0) * 10000],
        })
        st.bar_chart(compare_df.set_index("Metric"))
    else:
        st.info("A/B testing is disabled. Enable in config.py: `ab_test.enabled = True`")
        st.code("""
# In config.py, update:
@dataclass
class ABTestConfig:
    enabled: bool = True  # ← Change to True
    experiment_traffic_pct: float = 0.5
    control_threshold: float = 0.85
        """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Evaluation
# ═══════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.subheader("📋 Evaluation Suite")

    eval_col1, eval_col2, eval_col3 = st.columns(3)

    with eval_col1:
        st.markdown("### 🏁 Benchmark")
        st.caption("Compare against GPTCache, MeanCache, vLLM, SCALM, MinCache, and static thresholds")
        st.code("python -m evaluation.benchmark --dataset synthetic --n 300", language="bash")

        if st.button("View Benchmark Results"):
            try:
                with open("benchmark_synthetic_300.json") as f:
                    results = json.load(f)
                for r in results.get("results", []):
                    st.write(f"**{r['method']}**: HR={r['hit_rate_%']}% P={r['precision_%']}% F1={r['f1_%']}%")
            except FileNotFoundError:
                st.warning("No benchmark results found. Run the benchmark first.")

    with eval_col2:
        st.markdown("### 🔬 Ablation Study")
        st.caption("What happens if we remove each component?")
        st.code("python -m evaluation.ablation", language="bash")

        if st.button("View Ablation Results"):
            try:
                with open("ablation_results.json") as f:
                    results = json.load(f)
                for r in results:
                    emoji = "✅" if r.get("f1_%", 0) > 50 else "⚠️"
                    st.write(f"{emoji} **{r['ablation']}**: F1={r['f1_%']}% P={r['precision_%']}%")
            except FileNotFoundError:
                st.warning("No ablation results found. Run the study first.")

    with eval_col3:
        st.markdown("### 💥 Failure Modes")
        st.caption("Where does the system break?")
        st.code("python -m evaluation.failure_modes", language="bash")

        if st.button("View Failure Analysis"):
            try:
                with open("failure_analysis.json") as f:
                    data = json.load(f)
                report = data.get("single", {})
                st.metric("Accuracy", f"{report.get('accuracy_%', 0)}%")
                st.metric("False Positives", report.get("false_positives", 0))
                st.metric("False Negatives", report.get("false_negatives", 0))
                for cat, info in report.get("categories", {}).items():
                    emoji = "✅" if info["failures"] == 0 else "❌"
                    st.write(f"{emoji} **{cat}**: {info['accuracy_%']}% ({info['failures']}/{info['total']})")
            except FileNotFoundError:
                st.warning("No failure analysis found. Run the analysis first.")

    st.divider()

    # Competitive positioning table
    st.subheader("📊 Competitive Positioning")
    st.markdown("""
    | Method | Threshold | Quality Gate | Adaptive | Domain-Aware | Production Ready |
    |--------|-----------|-------------|----------|--------------|-----------------|
    | **GPTCache** | Static | ❌ | ❌ | ❌ | ✅ |
    | **MeanCache** | Static | ❌ | ❌ | ❌ | Partial |
    | **vLLM Prefix** | Token-level | ❌ | ❌ | ❌ | ✅ |
    | **SCALM** | Static | ❌ | ❌ | Cluster | Partial |
    | **MinCache** | Static | ❌ | ❌ | ❌ | Partial |
    | **Ours** | **Adaptive MAB** | **✅** | **✅** | **✅** | **✅** |
    """)

    st.markdown("""
    **Key Differentiators:**
    - 🧠 **Adaptive thresholds** via Thompson Sampling (GPTCache/MeanCache use fixed τ)
    - ✅ **Quality gate** catches false positives that pure similarity misses
    - 🗺️ **Domain-aware** context (code needs τ≈0.90, factual tolerates τ≈0.82)
    - 🛡️ **Production resilience**: circuit breaker, request dedup, A/B testing
    """)

    # Paper references
    with st.expander("📚 Paper References"):
        st.markdown("""
        - **GPTCache**: Bang et al. (2023). *GPTCache: An Open-Source Semantic Cache for LLM Applications.* [arXiv:2309.05534](https://arxiv.org/abs/2309.05534)
        - **MeanCache**: Gill et al. (2025). *MeanCache: User-Centric Semantic Caching for LLM Web Services.* [arXiv:2403.02694](https://arxiv.org/abs/2403.02694)
        - **CacheBlend**: Yao et al. (2024). *CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion.* [arXiv:2405.16444](https://arxiv.org/abs/2405.16444)
        - **vLLM**: Kwon et al. (2023). *Efficient Memory Management for LLM Serving with PagedAttention.* [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
        - **SCALM**: Li et al. (2024). *SCALM: Towards Semantic Caching for Automated Chat Services with LLMs.* IWQoS 2024.
        - **MinCache**: Haqiq et al. (2025). *MinCache: An Efficient Caching Framework for LLMs.* FGCS 2025.
        - **Thompson Sampling**: Chapelle & Li (2011). *An Empirical Evaluation of Thompson Sampling.* NeurIPS 2011.
        """)
