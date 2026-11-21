"""Tab 5 — Comparative analysis & live benchmark."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from rag import config
from rag.service import load_questions, run_benchmark

from .state import get_flags, get_hp

REFERENCE = pd.DataFrame([
    ["Simple keyword queries", "High precision / low latency", "Moderate (HyDE overhead)", "Moderate (graph overhead)"],
    ["Multi-hop relational queries", "Poor (misses indirect relationships)", "Moderate (semantic expansion)", "Superior (explores k-hop edges)"],
    ["Domain vocabulary mismatch", "Poor (exact vector distance)", "Superior (synthesises domain terms)", "High (entity resolution maps synonyms)"],
    ["Average latency (reference)", "~450 ms", "~1,200 ms", "~850 ms"],
    ["Hallucination risk", "Moderate (out-of-context chunks)", "Moderate (synthetic passage drift)", "Lowest (entity-relation grounding)"],
    ["Best enterprise use case", "Single-document lookup (SOPs)", "Vague regulatory inquiries", "Complex tax / M&A due diligence"],
], columns=["Metric", "Standard RAG", "HyDE RAG", "Graph RAG (LangGraph)"])
NAMES = {"standard": "Standard RAG", "hyde": "HyDE RAG", "graph": "Graph RAG"}


def render() -> None:
    s = st.session_state
    kb = s.kb
    qs = load_questions()
    top = st.container(horizontal=True, vertical_alignment="bottom")
    n_calls = len(qs) * 3
    top.caption(f"{len(qs)} benchmark questions × 3 engines" +
                (f" ≈ {n_calls * 3}+ LLM calls via OpenRouter (answers, HyDE, judge)." if config.has_api_key()
                 else " — no API key: retrieval metrics only, lexical faithfulness."))
    judge = top.toggle("LLM faithfulness judge", value=config.has_api_key(), key="bench_judge",
                       disabled=not config.has_api_key())
    run = top.button("Run benchmark", icon=":material/speed:", type="primary", disabled=kb is None, key="run_bench")
    if kb is None:
        st.info("Index the sample corpus first (benchmark questions target it).", icon=":material/info:")
    if run and kb is not None:
        bar = st.progress(0.0, text="Starting…")
        s.bench_rows = run_benchmark(kb, get_hp(), qs, flags=get_flags(), judge_llm=judge,
                                     progress=lambda p, t: bar.progress(p, text=t))
        bar.empty()

    if s.bench_rows:
        df = pd.DataFrame(s.bench_rows)
        df["Engine"] = df.engine.map(NAMES)
        agg = df.groupby("Engine")[["rcr", "faithfulness", "total_ms", "retrieval_ms", "graph_ms", "llm_ms"]].mean()
        agg = agg.reindex(list(NAMES.values())).reset_index()
        st.markdown("##### Measured results")
        cols = st.columns(3)
        for col, (_, r) in zip(cols, agg.iterrows()):
            with col, st.container(border=True):
                st.markdown(f"**{r.Engine}**")
                a, b = st.columns(2)
                a.metric("Recall", f"{r.rcr:.0%}")
                b.metric("Faithfulness", f"{r.faithfulness:.2f}")
                st.metric("Latency", f"{r.total_ms:,.0f} ms")
        c1, c2 = st.columns(2)
        with c1:
            q = agg.melt(id_vars="Engine", value_vars=["rcr", "faithfulness"], var_name="KPI", value_name="score")
            q["KPI"] = q.KPI.map({"rcr": "Context recall", "faithfulness": "Faithfulness"})
            piv = q.pivot(index="KPI", columns="Engine", values="score").reset_index()
            st.echarts_chart({"legend": {}, "tooltip": {"trigger": "axis", "valueFormatter": "%.2f"},
                              "dataset": {"source": piv}, "xAxis": {"type": "category"}, "yAxis": {"max": 1},
                              "series": [{"type": "bar"} for _ in NAMES]}, height=300,
                             alt="Grouped bars of context recall and faithfulness for the three engines")
        with c2:
            lat = agg[["Engine", "retrieval_ms", "graph_ms", "llm_ms"]]
            lat = lat.rename(columns={"retrieval_ms": "Retrieval", "graph_ms": "Graph traversal", "llm_ms": "LLM"})
            st.echarts_chart({"legend": {}, "tooltip": {"trigger": "axis", "valueFormatter": "%,.0f ms"},
                              "dataset": {"source": lat}, "xAxis": {"type": "category"}, "yAxis": {"name": "ms"},
                              "series": [{"type": "bar", "stack": "t"} for _ in range(3)]}, height=300,
                             alt="Stacked latency breakdown per engine in milliseconds")
        st.markdown("##### By query type")
        bt = df.groupby(["type", "Engine"])[["rcr", "faithfulness", "total_ms"]].mean().reset_index()
        piv = bt.pivot(index="type", columns="Engine", values="rcr").reset_index().rename(columns={"type": "Query type"})
        st.dataframe(piv, hide_index=True, alt="Context recall by query type and engine",
                     column_config={n: st.column_config.ProgressColumn(n, min_value=0, max_value=1, format="percent")
                                    for n in NAMES.values() if n in piv})
        with st.expander("Per-question results", icon=":material/table:"):
            st.dataframe(df[["id", "type", "Engine", "rcr", "faithfulness", "total_ms"]], hide_index=True,
                         alt="Per-question benchmark results")

    st.markdown("##### Reference expectations (from the architecture specification)")
    st.dataframe(REFERENCE, hide_index=True, alt="Reference comparison matrix of the three RAG architectures")
