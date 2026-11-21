"""Tab 4 — Hyperparameters & KPI telemetry."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from rag.service import er_accuracy

from .state import get_hp, index_documents
from rag.service import load_sample_docs


def render() -> None:
    s = st.session_state
    st.markdown("##### Hyperparameters")
    with st.container(border=True):
        c = st.columns(4, gap="large")
        with c[0]:
            st.slider("Chunk size (tokens)", 200, 1000, key="hp_chunk", step=50,
                      help="Statutory clause integrity: 600 recommended.")
            st.slider("Overlap (tokens)", 0, 200, key="hp_overlap", step=10)
        with c[1]:
            st.slider("Multi-HyDE passages (N)", 1, 6, key="hp_n", help="Synthetic documents from distinct regulatory angles.")
        with c[2]:
            st.slider("Graph hop radius (k)", 1, 4, key="hp_k", help="Entity-relation expansion depth in the graph index.")
        with c[3]:
            st.slider("RRF constant (α)", 1, 120, key="hp_alpha", help="score = Σ 1/(α + rank) over BM25, vector and graph lists.")
        hp = get_hp()
        stale = s.kb is not None and s.kb_hp != (hp.chunk_size, hp.chunk_overlap)
        if stale:
            st.warning("Chunk settings changed — re-index to apply them.", icon=":material/sync_problem:")
            if st.button("Re-index sample corpus", icon=":material/refresh:", key="reindex"):
                index_documents(load_sample_docs(), "sample corpus")
                st.rerun()

    st.markdown("##### KPI telemetry")
    rows = s.bench_rows
    eng = s.engine
    df = pd.DataFrame(rows) if rows else None
    cur = df[df.engine == eng] if df is not None else None
    last = s.last_result
    kb = s.kb
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Retrieval context recall", f"{cur.rcr.mean():.0%}" if cur is not None and len(cur) else "—",
              help="Benchmark mean for the selected engine", border=True)
    k2.metric("Faithfulness", f"{cur.faithfulness.mean():.2f}" if cur is not None and len(cur) else "—",
              help="0.0–1.0, supported claims / total claims", border=True)
    k3.metric("Entity resolution accuracy", f"{er_accuracy(kb):.0%}" if kb else "—", border=True)
    k4.metric("Last query latency", f"{last['timings']['total_ms']:.0f} ms" if last else "—", border=True)
    if cur is None:
        st.caption("Run the benchmark in the last tab to populate recall and faithfulness.")

    st.markdown("##### Latency breakdown")
    if last:
        m = last["timings"]
        lat = pd.DataFrame({"component": ["Retrieval", "Graph traversal", "LLM inference"],
                            "ms": [m["retrieval_ms"], m["graph_ms"], m["llm_ms"]]})
        st.echarts_chart({"tooltip": {"trigger": "axis", "valueFormatter": "%,.0f ms"},
                          "dataset": {"source": lat}, "grid": {"left": 110},
                          "xAxis": {"type": "value", "name": "ms"}, "yAxis": {"type": "category", "inverse": True},
                          "series": [{"type": "bar", "encode": {"x": "ms", "y": "component"}, "colorBy": "data",
                                      "label": {"show": True, "position": "right", "formatter": "{@ms} ms"}}]},
                         height=240, alt=f"Latency breakdown of last {last['engine']} query: total {m['total_ms']:.0f} ms")
    else:
        st.caption("Ask a question in the chat console to see where time is spent.")
