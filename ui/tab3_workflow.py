"""Tab 3 — LangGraph workflow canvas: toggle nodes / edges and test pipeline variations."""
from __future__ import annotations

import streamlit as st

from rag.pipelines import run_query

from .state import get_flags, get_hp

# (id, label, x, y, toggle-key or None for mandatory)
NODES = [
    ("question", "User question", 0, 50, None),
    ("query_expansion", "Query expansion", 1, 50, "query_expansion"),
    ("vector", "Dense vector\n(HNSW)", 2, 15, "vector"),
    ("bm25", "Sparse BM25", 2, 50, "bm25"),
    ("graph", "Graph k-hop\ntraversal", 2, 85, "graph"),
    ("rrf_fusion", "RRF fusion", 3, 50, "rrf_fusion"),
    ("rerank", "Rerank", 4, 50, "rerank"),
    ("synthesis", "Synthesis\n(Claude Sonnet 5.5)", 5, 50, None),
]
EDGES = [("question", "query_expansion"), ("query_expansion", "vector"), ("query_expansion", "bm25"),
         ("query_expansion", "graph"), ("vector", "rrf_fusion"), ("bm25", "rrf_fusion"), ("graph", "rrf_fusion"),
         ("rrf_fusion", "rerank"), ("rerank", "synthesis")]


def _spec(flags: dict[str, bool]) -> dict:
    def on(n):  # node enabled?
        key = next((k for i, _, _, _, k in NODES if i == n), None)
        return True if key is None else flags[key]
    data = [{"name": nid, "value": label, "x": x * 190, "y": y * 4.2, "symbolSize": [118, 46], "symbol": "roundRect",
             "label": {"show": True, "formatter": label, "fontSize": 12},
             "itemStyle": {"opacity": 1.0 if on(nid) else 0.25, "borderType": "solid" if on(nid) else "dashed",
                           "borderWidth": 2, "borderColor": "#FF7A3D" if nid in ("question", "synthesis") else "#A39B94"}}
            for nid, label, x, y, _ in NODES]
    links = [{"source": a, "target": b,
              "lineStyle": {"type": "solid" if on(a) and on(b) else "dashed", "opacity": 1.0 if on(a) and on(b) else 0.2,
                            "width": 2, "curveness": 0.0}} for a, b in EDGES]
    links.append({"source": "rerank", "target": "graph", "lineStyle": {"type": "dotted", "curveness": 0.55, "width": 1.5,
                  "opacity": 0.9 if flags["graph"] else 0.2}, "label": {"show": True, "formatter": "evidence thin → widen k (cycle)", "fontSize": 10}})
    return {"tooltip": {}, "animation": True,
            "series": [{"type": "graph", "layout": "none", "roam": False, "edgeSymbol": ["none", "arrow"],
                        "edgeSymbolSize": 8, "data": data, "links": links, "lineStyle": {"color": "source"},
                        "left": 40, "right": 40, "top": 30, "bottom": 30}]}


def render() -> None:
    s = st.session_state
    st.caption("LangGraph state graph for Graph RAG. Toggle a node or a retrieval edge and re-run a test query to "
               "compare pipeline variations. Settings also apply to the chat console.")
    c = st.columns(6)
    labels = [("query_expansion", "Query expansion"), ("vector", "Vector"), ("bm25", "BM25"), ("graph", "Graph"),
              ("rrf_fusion", "RRF fusion"), ("rerank", "Rerank")]
    for col, (k, lab) in zip(c, labels):
        with col:
            st.toggle(lab, key=f"flag_{k}")
    flags = {k: s[f"flag_{k}"] for k, _ in labels}
    if not any(flags[k] for k in ("vector", "bm25", "graph")):
        st.error("At least one retrieval source must stay on.", icon=":material/error:")
    st.echarts_chart(_spec(flags), height=360, key="wf_chart", alt="LangGraph workflow: question to query expansion, "
                     "three parallel retrievers, RRF fusion, rerank and synthesis, with a widen-k loop.")

    st.markdown("##### Test this variation")
    kb = s.kb
    if kb is None:
        st.info("Index a corpus first.", icon=":material/info:")
        return
    q = st.text_input("Test query", value="Which internal control maps to the STR obligation and who owns it?",
                      key="wf_query")
    if st.button("Run Graph RAG with these settings", icon=":material/play_arrow:", type="primary",
                 disabled=not any(flags[k] for k in ("vector", "bm25", "graph"))):
        with st.spinner("Running LangGraph pipeline…"):
            r = run_query("graph", kb, q, s.area, get_hp(), get_flags())
        s.last_result = r
        s.wf_result = r
    r = s.get("wf_result")
    if r:
        m = r["timings"]
        a, b, c2, d = st.columns(4)
        a.metric("Total", f"{m['total_ms']:.0f} ms", border=True)
        b.metric("Retrieval", f"{m['retrieval_ms']:.0f} ms", border=True)
        c2.metric("Graph traversal", f"{m['graph_ms']:.0f} ms", border=True)
        d.metric("LLM", f"{m['llm_ms']:.0f} ms", border=True)
        with st.expander("Execution trace", expanded=True, icon=":material/account_tree:"):
            for step in r["trace"]:
                st.markdown(f"- {step}")
        st.markdown(r["answer"])
