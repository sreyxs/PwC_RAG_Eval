"""LangGraph pipelines for Standard, HyDE and Graph RAG (the latter with cyclic sub-graph traversal)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config
from .config import Hyperparams
from .llm import chat, ensure_token_footer
from .prompts import system_prompt
from .retrievers import expand_query, hyde_passages, rerank, rrf
from .stores import KnowledgeBase

TOP_CTX = 6

# Node catalogue for the visual workflow canvas (Tab 3)
GRAPH_NODES = [
    ("query_expansion", "Query expansion", True),
    ("hybrid_search", "Hybrid vector / graph search", False),
    ("rrf_fusion", "RRF fusion", True),
    ("rerank", "Rerank", True),
    ("synthesis", "Synthesis", False),
]
SOURCE_EDGES = [("vector", "Dense vector (HNSW)"), ("bm25", "Sparse BM25"), ("graph", "Graph k-hop traversal")]


@dataclass
class Flags:
    """User-toggleable pipeline variations (Tab 3) and retrieval filters (Tab 2)."""
    query_expansion: bool = True
    rrf_fusion: bool = True
    rerank: bool = True
    vector: bool = True
    bm25: bool = True
    graph: bool = True
    allowed_nodes: set[str] | None = None
    allowed_relations: set[str] | None = None

    def key(self) -> tuple:
        return (self.query_expansion, self.rrf_fusion, self.rerank, self.vector, self.bm25, self.graph,
                None if self.allowed_nodes is None else tuple(sorted(self.allowed_nodes)),
                None if self.allowed_relations is None else tuple(sorted(self.allowed_relations)))


class State(TypedDict, total=False):
    question: str
    practice_area: str
    queries: list[str]
    hits: dict[str, list[tuple[int, float]]]
    fused: list[tuple[int, float]]
    final: list[tuple[int, float]]
    seeds: list[str]
    paths: list[tuple[str, str, str]]
    hop: int
    loops: int
    answer: str
    contexts: list[dict]
    trace: list[str]
    timings: dict[str, float]


def _ms(state: State, key: str, t0: float) -> dict:
    t = dict(state.get("timings", {}))
    t[key] = t.get(key, 0.0) + (time.perf_counter() - t0) * 1000
    return t


def _contexts(kb: KnowledgeBase, ranked: list[tuple[int, float]]) -> list[dict]:
    return [{"tag": f"S{n}", "chunk_id": kb.chunks[i].metadata["chunk_id"], "source": kb.chunks[i].metadata["source"],
             "page": kb.chunks[i].metadata.get("page"), "text": kb.chunks[i].page_content, "score": round(float(s), 4),
             "idx": i} for n, (i, s) in enumerate(ranked, start=1)]


def _synthesis_node(kb: KnowledgeBase, engine: str):
    def node(state: State) -> dict:
        t0 = time.perf_counter()
        ctxs = _contexts(kb, state.get("final", [])[:TOP_CTX])
        block = "\n\n".join(f"[{c['tag']}] (source: {c['source']}, p.{c['page']})\n{c['text']}" for c in ctxs)
        facts = ""
        if engine == "graph" and state.get("paths"):
            facts = "\n\nKNOWLEDGE-GRAPH RELATIONS (entity -[relation]-> entity):\n" + "\n".join(
                f"- {a} -[{r}]-> {b}" for a, r, b in state["paths"][:25])
        user = (f"CONTEXT:\n{block or '(no context retrieved)'}{facts}\n\nQUESTION: {state['question']}\n\n"
                "Answer in structured markdown with inline [S#] citations and a 'Gaps & limitations' section.")
        if config.has_api_key():
            try:
                ans = chat(system_prompt(state["practice_area"]), user, temperature=0.1, max_tokens=1600)
            except Exception as e:
                ans = f"**LLM call failed:** `{type(e).__name__}: {str(e)[:200]}`\n\nRetrieved context is shown below."
        else:
            ans = ("**LLM not configured** — add `OPENROUTER_API_KEY` to `.env.example`/`.env` to enable synthesis. "
                   "Retrieved evidence:\n\n" + "\n".join(f"- [{c['tag']}] {c['text'][:220]}…" for c in ctxs))
        return {"answer": ensure_token_footer(ans), "contexts": ctxs, "timings": _ms(state, "llm_ms", t0),
                "trace": state.get("trace", []) + [f"Synthesis: {len(ctxs)} chunks → {config.MODEL_NAME}"]}
    return node


def build_standard(kb: KnowledgeBase, hp: Hyperparams, flags: Flags):
    def retrieve(state: State) -> dict:
        t0 = time.perf_counter()
        hits = kb.vector_search(state["question"], TOP_CTX)
        return {"final": hits, "timings": _ms(state, "retrieval_ms", t0),
                "trace": state.get("trace", []) + [f"Vector search (top {TOP_CTX}) over {len(kb.chunks)} chunks"]}
    g = StateGraph(State)
    g.add_node("retrieve", retrieve); g.add_node("synthesis", _synthesis_node(kb, "standard"))
    g.add_edge(START, "retrieve"); g.add_edge("retrieve", "synthesis"); g.add_edge("synthesis", END)
    return g.compile()


def build_hyde(kb: KnowledgeBase, hp: Hyperparams, flags: Flags):
    def generate(state: State) -> dict:
        t0 = time.perf_counter()
        docs = hyde_passages(state["question"], state["practice_area"], hp.hyde_n)
        return {"queries": docs, "timings": _ms(state, "llm_ms", t0),
                "trace": state.get("trace", []) + [f"Multi-HyDE: generated N={hp.hyde_n} hypothetical passages"]}

    def retrieve(state: State) -> dict:
        t0 = time.perf_counter()
        lists = [[i for i, _ in kb.vector_search(state["question"], 10)]]
        embs = kb.embedder.encode(state["queries"])  # hypothetical docs are embedded as documents (no query prefix)
        lists += [[i for i, _ in kb.vector_search_emb(e, 10)] for e in embs]
        fused = rrf(lists, hp.rrf_alpha)[:TOP_CTX]
        return {"final": fused, "timings": _ms(state, "retrieval_ms", t0),
                "trace": state.get("trace", []) + [f"Vector search × {len(lists)} (query + {hp.hyde_n} HyDE), RRF α={hp.rrf_alpha}"]}
    g = StateGraph(State)
    g.add_node("hyde", generate); g.add_node("retrieve", retrieve); g.add_node("synthesis", _synthesis_node(kb, "hyde"))
    g.add_edge(START, "hyde"); g.add_edge("hyde", "retrieve"); g.add_edge("retrieve", "synthesis")
    g.add_edge("synthesis", END)
    return g.compile()


def build_graph_rag(kb: KnowledgeBase, hp: Hyperparams, flags: Flags):
    def expansion(state: State) -> dict:
        t0 = time.perf_counter()
        qs = expand_query(state["question"]) if flags.query_expansion else [state["question"]]
        return {"queries": qs, "hop": hp.hop_k, "loops": 0, "timings": _ms(state, "llm_ms", t0),
                "trace": state.get("trace", []) + [
                    f"Query expansion: {len(qs)} queries" if flags.query_expansion else "Query expansion: skipped (toggled off)"]}

    def hybrid(state: State) -> dict:
        t0 = time.perf_counter()
        hits: dict[str, list] = {}
        qs = state["queries"]
        if flags.vector:
            hits["vector"] = [h for q in qs for h in kb.vector_search(q, 10)]
        if flags.bm25:
            hits["bm25"] = [h for q in qs for h in kb.bm25_search(q, 10)]
        t_vec = (time.perf_counter() - t0) * 1000
        seeds, paths, t_graph = [], [], 0.0
        if flags.graph:
            t1 = time.perf_counter()
            ranked, seeds, paths = kb.graph_search(" ".join(qs[:2]), state["hop"], flags.allowed_nodes,
                                                   flags.allowed_relations, 10)
            hits["graph"] = ranked
            t_graph = (time.perf_counter() - t1) * 1000
        t = dict(state.get("timings", {}))
        t["retrieval_ms"] = t.get("retrieval_ms", 0.0) + t_vec
        t["graph_ms"] = t.get("graph_ms", 0.0) + t_graph
        return {"hits": hits, "seeds": seeds, "paths": paths, "timings": t,
                "trace": state.get("trace", []) + [
                    f"Hybrid search: " + ", ".join(f"{k}={len(v)}" for k, v in hits.items()) +
                    (f" | seeds={seeds} | k={state['hop']}" if flags.graph else "")]}

    def fusion(state: State) -> dict:
        t0 = time.perf_counter()
        hits = state.get("hits", {})
        if flags.rrf_fusion:
            lists = []
            for src, h in hits.items():
                seen, lst = set(), []
                for i, _ in sorted(h, key=lambda x: -x[1]):
                    if i not in seen:
                        seen.add(i); lst.append(i)
                lists.append(lst)
            fused = rrf(lists, hp.rrf_alpha)[:15]
            msg = f"RRF fusion over {len(lists)} rank lists (α={hp.rrf_alpha}) → {len(fused)} candidates"
        else:
            best: dict[int, float] = {}
            for h in hits.values():
                for i, s in h:
                    best[i] = max(best.get(i, 0.0), s)
            fused = sorted(best.items(), key=lambda x: -x[1])[:15]
            msg = "RRF fusion: skipped — raw max-score merge"
        return {"fused": fused, "timings": _ms(state, "retrieval_ms", t0), "trace": state.get("trace", []) + [msg]}

    def rerank_node(state: State) -> dict:
        t0 = time.perf_counter()
        if flags.rerank:
            final = rerank(kb, state["question"], state.get("fused", []), state.get("seeds", []), TOP_CTX)
            msg = f"Rerank (bi-encoder + entity bonus) → top {len(final)}"
        else:
            final = state.get("fused", [])[:TOP_CTX]
            msg = "Rerank: skipped (toggled off)"
        return {"final": final, "timings": _ms(state, "retrieval_ms", t0), "trace": state.get("trace", []) + [msg]}

    def route(state: State) -> str:
        # Cyclic sub-graph traversal: if the evidence is thin, widen the hop radius once and search again.
        if flags.graph and state.get("loops", 0) < 1 and len(state.get("final", [])) < 3:
            return "widen"
        return "synthesis"

    def widen(state: State) -> dict:
        return {"hop": state["hop"] + 1, "loops": state.get("loops", 0) + 1,
                "trace": state.get("trace", []) + [f"Evidence thin → widening hop radius to k={state['hop'] + 1} (cycle)"]}

    g = StateGraph(State)
    for name, fn in [("query_expansion", expansion), ("hybrid_search", hybrid), ("rrf_fusion", fusion),
                     ("rerank", rerank_node), ("widen", widen), ("synthesis", _synthesis_node(kb, "graph"))]:
        g.add_node(name, fn)
    g.add_edge(START, "query_expansion"); g.add_edge("query_expansion", "hybrid_search")
    g.add_edge("hybrid_search", "rrf_fusion"); g.add_edge("rrf_fusion", "rerank")
    g.add_conditional_edges("rerank", route, {"widen": "widen", "synthesis": "synthesis"})
    g.add_edge("widen", "hybrid_search"); g.add_edge("synthesis", END)
    return g.compile()


BUILDERS = {"standard": build_standard, "hyde": build_hyde, "graph": build_graph_rag}


def run_query(engine: str, kb: KnowledgeBase, question: str, practice_area: str, hp: Hyperparams,
              flags: Flags | None = None) -> dict[str, Any]:
    flags = flags or Flags()
    app = BUILDERS[engine](kb, hp, flags)
    t0 = time.perf_counter()
    out = app.invoke({"question": question, "practice_area": practice_area, "timings": {}, "trace": []})
    total = (time.perf_counter() - t0) * 1000
    tm = out.get("timings", {})
    tm = {"retrieval_ms": tm.get("retrieval_ms", 0.0), "graph_ms": tm.get("graph_ms", 0.0),
          "llm_ms": tm.get("llm_ms", 0.0), "total_ms": total}
    return {"engine": engine, "question": question, "answer": out.get("answer", ""), "contexts": out.get("contexts", []),
            "trace": out.get("trace", []), "timings": tm, "seeds": out.get("seeds", []), "paths": out.get("paths", [])}
