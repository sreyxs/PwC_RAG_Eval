"""Session-state bootstrap, cached resources and shared helpers."""
from __future__ import annotations

import streamlit as st

from rag import config
from rag.config import Hyperparams
from rag.embeddings import Embedder
from rag.pipelines import Flags
from rag.service import build_kb, load_sample_docs


@st.cache_resource(show_spinner="Loading embedding model (first run downloads weights)…")
def get_embedder() -> Embedder:
    return Embedder()


def init_state() -> None:
    d = st.session_state
    d.setdefault("kb", None)
    d.setdefault("kb_hp", None)
    d.setdefault("messages", [])
    d.setdefault("last_result", None)
    d.setdefault("bench_rows", None)
    d.setdefault("hp_chunk", 600)
    d.setdefault("hp_overlap", 60)
    d.setdefault("hp_n", 3)
    d.setdefault("hp_k", 2)
    d.setdefault("hp_alpha", 60)
    d.setdefault("active_nodes", None)      # None = all nodes active
    d.setdefault("active_rels", None)
    d.setdefault("pending_docs", [])
    for k in ("query_expansion", "rrf_fusion", "rerank", "vector", "bm25", "graph"):
        d.setdefault(f"flag_{k}", True)


def get_hp() -> Hyperparams:
    s = st.session_state
    return Hyperparams(chunk_size=s.hp_chunk, chunk_overlap=s.hp_overlap, hyde_n=s.hp_n, hop_k=s.hp_k,
                       rrf_alpha=s.hp_alpha)


def get_flags() -> Flags:
    s = st.session_state
    return Flags(**{k: s[f"flag_{k}"] for k in ("query_expansion", "rrf_fusion", "rerank", "vector", "bm25", "graph")},
                 allowed_nodes=set(s.active_nodes) if s.active_nodes is not None else None,
                 allowed_relations=set(s.active_rels) if s.active_rels is not None else None)


def index_documents(docs, label: str) -> None:
    """Build (or rebuild) the dual index and store it in session state."""
    hp = get_hp()
    with st.status(f"Indexing {label}…", expanded=True) as status:
        st.write("Chunking (sub-word token windows) …")
        bar = st.progress(0.0, text="Extracting entities & relations")

        def cb(p: float) -> None:
            bar.progress(min(p, 1.0), text="Extracting entities & relations")
        kb = build_kb(docs, hp, get_embedder(), use_llm=config.has_api_key(), progress=cb)
        bar.progress(1.0, text="Entity extraction complete")
        s = kb.stats()
        st.write(f"{s['chunks']} chunks · {s['nodes']} entities · {s['edges']} relations · "
                 f"{s['vector_backend']} · extraction: {s['extraction']}")
        status.update(label=f"Indexed {label}", state="complete", expanded=False)
    st.session_state.kb = kb
    st.session_state.kb_hp = (hp.chunk_size, hp.chunk_overlap)
    st.session_state.active_nodes = None
    st.session_state.active_rels = None
    st.session_state.bench_rows = None


def load_sample() -> None:
    index_documents(load_sample_docs(), "sample corpus")
