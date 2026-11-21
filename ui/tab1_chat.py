"""Tab 1 — Data ingestion & multi-RAG chat console."""
from __future__ import annotations

import time

import streamlit as st

from rag import config
from rag.ingest import load_bytes
from rag.pipelines import run_query

from .state import get_flags, get_hp, index_documents, load_sample


def _stream(text: str):
    for w in text.split(" "):
        yield w + " "
        time.sleep(0.008)


def _render_result(r: dict) -> None:
    with st.expander(f"Retrieval trace · {r['timings']['total_ms']:.0f} ms", icon=":material/account_tree:"):
        for step in r["trace"]:
            st.markdown(f"- {step}")
        if r["paths"]:
            st.caption("Graph relations traversed")
            st.code("\n".join(f"{a} -[{rel}]-> {b}" for a, rel, b in r["paths"][:15]), language="text")
    if r["contexts"]:
        with st.expander(f"Sources ({len(r['contexts'])})", icon=":material/menu_book:"):
            for c in r["contexts"]:
                st.markdown(f":blue-badge[{c['tag']}] **{c['source']}** · p.{c['page']} · score {c['score']}")
                st.caption(c["text"][:420] + ("…" if len(c["text"]) > 420 else ""))


def render() -> None:
    s = st.session_state
    left, right = st.columns([1, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader(":material/upload_file: Data ingestion")
            st.caption("PDF, DOCX, JSON, CSV or TXT. Provenance metadata is attached to every chunk.")
            files = st.file_uploader("Regulatory documents", type=["pdf", "docx", "json", "csv", "txt", "md"],
                                     accept_multiple_files=True, key="uploader", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            if c1.button("Index uploads", icon=":material/database:", width="stretch", type="primary",
                         disabled=not files, key="index_uploads"):
                docs = []
                for f in files:
                    try:
                        docs += load_bytes(f.name, f.getvalue(), s.get("area", "tax"))
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
                if docs:
                    index_documents(docs, f"{len(files)} file(s)")
            if c2.button("Load sample corpus", icon=":material/dataset:", width="stretch", key="load_sample"):
                load_sample()
        kb = s.kb
        if kb:
            st.markdown("**Index status**")
            m = kb.stats()
            a, b, c = st.columns(3)
            a.metric("Chunks", m["chunks"], border=True)
            b.metric("Entities", m["nodes"], border=True)
            c.metric("Relations", m["edges"], border=True)
            st.caption(f"{m['embedder']} · {m['dim']}d · {m['vector_backend']} · graph extraction: {m['extraction']}")

    with right:
        header = st.container(horizontal=True, vertical_alignment="center")
        header.subheader(":material/forum: Advisory chat")
        header.badge(config.PRACTICE_AREAS[s.area], icon=":material/domain:", color="orange")
        header.badge(config.ENGINES[s.engine], icon=":material/hub:", color="blue")
        if not config.has_api_key():
            st.warning("Add your key to `OPENROUTER_API_KEY` in `.env.example` and restart. Retrieval still works "
                       "without it, but answers are not synthesised.", icon=":material/key:")
        if not s.messages and not kb:
            st.info("Load the sample corpus or upload documents to begin.", icon=":material/info:")

        for m in s.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("result"):
                    _render_result(m["result"])

        prompt = st.chat_input("Ask a compliance question…", disabled=kb is None, submit_mode="disable",
                               key="chat_prompt")
        if prompt:
            s.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.status(f"{config.ENGINES[s.engine]} running…", expanded=True) as status:
                    r = run_query(s.engine, kb, prompt, s.area, get_hp(), get_flags())
                    for step in r["trace"]:
                        st.write(step)
                    status.update(label=f"Done in {r['timings']['total_ms']:.0f} ms", state="complete", expanded=False)
                st.write_stream(_stream(r["answer"]))
                _render_result(r)
            s.messages.append({"role": "assistant", "content": r["answer"], "result": r})
            s.last_result = r
        if s.messages:
            if st.button("Clear conversation", icon=":material/delete:", key="clear_chat"):
                s.messages = []
                st.rerun()
