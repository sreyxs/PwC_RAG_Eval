"""High-level service: build a KnowledgeBase from documents; run the comparative benchmark."""
from __future__ import annotations

import json
import time

from langchain_core.documents import Document

from . import config
from .chunking import chunk_documents
from .config import Hyperparams
from .embeddings import Embedder
from .entities import extract_graph_data, register_aliases
from .ingest import load_path
from .metrics import entity_resolution_accuracy, faithfulness, retrieval_context_recall
from .pipelines import Flags, run_query
from .stores import KnowledgeBase


def load_sample_docs() -> list[Document]:
    """Each sample file bundles separate circulars/clauses; split on blank lines so every section becomes its
    own page-level document (as in a real corpus of PDFs)."""
    area_by_file = {"tax_circulars.txt": "tax", "grc_rbi_kyc.txt": "grc", "deal_spa.txt": "deals"}
    docs: list[Document] = []
    for f, area in area_by_file.items():
        base = load_path(config.SAMPLE_DIR / f, area)[0]
        blocks = [b.strip() for b in base.page_content.split("\n\n") if b.strip()][1:]  # drop banner line
        for i, b in enumerate(blocks, start=1):
            docs.append(Document(page_content=b, metadata={**base.metadata, "page": i}))
    return docs


def load_aliases() -> dict[str, list[str]]:
    p = config.SAMPLE_DIR / "aliases.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_questions() -> list[dict]:
    return json.loads((config.SAMPLE_DIR / "benchmark_questions.json").read_text())


def build_kb(docs: list[Document], hp: Hyperparams, embedder: Embedder, use_llm: bool = True,
             progress=None) -> KnowledgeBase:
    register_aliases(load_aliases())
    chunks = chunk_documents(docs, hp.chunk_size, hp.chunk_overlap)
    extraction, mode = extract_graph_data(chunks, use_llm=use_llm, progress=progress)
    kb = KnowledgeBase(embedder).build(chunks, extraction)
    kb.extraction_mode = mode
    kb.corpus_text = "\n".join(d.page_content for d in docs)  # type: ignore[attr-defined]
    return kb


def run_benchmark(kb: KnowledgeBase, hp: Hyperparams, questions: list[dict], engines=("standard", "hyde", "graph"),
                  flags: Flags | None = None, judge_llm: bool = True, progress=None) -> list[dict]:
    rows = []
    total = len(questions) * len(engines)
    n = 0
    for q in questions:
        for eng in engines:
            r = run_query(eng, kb, q["question"], q.get("area", "tax"), hp, flags)
            rows.append({"id": q["id"], "type": q["type"], "engine": eng, "question": q["question"],
                         "rcr": retrieval_context_recall(r["contexts"], q["facts"]),
                         "faithfulness": faithfulness(r["answer"], r["contexts"], judge_llm),
                         **r["timings"]})
            n += 1
            if progress:
                progress(n / total, f"{q['id']} · {eng}")
    return rows


def er_accuracy(kb: KnowledgeBase) -> float:
    return entity_resolution_accuracy(load_aliases(), set(kb.graph.nodes), getattr(kb, "corpus_text", ""))
