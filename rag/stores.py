"""STEP 5/6 — Dual index: Chroma HNSW vector index + NetworkX graph adjacency index (+ BM25 sparse)."""
from __future__ import annotations

import re
import uuid
from collections import defaultdict, deque

import networkx as nx
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .embeddings import Embedder, node2vec_embeddings
from .entities import normalize_entity

_TOK = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text)]


class KnowledgeBase:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Document] = []
        self.emb = np.zeros((0, embedder.dim), dtype="float32")
        self.graph = nx.MultiDiGraph()
        self.entity_chunks: dict[str, set[int]] = defaultdict(set)
        self.node_vecs: dict[str, np.ndarray] = {}
        self.node_text_emb: dict[str, np.ndarray] = {}
        self.extraction_mode = "none"
        self._collection = None
        self._bm25: BM25Okapi | None = None
        self.vector_backend = "numpy"

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Document], extraction: dict) -> "KnowledgeBase":
        self.chunks = chunks
        self.emb = self.embedder.encode([c.page_content for c in chunks])
        self._build_vector_index()
        self._bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks]) if chunks else None
        self._build_graph(extraction)
        return self

    def _build_vector_index(self) -> None:
        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self._collection = client.create_collection(
                f"kb_{uuid.uuid4().hex[:10]}", metadata={"hnsw:space": "cosine"})
            ids = [c.metadata["chunk_id"] for c in self.chunks]
            for i in range(0, len(ids), 200):
                self._collection.add(ids=ids[i:i + 200], embeddings=self.emb[i:i + 200].tolist(),
                                     documents=[c.page_content for c in self.chunks[i:i + 200]],
                                     metadatas=[{"source": str(c.metadata["source"]), "idx": i + j}
                                                for j, c in enumerate(self.chunks[i:i + 200])])
            self.vector_backend = "chromadb-hnsw"
        except Exception:
            self._collection = None
            self.vector_backend = "numpy-bruteforce"

    def _build_graph(self, extraction: dict) -> None:
        g = nx.MultiDiGraph()
        id2idx = {c.metadata["chunk_id"]: i for i, c in enumerate(self.chunks)}
        for cid, (ents, rels) in extraction.items():
            if cid not in id2idx:
                continue
            ci = id2idx[cid]
            for e in ents:
                nm = normalize_entity(str(e.get("name", "")))
                if not nm:
                    continue
                if nm not in g:
                    g.add_node(nm, type=e.get("type", "Entity"), aliases=set(), mentions=0)
                g.nodes[nm]["mentions"] += 1
                if str(e.get("name")) != nm:
                    g.nodes[nm]["aliases"].add(str(e["name"]))
                self.entity_chunks[nm].add(ci)
            for r in rels:
                s, t = normalize_entity(str(r.get("source", ""))), normalize_entity(str(r.get("target", "")))
                if not s or not t or s == t:
                    continue
                for n in (s, t):
                    if n not in g:
                        g.add_node(n, type="Entity", aliases=set(), mentions=0)
                    self.entity_chunks[n].add(ci)
                g.add_edge(s, t, relation=str(r.get("relation", "related_to"))[:40], chunk=ci)
        self.graph = g
        adj = {n: sorted(set(g.successors(n)) | set(g.predecessors(n))) for n in g.nodes}
        self.node_vecs = node2vec_embeddings(adj)
        names = list(g.nodes)
        if names:
            vecs = self.embedder.encode(names)
            self.node_text_emb = {n: v for n, v in zip(names, vecs)}

    # ----------------------------------------------------------------- search
    def vector_search(self, query: str, n: int = 8) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query], query=True)[0]
        return self.vector_search_emb(q, n)

    def vector_search_emb(self, q: np.ndarray, n: int = 8) -> list[tuple[int, float]]:
        n = min(n, len(self.chunks))
        if self._collection is not None:
            try:
                r = self._collection.query(query_embeddings=[q.tolist()], n_results=n)
                return [(int(m["idx"]), 1 - float(d)) for m, d in zip(r["metadatas"][0], r["distances"][0])]
            except Exception:
                pass
        sims = self.emb @ q
        top = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in top]

    def bm25_search(self, query: str, n: int = 8) -> list[tuple[int, float]]:
        if not self._bm25:
            return []
        sc = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-sc)[:n]
        return [(int(i), float(sc[i])) for i in top if sc[i] > 0]

    def find_seeds(self, query: str, allowed: set[str] | None = None, max_seeds: int = 4) -> list[str]:
        """Entity linking: alias/string match first, then embedding similarity to node names."""
        ql = query.lower()
        pool = [n for n in self.graph.nodes if (allowed is None or n in allowed)]
        hits = [n for n in pool if n.lower() in ql or
                any(a.lower() in ql for a in self.graph.nodes[n].get("aliases", ()))]
        from .entities import _ALIAS_LOOKUP
        for alias, canon in _ALIAS_LOOKUP.items():
            if len(alias) > 3 and re.search(rf"\b{re.escape(alias)}\b", ql) and canon in pool and canon not in hits:
                hits.append(canon)
        for m in re.finditer(r"(?:section|sec\.?|rule|clause)\s*(\d+[a-z]*)", ql):
            for n in pool:
                if n.lower().endswith(m.group(1)) and n not in hits:
                    hits.append(n)
        if len(hits) < max_seeds and self.node_text_emb:
            q = self.embedder.encode([query], query=True)[0]
            scored = sorted(((float(q @ v), n) for n, v in self.node_text_emb.items() if n in pool), reverse=True)
            for s, n in scored[:max_seeds]:
                if s > 0.45 and n not in hits:
                    hits.append(n)
        return hits[:max_seeds]

    def graph_search(self, query: str, k: int = 2, allowed_nodes: set[str] | None = None,
                     allowed_relations: set[str] | None = None, n: int = 10):
        """k-hop BFS from linked seed entities. Returns (ranked chunks, seeds, visited paths)."""
        seeds = self.find_seeds(query, allowed_nodes)
        g = self.graph
        dist: dict[str, int] = {s: 0 for s in seeds}
        paths: list[tuple[str, str, str]] = []
        dq = deque(seeds)
        while dq:
            cur = dq.popleft()
            if dist[cur] >= k:
                continue
            for u, v, d in list(g.out_edges(cur, data=True)) + list(g.in_edges(cur, data=True)):
                if allowed_relations is not None and d["relation"] not in allowed_relations:
                    continue
                nxt = v if u == cur else u
                if allowed_nodes is not None and nxt not in allowed_nodes:
                    continue
                paths.append((u, d["relation"], v))
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    dq.append(nxt)
        score: dict[int, float] = defaultdict(float)
        for node, h in dist.items():
            for ci in self.entity_chunks.get(node, ()):
                score[ci] += 1.0 / (1 + h)
        for u, v, d in g.edges(data=True):
            if u in dist and v in dist and (allowed_relations is None or d["relation"] in allowed_relations):
                score[d["chunk"]] += 0.5 / (1 + min(dist[u], dist[v]))
        ranked = sorted(score.items(), key=lambda x: -x[1])[:n]
        return ranked, seeds, list(dict.fromkeys(paths))[:40]

    # ------------------------------------------------------------ UI helpers
    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(), "vector_backend": self.vector_backend,
                "embedder": self.embedder.mode, "dim": self.embedder.dim, "extraction": self.extraction_mode}
