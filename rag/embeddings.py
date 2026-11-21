"""STEP 4 — Multi-modal embeddings: bge-large-en-v1.5 text vectors (1024d) + Node2Vec-style graph vectors."""
from __future__ import annotations

import numpy as np

from . import config

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Dense text embedder. Falls back to hashed TF vectors (1024d) if the HF model cannot load."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL
        self.mode = "bge"
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dim = getattr(self.model, 'get_embedding_dimension', self.model.get_sentence_embedding_dimension)()
        except Exception as e:  # offline / missing weights
            from sklearn.feature_extraction.text import HashingVectorizer
            self.mode = f"hashing-fallback ({type(e).__name__})"
            self.dim = 1024
            self._hv = HashingVectorizer(n_features=1024, alternate_sign=False, norm="l2",
                                         ngram_range=(1, 2), stop_words="english")

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if self.model is not None:
            if query and "bge" in self.model_name.lower():
                texts = [BGE_QUERY_PREFIX + t for t in texts]
            return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=16,
                                                show_progress_bar=False), dtype="float32")
        return self._hv.transform(texts).toarray().astype("float32")


def node2vec_embeddings(adj: dict[str, list[str]], dim: int = 32, walk_len: int = 10, walks: int = 12,
                        p: float = 1.0, q: float = 0.5, seed: int = 7) -> dict[str, np.ndarray]:
    """Compact Node2Vec: biased 2nd-order random walks -> windowed PPMI co-occurrence -> truncated SVD."""
    rng = np.random.default_rng(seed)
    nodes = sorted(adj)
    if not nodes:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    co = np.zeros((len(nodes), len(nodes)), dtype="float64")
    for start in nodes:
        for _ in range(walks):
            walk = [start]
            while len(walk) < walk_len:
                cur = walk[-1]
                nbrs = adj.get(cur, [])
                if not nbrs:
                    break
                if len(walk) == 1:
                    walk.append(nbrs[rng.integers(len(nbrs))])
                    continue
                prev = walk[-2]
                w = np.array([1 / p if n == prev else (1.0 if n in adj.get(prev, []) else 1 / q) for n in nbrs])
                walk.append(nbrs[rng.choice(len(nbrs), p=w / w.sum())])
            for i, a in enumerate(walk):
                for b in walk[i + 1:i + 4]:
                    co[idx[a], idx[b]] += 1; co[idx[b], idx[a]] += 1
    total = co.sum() or 1.0
    pr = co.sum(1, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co / total) / (pr @ pr.T))
    ppmi = np.nan_to_num(np.maximum(pmi, 0), neginf=0, posinf=0)
    d = min(dim, len(nodes))
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    emb = u[:, :d] * np.sqrt(s[:d])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return {n: emb[idx[n]].astype("float32") for n in nodes}
