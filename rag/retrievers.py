"""Retrieval primitives: Reciprocal Rank Fusion, embedding rerank, HyDE / query expansion."""
from __future__ import annotations

import numpy as np

from . import config
from .entities import ALIAS_MAP
from .llm import chat
from .prompts import EXPAND_PROMPT, HYDE_ANGLES, HYDE_PROMPT
from .stores import KnowledgeBase


def rrf(rank_lists: list[list[int]], alpha: int = 60) -> list[tuple[int, float]]:
    """score(d) = sum_i 1 / (alpha + rank_i(d)); ranks are 1-based."""
    score: dict[int, float] = {}
    for lst in rank_lists:
        for r, d in enumerate(lst, start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (alpha + r)
    return sorted(score.items(), key=lambda x: -x[1])


def rerank(kb: KnowledgeBase, query: str, cands: list[tuple[int, float]], seeds: list[str], top_n: int = 6):
    if not cands:
        return []
    q = kb.embedder.encode([query], query=True)[0]
    mx = max(s for _, s in cands) or 1.0
    out = []
    for i, s in cands:
        sim = float(kb.emb[i] @ q)
        text = kb.chunks[i].page_content.lower()
        bonus = 0.08 * sum(1 for e in seeds if e.lower() in text)
        out.append((i, 0.7 * sim + 0.3 * (s / mx) + min(bonus, 0.2)))
    return sorted(out, key=lambda x: -x[1])[:top_n]


def hyde_passages(question: str, practice_area: str, n: int) -> list[str]:
    """Multi-HyDE: N hypothetical documents, each from a distinct regulatory angle."""
    area = config.PRACTICE_AREAS[practice_area]
    angles = [HYDE_ANGLES[i % len(HYDE_ANGLES)] for i in range(n)]
    if not config.has_api_key():
        return [f"{question} ({a})" for a in angles]
    from concurrent.futures import ThreadPoolExecutor

    def one(a: str) -> str:
        try:
            return chat("You are a PwC India regulatory domain expert.",
                        HYDE_PROMPT.format(angle=a, area=area, question=question), temperature=0.4, max_tokens=300)
        except Exception:
            return f"{question} ({a})"
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(one, angles))


def expand_query(question: str, n: int = 3) -> list[str]:
    """Query expansion with regulatory synonyms; deterministic alias fallback if the LLM is unavailable."""
    qs: list[str] = []
    if config.has_api_key():
        try:
            txt = chat("You are a PwC India regulatory search specialist.",
                       EXPAND_PROMPT.format(n=n, question=question), temperature=0.3, max_tokens=200)
            qs = [l.strip(" -*0123456789.") for l in txt.splitlines() if l.strip()][:n]
        except Exception:
            qs = []
    if not qs:
        low = question.lower()
        for canon, aliases in ALIAS_MAP.items():
            if any(a in low for a in aliases if len(a) > 3) or canon.lower() in low:
                qs.append(question + " " + " ".join([canon] + aliases[:3]))
    return [question] + qs[:n]
