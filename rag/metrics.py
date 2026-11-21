"""KPI computation: RCR, faithfulness, entity-relation resolution accuracy, latency."""
from __future__ import annotations

import re

from . import config
from .entities import normalize_entity
from .llm import chat, parse_json
from .prompts import JUDGE_PROMPT


def retrieval_context_recall(contexts: list[dict], facts: list[list[str]]) -> float:
    """% of ground-truth facts present in retrieved chunks. A fact is a list of required phrases (ALL must appear
    in the retrieved text); a phrase may list alternatives separated by '|'."""
    if not facts:
        return 0.0
    text = " ".join(c["text"] for c in contexts).lower()
    hit = sum(1 for group in facts if all(any(alt.strip().lower() in text for alt in ph.split("|")) for ph in group))
    return hit / len(facts)


def _overlap_faithfulness(answer: str, contexts: list[dict]) -> float:
    ctx = set(re.findall(r"[a-z0-9]{4,}", " ".join(c["text"] for c in contexts).lower()))
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"---\n\*Estimated.*", "", answer)) if len(s.split()) > 5]
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        toks = set(re.findall(r"[a-z0-9]{4,}", s.lower()))
        if toks and len(toks & ctx) / len(toks) >= 0.6:
            ok += 1
    return ok / len(sents)


def faithfulness(answer: str, contexts: list[dict], use_llm: bool = True) -> float:
    """Ratio of answer claims supported by the retrieved context (0.0-1.0)."""
    if not contexts or not answer:
        return 0.0
    if use_llm and config.has_api_key():
        try:
            ctx = "\n\n".join(f"[{c['tag']}] {c['text'][:1200]}" for c in contexts)
            d = parse_json(chat("You are a strict fact-checking judge.", JUDGE_PROMPT.format(context=ctx, answer=answer),
                                temperature=0.0, max_tokens=120))
            if d and d.get("total"):
                return max(0.0, min(1.0, float(d["supported"]) / float(d["total"])))
        except Exception:
            pass
    return _overlap_faithfulness(answer, contexts)


def entity_resolution_accuracy(aliases: dict[str, list[str]], graph_nodes: set[str], corpus_text: str) -> float:
    """Gold alias groups: every alias that occurs in the corpus must resolve to the canonical name,
    and the canonical node must exist in the graph."""
    low = corpus_text.lower()
    total = ok = 0
    for canon, al in aliases.items():
        for a in al:
            if a.lower() not in low and canon.lower() not in low:
                continue
            total += 1
            if normalize_entity(a) == normalize_entity(canon) and normalize_entity(canon) in graph_nodes:
                ok += 1
    return ok / total if total else 0.0
