"""STEP 2/3 — LLM-guided entity-relation extraction + entity normalisation (alias resolution)."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from . import config
from .llm import chat, parse_json
from .prompts import ENTITY_PROMPT

# canonical -> aliases (all lower-case matching). Extendable via data/sample/aliases.json.
ALIAS_MAP: dict[str, list[str]] = {
    "CBDT": ["cbdt", "central board of direct taxes", "income tax dept", "income tax department",
             "income-tax department", "tax authority", "the department", "it department", "itd"],
    "CBIC": ["cbic", "central board of indirect taxes and customs", "gst authority", "customs and gst board"],
    "RBI": ["rbi", "reserve bank of india", "reserve bank", "the regulator", "central bank"],
    "SEBI": ["sebi", "securities and exchange board of india", "capital markets regulator"],
    "Income Tax Act": ["income tax act", "income-tax act", "it act", "income tax act, 1961", "the act"],
    "GST Council": ["gst council", "goods and services tax council"],
    "PMLA": ["pmla", "prevention of money laundering act", "prevention of money-laundering act", "pml act"],
    "FIU-IND": ["fiu-ind", "fiu ind", "financial intelligence unit", "financial intelligence unit india"],
    "KYC": ["kyc", "know your customer", "know-your-customer"],
    "Companies Act 2013": ["companies act 2013", "companies act, 2013", "companies act"],
    "FEMA": ["fema", "foreign exchange management act"],
    "MAC": ["mac", "material adverse change", "material adverse effect", "mae"],
    "SPA": ["spa", "share purchase agreement", "the agreement"],
    "Purchaser": ["purchaser", "buyer", "acquirer"],
    "Seller": ["seller", "vendor", "promoter seller"],
}
_ALIAS_LOOKUP: dict[str, str] = {}


def register_aliases(extra: dict[str, list[str]] | None = None) -> None:
    if extra:
        for canon, al in extra.items():
            ALIAS_MAP.setdefault(canon, [])
            ALIAS_MAP[canon] = list({*ALIAS_MAP[canon], *[a.lower() for a in al], canon.lower()})
    _ALIAS_LOOKUP.clear()
    for canon, al in ALIAS_MAP.items():
        _ALIAS_LOOKUP[canon.lower()] = canon
        for a in al:
            _ALIAS_LOOKUP[a.lower()] = canon


register_aliases()
_SEC_RE = re.compile(r"^(sec(?:tion)?|s)\.?\s*(\d+[a-z]{0,3}(?:\(\d+\))?)$", re.I)
_RULE_RE = re.compile(r"^rule\s*(\d+[a-z]?)$", re.I)


def normalize_entity(name: str) -> str:
    """Map aliases / abbreviations / typographic variants to one canonical node name."""
    n = re.sub(r"\s+", " ", name.strip().strip(".,;:()\"'")).strip()
    if not n:
        return n
    low = n.lower()
    if low in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[low]
    if low.startswith("the ") and low[4:] in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[low[4:]]
    m = _SEC_RE.match(n)
    if m:
        return f"Section {m.group(2).upper()}"
    m = _RULE_RE.match(n)
    if m:
        return f"Rule {m.group(1).upper()}"
    return n


# ---------------- heuristic fallback (no API key / LLM failure) ----------------
_PATTERNS = [
    (re.compile(r"\b(?:Section|Sec\.|Rule|Clause|Regulation|Article)\s+\d+[A-Za-z]*(?:\.\d+)*(?:\(\w+\))?"), "Section"),
    (re.compile(r"\b[A-Z][A-Za-z&\-]+(?:[ ](?:(?:of|and|for|the)[ ])?[A-Z][A-Za-z&\-]+){1,5}\b"), "Entity"),
    (re.compile(r"\b[A-Z]{2,}(?:-[A-Z]{2,})?\b"), "Authority"),
]
_STOP = {"The", "This", "That", "These", "Such", "Where", "Any", "Each", "If", "In", "On", "For", "A", "An", "It"}


def _heuristic(chunk: Document) -> tuple[list[dict], list[dict]]:
    ents: dict[str, str] = {}
    rels: list[dict] = []
    for sent in re.split(r"(?<=[.;])\s+|\n+", chunk.page_content):
        found: list[str] = []
        for rx, typ in _PATTERNS:
            for m in rx.finditer(sent):
                t = m.group(0).strip()
                if t in _STOP or len(t) < 3:
                    continue
                nm = normalize_entity(t)
                ents.setdefault(nm, typ)
                if nm not in found:
                    found.append(nm)
        for a, b in zip(found, found[1:]):
            if a != b:
                rels.append({"source": a, "relation": "related_to", "target": b})
    return [{"name": k, "type": v} for k, v in list(ents.items())[:12]], rels[:12]


def _llm_batch(batch: list[Document]) -> dict[str, tuple[list[dict], list[dict]]]:
    payload = "\n\n".join(f"[chunk id: {c.metadata['chunk_id']}]\n{c.page_content[:1800]}" for c in batch)
    data = parse_json(chat("You extract precise regulatory knowledge graphs as JSON.",
                           ENTITY_PROMPT.format(chunks=payload), temperature=0.0, max_tokens=2500))
    out: dict = {}
    for item in (data or {}).get("chunks", []):
        out[item.get("id")] = (item.get("entities", []) or [], item.get("relations", []) or [])
    return out


def extract_graph_data(chunks: list[Document], use_llm: bool = True, batch_size: int = 3, max_llm_chunks: int = 60,
                       progress=None) -> tuple[dict[str, tuple[list[dict], list[dict]]], str]:
    """Returns ({chunk_id: (entities, relations)}, mode) where mode is 'llm', 'hybrid' or 'heuristic'."""
    results: dict[str, tuple[list[dict], list[dict]]] = {}
    mode = "heuristic"
    if use_llm and config.has_api_key():
        targets = chunks[:max_llm_chunks]
        batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
        done = 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_llm_batch, b) for b in batches]
            for f in futs:
                try:
                    results.update(f.result())
                    mode = "llm"
                except Exception:
                    pass
                done += 1
                if progress:
                    progress(done / len(futs))
    for c in chunks:
        cid = c.metadata["chunk_id"]
        if cid not in results or not results[cid][0]:
            results[cid] = _heuristic(c)
            if mode == "llm":
                mode = "hybrid"
    return results, mode
