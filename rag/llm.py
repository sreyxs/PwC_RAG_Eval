"""OpenRouter-backed Claude Sonnet 5.5 client + deterministic token-estimate footer."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import config

_FOOTER_RE = re.compile(r"\n*\s*(?:-{3,}\s*\n+)?\s*\*?Estimated Output Tokens:[^\n]*\s*$", re.I)


def get_llm(temperature: float = 0.1, max_tokens: int = 1800) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.api_key(),
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=2,
        default_headers={"HTTP-Referer": "https://pwc.in/advisory-rag-benchmark", "X-Title": "PwC RAG Benchmark"},
    )


def estimate_tokens(text: str) -> int:
    """Heuristic from the spec: chars/4 or words*1.33, +12% for code/JSON/non-English content."""
    chars = len(text)
    words = len(text.split())
    est = max(chars / 4.0, words * 1.33)
    heavy = text.count("```") >= 2 or text.count("{") > 8 or any(ord(c) > 127 for c in text)
    if heavy:
        est *= 1.12
    return max(1, round(est))


def ensure_token_footer(text: str) -> str:
    """Strip any model-produced footer and append a deterministic one at the very bottom."""
    body = _FOOTER_RE.sub("", text).rstrip()
    n = estimate_tokens(body)
    return f"{body}\n\n---\n*Estimated Output Tokens: ~{n} tokens*"


def chat(system: str, user: str, **kw) -> str:
    resp = get_llm(**kw).invoke([SystemMessage(content=system), HumanMessage(content=user)])
    c = resp.content
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict))


def parse_json(text: str):
    """Tolerant JSON extraction from an LLM reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
