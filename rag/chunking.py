"""STEP 2/3 — Semantic chunking with sub-word (HF tokenizer) length function; clause-aware separators."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

# Prefer splitting on clause / section boundaries before sentences so legal clauses stay whole.
LEGAL_SEPARATORS = ["\n\n", "\nSection ", "\nClause ", "\nRule ", "\n", ". ", "; ", " ", ""]


@lru_cache(maxsize=1)
def get_tokenizer():
    """BAAI/bge-large-en-v1.5 WordPiece tokenizer; None if unavailable (word-based fallback)."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    except Exception:
        return None


def token_len(text: str) -> int:
    tok = get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False, truncation=False, verbose=False))
    return int(len(text.split()) * 1.3)


def chunk_documents(docs: list[Document], chunk_size: int = 600, chunk_overlap: int = 60) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=token_len, keep_separator="start")
    out: list[Document] = []
    counters: dict[str, int] = {}
    for d in docs:
        for piece in splitter.split_text(d.page_content):
            src = d.metadata["source"]
            counters[src] = counters.get(src, 0) + 1
            cid = f"{src}::c{counters[src]}"
            out.append(Document(page_content=piece.strip(),
                                metadata={**d.metadata, "chunk_id": cid, "tokens": token_len(piece)}))
    return [c for c in out if c.page_content]
