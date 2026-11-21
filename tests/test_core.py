import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag.chunking import chunk_documents, token_len
from rag.entities import normalize_entity
from rag.llm import ensure_token_footer, estimate_tokens
from rag.metrics import retrieval_context_recall
from rag.retrievers import rrf


def test_rrf_math():
    out = dict(rrf([[1, 2], [2, 1]], alpha=60))
    assert abs(out[1] - (1 / 61 + 1 / 62)) < 1e-9 and abs(out[1] - out[2]) < 1e-9
    assert rrf([[5, 6], [5]], 60)[0][0] == 5


def test_alias_normalisation():
    assert normalize_entity("Income Tax Dept") == "CBDT"
    assert normalize_entity("Tax Authority") == "CBDT"
    assert normalize_entity("Sec. 194c") == "Section 194C"
    assert normalize_entity("Reserve Bank of India") == "RBI"


def test_token_footer():
    out = ensure_token_footer("Answer text.\n\n---\n*Estimated Output Tokens: ~999 tokens*")
    assert out.count("Estimated Output Tokens") == 1
    assert out.rstrip().endswith("tokens*") and "\n---\n*Estimated Output Tokens: ~" in out
    assert estimate_tokens("a" * 400) >= 100


def test_chunk_size_limit():
    text = "\n\n".join(f"Section {i} The CBDT clarifies that tax shall be deducted at source. " * 12 for i in range(30))
    chunks = chunk_documents([Document(page_content=text, metadata={"source": "x"})], 600, 60)
    assert chunks and all(token_len(c.page_content) <= 620 for c in chunks)


def test_rcr():
    ctx = [{"text": "Rate is 10% for professional fees"}]
    assert retrieval_context_recall(ctx, [["10%", "professional"], ["missing"]]) == 0.5
    assert retrieval_context_recall(ctx, [["five years|10%"]]) == 1.0
