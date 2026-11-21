"""STEP 1 — Data format & ingestion: PDF / DOCX / JSON / CSV / TXT -> LangChain Documents with provenance."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document


def _meta(name: str, practice_area: str, **extra) -> dict:
    return {"source": name, "practice_area": practice_area,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}


def _load_pdf(name: str, data: bytes, area: str) -> list[Document]:
    try:  # spec asks for UnstructuredPDFLoader; use it when the heavy dependency is installed
        import tempfile
        from langchain_community.document_loaders import UnstructuredPDFLoader  # type: ignore
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(data); tmp.flush()
            docs = UnstructuredPDFLoader(tmp.name).load()
        text = "\n".join(d.page_content for d in docs)
        return [Document(page_content=text, metadata=_meta(name, area, page=1, loader="unstructured"))]
    except Exception:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [Document(page_content=(p.extract_text() or ""), metadata=_meta(name, area, page=i + 1, loader="pypdf"))
                for i, p in enumerate(reader.pages) if (p.extract_text() or "").strip()]


def _load_docx(name: str, data: bytes, area: str) -> list[Document]:
    import docx
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [Document(page_content="\n".join(parts), metadata=_meta(name, area, page=1, loader="python-docx"))]


def _load_json(name: str, data: bytes, area: str) -> list[Document]:
    obj = json.loads(data.decode("utf-8", "ignore"))
    rows = obj if isinstance(obj, list) else [obj]
    docs = []
    for i, r in enumerate(rows):
        text = "\n".join(f"{k}: {v}" for k, v in r.items()) if isinstance(r, dict) else str(r)
        docs.append(Document(page_content=text, metadata=_meta(name, area, page=i + 1, loader="json")))
    return docs


def _load_csv(name: str, data: bytes, area: str) -> list[Document]:
    rdr = csv.DictReader(io.StringIO(data.decode("utf-8", "ignore")))
    return [Document(page_content="\n".join(f"{k}: {v}" for k, v in r.items()),
                     metadata=_meta(name, area, page=i + 1, loader="csv")) for i, r in enumerate(rdr)]


def _load_text(name: str, data: bytes, area: str) -> list[Document]:
    return [Document(page_content=data.decode("utf-8", "ignore"), metadata=_meta(name, area, page=1, loader="text"))]


LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".json": _load_json, ".csv": _load_csv,
           ".txt": _load_text, ".md": _load_text}


def load_bytes(name: str, data: bytes, practice_area: str) -> list[Document]:
    ext = Path(name).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return [d for d in LOADERS[ext](name, data, practice_area) if d.page_content.strip()]


def load_path(path: Path, practice_area: str) -> list[Document]:
    return load_bytes(path.name, path.read_bytes(), practice_area)
