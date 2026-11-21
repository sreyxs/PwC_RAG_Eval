"""Central configuration: environment, model ids and the 4 system hyperparameters."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# .env wins; .env.example is a convenience fallback because the key is pasted there.
load_dotenv(ROOT / ".env.example")
load_dotenv(ROOT / ".env", override=True)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-sonnet-5.5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
SAMPLE_DIR = ROOT / "data" / "sample"


def api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def has_api_key() -> bool:
    return len(api_key()) > 8


@dataclass
class Hyperparams:
    """The 4 configurable hyperparameters from the specification."""
    chunk_size: int = 600      # tokens
    chunk_overlap: int = 60    # tokens
    hyde_n: int = 3            # Multi-HyDE synthetic passages
    hop_k: int = 2             # graph traversal radius
    rrf_alpha: int = 60        # RRF constant

    def as_dict(self) -> dict:
        return asdict(self)


PRACTICE_AREAS = {
    "tax": "Tax & Regulatory Services",
    "grc": "Risk Advisory & Governance (GRC)",
    "deals": "Deal Advisory & Corporate Finance",
}
ENGINES = {"standard": "Standard RAG", "hyde": "HyDE RAG", "graph": "Graph RAG"}
