# PwC India Advisory — Standard vs HyDE vs Graph RAG benchmark

LangChain + LangGraph app with a 5-tab Streamlit UI. LLM: Claude Sonnet 5.5 via OpenRouter.

## Run
```bash
pip install -r requirements.txt
# paste your key into .env.example (OPENROUTER_API_KEY=...), or create .env (takes precedence)
streamlit run app.py
```
In tab 1 click **Load sample corpus** (synthetic Tax / GRC / Deal documents, not real law), then chat or run the benchmark in tab 5.

## Layout
- `rag/` — ingestion, token-based chunking (600/60), entity extraction + alias normalisation, bge-large-en-v1.5 + Node2Vec embeddings,
  Chroma (HNSW) + NetworkX + BM25 dual index, RRF/rerank, LangGraph pipelines, KPI metrics.
- `ui/` — one module per tab. `.streamlit/config.toml` holds the theme.
- Every answer ends with `*Estimated Output Tokens: ~N tokens*` (appended deterministically; the rule is also in the system prompt).

Neo4j / Qdrant are not wired in; the spec allows Chroma + NetworkX as the alternative. `pytest tests` runs unit tests.
