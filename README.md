# ContexDex – Enterprise Knowledge RAG System

A small retrieval-augmented generation (RAG) system: it ingests PDFs and web pages, stores chunk embeddings in PostgreSQL + pgvector, retrieves with hybrid search (BM25 + vectors + reranking), and answers questions through a Flask API.

> Work in progress – built step by step.

## Stack
Python · Flask · PostgreSQL + pgvector · sentence-transformers · rank-bm25 · Ollama (llama3.2:3b, runs locally, free)

## Setup

```bash
# 1. Start Postgres with pgvector
docker compose up -d

# 2. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env       # macOS/Linux: cp .env.example .env

# 4. Check the database connection
python db.py

# 5. Install Ollama (https://ollama.com/download) and pull the model
ollama pull llama3.2:3b
```

## Ingesting documents

- PDFs: put them in `data/<category>/` (e.g. `data/hr/leave-policy.pdf`). The folder name becomes the category.
- Web pages: add `<category> <url>` lines to `data/urls.txt`.

```bash
python ingest.py            # add or refresh sources
python ingest.py --reset    # start from an empty table
```

Each document is split into sections (PDF headings / HTML `h1`-`h3`), then into ~350-word chunks with 50-word overlap. Chunks are embedded with `all-MiniLM-L6-v2` and stored with their source, section and category.

## Retrieval

```bash
python retrieve.py "Can I expense a laptop?"
python retrieve.py "Can I expense a laptop?" --category finance --mode hybrid
```

1. **Vector search**: cosine similarity in pgvector (HNSW index), top 20
2. **BM25**: keyword search over chunk text + section heading, top 20
3. **Reciprocal Rank Fusion**: merges both lists (`score = Σ 1/(60 + rank)`)
4. **Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2` re-scores the fused candidates and keeps the top k

`--category` restricts every stage to one category (metadata filtering). `--mode vector|bm25|hybrid|rerank` exists so each stage can be compared in evaluation.

## Query classification

```bash
python classify.py "How do I reset my password?"
python retrieve.py "What is my last day process?" --category auto
```

The classifier picks the category so retrieval only searches the relevant documents:

1. **Keyword rules**: if one category's keywords clearly win, use it (fast, explainable).
2. **Embedding fallback**: compare the query with each category's centroid (`AVG(embedding)` in pgvector).
3. **Uncertain**: if the top two categories are within 0.05 similarity, search all categories instead of guessing.

## Answer generation

```bash
python llm.py "Can I expense a laptop?"
```

Runs the full pipeline: classify → hybrid retrieve → rerank → local LLM. The top chunks are numbered `[1]..[n]` in the prompt, and the model must answer only from them, cite the numbers, and reply *"I couldn't find this in the documents."* otherwise. Context is capped at ~1,500 words to limit tokens. Each answer returns cited sources, token counts and LLM latency.

## REST API

```bash
python app.py        # serves on http://127.0.0.1:5000
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Checks the database and Ollama are reachable |
| POST | `/ask` | Answers a question |
| POST | `/reload` | Rebuilds the BM25 index and category centroids after `ingest.py` |

`/ask` body: `{"question": "...", "k": 4, "category": "auto", "mode": "rerank"}`. Only `question` is required.

```bash
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d "{\"question\": \"How much can I spend on books?\"}"
```

Example response:

```json
{
  "question": "How much can I spend on books?",
  "answer": "According to [1] (4.1 NON-TRAVEL RELATED EXPENSES), books are reimbursable if used to optimize your job position, with a limit set to $60 per year.",
  "category": "finance",
  "classified_by": "keywords",
  "sources": [
    {"ref": 1, "section": "4.1 NON-TRAVEL RELATED EXPENSES", "category": "finance",
     "source": "https://handbook.gitlab.com/handbook/finance/expenses/"}
  ],
  "tokens": {"prompt": 1491, "completion": 43},
  "latency_ms": {"retrieval": 1624, "llm": 60165, "total": 61790}
}
```

LLM latency is from `llama3.2:3b` running on a laptop CPU.

## Roadmap
- [x] Project setup, Postgres + pgvector
- [x] Ingestion: PDFs/URLs → chunks → embeddings
- [x] Hybrid retrieval (BM25 + vector + RRF) and cross-encoder reranking
- [x] Query classification + metadata filtering
- [x] LLM answer generation with citations
- [x] Flask `/ask` endpoint
- [ ] Evaluation (hit@k, latency, token usage)
