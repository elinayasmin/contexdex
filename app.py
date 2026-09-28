"""Step 6: Flask REST API.

Run:
  python app.py

Endpoints:
  GET  /health   -> checks the database and Ollama are reachable
  POST /ask      -> {"question": "...", "k": 2, "category": "auto", "mode": "rerank"}
                    only "question" is required
  POST /reload   -> rebuild the BM25 index and category centroids after running ingest.py
"""
import time

import requests
from flask import Flask, jsonify, request

from classify import QueryClassifier
from llm import LLM_MODEL, OLLAMA_URL, generate
from retrieve import Retriever

app = Flask(__name__)

# Models and indexes are loaded once at startup, not per request.
retriever = Retriever()
classifier = QueryClassifier(retriever.embedder, retriever.conn)

MODES = {"vector", "bm25", "hybrid", "rerank"}


@app.get("/health")
def health():
    status = {"database": "ok", "ollama": "ok", "chunks": len(retriever.chunks), "model": LLM_MODEL}
    try:
        retriever.conn.execute("SELECT 1")
    except Exception as e:
        status["database"] = f"error: {e}"
    try:
        requests.get(OLLAMA_URL, timeout=3).raise_for_status()
    except Exception:
        status["ollama"] = "unreachable - is Ollama running?"
    code = 200 if status["database"] == "ok" and status["ollama"] == "ok" else 503
    return jsonify(status), code


@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "'question' is required"}), 400

    k = body.get("k", 2)   # k=2 matched k=4 accuracy at ~half the tokens (see eval/results.md)
    mode = body.get("mode", "rerank")
    category = body.get("category", "auto")
    if not isinstance(k, int) or not 1 <= k <= 10:
        return jsonify({"error": "'k' must be an integer from 1 to 10"}), 400
    if mode not in MODES:
        return jsonify({"error": f"'mode' must be one of {sorted(MODES)}"}), 400

    start = time.perf_counter()
    if category == "auto":
        category, classified_by, _ = classifier.classify(question)
    elif category and category not in classifier.centroids:
        return jsonify({"error": f"unknown category; use one of {sorted(classifier.centroids)} or 'auto'"}), 400
    else:
        classified_by = "request"

    chunks = retriever.search(question, k=k, category=category, mode=mode)
    retrieval_ms = round((time.perf_counter() - start) * 1000)

    try:
        result = generate(question, chunks)
    except requests.RequestException as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 503

    return jsonify({
        "question": question,
        "answer": result["answer"],
        "category": category or "all",
        "classified_by": classified_by,
        "sources": result["sources"],
        "tokens": result["tokens"],
        "latency_ms": {
            "retrieval": retrieval_ms,
            "llm": result["llm_ms"],
            "total": round((time.perf_counter() - start) * 1000),
        },
    })


@app.post("/reload")
def reload_index():
    retriever.refresh()
    classifier.refresh()
    return jsonify({"chunks": len(retriever.chunks), "categories": sorted(classifier.centroids)})


if __name__ == "__main__":
    # threaded=False: the shared DB connection and models are used by one request at a time.
    app.run(host="127.0.0.1", port=5000, threaded=False)
