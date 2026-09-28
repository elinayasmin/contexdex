"""Step 3: hybrid retrieval = BM25 (keywords) + pgvector (meaning), merged with RRF, then reranked.

Run:
  python retrieve.py "Can I expense a laptop?"
  python retrieve.py "Can I expense a laptop?" --category finance --mode hybrid

Modes (useful for comparing in the evaluation step):
  vector  - embedding similarity only
  bm25    - keyword matching only
  hybrid  - vector + BM25 merged with Reciprocal Rank Fusion
  rerank  - hybrid, then a cross-encoder re-scores the candidates (default)
"""
import argparse
import os
import re

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from db import get_conn

load_dotenv()

CANDIDATES = 20   # how many results each retriever contributes before fusion/reranking
RRF_K = 60        # standard constant from the RRF paper; dampens the effect of top ranks

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "be",
    "i", "you", "we", "it", "can", "do", "does", "how", "what", "when", "my", "our", "with", "at",
}


def tokenize(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


class Retriever:
    """Loads models and the BM25 index once, then answers many queries."""

    def __init__(self):
        self.embedder = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        self.reranker = CrossEncoder(os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        self.conn = get_conn(autocommit=True)  # long-lived read-only connection
        self.refresh()

    def refresh(self):
        """(Re)build the in-memory BM25 index. Call again after running ingest.py."""
        rows = self.conn.execute(
            "SELECT id, content, source, section, category FROM chunks ORDER BY id"
        ).fetchall()
        self.chunks = {
            r[0]: {"id": r[0], "content": r[1], "source": r[2], "section": r[3], "category": r[4]}
            for r in rows
        }
        self.ids = [r[0] for r in rows]
        # Include the section heading so keyword matches on headings count too.
        self.bm25 = BM25Okapi([tokenize(f"{r[3]} {r[1]}") for r in rows]) if rows else None

    # ---------- individual retrievers: each returns a ranked list of chunk ids ----------

    def vector_search(self, query, category=None, n=CANDIDATES):
        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        sql = "SELECT id FROM chunks"
        params = []
        if category:
            sql += " WHERE category = %s"
            params.append(category)
        sql += " ORDER BY embedding <=> %s LIMIT %s"   # <=> is cosine distance
        params += [q_vec, n]
        return [r[0] for r in self.conn.execute(sql, params)]

    def bm25_search(self, query, category=None, n=CANDIDATES):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, score in ranked
                if score > 0 and (not category or self.chunks[cid]["category"] == category)][:n]

    # ---------- combining ----------

    @staticmethod
    def rrf(*ranked_lists):
        """Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank) over every list a chunk appears in."""
        scores = {}
        for ranked in ranked_lists:
            for rank, cid in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        return sorted(scores, key=scores.get, reverse=True)

    def rerank(self, query, ids):
        """Cross-encoder reads (query, chunk) together - slower but more accurate than embeddings."""
        if not ids:
            return []
        pairs = [(query, f"{self.chunks[cid]['section']}: {self.chunks[cid]['content']}") for cid in ids]
        scores = self.reranker.predict(pairs)
        return [cid for cid, _ in sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)]

    def search(self, query, k=5, category=None, mode="rerank"):
        if mode == "vector":
            ids = self.vector_search(query, category)
        elif mode == "bm25":
            ids = self.bm25_search(query, category)
        else:
            ids = self.rrf(self.vector_search(query, category), self.bm25_search(query, category))
            if mode == "rerank":
                ids = self.rerank(query, ids[:CANDIDATES])
        return [self.chunks[cid] for cid in ids[:k]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--category", help="a category name, or 'auto' to let the classifier pick")
    parser.add_argument("--mode", default="rerank", choices=["vector", "bm25", "hybrid", "rerank"])
    args = parser.parse_args()

    retriever = Retriever()
    category = args.category
    if category == "auto":
        from classify import QueryClassifier
        category, method, _ = QueryClassifier(retriever.embedder, retriever.conn).classify(args.query)
        print(f"classified as: {category or 'all categories'} (via {method})\n")

    for i, c in enumerate(retriever.search(args.query, args.k, category, args.mode), start=1):
        print(f"{i}. [{c['category']}] {c['section']}  ({c['source']})")
        print(f"   {c['content'][:150]}...\n")
