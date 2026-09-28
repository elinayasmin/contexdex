"""Step 4: query classification - pick which category (hr / it / finance) a question belongs to,
so retrieval only searches the right part of the knowledge base.

Two stages:
  1. Keyword rules  - fast and explainable; used when exactly one category matches best.
  2. Embedding fallback - compare the question with each category's centroid (the average embedding
     of all its chunks, computed by pgvector). Only trusted when one category clearly wins;
     otherwise we return None and search every category rather than risk filtering out the answer.

Run:
  python classify.py "How do I reset my password?"
"""
import re

import numpy as np

KEYWORDS = {
    "hr": ["leave", "vacation", "pto", "time off", "holiday", "sick", "parental", "resign",
           "offboard", "terminat", "benefit", "onboard", "manager", "performance", "hiring"],
    "it": ["password", "vpn", "security", "2fa", "mfa", "phishing", "access", "sso", "okta",
           "device", "encrypt", "incident", "malware", "login"],
    "finance": ["expense", "reimburs", "receipt", "invoice", "travel", "payment", "budget",
                "cost", "per diem", "navan", "card", "flight", "hotel", "spend"],
}

MIN_MARGIN = 0.05   # centroid similarity gap needed between 1st and 2nd category to trust the fallback


class QueryClassifier:
    def __init__(self, embedder, conn):
        self.embedder = embedder
        self.conn = conn
        self.refresh()

    def refresh(self):
        """Load one centroid per category. pgvector can average vectors directly in SQL."""
        rows = self.conn.execute("SELECT category, AVG(embedding) FROM chunks GROUP BY category").fetchall()
        self.centroids = {}
        for cat, vec in rows:
            vec = np.asarray(vec.to_numpy() if hasattr(vec, "to_numpy") else vec, dtype=np.float32)
            self.centroids[cat] = vec / np.linalg.norm(vec)

    def keyword_scores(self, query):
        q = query.lower()
        return {
            cat: sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}", q))
            for cat, kws in KEYWORDS.items() if cat in self.centroids
        }

    def classify(self, query):
        """Return (category or None, method, details)."""
        scores = self.keyword_scores(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if ranked and ranked[0][1] > 0 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            return ranked[0][0], "keywords", scores

        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        sims = {cat: float(q_vec @ c) for cat, c in self.centroids.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= MIN_MARGIN:
            return ranked[0][0], "embedding", sims
        return None, "uncertain", sims


if __name__ == "__main__":
    import argparse

    from retrieve import Retriever

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()

    r = Retriever()
    category, method, details = QueryClassifier(r.embedder, r.conn).classify(args.query)
    print(f"category: {category or 'all'}  (via {method})")
    print({k: round(v, 3) for k, v in details.items()})
