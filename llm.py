"""Step 5: answer generation with a local LLM (Ollama), grounded in the retrieved chunks.

The prompt numbers each chunk [1], [2], ... and tells the model to answer ONLY from them and cite
the numbers, so every answer can be traced back to a source - and to say so when the answer isn't there.

Run (needs Ollama running and the model pulled: `ollama pull llama3.2:3b`):
  python llm.py "Can I expense a laptop?"
"""
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
MAX_CONTEXT_WORDS = 1500   # cap on retrieved text sent to the model (~2k tokens) to keep answers fast

SYSTEM_PROMPT = """You are ContexDex, an assistant that answers employee questions using company documents.
Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- Cite the sources you used with their numbers in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "I couldn't find this in the documents."
- Be concise: 2-5 sentences."""


def build_prompt(question, chunks):
    """Number the chunks and stop adding them once the word budget is used up."""
    parts, used, words = [], [], 0
    for chunk in chunks:
        n = len(chunk["content"].split())
        if used and words + n > MAX_CONTEXT_WORDS:
            break
        used.append(chunk)
        words += n
        parts.append(f"[{len(used)}] ({chunk['section']})\n{chunk['content']}")
    context = "\n\n".join(parts)
    return f"Sources:\n\n{context}\n\nQuestion: {question}", used


def generate(question, chunks):
    prompt, used = build_prompt(question, chunks)
    start = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data["message"]["content"].strip()

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(used)})
    return {
        "answer": answer,
        "sources": [
            {"ref": i, "source": c["source"], "section": c["section"], "category": c["category"]}
            for i, c in enumerate(used, start=1) if i in cited
        ],
        "tokens": {"prompt": data.get("prompt_eval_count", 0), "completion": data.get("eval_count", 0)},
        "llm_ms": round((time.perf_counter() - start) * 1000),
    }


if __name__ == "__main__":
    import argparse

    from classify import QueryClassifier
    from retrieve import Retriever

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    retriever = Retriever()
    category, method, _ = QueryClassifier(retriever.embedder, retriever.conn).classify(args.question)
    chunks = retriever.search(args.question, k=args.k, category=category)
    result = generate(args.question, chunks)

    print(f"category: {category or 'all'} (via {method})\n")
    print(result["answer"], "\n")
    for s in result["sources"]:
        print(f"  [{s['ref']}] {s['section']} - {s['source']}")
    print(f"\ntokens: {result['tokens']}  llm time: {result['llm_ms']} ms")
