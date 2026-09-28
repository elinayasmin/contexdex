"""Step 7: evaluation.

Retrieval (fast, no LLM) - for each mode, on the answerable questions:
  hit@k  = share of questions where a chunk containing the evidence text is in the top k
  MRR    = mean of 1/rank of the first correct chunk (rewards ranking it higher)
  also: query-classification accuracy and retrieval latency.

End-to-end (with --llm, slow on CPU) - full pipeline (auto category + rerank) at different top-k:
  answer accuracy  = answer contains an expected fact (and is not a refusal)
  refusal accuracy = unanswerable questions correctly answered with "I couldn't find this..."
  plus average prompt tokens and latency, to show the cost/quality trade-off of top-k.

Run from the project folder:
  python eval/run_eval.py              # retrieval only (~1 min)
  python eval/run_eval.py --llm        # + end-to-end with the LLM (~1 min per question per k on CPU)
Results are printed and written to eval/results.md.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classify import QueryClassifier  # noqa: E402
from llm import LLM_MODEL, generate  # noqa: E402
from retrieve import Retriever  # noqa: E402

REFUSAL = "couldn't find this in the documents"


def first_hit_rank(results, evidence):
    for rank, chunk in enumerate(results, start=1):
        if evidence in chunk["content"]:
            return rank
    return None


def eval_retrieval(retriever, classifier, questions):
    answerable = [q for q in questions if q["evidence"]]
    configs = [
        ("vector", "vector", False),
        ("bm25", "bm25", False),
        ("hybrid (RRF)", "hybrid", False),
        ("hybrid + rerank", "rerank", False),
        ("hybrid + rerank + auto category", "rerank", True),
    ]
    rows = []
    for name, mode, use_category in configs:
        ranks, times = [], []
        for q in answerable:
            start = time.perf_counter()
            category = classifier.classify(q["question"])[0] if use_category else None
            results = retriever.search(q["question"], k=10, category=category, mode=mode)
            times.append((time.perf_counter() - start) * 1000)
            ranks.append(first_hit_rank(results, q["evidence"]))
        n = len(answerable)
        rows.append({
            "config": name,
            "hit@1": sum(r == 1 for r in ranks if r) / n,
            "hit@3": sum(r <= 3 for r in ranks if r) / n,
            "hit@5": sum(r <= 5 for r in ranks if r) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "ms": statistics.median(times),
        })
        print(f"  {name:32} hit@1 {rows[-1]['hit@1']:.2f}  hit@3 {rows[-1]['hit@3']:.2f}  "
              f"hit@5 {rows[-1]['hit@5']:.2f}  MRR {rows[-1]['mrr']:.2f}  {rows[-1]['ms']:.0f} ms")

    correct = sum(classifier.classify(q["question"])[0] in (q["category"], None) for q in answerable)
    wrong = [q["question"] for q in answerable
             if classifier.classify(q["question"])[0] not in (q["category"], None)]
    unsure = sum(classifier.classify(q["question"])[0] is None for q in answerable)
    print(f"  classification: {correct}/{len(answerable)} not misfiltered ({unsure} searched all categories)")
    return rows, {"ok": correct, "n": len(answerable), "unsure": unsure, "wrong": wrong}


def eval_end_to_end(retriever, classifier, questions, k):
    details, tokens, llm_ms, total_ms = [], [], [], []
    for i, q in enumerate(questions, start=1):
        start = time.perf_counter()
        category = classifier.classify(q["question"])[0]
        chunks = retriever.search(q["question"], k=k, category=category)
        out = generate(q["question"], chunks)
        total_ms.append((time.perf_counter() - start) * 1000)
        llm_ms.append(out["llm_ms"])
        tokens.append(out["tokens"]["prompt"] + out["tokens"]["completion"])

        answer = out["answer"].lower()
        refused = REFUSAL in answer
        if q["answer_any"]:
            ok = not refused and any(a.lower() in answer for a in q["answer_any"])
        else:
            ok = refused
        details.append({"question": q["question"], "ok": ok, "answer": out["answer"]})
        print(f"    [k={k}] {i:2}/{len(questions)} {'OK ' if ok else 'XX '} {q['question']}")

    answerable = [d for d, q in zip(details, questions) if q["answer_any"]]
    unanswerable = [d for d, q in zip(details, questions) if not q["answer_any"]]
    return {
        "k": k,
        "accuracy": sum(d["ok"] for d in answerable) / len(answerable),
        "refusal": sum(d["ok"] for d in unanswerable) / len(unanswerable) if unanswerable else None,
        "tokens": statistics.mean(tokens),
        "llm_s": statistics.median(llm_ms) / 1000,
        "total_s": statistics.median(total_ms) / 1000,
        "details": details,
    }


def write_report(retrieval_rows, cls, e2e, n_questions):
    lines = [
        "# ContexDex evaluation results",
        "",
        f"Run on {date.today()} · {n_questions} questions "
        f"({cls['n']} answerable, {n_questions - cls['n']} unanswerable) · 102 chunks "
        "from 4 GitLab handbook pages · CPU laptop",
        "",
        "## Retrieval (answerable questions)",
        "",
        "| Configuration | hit@1 | hit@3 | hit@5 | MRR | median latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in retrieval_rows:
        lines.append(f"| {r['config']} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} "
                     f"| {r['mrr']:.2f} | {r['ms']:.0f} ms |")
    lines += [
        "",
        f"Query classification: {cls['ok']}/{cls['n']} questions routed to the right category "
        f"or to all categories ({cls['unsure']} fell back to searching everything).",
    ]
    if cls["wrong"]:
        lines.append("Misrouted: " + "; ".join(f"“{w}”" for w in cls["wrong"]))

    if e2e:
        lines += [
            "",
            f"## End-to-end answers (auto category + hybrid + rerank, `{LLM_MODEL}` via Ollama)",
            "",
            "| top-k chunks | answer accuracy | correct refusals | avg tokens / question | median LLM time | median total time |",
            "|---|---|---|---|---|---|",
        ]
        for r in e2e:
            refusal = f"{r['refusal']:.0%}" if r["refusal"] is not None else "-"
            lines.append(f"| {r['k']} | {r['accuracy']:.0%} | {refusal} | {r['tokens']:.0f} "
                         f"| {r['llm_s']:.1f} s | {r['total_s']:.1f} s |")
        lines += ["", "### Failed answers", ""]
        for r in e2e:
            for d in r["details"]:
                if not d["ok"]:
                    lines.append(f"- **k={r['k']}** {d['question']} → {d['answer'][:200]}")

    path = ROOT / "eval" / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also run end-to-end answer evaluation")
    parser.add_argument("--ks", default="4,2", help="comma-separated top-k values for --llm")
    args = parser.parse_args()

    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    retriever = Retriever()
    classifier = QueryClassifier(retriever.embedder, retriever.conn)

    print("Retrieval:")
    retrieval_rows, cls = eval_retrieval(retriever, classifier, questions)

    e2e = []
    if args.llm:
        print("End-to-end:")
        for k in [int(x) for x in args.ks.split(",")]:
            e2e.append(eval_end_to_end(retriever, classifier, questions, k))
            r = e2e[-1]
            print(f"  k={k}: accuracy {r['accuracy']:.0%}, refusals {r['refusal']:.0%}, "
                  f"avg tokens {r['tokens']:.0f}, median total {r['total_s']:.1f} s")

    write_report(retrieval_rows, cls, e2e, len(questions))


if __name__ == "__main__":
    main()
