# ContexDex evaluation results

Run on 2026-09-28 · 25 questions (22 answerable, 3 unanswerable) · 102 chunks from 4 GitLab handbook pages · CPU laptop

## Retrieval (answerable questions)

| Configuration | hit@1 | hit@3 | hit@5 | MRR | median latency |
|---|---|---|---|---|---|
| vector | 73% | 91% | 95% | 0.83 | 29 ms |
| bm25 | 55% | 91% | 95% | 0.73 | 1 ms |
| hybrid (RRF) | 82% | 95% | 100% | 0.89 | 26 ms |
| hybrid + rerank | 82% | 95% | 100% | 0.90 | 2522 ms |
| hybrid + rerank + auto category | 82% | 100% | 100% | 0.90 | 2304 ms |

Query classification: 22/22 questions routed to the right category or to all categories (1 fell back to searching everything).

## End-to-end answers (auto category + hybrid + rerank, `llama3.2:3b` via Ollama)

| top-k chunks | answer accuracy | correct refusals | avg tokens / question | median LLM time | median total time |
|---|---|---|---|---|---|
| 4 | 95% | 100% | 1384 | 46.3 s | 48.5 s |
| 2 | 95% | 100% | 728 | 6.5 s | 8.7 s |

### Failed answers

- **k=4** Can I get my ExpressVPN subscription reimbursed? → I couldn't find this in the documents.
- **k=2** Do I have to give back company equipment when I leave? → I couldn't find this in the documents.
