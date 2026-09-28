# ContexDex evaluation results

Run on 2026-09-28 · 25 questions (22 answerable, 3 unanswerable) · 102 chunks from 4 GitLab handbook pages · CPU laptop

## Retrieval (answerable questions)

| Configuration | hit@1 | hit@3 | hit@5 | MRR | median latency |
|---|---|---|---|---|---|
| vector | 73% | 91% | 95% | 0.83 | 20 ms |
| bm25 | 55% | 91% | 95% | 0.73 | 0 ms |
| hybrid (RRF) | 82% | 95% | 100% | 0.89 | 15 ms |
| hybrid + rerank | 82% | 95% | 100% | 0.90 | 2344 ms |
| hybrid + rerank + auto category | 82% | 100% | 100% | 0.90 | 2443 ms |

Query classification: 22/22 questions routed to the right category or to all categories (1 fell back to searching everything).
