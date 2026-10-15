# Original vs. Cohere RAG pipeline

104 test questions, both systems judged by `llama3:8b` (local, via Ollama). `n=` shows how many questions got a valid score when the judge failed to parse some outputs; those are left out of the average.

| Metric | Original | Cohere |
|---|---|---|
| Answer score (1-5) | 4.64 | 4.91 |
| Answer hit rate (score ≥ 4) | 95.2% | 99.0% |

RAGAS metrics were not run (see `--ragas`).
Cohere answers with citations: 96/105.
Cohere pipeline latency: 3.75s per answer on average (retrieval 1.65s, generation 2.10s), via the API.
The original pipeline's 2.16s (dissertation) was measured on a Colab GPU, so latencies are not directly comparable.
