# Cohere RAG: the Sheffield research chatbot rebuilt on Cohere

This branch rebuilds the [original RAG chatbot](../README.md) (my BSc dissertation) on Cohere's models
and measures the difference with the same evaluation method, on the same test questions.

| Stage | Original (`main.ipynb`) | Cohere version (`cohere_rag/`) |
|---|---|---|
| Corpus | 499 LLM-cleaned pages of the Sheffield Computer Science department | same pages, same chunking (800 chars, 80 overlap → 1,155 chunks) |
| Retrieval | BM25 + Chroma (`nomic-embed-text-v1.5`), top 10 each | BM25 + **Cohere Embed** (`embed-v5.0-pro`), top 10 each |
| Reranking | `BAAI/bge-reranker-v2-m3`, score > 0.2, top 3 | **Cohere Rerank** (`rerank-v4.0-pro`), score ≥ 0.1, top 5 |
| Generation | Llama 3.2 3B (local) + a second "refine" pass | **Command A+** (`command-a-plus-05-2026`), one grounded call |
| Citations | none | every claim is linked to the chunk it came from |

## Results

Both systems answered the same **104 test questions** and were scored by the **same judge** (Llama 3 8B, run
locally), using the answer-grader prompt from the dissertation (1 = completely wrong, 5 = fully correct, no hallucinations).

| | Original (Llama 3.2 3B) | Cohere (Command A+) |
|---|---|---|
| Average answer score (1-5) | 4.64 | **4.91** |
| Answers scored ≥ 4 | 95.2% | **99.0%** |
| Answers scored 2 (incorrect or hallucinated) | 5 | **1** |
| Answers with citations to source chunks | – | **96 / 105** |

- The one question the corpus can't answer (BSc entry requirements) gets an explicit "the documents don't say",
  not a guess.
- Cohere answers average 149 words vs 70 for the original, even with the original prompt's "no additional
  explanations" instruction (the original also had a second rewriting pass). Longer answers are not penalised by
  the answer score, but would be by RAGAS factual correctness (F1).
- Latency: 3.75 s per answer via the API (retrieval 1.65 s, including trial-key rate limiting; generation 2.10 s).
  The dissertation's 2.16 s was measured on a Colab GPU, so the two are not directly comparable.
- These scores are higher than the dissertation's (3.79 for the original) because the dissertation graded answers with
  the smaller Llama 3.2 3B; only the two columns above are comparable with each other.

Full numbers: [`results/summary.md`](results/summary.md); per-question scores and judge explanations:
`results/grades_*.jsonl`; every Cohere answer with its citations and sources: `results/cohere_answers.json`.

## How the comparison was kept fair

- **Same questions and references:** the 104 test questions shared by both runs (`../questions.txt`, `../answers.txt`).
  One question in the file is a near-duplicate of another and was missing from the original run, so it is left out.
- **Same judge for both systems:** Llama 3 8B running locally (Ollama), the RAGAS judge used in the dissertation.
  Cohere never grades its own answers.
- **Same prompt and scoring:** the Cohere system prompt keeps the original instructions, and answers are graded with
  the dissertation's answer-grader prompt. RAGAS (context recall/precision, faithfulness, factual correctness) is
  available with `python judge.py --ragas N` on a sample of N questions; it was not run for the table above because
  RAGAS with a local 8B judge takes hours on a laptop.
- **Original answers are the dissertation's saved outputs**, re-scored with the same judge in the same run, rather than
  numbers copied from the report.

## Run it

```bash
cd cohere_rag
uv venv && uv pip install -r requirements.txt     # or: python -m venv .venv && pip install -r requirements.txt
cp .env.example .env                               # add COHERE_API_KEY, DATA_DIR and BASELINE_JSON

python ask.py "What are Dr. Yoshi Gotoh's research interests?"   # one cited answer
python run_eval.py      # answer the test questions → results/cohere_answers.json
ollama pull llama3:8b   # local judge
python judge.py         # score both systems → results/summary.md (add --ragas 20 for RAGAS on 20 questions)
```

A free Cohere trial key is enough: indexing takes ~13 calls (cached afterwards) and the full run ~315.
`run_eval.py` waits 6.5 s between questions to respect the trial limit of 10 rerank calls per minute.

The corpus (`final_llm_parsed_data`) is not in the repository: the pages were collected for the dissertation and
are not mine to publish. For the same reason the retrieved passages are kept out of the published results
(`results/private/` is git-ignored).

## Example

```
$ python ask.py "What are Dr. Yoshi Gotoh's research interests?"

…
Dr. Gotoh is a Lecturer in the Department of Computer Science[1] and a member of the Speech and Hearing (SpandH)
research group[1] at the University of Sheffield. His expertise spans both speech and spoken language processing[1]
and audio-visual processing[1], with recent work focusing particularly on video analysis and retrieval systems.[1]

  [1] https://www.sheffield.ac.uk/dcs/people/academic/yoshi-gotoh  (relevance 0.98)
  [2] http://staffwww.dcs.shef.ac.uk/people/Y.Gotoh/  (relevance 0.98)
  [3] https://www.sheffield.ac.uk/dcs/people/academic/yoshi-gotoh  (relevance 0.90)
  [4] https://www.sheffield.ac.uk/dcs/research/groups/spandh  (relevance 0.77)
  [5] https://www.sheffield.ac.uk/dcs/people/academic/yoshi-gotoh  (relevance 0.60)
```
