"""Score the original pipeline and the Cohere pipeline with the SAME judge, on the SAME questions.

Judge: Llama 3 8B served locally by Ollama (the RAGAS judge used in the dissertation), so no system grades itself.
  1. Answer score (1-5) with the dissertation's answer-grader prompt (main.ipynb)
  2. RAGAS: context recall, faithfulness, factual correctness (F1), context precision with/without reference

Inputs : BASELINE_JSON  = the original pipeline's saved outputs (RAG_chat_data.json from the dissertation run)
         results/private/cohere_outputs.json (from run_eval.py)
Outputs: results/grades_<system>.jsonl, results/ragas_<system>.csv, results/summary.md
Every step is cached, so the (slow, local) judging can be stopped and resumed.

    python judge.py              # answer scores only (~20 min on an M2 Pro)
    python judge.py --ragas 20   # also RAGAS on a fixed random sample of 20 questions (slow on a laptop)
"""
import argparse
import random
import json
import os
import re
from pathlib import Path
from statistics import mean

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithoutReference,
                           LLMContextPrecisionWithReference, LLMContextRecall)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
load_dotenv(ROOT / ".env")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3:8b")

# Answer grader prompt, unchanged from main.ipynb
GRADER_INSTRUCTIONS = """You are a grader assessing the accuracy of a generated answer to a user question.

    **Grading Criteria (Score from 1 to 5):**
    - `5`: Fully correct, no hallucinations, and complete.
    - `4`: Mostly correct, minor inaccuracies but still relevant.
    - `3`: Partially correct, some key facts missing or slightly incorrect.
    - `2`: Incorrect or contains significant hallucinations.
    - `1`: Completely wrong or entirely made up.

    **Considerations:**
    - Compare the generated answer to the expected answer.
    - If retrieved documents are provided, ensure the generated answer is supported by them.
    - Flag hallucinations if the answer includes details **not present in the retrieved documents**.
    - If the generated answer does not answer the question or say I dont know you should give it a low score.

    **Return a JSON object with:**
    - `answer_score`: An integer from 1 to 5.
    - `explanation`: A brief justification for the score.
    """

METRICS = {
    "context_recall": "Context recall",
    "llm_context_precision_with_reference": "Context precision (w/ ref)",
    "llm_context_precision_without_reference": "Context precision (w/o ref)",
    "faithfulness": "Faithfulness",
    "factual_correctness(mode=f1)": "Factual correctness (F1)",
}


def norm(q):
    return re.sub(r"\W+", " ", q.lower()).strip()


def load_systems():
    baseline = json.loads(Path(os.environ["BASELINE_JSON"]).read_text(encoding="utf-8"))
    cohere = json.loads((RESULTS / "private" / "cohere_outputs.json").read_text(encoding="utf-8"))
    by_q = {"original": {norm(d["user_input"]): d for d in baseline},
            "cohere": {norm(d["user_input"]): d for d in cohere}}
    common = [k for k in by_q["original"] if k in by_q["cohere"]]  # same questions for both systems
    systems = {}
    for name, rows in by_q.items():
        systems[name] = [{
            "user_input": rows[k]["user_input"],
            "response": rows[k]["response"],
            # one joined context string per question, as in the original RAGAS run
            "retrieved_contexts": ["\n\n".join(rows[k]["retrieved_contexts"])],
            "reference": rows[k]["reference"],
        } for k in common]
    return systems


def grade_answers(name, rows, judge):
    path = RESULTS / f"grades_{name}.jsonl"
    done = {}
    if path.exists():
        done = {g["user_input"]: g for g in map(json.loads, path.read_text().splitlines())}
    with path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            if r["user_input"] in done:
                continue
            prompt = (f"QUESTION: \n\n{r['user_input']}\n\n"
                      f"    EXPECTED ANSWER: \n\n{r['reference']}\n\n"
                      f"    GENERATED ANSWER: \n\n{r['response']}\n\n"
                      f"    RETRIEVED DOCUMENTS:\n\n{r['retrieved_contexts'][0]}\n\n\n"
                      "    **Return a JSON object with `answer_score` (1-5) and `explanation` fields.**")
            try:
                out = json.loads(judge.invoke([SystemMessage(content=GRADER_INSTRUCTIONS),
                                               HumanMessage(content=prompt)]).content)
                score = int(out["answer_score"])
            except (ValueError, KeyError, TypeError):
                out, score = {}, None
            g = {"user_input": r["user_input"], "answer_score": score, "explanation": out.get("explanation", "")}
            done[r["user_input"]] = g
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{name} {i}/{len(rows)}] answer score {score}")
    return [done[r["user_input"]]["answer_score"] for r in rows]


def run_ragas(name, rows, judge):
    path = RESULTS / f"ragas_{name}_{len(rows)}.csv"
    if path.exists():
        return pd.read_csv(path)
    print(f"  RAGAS on {name} ({len(rows)} questions)…")
    result = evaluate(dataset=EvaluationDataset.from_list(rows),
                      metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(),
                               LLMContextPrecisionWithReference(), LLMContextPrecisionWithoutReference()],
                      llm=LangchainLLMWrapper(judge), run_config=RunConfig(timeout=900, max_workers=2))
    df = result.to_pandas()
    df.to_csv(path, index=False)
    return df


def summary(scores, ragas, latency, citations):
    names = list(scores)
    lines = [f"| Metric | {' | '.join(n.capitalize() for n in names)} |", "|---|" + "---|" * len(names)]

    def row(label, values):
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    def avg(xs):
        xs = [x for x in xs if x is not None and x == x]  # drop None/NaN (judge parse failures)
        return mean(xs) if xs else float("nan"), len(xs)

    row("Answer score (1-5)", [f"{avg(scores[n])[0]:.2f}" for n in names])
    row("Answer hit rate (score ≥ 4)",
        [f"{100 * sum(1 for s in scores[n] if s and s >= 4) / len(scores[n]):.1f}%" for n in names])
    for col, label in (METRICS.items() if ragas else []):
        cells = []
        for n in names:
            m, k = avg(ragas[n][col].tolist()) if col in ragas[n] else (float("nan"), 0)
            cells.append(f"{m:.3f}" + (f" (n={k})" if k < len(ragas[n]) else ""))
        row(label, cells)
    n_q = len(next(iter(scores.values())))
    n_r = len(next(iter(ragas.values()))) if ragas else 0
    ragas_note = (f"RAGAS rows use a fixed random sample of {n_r} of the questions.\n" if ragas and n_r < n_q
                  else "" if ragas else "RAGAS metrics were not run (see `--ragas`).\n")
    text = (f"# Original vs. Cohere RAG pipeline\n\n{n_q} test questions, both systems judged by `{JUDGE_MODEL}` "
            "(local, via Ollama). `n=` shows how many questions got a valid score when the judge failed to parse "
            "some outputs; those are left out of the average.\n\n" + "\n".join(lines) + "\n\n" + ragas_note +
            f"Cohere answers with citations: {citations[0]}/{citations[1]}.\n"
            f"Cohere pipeline latency: {latency['total']:.2f}s per answer on average "
            f"(retrieval {latency['retrieval']:.2f}s, generation {latency['generation']:.2f}s), via the API.\n"
            "The original pipeline's 2.16s (dissertation) was measured on a Colab GPU, so latencies are not "
            "directly comparable.\n")
    (RESULTS / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--ragas", type=int, default=0, metavar="N",
                      help="also run RAGAS on a random sample of N questions (0 = skip, the default)")
    args = args.parse_args()
    systems = load_systems()
    print(f"Judging {len(systems['original'])} common questions with {JUDGE_MODEL}")
    grader = ChatOllama(model=JUDGE_MODEL, temperature=0.2, format="json")
    ragas_judge = ChatOllama(model=JUDGE_MODEL, temperature=0.2)
    scores = {n: grade_answers(n, rows, grader) for n, rows in systems.items()}
    ragas = {}
    if args.ragas:
        sample = sorted(random.Random(42).sample(range(len(systems["original"])), min(args.ragas, len(systems["original"]))))
        ragas = {n: run_ragas(n, [rows[i] for i in sample], ragas_judge) for n, rows in systems.items()}
    cohere = json.loads((RESULTS / "private" / "cohere_outputs.json").read_text())
    latency = {k: mean(d["latency"][k] for d in cohere) for k in ("total", "retrieval", "generation")}
    summary(scores, ragas, latency, (sum(1 for d in cohere if d["citations"]), len(cohere)))


if __name__ == "__main__":
    main()
