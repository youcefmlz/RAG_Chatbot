"""Answer the 105 test questions (../questions.txt) with the Cohere pipeline.

Writes results/private/cohere_outputs.json in the same format as the original evaluation data
(user_input, response, retrieved_contexts, reference) plus citations and latency. That file contains text from the
corpus, which is private, so it is git-ignored; results/cohere_answers.json is the shareable copy without it.
Safe to stop and re-run: questions already answered are skipped, so no API calls are wasted.
"""
import json
import time
from pathlib import Path

from rag import CohereRAG

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "private" / "cohere_outputs.json"
PUBLIC = ROOT / "results" / "cohere_answers.json"
MIN_SECONDS_PER_QUESTION = 6.5  # trial keys allow 10 rerank calls per minute


def load_questions():
    questions = [l.strip() for l in (ROOT.parent / "questions.txt").open(encoding="utf-8") if l.strip()]
    answers = [l.strip() for l in (ROOT.parent / "answers.txt").open(encoding="utf-8") if l.strip()]
    assert len(questions) == len(answers), (len(questions), len(answers))
    return list(zip(questions, answers))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(OUT.read_text()) if OUT.exists() else []
    seen = {d["user_input"] for d in done}
    todo = [(q, a) for q, a in load_questions() if q not in seen]
    print(f"{len(seen)} already answered, {len(todo)} to go")
    rag = CohereRAG()
    for n, (question, reference) in enumerate(todo, 1):
        started = time.monotonic()
        ans = rag.answer(question)
        done.append({
            "user_input": question,
            "response": ans.text,
            "retrieved_contexts": [c["text"] for c in ans.chunks],
            "reference": reference,
            "sources": [c["source"] for c in ans.chunks],
            "citations": ans.citations,
            "latency": ans.latency,
        })
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=2))  # save after every question
        print(f"[{n}/{len(todo)}] {ans.latency['total']:.1f}s  {len(ans.citations)} citations  {question[:60]}")
        time.sleep(max(0.0, MIN_SECONDS_PER_QUESTION - (time.monotonic() - started)))
    write_public(done)
    print(f"saved {len(done)} answers to {OUT} (shareable copy: {PUBLIC})")


def write_public(rows):
    """Everything except the retrieved corpus text, which can't be published."""
    PUBLIC.write_text(json.dumps([{k: v for k, v in r.items() if k != "retrieved_contexts"} for r in rows],
                                 ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
