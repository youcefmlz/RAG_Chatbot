"""Ask the Cohere RAG chatbot a question and print a cited answer.

    python ask.py "What are Dr. Yoshi Gotoh's research interests?"
    python ask.py            # interactive
"""
import sys

from rag import CohereRAG


def show(rag, question):
    ans = rag.answer(question)
    print("\n" + ans.with_markers() + "\n")
    for i, c in enumerate(ans.chunks, 1):
        print(f"  [{i}] {c['source'] or 'unknown source'}  (relevance {c['relevance']:.2f})")
    print(f"\n  {ans.latency['total']:.2f}s (retrieval {ans.latency['retrieval']:.2f}s, "
          f"generation {ans.latency['generation']:.2f}s)\n")


if __name__ == "__main__":
    rag = CohereRAG()
    if len(sys.argv) > 1:
        show(rag, " ".join(sys.argv[1:]))
    else:
        while (q := input("Question (empty to quit): ").strip()):
            show(rag, q)
