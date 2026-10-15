"""Cohere version of the Sheffield research RAG chatbot.

Same corpus and chunking as ../main.ipynb, so the two systems answer from identical knowledge:
  retrieval  : BM25 (keywords) + Cohere Embed (semantic), top 10 each, merged
  reranking  : Cohere Rerank keeps the most relevant chunks
  generation : Command A+ grounded generation, which returns citations into the chunks it used
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cohere
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

load_dotenv(Path(__file__).with_name(".env"))

EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v5.0-pro")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-pro")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-a-plus-05-2026")
CACHE_DIR = Path(__file__).with_name(".cache")

K_EACH = 10          # chunks taken from BM25 and from the embedding search (as in the original)
TOP_N = 5            # chunks kept after reranking
MIN_RELEVANCE = 0.1  # reranked chunks below this score are dropped

# Same splitter settings as main.ipynb
SEPARATORS = ["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""]
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, add_start_index=True,
                                          strip_whitespace=True, separators=SEPARATORS)

# Same instructions as the original prompt in main.ipynb, so differences come from the models, not the prompt
SYSTEM_PROMPT = """You are an assistant for answering questions about research and computer science at the \
University of Sheffield. Answer the user's question based **only** on the provided documents.
- Answer the question in a natural, coherent way.
- Provide a **clear, precise, and complete** answer to the question without additional explanations and information.
- Ensure your response is **well-structured and informative**.
- Make sure to use all the information in the documents that is relevant to the question.
- If the documents do not contain the answer, say that you don't know instead of guessing."""

FALLBACK = ("I'm a chatbot designed to answer questions about research in the Department of Computer Science "
            "at the University of Sheffield. Please ask a relevant question related to this topic.")


def load_chunks(data_dir):
    """Parse the LLM-cleaned pages ([SOURCE] ... [LAST UPDATED] ... [CONTENT] ...) and split them into chunks."""
    docs = []
    for name in sorted(os.listdir(data_dir)):
        raw = Path(data_dir, name).read_text(encoding="utf-8", errors="replace")
        # 49 personal-site pages have no [LAST UPDATED] line, so the source also ends at [CONTENT]
        source = re.search(r"\[SOURCE\]\s*(.*?)\s*\[(?:LAST UPDATED|CONTENT)\]", raw, re.S)
        content = re.search(r"\[CONTENT\]\s*(.*)", raw, re.S)
        docs.append(Document(page_content=content.group(1) if content else "",
                             metadata={"source": source.group(1) if source else ""}))
    return [c for c in SPLITTER.split_documents(docs) if c.page_content.strip()]


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def _retry(fn, *args, **kwargs):
    """Call the API, waiting and retrying when a trial-key rate limit is hit."""
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except cohere.TooManyRequestsError:
            wait = 15 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s…")
            time.sleep(wait)
    return fn(*args, **kwargs)


@dataclass
class Answer:
    text: str
    citations: list = field(default_factory=list)   # [{start, end, text, doc_ids}]
    chunks: list = field(default_factory=list)      # [{id, source, text, relevance}]
    latency: dict = field(default_factory=dict)     # seconds per stage

    def with_markers(self):
        """The answer with [n] markers after each cited span, n = position of the chunk in `chunks`."""
        number = {c["id"]: i + 1 for i, c in enumerate(self.chunks)}
        out, last = [], 0
        for c in sorted(self.citations, key=lambda c: c["end"]):
            refs = "".join(f"[{number[d]}]" for d in dict.fromkeys(c["doc_ids"]) if d in number)
            out.append(self.text[last:c["end"]] + refs)
            last = c["end"]
        return "".join(out) + self.text[last:]


class CohereRAG:
    def __init__(self, data_dir=None, api_key=None):
        data_dir = data_dir or os.environ["DATA_DIR"]
        self.co = cohere.ClientV2(api_key or os.environ["COHERE_API_KEY"])
        self.chunks = load_chunks(data_dir)
        self.texts = [c.page_content for c in self.chunks]
        self.bm25 = BM25Okapi([_tokens(t) for t in self.texts])
        self.vectors = self._embed_corpus()

    def _embed_corpus(self):
        """Embed every chunk once; the vectors are cached on disk so later runs cost no API calls."""
        key = hashlib.sha256((EMBED_MODEL + "\0" + "\0".join(self.texts)).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"embeddings-{key}.npy"
        if path.exists():
            return np.load(path)
        print(f"Embedding {len(self.texts)} chunks with {EMBED_MODEL} (one-time, cached)…")
        vectors = []
        for i in range(0, len(self.texts), 96):  # API limit: 96 texts per call
            res = _retry(self.co.embed, model=EMBED_MODEL, input_type="search_document",
                         texts=self.texts[i:i + 96], embedding_types=["float"])
            vectors.extend(res.embeddings.float_)
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(path, matrix)
        return matrix

    def retrieve(self, question):
        """Hybrid candidates (BM25 + embeddings), reranked by Cohere Rerank."""
        bm25_top = np.argsort(self.bm25.get_scores(_tokens(question)))[::-1][:K_EACH]
        q = _retry(self.co.embed, model=EMBED_MODEL, input_type="search_query", texts=[question],
                   embedding_types=["float"]).embeddings.float_[0]
        q = np.asarray(q, dtype=np.float32)
        dense_top = np.argsort(self.vectors @ (q / np.linalg.norm(q)))[::-1][:K_EACH]
        candidates = list(dict.fromkeys([*bm25_top.tolist(), *dense_top.tolist()]))
        ranked = _retry(self.co.rerank, model=RERANK_MODEL, query=question,
                        documents=[self.texts[i] for i in candidates], top_n=TOP_N)
        return [{"id": f"doc{candidates[r.index]}", "source": self.chunks[candidates[r.index]].metadata["source"],
                 "text": self.texts[candidates[r.index]], "relevance": round(r.relevance_score, 4)}
                for r in ranked.results if r.relevance_score >= MIN_RELEVANCE]

    def answer(self, question):
        t0 = time.perf_counter()
        chunks = self.retrieve(question)
        t1 = time.perf_counter()
        if not chunks:
            return Answer(FALLBACK, latency={"retrieval": t1 - t0, "generation": 0.0, "total": t1 - t0})
        res = _retry(self.co.chat, model=CHAT_MODEL, temperature=0.3, seed=42,
                     messages=[{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": question}],
                     documents=[{"id": c["id"], "data": {"source": c["source"], "text": c["text"]}} for c in chunks])
        t2 = time.perf_counter()
        citations = [{"start": c.start, "end": c.end, "text": c.text,
                      "doc_ids": [s.id for s in (c.sources or []) if getattr(s, "id", None)]}
                     for c in (res.message.citations or [])]
        # Command A+ may return a "thinking" item before the answer; citations point into the text item
        text = next(c.text for c in res.message.content if getattr(c, "type", "") == "text")
        return Answer(text, citations, chunks,
                      {"retrieval": t1 - t0, "generation": t2 - t1, "total": t2 - t0})


if __name__ == "__main__":
    import sys
    print(json.dumps(CohereRAG().answer(" ".join(sys.argv[1:]) or "Who is Dr. Yoshi Gotoh?").__dict__, indent=2)[:3000])
