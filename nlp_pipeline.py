"""
Merekos NLP system - the whole pipeline in one module.

  documents/ -> clean -> chunk -> embed -> FAISS (vector_store/)
  question   -> hybrid retrieval (FAISS + BM25 -> RRF -> cross-encoder rerank)
             -> router: factoid  -> fine-tuned BERT extractive reader
                        procedural -> fine-tuned T5 generator
             -> citation attribution + grounding guard -> answer with sources
  document   -> hierarchical T5 summarization (per-section + overall)

CLI:  python nlp_pipeline.py build
      python nlp_pipeline.py ask "How do I replace the hydraulic filter?"
      python nlp_pipeline.py summarize hp200_service_manual
Heavy libraries are imported lazily so the preprocessing functions work without torch/faiss.
"""
import json
import re
import sys
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np

import config as C

# =========================================================================================
# 1. PREPROCESSING: load -> clean -> structure -> chunk
# =========================================================================================
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+[A-Z][^\n.]{2,80}|[A-Z][A-Z0-9 \-/&]{4,80})$")
LIST_RE = re.compile(r"^\s*(?:step\s*\d+|\d+[.)]|[-•*])\s*", re.I)
PAGE_NUM_RE = re.compile(r"\s*(page\s*)?\d+(\s*of\s*\d+)?\s*", re.I)


def split_sentences(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out.extend(s.strip() for s in SENT_RE.split(line) if s.strip())
    return out


def load_pages(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # scanned PDFs need OCR first (e.g. ocrmypdf)
        return [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.split("\f") if "\f" in text else [text]


def _norm(line):
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages):
    boiler = set()
    if len(pages) >= 3:  # repeated headers / footers
        counts = Counter()
        for p in pages:
            counts.update({_norm(x) for x in p.splitlines() if x.strip()})
        boiler = {l for l, c in counts.items() if c >= 0.4 * len(pages) and len(l) < 80}
    out = []
    for p in pages:
        lines = [l for l in p.splitlines() if _norm(l) not in boiler and not PAGE_NUM_RE.fullmatch(l)]
        t = "\n".join(lines)
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)       # de-hyphenate line breaks
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        out.append(t.strip())
    return out


def to_units(pages):
    """Paragraph units tagged with page + current section heading (numbered steps kept intact)."""
    units, section = [], "Introduction"
    for pno, text in enumerate(pages, 1):
        for para in re.split(r"\n\s*\n", text):
            lines = [l for l in para.strip().splitlines() if l.strip()]
            if not lines:
                continue
            if len(lines) == 1 and HEADING_RE.match(lines[0].strip()):
                section = lines[0].strip()
                continue
            is_list = sum(bool(LIST_RE.match(l)) for l in lines) >= 2
            body = "\n".join(l.strip() for l in lines) if is_list else " ".join(l.strip() for l in lines)
            units.append({"page": pno, "section": section, "text": body})
    return units


def _split_long(unit, max_words):
    if len(unit["text"].split()) <= max_words:
        return [unit]
    res, cur, n = [], [], 0
    for s in SENT_RE.split(unit["text"]):
        w = len(s.split())
        if cur and n + w > max_words:
            res.append({**unit, "text": " ".join(cur)})
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        res.append({**unit, "text": " ".join(cur)})
    return res


def chunk_units(units, max_words=C.CHUNK_MAX_WORDS, overlap_words=C.CHUNK_OVERLAP_WORDS):
    """Pack units into chunks; never cross section boundaries; keep overlap inside a section."""
    flat = [u2 for u in units for u2 in _split_long(u, max_words)]
    groups, cur, n = [], [], 0
    for u in flat:
        w = len(u["text"].split())
        if cur and (n + w > max_words or u["section"] != cur[0]["section"]):
            groups.append(cur)
            keep, k = [], 0
            if u["section"] == cur[0]["section"]:
                for prev in reversed(cur):
                    pw = len(prev["text"].split())
                    if k + pw > overlap_words:
                        break
                    keep.insert(0, prev)
                    k += pw
            cur, n = keep, k
        cur.append(u)
        n += w
    if cur:
        groups.append(cur)
    return [{"section": g[0]["section"], "page": g[0]["page"],
             "text": "\n".join(u["text"] for u in g)} for g in groups]


def load_and_chunk_documents(doc_dir=C.DOCUMENTS_DIR):
    rows = []
    for f in sorted(Path(doc_dir).rglob("*")):
        if f.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        chunks = chunk_units(to_units(clean_pages(load_pages(f))))
        for i, c in enumerate(chunks):
            rows.append({"chunk_id": f"{f.stem}::{i}", "doc_id": f.stem, **c})
        print(f"[ingest] {f.name}: {len(chunks)} chunks")
    return rows


def embed_text(chunk):
    """Section title gives each chunk context for both dense and BM25 retrieval."""
    return f"{chunk['doc_id']} | {chunk['section']}\n{chunk['text']}"


# =========================================================================================
# 2. QUERY ROUTING
# =========================================================================================
PROC = re.compile(r"\b(how (do|to|can|should)|steps?|procedure|replace|install|remove|calibrate|"
                  r"troubleshoot|reset|fix|diagnose|set up|perform)\b", re.I)
FACT = re.compile(r"^(what|which|who|when|where|how (much|many|long|often|high|low))\b", re.I)
ABSTAIN = "I couldn't find reliable information about that in the indexed manuals."


def route(q):
    """factoid -> BERT extractive lookup; procedural -> T5 generation over several chunks."""
    if PROC.search(q):
        return "procedural"
    return "factoid" if FACT.match(q.strip()) else "procedural"


def decompose(q):
    parts = [p.strip() for p in re.split(r"\?\s+|\s+and then\s+|;\s+", q) if p.strip()]
    return parts if 1 < len(parts) <= 3 else [q]


def _tok(s):
    return re.findall(r"\w+", s.lower())


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores, key=scores.get, reverse=True)


# =========================================================================================
# 3. THE SYSTEM
# =========================================================================================
class NLPSystem:
    def __init__(self):
        self.device = C.get_device()
        self.index = self.chunks = self.bm25 = None
        self._embedder = self._reranker = self._t5 = self._reader = None

    # ---- lazy model loaders -------------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(C.EMBEDDING_MODEL, device=self.device)
        return self._embedder

    def encode_docs(self, texts, progress=False):
        return self.embedder.encode(texts, batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=progress).astype("float32")

    def encode_query(self, q):
        return self.encode_docs([C.EMBEDDING_QUERY_PREFIX + q])

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(C.RERANK_MODEL, device=self.device)
        return self._reranker

    @property
    def t5(self):
        if self._t5 is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            path = C.resolve_model(C.T5_PATH, C.T5_FALLBACK)
            self._t5 = (AutoTokenizer.from_pretrained(path),
                        AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device).eval())
        return self._t5

    @property
    def reader(self):
        if self._reader is None:
            from transformers import pipeline
            path = C.resolve_model(C.BERT_PATH, C.BERT_FALLBACK)
            self._reader = pipeline("question-answering", model=path,
                                    device=0 if self.device == "cuda" else -1)
        return self._reader

    # ---- indexing -----------------------------------------------------------------------
    def build_index(self):
        import faiss
        chunks = load_and_chunk_documents()
        if not chunks:
            raise SystemExit(f"No documents found in {C.DOCUMENTS_DIR}")
        emb = self.encode_docs([embed_text(c) for c in chunks], progress=True)
        d = emb.shape[1]
        if len(chunks) >= C.HNSW_THRESHOLD:
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction, index.hnsw.efSearch = 200, 64
        else:
            index = faiss.IndexFlatIP(d)  # cosine similarity (vectors are normalized)
        index.add(emb)
        C.VECTOR_STORE_DIR.mkdir(exist_ok=True)
        faiss.write_index(index, str(C.VECTOR_STORE_DIR / "faiss.index"))
        (C.VECTOR_STORE_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        print(f"[index] {index.ntotal} vectors (dim {d}) saved to vector_store/")
        self.index, self.chunks, self.bm25 = None, None, None  # force reload

    def load_index(self):
        if self.index is not None:
            return
        import faiss
        from rank_bm25 import BM25Okapi
        fp, cp = C.VECTOR_STORE_DIR / "faiss.index", C.VECTOR_STORE_DIR / "chunks.json"
        if not (fp.exists() and cp.exists()):
            raise FileNotFoundError("Vector store is empty. Run: python nlp_pipeline.py build")
        self.index = faiss.read_index(str(fp))
        self.chunks = json.loads(cp.read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([_tok(embed_text(c)) for c in self.chunks])

    def list_documents(self):
        self.load_index()
        return sorted({c["doc_id"] for c in self.chunks})

    # ---- retrieval ----------------------------------------------------------------------
    def retrieve(self, query, k=C.FINAL_K, doc_id=None):
        self.load_index()
        _, ids = self.index.search(self.encode_query(query), C.TOP_K_DENSE)
        dense = [int(i) for i in ids[0] if i >= 0]
        bm = self.bm25.get_scores(_tok(query))
        sparse = [int(i) for i in np.argsort(-bm)[:C.TOP_K_BM25] if bm[i] > 0]
        cand = rrf([dense, sparse])[:C.TOP_K_DENSE]
        if doc_id:
            cand = [i for i in cand if self.chunks[i]["doc_id"] == doc_id]
        if not cand:
            return []
        scores = self.reranker.predict([(query, embed_text(self.chunks[i])) for i in cand])
        order = np.argsort(-scores)[:k]
        return [{**self.chunks[cand[j]], "score": float(scores[j])} for j in order]

    # ---- readers ------------------------------------------------------------------------
    def _extractive(self, question, hits):
        best = None
        for i, h in enumerate(hits[:3]):
            r = self.reader(question=question, context=h["text"], handle_impossible_answer=True)
            if r["answer"].strip() and (best is None or r["score"] > best[0]["score"]):
                best = (r, i)
        if not best or best[0]["score"] < C.READER_MIN_SCORE:
            return None
        r, i = best
        return next((s for s in split_sentences(hits[i]["text"]) if r["answer"] in s), r["answer"])

    def _generate(self, question, hits):
        import torch
        tok, model = self.t5
        ctxs = [f"({h['section']}) {h['text']}" for h in hits]
        inp = tok(C.qa_prompt(question, ctxs), return_tensors="pt", truncation=True,
                  max_length=1024).to(self.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=C.MAX_NEW_TOKENS, num_beams=4,
                                 no_repeat_ngram_size=3, early_stopping=True)
        return tok.decode(out[0], skip_special_tokens=True)

    # ---- grounding + citations ----------------------------------------------------------
    def _attribute(self, answer, hits):
        sents = split_sentences(answer)
        if not sents:
            return answer, set(), 0.0
        sims = self.encode_docs(sents) @ self.encode_docs([h["text"] for h in hits]).T
        kept, cited, supported = [], set(), 0
        for s, row in zip(sents, sims):
            j = int(np.argmax(row))
            if row[j] >= C.ATTRIBUTION_THRESHOLD:
                kept.append(f"{s} [{j + 1}]")
                cited.add(j)
                supported += 1
            elif not C.STRICT_GROUNDING:
                kept.append(s)
        if not kept:
            return ABSTAIN, set(), 0.0
        return ("\n".join(kept) if "\n" in answer else " ".join(kept)), cited, supported / len(sents)

    # ---- public: QA ---------------------------------------------------------------------
    def ask(self, question, doc_id=None):
        t0 = time.time()
        hits, seen = [], set()
        subqs = decompose(question)
        for q in subqs:
            for h in self.retrieve(q, doc_id=doc_id):
                if h["chunk_id"] not in seen:
                    seen.add(h["chunk_id"])
                    hits.append(h)
        hits = hits[:8]
        if not hits or max(h["score"] for h in hits) < C.MIN_RERANK_SCORE:
            return {"answer": ABSTAIN, "route": "abstain", "sources": [], "grounded_ratio": 0.0,
                    "latency_s": round(time.time() - t0, 2)}
        mode = route(question) if len(subqs) == 1 else "procedural"
        answer = self._extractive(question, hits) if mode == "factoid" else None
        if answer is None:
            mode, answer = "procedural", self._generate(question, hits)
        answer, cited, ratio = self._attribute(answer, hits)
        sources = [{"n": i + 1, "chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "page": h["page"],
                    "section": h["section"], "score": round(h["score"], 3), "cited": i in cited,
                    "text": h["text"]} for i, h in enumerate(hits)]
        return {"answer": answer, "route": mode, "sources": sources,
                "grounded_ratio": round(ratio, 3), "latency_s": round(time.time() - t0, 2)}

    # ---- public: summarization ----------------------------------------------------------
    def _summ_batch(self, texts, max_new_tokens):
        import torch
        tok, model = self.t5
        inp = tok([C.SUMMARY_PREFIX + t for t in texts], return_tensors="pt", padding=True,
                  truncation=True, max_length=768).to(self.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=4,
                                 no_repeat_ngram_size=3, early_stopping=True)
        return tok.batch_decode(out, skip_special_tokens=True)

    @staticmethod
    def _windows(text, words):
        w = text.split()
        return [" ".join(w[i:i + words]) for i in range(0, len(w), words)] or [""]

    def summarize_text(self, text, max_new_tokens=C.SUMMARY_MAX_NEW_TOKENS):
        """Hierarchical map-reduce: summarize windows, then summarize the summaries."""
        pieces = self._windows(text, C.SUMMARY_WINDOW_WORDS)
        summaries = pieces
        for _ in range(4):
            summaries = [s for i in range(0, len(pieces), 4)
                         for s in self._summ_batch(pieces[i:i + 4], max_new_tokens)]
            if len(summaries) == 1:
                return summaries[0]
            pieces = self._windows(" ".join(summaries), C.SUMMARY_WINDOW_WORDS)
        return " ".join(summaries)

    def summarize_document(self, doc_id):
        self.load_index()
        chunks = [c for c in self.chunks if c["doc_id"] == doc_id]
        if not chunks:
            raise ValueError(f"Unknown document: {doc_id}")
        sections = []
        for sec, grp in groupby(chunks, key=lambda c: c["section"]):
            grp = list(grp)
            sections.append({"section": sec, "page": grp[0]["page"],
                             "summary": self.summarize_text(" ".join(c["text"] for c in grp), 80)})
        return {"doc_id": doc_id, "overall": self.summarize_text(" ".join(s["summary"] for s in sections)),
                "sections": sections}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"build", "ask", "summarize"}:
        sys.exit(__doc__)
    system = NLPSystem()
    cmd = sys.argv[1]
    if cmd == "build":
        system.build_index()
    elif cmd == "ask":
        print(json.dumps(system.ask(" ".join(sys.argv[2:])), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(system.summarize_document(sys.argv[2]), indent=2, ensure_ascii=False))
