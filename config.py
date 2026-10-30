"""Central configuration + shared helpers for the Merekos NLP system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- paths ----------------------------------------------------------------------------
DOCUMENTS_DIR = ROOT / "documents"
VECTOR_STORE_DIR = ROOT / "vector_store"
T5_PATH = ROOT / "trained_models" / "t5_manufacturing"
BERT_PATH = ROOT / "trained_models" / "bert_manufacturing_qa"
SUMMARIZATION_DATA = ROOT / "training" / "datasets" / "summarization.json"
QA_DATA = ROOT / "training" / "datasets" / "qa.json"
TEST_QUESTIONS = ROOT / "evaluation" / "test_questions.json"

# ---- models (public fallbacks are used when fine-tuned models are not trained yet) ----
T5_FALLBACK = "google/flan-t5-base"
BERT_FALLBACK = "distilbert-base-cased-distilled-squad"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
WEAK_LABEL_MODEL = "google/flan-t5-large"      # used only to bootstrap training data

# ---- chunking -------------------------------------------------------------------------
CHUNK_MAX_WORDS = 160
CHUNK_OVERLAP_WORDS = 30

# ---- retrieval ------------------------------------------------------------------------
TOP_K_DENSE = 20
TOP_K_BM25 = 20
FINAL_K = 4
MIN_RERANK_SCORE = -4.0        # abstain below this cross-encoder logit
HNSW_THRESHOLD = 100_000       # chunks; above this use an HNSW index instead of exact search

# ---- QA / generation ------------------------------------------------------------------
READER_MIN_SCORE = 0.35
MAX_NEW_TOKENS = 256
ATTRIBUTION_THRESHOLD = 0.45   # min cosine sim for an answer sentence to count as grounded
STRICT_GROUNDING = True        # drop unsupported sentences from answers

# ---- summarization --------------------------------------------------------------------
SUMMARY_PREFIX = "summarize: "
SUMMARY_WINDOW_WORDS = 350
SUMMARY_MAX_NEW_TOKENS = 128


def qa_prompt(question, contexts):
    """Single source of truth for the QA prompt (used at training AND inference time)."""
    ctx = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return ("Answer the technical question using only the numbered context. "
            "If it is a procedure, give the steps in order.\n"
            f"Question: {question}\nContext:\n{ctx}\nAnswer:")


def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_model(local_path, fallback):
    p = Path(local_path)
    if p.exists() and any(f for f in p.iterdir() if f.name != ".gitkeep"):
        return str(p)
    print(f"[info] fine-tuned model not found at {p.relative_to(ROOT)} -> using '{fallback}'")
    return fallback
