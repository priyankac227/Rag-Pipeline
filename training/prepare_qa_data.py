"""Build training/datasets/qa.json -> [{"question","context","answer","answer_start","answer_sentence","source"}]

For each chunk: a zero-shot model writes a question, a SQuAD-trained reader extracts the answer span
(kept only if confidence > 0.5). The same file trains BERT (span) and T5 (answer_sentence).
Rows with "source": "human" in the existing file are preserved - add real engineer questions there
(answer_start = context.find(answer)).
Usage: python training/prepare_qa_data.py [--bs 8] [--min_words 40]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from nlp_pipeline import load_and_chunk_documents, split_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--min_words", type=int, default=40)
    ap.add_argument("--model", default=C.WEAK_LABEL_MODEL)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    device = C.get_device()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(device).eval()
    reader = pipeline("question-answering", model=C.BERT_FALLBACK, device=0 if device == "cuda" else -1)

    keep = []
    if C.QA_DATA.exists():
        keep = [r for r in json.loads(C.QA_DATA.read_text()) if r.get("source") == "human"]
    chunks = [c for c in load_and_chunk_documents() if len(c["text"].split()) >= a.min_words]
    rows = list(keep)
    for i in tqdm(range(0, len(chunks), a.bs)):
        texts = [c["text"] for c in chunks[i:i + a.bs]]
        prompts = [f"Write one specific question that this passage answers:\n{t}" for t in texts]
        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=40, num_beams=4, no_repeat_ngram_size=3)
        for t, q in zip(texts, tok.batch_decode(out, skip_special_tokens=True)):
            r = reader(question=q, context=t)
            if r["score"] > 0.5 and r["answer"].strip():
                sent = next((s for s in split_sentences(t) if r["answer"] in s), r["answer"])
                rows.append({"question": q, "context": t, "answer": r["answer"],
                             "answer_start": r["start"], "answer_sentence": sent, "source": "weak"})
    C.QA_DATA.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows ({len(keep)} human) -> {C.QA_DATA}")


if __name__ == "__main__":
    main()
