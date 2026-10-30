"""Build training/datasets/summarization.json  ->  [{"text", "summary", "source"}]

Weak labels: a strong zero-shot model summarizes every chunk of documents/.
Rows with "source": "human" already present in the output file are PRESERVED - add your own
engineer-written / corrected summaries there. Aim to review a few hundred weak rows before training.
Usage: python training/prepare_summarization_data.py [--bs 8] [--min_words 40]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from nlp_pipeline import load_and_chunk_documents  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--min_words", type=int, default=40)
    ap.add_argument("--model", default=C.WEAK_LABEL_MODEL)
    a = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = C.get_device()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(device).eval()

    keep = []
    if C.SUMMARIZATION_DATA.exists():
        keep = [r for r in json.loads(C.SUMMARIZATION_DATA.read_text()) if r.get("source") == "human"]
    chunks = [c for c in load_and_chunk_documents() if len(c["text"].split()) >= a.min_words]
    rows = list(keep)
    for i in tqdm(range(0, len(chunks), a.bs)):
        texts = [c["text"] for c in chunks[i:i + a.bs]]
        prompts = [f"Summarize this technical manual passage in 2-3 sentences:\n{t}" for t in texts]
        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=96, num_beams=4, no_repeat_ngram_size=3)
        rows += [{"text": t, "summary": s, "source": "weak"}
                 for t, s in zip(texts, tok.batch_decode(out, skip_special_tokens=True))]
    C.SUMMARIZATION_DATA.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows ({len(keep)} human) -> {C.SUMMARIZATION_DATA}")


if __name__ == "__main__":
    main()
