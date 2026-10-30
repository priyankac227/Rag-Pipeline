"""Evaluate the full RAG system on evaluation/test_questions.json.

Metrics
  - RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
           (needs a judge LLM: export OPENAI_API_KEY=...)
  - NLI faithfulness (offline fallback): share of answer sentences entailed by retrieved context
  - retrieval hit@k and MRR (via "gold_substring"), answer token-F1, latency p50/p95
Usage: python evaluation/evaluate_ragas.py [--questions path] [--out evaluation/results.json]"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import config as C  # noqa: E402
from nlp_pipeline import NLPSystem, split_sentences  # noqa: E402


class NLIFaithfulness:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, C.get_device()
        self.tok = AutoTokenizer.from_pretrained(C.NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(C.NLI_MODEL).to(self.device).eval()
        self.ent = [i for i, l in self.model.config.id2label.items() if l.lower() == "entailment"][0]

    def score(self, answer, contexts):
        sents = [re.sub(r"\s*\[\d+\]", "", s) for s in split_sentences(answer)]
        if not sents or not contexts:
            return 0.0
        ok = 0
        with self.torch.inference_mode():
            for s in sents:
                enc = self.tok(contexts, [s] * len(contexts), return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(self.device)
                p = self.model(**enc).logits.softmax(-1)[:, self.ent]
                ok += int(p.max().item() > 0.5)
        return ok / len(sents)


def token_f1(pred, gold):
    p, g = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", gold.lower())
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def ragas_scores(rows):
    from datasets import Dataset
    from ragas import evaluate as ragas_eval
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    ds = Dataset.from_list([{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"],
                             "ground_truth": r["ground_truth"]} for r in rows])
    res = ragas_eval(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    return res.to_pandas().mean(numeric_only=True).round(4).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(C.TEST_QUESTIONS))
    ap.add_argument("--out", default=str(C.ROOT / "evaluation" / "results.json"))
    a = ap.parse_args()

    system, nli = NLPSystem(), NLIFaithfulness()
    rows = []
    for ex in json.loads(Path(a.questions).read_text()):
        res = system.ask(ex["question"])
        rows.append({**ex, "answer": res["answer"], "route": res["route"], "latency": res["latency_s"],
                     "contexts": [s["text"] for s in res["sources"]]})
        print(f"- {ex['question']}\n  -> [{res['route']}] {res['answer'][:140]}")

    m = {"n_questions": len(rows)}
    gold = [r for r in rows if r.get("gold_substring")]
    if gold:
        ranks = []
        for r in gold:
            pos = [i for i, c in enumerate(r["contexts"]) if r["gold_substring"].lower() in c.lower()]
            ranks.append(pos[0] + 1 if pos else None)
        m["retrieval_hit@k"] = round(float(np.mean([x is not None for x in ranks])), 4)
        m["retrieval_MRR"] = round(float(np.mean([1 / x if x else 0 for x in ranks])), 4)
    m["answer_token_f1"] = round(float(np.mean([token_f1(r["answer"], r.get("ground_truth", "")) for r in rows])), 4)
    m["nli_faithfulness"] = round(float(np.mean([nli.score(r["answer"], r["contexts"]) for r in rows])), 4)
    m["latency_p50_s"] = round(float(np.median([r["latency"] for r in rows])), 2)
    m["latency_p95_s"] = round(float(np.percentile([r["latency"] for r in rows], 95)), 2)

    if os.getenv("OPENAI_API_KEY"):
        try:
            m.update({f"ragas_{k}": v for k, v in ragas_scores(rows).items()})
        except Exception as e:  # ragas version / judge-LLM issues must not kill the run
            print("[warn] RAGAS failed:", e)
    else:
        print("[info] OPENAI_API_KEY not set -> RAGAS skipped; NLI faithfulness is the offline proxy.")

    Path(a.out).write_text(json.dumps({"metrics": m, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== METRICS ===")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
