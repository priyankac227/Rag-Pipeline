"""Fine-tune ONE T5 model for both tasks (multi-task via different prompts):
   - abstractive summarization  ("summarize: <text>")        from datasets/summarization.json
   - generative QA              (config.qa_prompt)            from datasets/qa.json
Saves to trained_models/t5_manufacturing/.
Usage: python training/train_t5.py [--base google/flan-t5-base] [--epochs 4] [--tasks both|summarization|qa]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.T5_FALLBACK)
    ap.add_argument("--tasks", choices=["both", "summarization", "qa"], default="both")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_in", type=int, default=768)
    ap.add_argument("--max_out", type=int, default=160)
    a = ap.parse_args()

    import evaluate
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    pairs = []  # (source text, target text)
    if a.tasks in ("both", "summarization"):
        for r in json.loads(C.SUMMARIZATION_DATA.read_text()):
            pairs.append((C.SUMMARY_PREFIX + r["text"], r["summary"]))
    if a.tasks in ("both", "qa"):
        for r in json.loads(C.QA_DATA.read_text()):
            pairs.append((C.qa_prompt(r["question"], [r["context"]]), r.get("answer_sentence") or r["answer"]))
    if len(pairs) < 10:
        sys.exit("Not enough training rows - run the prepare_* scripts (or add human-labelled rows).")
    print(f"Training on {len(pairs)} examples")

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.base)
    ds = Dataset.from_list([{"src": s, "tgt": t} for s, t in pairs]).train_test_split(test_size=0.1, seed=42)

    def prep(ex):
        enc = tok(ex["src"], max_length=a.max_in, truncation=True)
        enc["labels"] = tok(text_target=ex["tgt"], max_length=a.max_out, truncation=True)["input_ids"]
        return enc

    ds = ds.map(prep, batched=True, remove_columns=["src", "tgt"])
    rouge = evaluate.load("rouge")

    def metrics(p):
        preds, labels = p
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        res = rouge.compute(predictions=tok.batch_decode(preds, skip_special_tokens=True),
                            references=tok.batch_decode(labels, skip_special_tokens=True))
        return {k: round(v, 4) for k, v in res.items()}

    args = Seq2SeqTrainingArguments(
        output_dir=str(C.ROOT / "training" / "_checkpoints_t5"), num_train_epochs=a.epochs,
        learning_rate=a.lr, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="rougeL", greater_is_better=True, predict_with_generate=True,
        generation_max_length=a.max_out, warmup_ratio=0.05, weight_decay=0.01, logging_steps=20,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # T5 + fp16 is unstable
        report_to="none")
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                             processing_class=tok, data_collator=DataCollatorForSeq2Seq(tok, model=model),
                             compute_metrics=metrics)
    trainer.train()
    trainer.save_model(str(C.T5_PATH))
    tok.save_pretrained(str(C.T5_PATH))
    print("Final eval:", trainer.evaluate())
    print(f"Saved -> {C.T5_PATH}")


if __name__ == "__main__":
    main()
