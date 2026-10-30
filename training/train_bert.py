"""Fine-tune BERT for extractive QA (fact lookup) on datasets/qa.json -> trained_models/bert_manufacturing_qa/
Usage: python training/train_bert.py [--base bert-base-uncased] [--epochs 3]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bert-base-uncased")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    a = ap.parse_args()

    from datasets import Dataset
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, DefaultDataCollator,
                              Trainer, TrainingArguments)

    rows = json.loads(C.QA_DATA.read_text())
    rows = [{k: r[k] for k in ("question", "context", "answer", "answer_start")} for r in rows]
    if len(rows) < 10:
        sys.exit("Not enough QA rows - run training/prepare_qa_data.py (or add human-labelled rows).")
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForQuestionAnswering.from_pretrained(a.base)
    ds = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)

    def prep(ex):
        enc = tok(ex["question"], ex["context"], truncation="only_second", max_length=a.max_len,
                  stride=a.stride, return_overflowing_tokens=True, return_offsets_mapping=True,
                  padding="max_length")
        smap, offsets = enc.pop("overflow_to_sample_mapping"), enc.pop("offset_mapping")
        starts, ends = [], []
        for i, off in enumerate(offsets):
            ids, seq = enc["input_ids"][i], enc.sequence_ids(i)
            cls = ids.index(tok.cls_token_id)
            s = smap[i]
            a_start = ex["answer_start"][s]
            a_end = a_start + len(ex["answer"][s])
            ts = seq.index(1)
            te = len(ids) - 1 - seq[::-1].index(1)
            if not (off[ts][0] <= a_start and off[te][1] >= a_end):   # answer not inside this window
                starts.append(cls); ends.append(cls)
            else:
                while ts < len(off) and off[ts][0] <= a_start:
                    ts += 1
                starts.append(ts - 1)
                while off[te][1] >= a_end:
                    te -= 1
                ends.append(te + 1)
        enc["start_positions"], enc["end_positions"] = starts, ends
        return enc

    ds = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)
    args = TrainingArguments(output_dir=str(C.ROOT / "training" / "_checkpoints_bert"),
                             num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
                             eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                             load_best_model_at_end=True, weight_decay=0.01, warmup_ratio=0.1,
                             logging_steps=20, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                      data_collator=DefaultDataCollator(), processing_class=tok)
    trainer.train()
    trainer.save_model(str(C.BERT_PATH))
    tok.save_pretrained(str(C.BERT_PATH))
    print("Final eval:", trainer.evaluate())
    print(f"Saved -> {C.BERT_PATH}")


if __name__ == "__main__":
    main()
