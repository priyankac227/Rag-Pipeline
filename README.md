# Merekos NLP System — Manual Summarization + RAG QA Assistant

```
Merekos-NLP-System/
├── app.py                 Streamlit UI (chat with citations, document summaries, upload & re-index)
├── nlp_pipeline.py        Whole pipeline: clean → chunk → FAISS → hybrid retrieval → router → BERT/T5 → citations
├── config.py              Paths, model names, thresholds, shared QA prompt
├── training/
│   ├── prepare_summarization_data.py   weak-label bootstrap (+ keeps your human rows) → summarization.json
│   ├── train_t5.py                     multi-task T5 (summarization + generative QA) → trained_models/t5_manufacturing
│   ├── prepare_qa_data.py              question generation + span extraction → qa.json
│   ├── train_bert.py                   extractive QA BERT → trained_models/bert_manufacturing_qa
│   └── datasets/                       starter human-labelled samples (replace/extend with real data)
├── trained_models/        fine-tuned models land here and are auto-detected
├── documents/             put manuals / FAQs here (.pdf .txt .md)
├── vector_store/          FAISS index + chunk metadata (created by `build`)
└── evaluation/
    ├── test_questions.json
    └── evaluate_ragas.py  RAGAS + offline NLI faithfulness + hit@k/MRR + latency
```

## How it works
1. **Preprocess** – strip repeated headers/footers, de-hyphenate, detect headings, keep numbered steps intact, chunk inside sections with overlap, keep page + section metadata.
2. **Index** – bge-small embeddings → FAISS (exact; HNSW above 100k chunks).
3. **Retrieve** – FAISS dense + BM25 → reciprocal-rank fusion → cross-encoder rerank.
4. **Answer** – router: *factoid* → fine-tuned BERT extractive reader; *procedural / multi-part* → fine-tuned T5 generator over the top chunks. Every answer sentence gets a `[n]` citation; unsupported sentences are dropped and weak retrieval triggers an "I couldn't find it" abstention.
5. **Summarize** – hierarchical T5 map-reduce: per-section summaries + overall abstract.

## Run
```bash
pip install -r requirements.txt
python nlp_pipeline.py build                    # index documents/
streamlit run app.py                            # UI
python nlp_pipeline.py ask "How do I replace the hydraulic filter?"
python nlp_pipeline.py summarize hp200_service_manual
```
Works immediately with public fallbacks (flan-t5-base, distilbert-squad). 

## Fine-tune on your documents
```bash
python training/prepare_summarization_data.py   # weak labels; review them, add "source":"human" rows
python training/prepare_qa_data.py
python training/train_t5.py                     # → trained_models/t5_manufacturing
python training/train_bert.py                   # → trained_models/bert_manufacturing_qa
python nlp_pipeline.py build                    # (re)index if documents changed
python evaluation/evaluate_ragas.py             # RAGAS needs: export OPENAI_API_KEY=...
```
The shipped `datasets/*.json` contain a few hand-written rows for the demo manual only — real gains need
hundreds of reviewed examples from your own manuals and real engineer questions. Extend
`evaluation/test_questions.json` to 100+ real questions before quoting any metric.

## Tips
- Scanned PDFs: run `ocrmypdf` first. Tables: extract with `pdfplumber` and serialize rows as text.
- Wrong-manual-version answers: use the document filter (`doc_id`) per machine/revision.
