"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

import config as C
from nlp_pipeline import NLPSystem

st.set_page_config(page_title="Merekos NLP System", page_icon="🔧", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_system():
    return NLPSystem()


system = get_system()

with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader("Add manuals / FAQs", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads and st.button("Ingest & re-index"):
        C.DOCUMENTS_DIR.mkdir(exist_ok=True)
        for f in uploads:
            (C.DOCUMENTS_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Indexing..."):
            system.build_index()
        st.success("Done.")
        st.rerun()
    try:
        docs = ["(all documents)"] + system.list_documents()
    except FileNotFoundError:
        st.warning("Vector store is empty: upload documents above, or run `python nlp_pipeline.py build`.")
        st.stop()
    choice = st.selectbox("Restrict to document", docs)
    doc_id = None if choice == docs[0] else choice

tab_chat, tab_sum = st.tabs(["💬 Ask the manuals", "📄 Summaries"])

with tab_chat:
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if q := st.chat_input("e.g. How do I replace the hydraulic filter?"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching manuals..."):
                r = system.ask(q, doc_id=doc_id)
            st.markdown(r["answer"])
            st.caption(f"route: {r['route']} · grounded: {r['grounded_ratio']:.0%} · {r['latency_s']}s")
            for s in r["sources"]:
                if s["cited"]:
                    with st.expander(f"[{s['n']}] {s['doc_id']} · p.{s['page']} · {s['section']}"):
                        st.write(s["text"])
        st.session_state.messages.append({"role": "assistant", "content": r["answer"]})

with tab_sum:
    mode = st.radio("Summarize", ["A document", "Pasted text"], horizontal=True)
    if mode == "A document":
        if doc_id is None:
            st.info("Pick a document in the sidebar.")
        elif st.button("Summarize document"):
            with st.spinner("Summarizing..."):
                res = system.summarize_document(doc_id)
            st.subheader("Overview")
            st.write(res["overall"])
            for s in res["sections"]:
                st.markdown(f"**{s['section']}** (p.{s['page']}) — {s['summary']}")
    else:
        text = st.text_area("Text", height=250)
        if text and st.button("Summarize"):
            st.write(system.summarize_text(text))
