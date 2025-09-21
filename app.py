import streamlit as st
from indexer import build_index
from retriever import Retriever
from agent import DeepResearcherAgent
import os
import json
from utils import list_documents
from utils import read_pdf, read_text_file, clean_text
import markdown2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
st.set_page_config(page_title="Deep Researcher Agent", layout="wide")
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
st.title("🔎 Deep Researcher — Local Research Agent (Streamlit)")
with st.sidebar:
    st.header("Indexing / Data")
    data_dir = st.text_input("Documents folder (local)", value="data/docs")
    uploaded_files = st.file_uploader("Or upload files to index", accept_multiple_files=True, type=["pdf","txt","md"])
    if st.button("Index Documents"):
        os.makedirs(data_dir, exist_ok=True)
        for f in uploaded_files:
            save_path = os.path.join(data_dir, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.info("Building index — may take a while on first run.")
        try:
            build_index(data_dir, index_path=INDEX_PATH, meta_path=META_PATH)
            st.success("Index built successfully.")
        except Exception as e:
            st.error(f"Indexing failed: {e}")
    st.markdown("**Indexed files:**")
    try:
        docs = list_documents(data_dir)
        if docs:
            for d in docs:
                st.write(d)
        else:
            st.write("No documents found. Upload or set correct folder and click 'Index Documents'.")
    except Exception:
        st.write("No documents found or data_dir invalid.")
st.markdown("---")
@st.cache_resource
def load_agent():
    retr = Retriever(index_path=INDEX_PATH, meta_path=META_PATH)
    agent = DeepResearcherAgent(retriever=retr, gen_model_name="google/flan-t5-small")
    return agent
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    agent = load_agent()
else:
    agent = None
    st.warning("No index found. Please add documents and click 'Index Documents' in the sidebar.")
if agent:
    if "history" not in st.session_state:
        st.session_state.history = []
    st.subheader("Ask anything (research-style)")
    query = st.text_area("Enter your research question", height=120, key="query_box")
    col1, col2, col3 = st.columns(3)
    with col1:
        topk = st.number_input("Top evidence per subtask", min_value=1, max_value=10, value=4)
    with col2:
        subtasks = st.number_input("Max subtasks (planner)", min_value=1, max_value=8, value=4)
    with col3:
        st.write("")
        if st.button("Run Research"):
            if not query.strip():
                st.warning("Please type a question.")
            else:
                with st.spinner("Planning, retrieving, and synthesizing..."):
                    result = agent.research(query, top_k_per_subtask=topk, max_subtasks=subtasks)
                st.session_state.history.insert(0, {"q": query, "result": result})
                st.success("Done — answer synthesized.")
    for i, item in enumerate(st.session_state.history):
        with st.expander(f"Q: {item['q']}", expanded=(i==0)):
            st.markdown("**Answer:**")
            st.write(item["result"]["answer"])
            st.markdown("**Reasoning steps:**")
            for r in item["result"]["reasoning_steps"]:
                st.write(f"- {r}")
            st.markdown("**Evidence (top sources):**")
            for e in item["result"]["evidence"]:
                st.write(f"- Source: `{e['source']}` (score: {e['score']:.3f})")
                st.write(f"  > {e['text'][:400]}...")
            colA, colB = st.columns(2)
            with colA:
                if st.button("Export as Markdown", key=f"md_{i}"):
                    md = f"# Research: {item['q']}\n\n## Answer\n\n{item['result']['answer']}\n\n## Reasoning Steps\n\n"
                    for r in item["result"]["reasoning_steps"]:
                        md += f"- {r}\n"
                    md += "\n## Evidence\n"
                    for e in item["result"]["evidence"]:
                        md += f"\n### Source: {e['source']}\n\n{e['text']}\n"
                    b = md.encode("utf-8")
                    st.download_button("Download .md", data=b, file_name="research.md", mime="text/markdown")
            with colB:
                if st.button("Export as PDF", key=f"pdf_{i}"):
                    md = f"# Research: {item['q']}\n\n## Answer\n\n{item['result']['answer']}\n\n## Reasoning Steps\n\n"
                    for r in item["result"]["reasoning_steps"]:
                        md += f"- {r}\n"
                    md += "\n## Evidence\n"
                    for e in item["result"]["evidence"]:
                        md += f"\n### Source: {e['source']}\n\n{e['text']}\n"
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    c = canvas.Canvas(tmpf.name, pagesize=letter)
                    width, height = letter
                    textobject = c.beginText(40, height - 40)
                    for line in md.splitlines():
                        textobject.textLine(line)
                        if textobject.getY() < 40:
                            c.drawText(textobject)
                            c.showPage()
                            textobject = c.beginText(40, height - 40)
                    c.drawText(textobject)
                    c.save()
                    with open(tmpf.name, "rb") as f:
                        st.download_button("Download PDF", data=f, file_name="research.pdf", mime="application/pdf")
