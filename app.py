import os
from pathlib import Path
 
import streamlit as st
from dotenv import load_dotenv
 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
 
load_dotenv()
 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
# ---------------- Settings (same as notebook) ----------------
PDF_FOLDER = Path(r"D:\Helpdesk")
CHROMA_DIR = "RAG1"
TOP_N = 4
NOT_FOUND = "I could not find this in the knowledge base."
 
st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️", layout="wide")
 
 
# ---------------- Prompt (same as notebook) ----------------
prompt = ChatPromptTemplate.from_template("""
You are an IT Helpdesk Assistant.
 
Answer the user's question using ONLY the provided context.
 
Rules:
1. Start directly with the answer. Give clear, numbered troubleshooting
   steps that directly address the question.
2. Every step, command, setting, and fact must appear in the context.
   Do not add steps, commands, causes, or advice from your own knowledge,
   even if they are common or correct.
3. Copy commands exactly as written in the context. Do not change or
   add command options.
4. Cover all relevant steps found in the context for this question,
   so the answer is complete but not padded.
5. If the context applies only to a specific product or version,
   say so or leave it out. Do not generalize.
6. Leave out unrelated scenarios and side topics.
7. Do not mention document names, source names, NIST, or any other
   standard in the answer. Sources are shown separately.
8. Do not end with follow-up questions or offers of more help.
9. If the context covers only part of the question, answer only that part.
   If it has nothing relevant, reply exactly:
   "I could not find this in the knowledge base."
10. Keep the answer focused (maximum 8 steps).
 
Context:
{context}
 
Question:
{question}
 
Answer:
""")
 
 
# ---------------- Knowledge base helpers ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
 
 
def build_ensemble(kb):
    # BM25 has to be rebuilt whenever new chunks are added
    vector_retriever = kb["vector_store"].as_retriever(
        search_kwargs={"k": 6}
    )
 
    retriever = BM25Retriever.from_documents(kb["chunks"])
    retriever.k = 6
 
    kb["ensemble"] = EnsembleRetriever(
        retrievers=[retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
 
 
def load_pdf_chunks(pdf_path):
    pages = PyPDFLoader(str(pdf_path)).load()
    return text_splitter.split_documents(pages)
 
 
@st.cache_resource(show_spinner="Loading knowledge base (first run only)...")
def load_pipeline():
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=OPENAI_API_KEY,
        temperature=0
    )
 
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY,
        dimensions=1000
    )
 
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )
 
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
    kb = {
        "vector_store": vector_store,
        "chunks": [],
        "sources": set(),
        "ensemble": None
    }
 
    # Sources already stored in Chroma
    stored = vector_store.get(include=["metadatas"])
    indexed_sources = {
        m.get("source") for m in (stored["metadatas"] or []) if m
    }
 
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
 
    new_chunks = []
    for pdf in sorted(PDF_FOLDER.rglob("*.pdf")):
        try:
            pdf_chunks = load_pdf_chunks(pdf)
        except Exception as e:
            print("Failed:", pdf.name, "-", e)
            continue
 
        kb["chunks"].extend(pdf_chunks)
        kb["sources"].add(str(pdf))
 
        # PDF is in the folder but not in Chroma yet -> index it
        if str(pdf) not in indexed_sources:
            new_chunks.extend(pdf_chunks)
 
    if new_chunks:
        vector_store.add_documents(new_chunks)
 
    build_ensemble(kb)
 
    return model, kb, reranker
 
 
model, kb, reranker = load_pipeline()
 
 
def add_uploaded_pdfs(uploaded_files):
    added, skipped, failed = [], [], []
 
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
 
    for uploaded in uploaded_files:
        save_path = PDF_FOLDER / uploaded.name
 
        # Already in the knowledge base -> do not add twice
        if str(save_path) in kb["sources"] or save_path.exists():
            skipped.append(uploaded.name)
            continue
 
        try:
            save_path.write_bytes(uploaded.getvalue())
            pdf_chunks = load_pdf_chunks(save_path)
 
            if not pdf_chunks:
                save_path.unlink()
                failed.append(f"{uploaded.name} (no readable text)")
                continue
 
            kb["vector_store"].add_documents(pdf_chunks)
            kb["chunks"].extend(pdf_chunks)
            kb["sources"].add(str(save_path))
            added.append(uploaded.name)
        except Exception as e:
            if save_path.exists() and str(save_path) not in kb["sources"]:
                save_path.unlink()
            failed.append(f"{uploaded.name} ({e})")
 
    if added:
        build_ensemble(kb)
 
    return added, skipped, failed
 
 
# ---------------- RAG functions (same as notebook) ----------------
def rerank_documents(question, documents, top_n=TOP_N):
    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]
 
    scores = reranker.predict(pairs)
 
    ranked_docs = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )
 
    return [
        doc for doc, score in ranked_docs[:top_n]
    ]
 
 
def answer_from_docs(question, docs):
    context = "\n\n".join(
        doc.page_content for doc in docs
    )
 
    messages = prompt.invoke({
        "context": context,
        "question": question
    })
 
    response = model.invoke(messages)
    answer = response.content.strip()
 
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ))
 
    return answer, sources
 
 
def hybrid_rag_answer(question):
    retrieved_docs = kb["ensemble"].invoke(question)
    reranked_docs = rerank_documents(
        question, retrieved_docs, top_n=TOP_N
    )
 
    answer, sources = answer_from_docs(
        question, reranked_docs
    )
 
    return answer, sources, reranked_docs
 
 
# ---------------- UI ----------------
st.title("🛠️ IT Helpdesk Assistant")
 
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
 
with st.sidebar:
    st.subheader("Add new document")
 
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )
 
    if st.button("Add to knowledge base", use_container_width=True,
                 disabled=not uploaded_files):
        with st.spinner("Reading and indexing the document..."):
            added, skipped, failed = add_uploaded_pdfs(uploaded_files)
 
        st.session_state["upload_result"] = (added, skipped, failed)
        st.session_state["uploader_key"] += 1
        st.rerun()
 
    if "upload_result" in st.session_state:
        added, skipped, failed = st.session_state.pop("upload_result")
        if added:
            st.success("Added: " + ", ".join(added))
        if skipped:
            st.info("Already in knowledge base: " + ", ".join(skipped))
        if failed:
            st.error("Failed: " + ", ".join(failed))
 
    st.divider()
    st.subheader("Try these")
    examples = [
        "How do I troubleshoot DNS resolution problems on a Windows computer?",
        "What should I check when a printer is not responding?",
        "How can I troubleshoot Remote Desktop connection failures?",
        "How do I troubleshoot a Windows blue screen error?",
        "What are the phases of incident response?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["pending_question"] = example
 
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
 
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 
 
def show_sources(sources, docs):
    with st.expander("Sources"):
        for source in sources:
            st.write("•", Path(source).name)
        st.divider()
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "Unknown")
            st.markdown(f"**Chunk {i}** — {Path(doc.metadata.get('source', 'Unknown')).name} (page {page})")
            st.text(doc.page_content[:500])
 
 
# Show previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"], message["docs"])
 
# Get the question (typed or from sidebar example)
question = st.chat_input("Ask an IT helpdesk question...")
 
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
 
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
 
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            answer, sources, reranked_docs = hybrid_rag_answer(question)
 
        st.markdown(answer)
 
        # Do not show sources when the answer is "not found"
        if answer.strip() != NOT_FOUND:
            show_sources(sources, reranked_docs)
            saved_sources, saved_docs = sources, reranked_docs
        else:
            saved_sources, saved_docs = [], []
 
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": saved_sources,
        "docs": saved_docs
    })
