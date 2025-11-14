import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI RAG + KG System", layout="wide")

if not os.path.exists("documents"):
    os.makedirs("documents")

def load_docs(path="documents"):
    docs = []
    for f in os.listdir(path):
        full = os.path.join(path, f)
        if f.endswith(".pdf"):
            docs.extend(PyMuPDFLoader(full).load())
        elif f.endswith(".txt"):
            docs.extend(TextLoader(full).load())
        elif f.endswith(".md"):
            with open(full, "r", encoding="utf-8") as fp:
                docs.append(Document(page_content=fp.read(), metadata={"source": full}))
    return docs

def build_vector_db():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed).as_retriever(search_kwargs={"k": 3})

retriever = build_vector_db()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt_docs = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided document context."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(d):
    return "\n\n".join([x.page_content for x in d])

def answer_docs(q):
    d = retriever.invoke(q)
    if not d:
        return "No documents matched.", []
    context = format_docs(d)
    p = prompt_docs.invoke({"question": q, "context": context})
    ans = llm.invoke(p.to_messages()).content
    sources = list({x.metadata.get("source") for x in d})
    return ans, sources

KG = {
    "relations": [
        ("Bob", "works_on", "Alpha"),
        ("Carol", "works_on", "Alpha"),
        ("Alice", "manages", "Alpha"),
        ("Alice", "manages", "Beta"),
        ("Bob", "uses", "React"),
        ("Carol", "uses", "Python"),
        ("Alice", "belongs_to", "Engineering"),
        ("Bob", "belongs_to", "Engineering"),
        ("Carol", "belongs_to", "Data")
    ]
}

def query_kg(q):
    q = q.lower()
    m = []
    for s, r, t in KG["relations"]:
        if s.lower() in q or r.lower() in q or t.lower() in q:
            m.append((s, r, t))
    return m

def answer_kg(q):
    m = query_kg(q)
    if not m:
        return "No KG facts found.", []
    fact_text = "\n".join([f"{s} {r} {t}" for s, r, t in m])
    prompt = f"Use these facts:\n{fact_text}\nAnswer: {q}"
    ans = llm.invoke(prompt).content
    return ans, m

def answer_both(q):
    d_ans, d_src = answer_docs(q)
    k_ans, k_src = answer_kg(q)
    p = f"Question: {q}\nDocument Answer:\n{d_ans}\nKG Answer:\n{k_ans}\nCombine both clearly."
    final = llm.invoke(p).content
    return final, d_src, k_src

def router(q):
    ql = q.lower()
    kg_keys = ["works", "manages", "belongs", "uses", "project", "department"]
    doc_keys = ["document", "pdf", "file", "milestone", "architecture", "rag", "overview"]
    if any(k in ql for k in kg_keys):
        return "kg"
    if any(k in ql for k in doc_keys):
        return "docs"
    return "both"

st.title("Task 4: AI RAG + Knowledge Graph System")

uploaded = st.file_uploader("Upload Documents (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)

if uploaded:
    for file in uploaded:
        path = os.path.join("documents", file.name)
        with open(path, "wb") as f:
            f.write(file.read())
    retriever = build_vector_db()
    st.success("Documents added and vector database updated.")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        route = router(question)
        if route == "docs":
            ans, src = answer_docs(question)
            st.subheader("📘 Document Answer")
            st.write(ans)
            st.subheader("Sources")
            st.write(src)
        elif route == "kg":
            ans, src = answer_kg(question)
            st.subheader("🧠 Knowledge Graph Answer")
            st.write(ans)
            st.subheader("KG Source Triples")
            st.write(src)
        else:
            ans, ds, ks = answer_both(question)
            st.subheader("📘 + 🧠 Combined Answer")
            st.write(ans)
            st.subheader("Document Sources")
            st.write(ds)
            st.subheader("KG Sources")
            st.write(ks)

        st.subheader("Knowledge Graph Visualization")
        G = nx.DiGraph()
        for s, r, t in KG["relations"]:
            G.add_edge(s, t, label=r)
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, arrows=True)
        edge_labels = {(s, t): r for s, r, t in KG["relations"]}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        st.pyplot(plt)