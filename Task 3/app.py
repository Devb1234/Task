import os
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

def load_docs(path):
    docs = []
    for file in os.listdir(path):
        full = os.path.join(path, file)
        if file.endswith(".pdf"):
            docs.extend(PyMuPDFLoader(full).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(full).load())
        elif file.endswith(".md"):
            with open(full, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": full}))
    return docs

docs = load_docs("documents")

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(chunks, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt_docs = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided document context."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def answer_docs(question):
    retrieved = retriever.invoke(question)
    if not retrieved:
        return {"answer": "No documents found for this query.", "sources": []}
    sources = list({d.metadata.get("source") for d in retrieved})
    context = format_docs(retrieved)
    p = prompt_docs.invoke({"question": question, "context": context})
    ans = llm.invoke(p.to_messages()).content
    return {"answer": ans, "sources": sources}

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

def query_kg(question):
    q = question.lower()
    m = []
    for s, r, t in KG["relations"]:
        if s.lower() in q or r.lower() in q or t.lower() in q:
            m.append((s, r, t))
    return m

def answer_kg(question):
    matches = query_kg(question)
    if not matches:
        return {"answer": "No KG facts found.", "sources": []}
    facts = "\n".join([f"{s} {r} {t}" for s, r, t in matches])
    prompt = f"Use these facts:\n{facts}\nAnswer: {question}"
    ans = llm.invoke(prompt).content
    return {"answer": ans, "sources": matches}

def answer_both(question):
    d = answer_docs(question)
    k = answer_kg(question)
    prompt = (
        f"Question: {question}\n\n"
        f"Document Answer:\n{d['answer']}\n\n"
        f"KG Answer:\n{k['answer']}\n\n"
        "Combine both into a single clear answer."
    )
    final = llm.invoke(prompt).content
    return {
        "answer": final,
        "doc_sources": d["sources"],
        "kg_sources": k["sources"]
    }

def router(question):
    q = question.lower()
    kg_keys = ["works on", "manages", "belongs", "uses", "project", "department"]
    doc_keys = ["document", "pdf", "file", "milestone", "architecture", "rag", "explain", "overview"]
    if any(k in q for k in kg_keys):
        return "kg"
    if any(k in q for k in doc_keys):
        return "docs"
    return "both"

def ask(question):
    route = router(question)
    if route == "docs":
        out = answer_docs(question)
        print("\nQ:", question, "\n\nA:", out["answer"])
        print("\nSources (Docs):", out["sources"])
    elif route == "kg":
        out = answer_kg(question)
        print("\nQ:", question, "\n\nA:", out["answer"])
        print("\nSources (KG):", out["sources"])
    else:
        out = answer_both(question)
        print("\nQ:", question, "\n\nA:", out["answer"])
        print("\nSources (Docs):", out["doc_sources"])
        print("Sources (KG):", out["kg_sources"])

ask("Explain the system architecture of Project Beta.")
ask("What are the milestones of Project Alpha?")
ask("Who manages Project Beta?")
ask("What is a knowledge graph?")
ask("Explain the RAG pipeline.")