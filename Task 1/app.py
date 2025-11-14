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

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the retrieved document context."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def ask(q):
    retrieved = retriever.invoke(q)
    sources = [d.metadata.get("source") for d in retrieved]
    context = format_docs(retrieved)
    final_prompt = prompt.invoke({"question": q, "context": context})
    result = llm.invoke(final_prompt.to_messages())

    print("\n====================================")
    print("Q:", q)
    print("====================================\n")
    print(result.content)
    print("\nSources:")
    if sources:
        for s in sources:
            print(" -", s)
    else:
        print(" - No documents matched")

ask("What are the key milestones of Project Beta?")
ask("Explain Project Alpha milestones.")
ask("What is a knowledge graph?")