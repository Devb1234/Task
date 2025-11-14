# RAG Pipeline Explained

A Retrieval‑Augmented Generation (RAG) pipeline enhances LLM outputs by grounding them in external knowledge.

## Steps in a RAG Pipeline
1. **Document Loading** – Load PDFs, TXT, MD using appropriate loaders.
2. **Chunking** – Split documents into manageable chunks.
3. **Embedding** – Convert chunks into high‑dimensional embeddings.
4. **Vector Store Storage** – Store embeddings in FAISS or another vector DB.
5. **Retriever Querying** – Retrieve relevant chunks based on semantic similarity.
6. **LLM Generation** – Supply retrieved context to an LLM for grounded answers.

RAG improves factual accuracy, reduces hallucinations, and is widely used in enterprise AI solutions.