**Consolidated Project README**

**Setup Instructions**
- **Prerequisites:** Install Python 3.10+ and ensure `python` is on your PATH.
- **Create and activate a virtual environment (PowerShell):**

```powershell
conda create -p venv python==3.12 -y
.\.venv\Scripts\Activate.ps1
```
- **Install dependencies:** Run from the repository root. If you have a root `requirements.txt`:

```powershell
pip install -r requirements.txt
```

If there is no root `requirements.txt`, but individual task folders include one (for example `Task 1\requirements.txt`), install from that path instead:

```powershell
pip install -r "Task 1\requirements.txt"
```

- **Run a task app:** After activating the virtualenv, run a specific task's `app.py` like this:

```powershell
python "Task 1\app.py"
python "Task 2\app.py"
python "Task 3\app.py"
python "Task 4\app.py"
```

**Repository Architecture Overview**
- **Root:** contains `requirements.txt` (optional) and the `Task N` folders.
- **Task folders:** Each `Task N` folder contains:
  - `app.py`: small runnable example or demo for that task.
  - `documents/`: supporting files such as `knowledge_graph_basics.txt` or `rag_pipeline_explained.md`.
  - (optional) `requirements.txt` or other task-specific resources.
- **Design:** Each task is an independent, minimal example intended to be runnable in isolation. The consolidated README provides a single entry point to run and explore all tasks.

**Per-Task Quick Summary & Run Examples**
- **Task 1**
  - Purpose: Knowledge-graph basics and related docs.
  - Run: `python "Task 1\app.py"`
  - Docs: `Task 1\documents\knowledge_graph_basics.txt`, `Task 1\documents\milestones_alpha.md`.
- **Task 2**
  - Purpose: (See `Task 2\app.py`) Example app demonstrating Task 2 concept.
  - Run: `python "Task 2\app.py"`
- **Task 3**
  - Purpose: RAG pipeline explanation and demos.
  - Run: `python "Task 3\app.py"`
  - Docs: `Task 3\documents\rag_pipeline_explained.md`.
- **Task 4**
  - Purpose: Another RAG example or extension.
  - Run: `python "Task 4\app.py"`
  - Docs: `Task 4\documents\rag_pipeline_explained.md`.

**Sample Queries & Expected Outputs**
Use these sample queries when exploring the documents or building a QA/RAG wrapper around the content. These are examples you can paste into a simple command-line QA script or prompt to a model that has retrieved context from the `documents/` folders.

- **Query:** `What is a knowledge graph?`
  - **Expected output:** A knowledge graph is a graph-based data structure that represents entities and their relationships; it supports semantic queries, linking heterogeneous data, and enables reasoning over connected facts.
- **Query:** `List the core components of a RAG pipeline.`
  - **Expected output:** Document retriever, text splitter, embedding encoder, vector store (index), and a generator (language model) that composes answers using retrieved context.
- **Query:** `How do I build embeddings from documents?`
  - **Expected output:** Chunk documents into passages, embed each chunk with an embedding model, store vectors in a vector DB or index, and use nearest-neighbor search for retrieval.
- **Query:** `When should I use retrieval vs fine-tuning?`
  - **Expected output:** Use retrieval when the knowledge base is large or frequently changing; fine-tune for persistent, task-specific behavior when you have labeled data and a fixed domain.
- **Query:** `Summarize the knowledge graph basics from Task 1 docs.`
  - **Expected output:** Knowledge graphs model entities and relationships as nodes and edges, use an ontology/schema to define types, are useful for semantic queries, and often complement embedding-based retrieval in QA systems.

**Assumptions Made**
- The workspace root is `e:\Projects\Task` and tasks are in `Task 1`..`Task 4` as shown.
- You are using Windows PowerShell; commands in this README are PowerShell-compatible.
- `app.py` files are runnable Python scripts; their CLI contracts may vary.
- Dependencies are either centralized in a root `requirements.txt` or provided per task.

**Challenges Faced & Notes**
- **Unknown CLI contracts:** The `app.py` scripts do not have a uniform interface or documented CLI. I avoided making assumptions about arguments and provided direct `python "Task N\\app.py"` run commands. If you want, I can inspect each `app.py` and add per-task usage details.
- **Multiple `requirements.txt` locations:** Some tasks may include their own `requirements.txt`; the README explains both root and per-task install options.