import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

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
    matches = []

    for src, rel, tgt in KG["relations"]:
        if src.lower() in q or rel.lower() in q or tgt.lower() in q:
            matches.append((src, rel, tgt))

    return matches

def ask(question):
    matches = query_kg(question)

    context = "\n".join([f"{s} {r} {t}" for s, r, t in matches]) or "No graph matches found."

    prompt = f"Question: {question}\nGraph facts:\n{context}\nAnswer clearly:"
    result = llm.invoke(prompt)

    print("\nQ:", question)
    print("\nA:", result.content)

ask("Which employees use Python on Project Alpha?")
ask("Who manages Project Beta?")
ask("Which department does Carol belong to?")
ask("Who works on Project Alpha?")