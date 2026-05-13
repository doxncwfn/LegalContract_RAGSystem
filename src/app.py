import os
import json
from pathlib import Path

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

# Ensures all relative paths work regardless of where Streamlit is launched from
SRC_DIR = Path(__file__).resolve().parent
load_dotenv(SRC_DIR / ".env")

st.set_page_config(page_title="Legal Contract Clause Search", layout="wide")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Missing GEMINI_API_KEY. Create `src/.env` with GEMINI_API_KEY=... or export it in your shell."
    )
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

DB_DIR = str(SRC_DIR / "chroma_db")
SRL_PATH = SRC_DIR / "output" / "srl_results.json"
chroma_client = chromadb.PersistentClient(path=DB_DIR)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="legal_clauses",
    embedding_function=embedding_func
)

def load_and_embed_data():
    """Load SRL output into ChromaDB if collection is empty."""
    if collection.count() > 0:
        return

    st.info("First run detected: Embedding contract clauses... This takes a few seconds.")
    
    try:
        with open(SRL_PATH, "r", encoding="utf-8") as f:
            srl_data = json.load(f)
        
        ids = []
        documents = []
        metadatas = []

        for i, item in enumerate(srl_data):
            clause_text = item.get("clause", "")
            roles = item.get("roles", {})
            
            doc_text = f"Clause: {clause_text}\nPredicate: {item.get('predicate')}\nRoles: {json.dumps(roles)}"
            metadata = {
                "clause_id": str(i),
                "agent": roles.get("Agent", "Unknown"),
                "condition": roles.get("Condition", "None"),
            }

            ids.append(f"clause_{i}")
            documents.append(doc_text)
            metadatas.append(metadata)

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        st.success(f"Successfully embedded {len(documents)} clauses!")

    except FileNotFoundError:
        st.error(
            f"Data file not found: {SRL_PATH}. Run the pipeline first: "
            "`python src/run_full_pipeline.py --input SPONSORSHIP_AGREEMENT.txt`"
        )

def generate_response(user_query):
    results = collection.query(
        query_texts=[user_query],
        n_results=5
    )

    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]

    # Build the context to restrict LLM to only retrieved clauses
    context = ""
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
        context += f"\n--- Source [Clause ID: {meta['clause_id']}] ---\n{doc}\n"

    system_prompt = f"""
    You are a legal contract assistant. Answer the user's question using ONLY the provided context below.

    RULES:
    1. If the answer cannot be found in the context, say: "I cannot find the answer in the provided contract clauses."
    2. Do not use anything outside the supplied context.
    3. Cite your sources using the [Clause ID: X] format as in the context.

    CONTEXT:
    {context}
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content, retrieved_docs

st.title("⚖️ Contract Clause Q&A")
st.markdown("Ask questions about the Sponsorship Agreement. Answers reference specific clauses only.")

load_and_embed_data()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What happens if ISO fails to fund the Reserve Account?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching contract clauses..."):
            answer, sources = generate_response(prompt)
            st.markdown(answer)
            with st.expander("View Retrieved Clauses"):
                for source in sources:
                    st.text(source)

    st.session_state.messages.append({"role": "assistant", "content": answer})