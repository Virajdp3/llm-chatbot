import json
import logging
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Load configuration
# ----------------------------
with open("config.json", "r") as f:
    config = json.load(f)

embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
db_path = config.get("db_path", "vector_db")
llm_model_name = config.get("llm_model", "llama3")
search_k = config.get("search_k", 3)
log_file = config.get("log_file", "qa_logs.txt")

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ----------------------------
# Load persisted vector DB
# ----------------------------
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
db = Chroma(persist_directory=db_path, embedding_function=embedding_function)

# ----------------------------
# Connect Llama 3 (Ollama)
# ----------------------------
llm = Ollama(model=llm_model_name)

# ----------------------------
# Create retriever and QA chain
# ----------------------------
retriever = db.as_retriever(search_kwargs={"k": search_k})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Local Chatbot", layout="wide")
st.title("💬 Local Document Chatbot")

# Store history in session
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question:", "")

if st.button("Submit") and query:
    result = qa_chain.invoke(query)
    answer = result["result"]
    sources = result.get("source_documents", [])

    # Save to history
    st.session_state.history.append({
        "question": query,
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "excerpt": doc.page_content[:300]
            } for doc in sources
        ]
    })

# Display history
for chat in reversed(st.session_state.history):
    st.markdown(f"**❓ Question:** {chat['question']}")
    st.markdown(f"**💡 Answer:** {chat['answer']}")

    with st.expander("📂 Sources & Excerpts"):
        for src in chat["sources"]:
            st.markdown(f"- **File:** {src['file']}")
            st.markdown(f"  > {src['excerpt']}...")

# Download chat history
if st.session_state.history:
    history_json = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download Chat History",
        data=history_json,
        file_name="chat_history.json",
        mime="application/json"
    )
