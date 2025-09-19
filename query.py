import json
import logging
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
# 1. Load persisted vector DB
# ----------------------------
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
db = Chroma(persist_directory=db_path, embedding_function=embedding_function)

print(f"Docs in DB: {len(db._collection.get(include=['metadatas'])['metadatas'])}")

# ----------------------------
# 2. Connect Llama 3 (Ollama)
# ----------------------------
llm_params = config.get("llm_params", {})
llm = Ollama(model=llm_model_name, **llm_params)

# ----------------------------
# 3. Create retriever
# ----------------------------
retriever = db.as_retriever(search_kwargs={"k": search_k})

# ----------------------------
# 4. Build RetrievalQA chain
# ----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ----------------------------
# 5. Interactive Q&A loop
# ----------------------------
while True:
    query = input("Enter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Goodbye 👋")
        break

    # Run QA
    result = qa_chain.invoke(query)
    answer = result["result"]
    sources = result.get("source_documents", [])

    # Print structured response
    print("\n========================")
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    print("\nSources & Referenced Text:")
    if sources:
        for i, doc in enumerate(sources, 1):
            print(f"  {i}. {doc.metadata.get('source', 'Unknown Source')}")
            print(f"     >>> {doc.page_content[:300]}...\n")  # show first 300 chars
    else:
        print("  No sources found.")
    print("========================\n")

    # Save plain log file
    logging.info(
        f"Q: {query}\nA: {answer}\nSources: {[doc.metadata.get('source', 'Unknown') for doc in sources]}\n"
    )

    # Save structured chat history (JSON)
    with open("chat_history.json", "a", encoding="utf-8") as f:
        json.dump(
            {
                "question": query,
                "answer": answer,
                "sources": [
                    {
                        "file": doc.metadata.get('source', 'Unknown'),
                        "excerpt": doc.page_content[:300]
                    } for doc in sources
                ]
            },
            f,
            ensure_ascii=False
        )
        f.write("\n")
