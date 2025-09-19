from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Reload the same embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing vector DB
vectorstore = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

# Ask a test question
query = "What are the unique selling points of StrategicERP?"
results = vectorstore.similarity_search(query, k=3)

print("\n🔎 Query:", query)
print("\n📄 Top Matches:")
for idx, doc in enumerate(results, start=1):
    print(f"\nResult {idx}:")
    print(doc.page_content[:300], "...")
