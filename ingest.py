import os
import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------
# Step 1: Load Multiple PDFs & DOCX
# --------------------------
data_path = r"C:\Users\viraj\llm-chatbot\data"  

pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
docx_files = glob.glob(os.path.join(data_path, "*.docx"))

all_documents = []

# Load PDFs
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {os.path.basename(pdf_path)}")
    all_documents.extend(documents)

# Load DOCX
for docx_path in docx_files:
    loader = Docx2txtLoader(docx_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} sections from {os.path.basename(docx_path)}")
    all_documents.extend(documents)

print(f"\n✅ Total documents loaded: {len(all_documents)} from {len(pdf_files)+len(docx_files)} files")

# --------------------------
# Step 2: Split Text into Chunks
# --------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(all_documents)

print(f"✅ Created {len(docs)} chunks in total")

# --------------------------
# Step 3: Create Embeddings
# --------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------
# Step 4: Store in Vector DB (Chroma)
# --------------------------
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="vector_db"
)

vectorstore.persist()
print("🎉 Vector DB created and saved in /vector_db")
