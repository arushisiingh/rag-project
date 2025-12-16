from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. PDF load karo
pdf_path = "data/pdfs/ml_notes.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Text ko chunks me todo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. FREE embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. FAISS vector database banao
vector_store = FAISS.from_documents(chunks, embeddings)

# 5. Database save karo
vector_store.save_local("faiss_index")

print("✅ Vector store successfully created (FREE embeddings)")
