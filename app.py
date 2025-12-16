import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG AI Assistant")

st.title("📄 AI Document Question Answering (RAG)")
st.write("Ask questions from your uploaded ML notes")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector store
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = st.text_input("Ask a question from the document:")

if query:
    docs = vector_store.similarity_search(query, k=3)

    st.subheader("Answer (based on documents):")
    for i, doc in enumerate(docs):
        st.markdown(f"**Source {i+1}:**")
        st.write(doc.page_content[:500])
