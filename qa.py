from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

# Load embeddings (same as before)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector store
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Simple prompt
prompt = PromptTemplate.from_template(
    """
    Answer the question ONLY using the context below.
    If you don't know, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

# Local free LLM (Ollama)
llm = Ollama(model="llama2")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Ask question
query = "What is machine learning?"
result = chain.invoke(query)

print("\nQuestion:", query)
print("\nAnswer:", result)
