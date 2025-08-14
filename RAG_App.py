import streamlit as st
from dotenv import load_dotenv
import os
import time

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file if present
load_dotenv()

# Load Groq API key securely from Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", None)
if not groq_api_key:
    st.error("GROQ_API_KEY not found in secrets. Please add it to .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

# Configure Streamlit page
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")

# Try to show logo image - fallback if missing
try:
    st.image("PragyanAI_Transparent.png", width=200)
except FileNotFoundError:
    st.warning("Logo image not found, continuing without logo.")

st.title("Dynamic RAG with Groq, FAISS, and Llama3")

# Initialize session state variables
if "vector" not in st.session_state:
    st.session_state.vector = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Document upload & processing
with st.sidebar:
    st.header("Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF files", type="pdf", accept_multiple_files=True
    )
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for file in uploaded_files:
                    # Save uploaded file temporarily
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())

                    loader = PyPDFLoader(file.name)
                    docs.extend(loader.load())

                # Split documents into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                split_docs = splitter.split_documents(docs)

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                # Create FAISS vector store and save in session state
                st.session_state.vector = FAISS.from_documents(split_docs, embeddings)

                st.success("Documents processed and vector store created!")
        else:
            st.warning("Please upload at least one PDF document.")

# Main chat interface
st.header("Chat with your Documents")

# Define the prompt template with system + human messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who answers questions based solely on the provided context."
        ),
        (
            "human",
            "Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion:\n{input}"
        ),
    ]
)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user question input
if prompt_input := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector is None:
        st.warning("Please upload and process documents first!")
    else:
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt_input)
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})

        # Initialize Groq LLM with correct model name (check your Groq dashboard for exact name)
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # Build document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Invoke the chain and time it
        with st.spinner("Thinking..."):
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": prompt_input})
            elapsed = time.process_time() - start_time

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

        st.info(f"Response time: {elapsed:.2f} seconds")
