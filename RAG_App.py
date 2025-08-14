import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Load environment variables
load_dotenv()

# Set up Groq API key
groq_api_key = st.secrets["GROQ_API_KEY"]

# Set up Streamlit page config
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.image("PragyanAI_Transparent.png")
st.title("Dynamic RAG with Groq, FAISS, and Llama3")

# Initialize session state for vector store and chat history
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for file in uploaded_files:
                    # Write file to a temporary location
                    with open(file.name, "wb") as fo:
                        fo.write(file.getbuffer())
                    loader = PyPDFLoader(file.name)
                    docs.extend(loader.load())

                # Split the documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1888, chunk_overlap=288)
                final_documents = text_splitter.split_documents(docs)

                # Use pre-trained model from Hugging Face for embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                # Create the vector store using FAISS
                st.session_state.vector = FAISS.from_documents(final_documents, embeddings)

            st.success("Documents processed successfully!")
        else:
            st.warning("Please upload at least one document.")

# Main chat interface
st.header("Chat with your Documents")

# Create the prompt template
prompt_template = """
Answer the questions based on the provided context only.

Please provide the most accurate response based on the question.

<context>

{context}

Questions: {input}
"""

# Display previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt_input := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector is not None:
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})
        
        # Processing the question
        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(Lim, prompt_template)
            retriever = st.session_state.vector.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start_time = time.process_time()
            
            # Get the response
            response = retrieval_chain.invoke({"input": prompt_input})
            
            # Measure the response time
            response_time = time.process_time() - start_time
            
            # Display the assistant's response
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            
            # Display response time
            st.info(f"Response time: {response_time:.2f} seconds")
            
            # Update chat history with the assistant's response
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
    else:
        st.warning("Please process your documents before asking questions.")
