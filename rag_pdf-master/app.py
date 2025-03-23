import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)


from dotenv import load_dotenv
load_dotenv()  # Load environment variables before other imports

# Import our RAG utilities
from rag import DocumentRAG, OpenRouterLLM

# Set up page configuration
st.set_page_config(page_title="Document QA System", page_icon="📚", layout="wide")
st.title("📚 Document QA System")
st.subheader("Upload documents and ask questions about them")

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'file_names' not in st.session_state:
    st.session_state.file_names = []

# Sidebar for API key and model settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password")
    
    st.subheader("Document Processing")
    chunk_size = st.slider("Chunk Size", 100, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 10)
    
    st.subheader("Retrieval Settings")
    k_value = st.slider("Number of chunks to retrieve (k)", 1, 10, 6)
    
    # Add embedding model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-mpnet-base-v2"],
        index=0
    )
    
    # Add OpenRouter model selection
    or_model = st.selectbox(
        "OpenRouter Model",
        ["deepseek/deepseek-r1-zero:free", "anthropic/claude-3-haiku", "google/gemma-7b-it"],
        index=0
    )

# Function to identify document loader based on file extension
def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        return UnstructuredExcelLoader(file_path)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# Function to process documents
def process_documents(files):
    if not api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar.")
        return False
        
    documents = []
    file_names = []
    
    with st.spinner("Processing documents..."):
        for file in files:
            # Create a temporary file
            temp_dir = tempfile.TemporaryDirectory()
            temp_filepath = os.path.join(temp_dir.name, file.name)
            
            # Write the uploaded file to the temporary file
            with open(temp_filepath, "wb") as f:
                f.write(file.getbuffer())
            
            # Get the appropriate loader and load the documents
            loader = get_loader(temp_filepath)
            if loader:
                try:
                    documents.extend(loader.load())
                    file_names.append(file.name)
                    st.success(f"Successfully processed {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            # Clean up
            temp_dir.cleanup()
    
    if documents:
        st.info(f"Processing {len(documents)} document sections...")
        
        # Initialize RAG system if not already done
        if not st.session_state.rag_system:
            st.session_state.rag_system = DocumentRAG(api_key, embedding_model)
        
        # Process documents
        with st.spinner("Creating vector embeddings... This may take a few minutes."):
            success = st.session_state.rag_system.process_documents(
                documents, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            if success:
                st.session_state.documents_processed = True
                st.session_state.file_names = file_names
                return True
            else:
                st.error("Failed to process documents.")
                return False
    
    return False

# File uploading section
upload_container = st.container()
with upload_container:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT, CSV, XLS, XLSX)", 
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xls", "xlsx"]
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            success = process_documents(uploaded_files)
            if success:
                st.session_state.documents_processed = True

# Display processed files
if st.session_state.documents_processed:
    st.subheader("Processed Documents")
    for file_name in st.session_state.file_names:
        st.write(f"- {file_name}")

# Query section
query_container = st.container()
with query_container:
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question", placeholder="What is...")
    
    if not api_key and query:
        st.warning("Please enter your OpenRouter API Key in the sidebar.")
    
    if st.session_state.documents_processed and query and api_key:
        with st.spinner("Searching for relevant information and generating answer..."):
            try:
                # Update the model name if needed
                if hasattr(st.session_state.rag_system, 'model_name') and or_model:
                    st.session_state.rag_system.model_name = or_model
                
                # Query the RAG system
                response, docs = st.session_state.rag_system.query(query, k=k_value)
                
                # Display retrieved chunks
                with st.expander("View Retrieved Chunks"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
                
                # Display answer
                st.subheader("Answer")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    elif query and not st.session_state.documents_processed:
        st.warning("Please upload and process documents first.")

# Add explanation and instructions
with st.sidebar:
    st.markdown("---")
    st.subheader("How to use this app")
    st.markdown("""
    1. Enter your OpenRouter API key
    2. Upload document files (PDF, DOCX, TXT, CSV, XLS, XLSX)
    3. Click 'Process Documents' to analyze and index them
    4. Ask questions about the content in your documents
    """)
    
    st.markdown("---")
    st.markdown("This app uses:")
    st.markdown("- LangChain for document processing and RAG pipeline")
    st.markdown("- HuggingFace embeddings for semantic search")
    st.markdown("- OpenRouter API for generating responses")
    st.markdown("- Chroma vector database for storing embeddings")