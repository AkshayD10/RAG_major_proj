
# import os
# import tempfile
# import streamlit as st
# from langchain_community.document_loaders import (
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     CSVLoader,
#     UnstructuredExcelLoader
# )

# # Import RAG utilities
# from rag import DocumentRAG, OpenRouterLLM 

# # Set up page configuration
# st.set_page_config(page_title="Document QA System", page_icon="📚", layout="wide")
# st.title("📚 Document QA System")
# st.subheader("Upload documents and ask questions about them")

# # Create a lightweight requirements.txt file with dependencies needed
# if not os.path.exists("requirements.txt"):
#     with open("requirements.txt", "w") as f:
#         f.write("""
# langchain>=0.1.0
# langchain-community>=0.0.11
# streamlit>=1.24.0
# sentence-transformers>=2.2.2
# chromadb>=0.4.22
# pydantic>=2.5.0
# unstructured>=0.10.30
# pdf2image>=1.16.3
# python-docx>=0.8.11
# python-dotenv>=1.0.0
# openpyxl>=3.1.2
# docx2txt>=0.8
# requests>=2.31.0
# """)

# # Initialize session state with better error handling
# if 'rag_system' not in st.session_state:
#     st.session_state.rag_system = None
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'file_names' not in st.session_state:
#     st.session_state.file_names = []
# if 'processing_error' not in st.session_state:
#     st.session_state.processing_error = None

# # Add debug mode toggle in sidebar
# with st.sidebar:
#     st.header("Configuration")
#     debug_mode = st.checkbox("Debug Mode", value=False)
#     api_key = st.text_input("OpenRouter API Key", type="password")
    
#     st.subheader("Document Processing")
#     chunk_size = st.slider("Chunk Size", 100, 2000, 500, 100)  # Reduced default to 500
#     chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, 10)  # Reduced default to 100
    
#     st.subheader("Retrieval Settings")
#     k_value = st.slider("Number of chunks to retrieve (k)", 1, 10, 4)  # Reduced default to 4
    
#     # Add embedding model selection
#     embedding_model = st.selectbox(
#         "Embedding Model",
#         [
#             "sentence-transformers/all-mpnet-base-v2"
#               # Smaller, faster model option
#         ],
#         index=0  # Default to the smaller model
#     )
    
#     # Add OpenRouter model selection
#     or_model = st.selectbox(
#         "OpenRouter Model",
#         ["deepseek/deepseek-r1-zero:free"],
#         index=0
#     )

# # Function to identify document loader based on file extension
# def get_loader(file_path):
#     ext = os.path.splitext(file_path)[1].lower()
#     try:
#         if ext == ".pdf":
#             return PyPDFLoader(file_path)
#         elif ext == ".docx":
#             return Docx2txtLoader(file_path)
#         elif ext == ".txt":
#             return TextLoader(file_path)
#         elif ext == ".csv":
#             return CSVLoader(file_path)
#         elif ext in [".xls", ".xlsx"]:
#             return UnstructuredExcelLoader(file_path)
#         else:
#             st.error(f"Unsupported file format: {ext}")
#             return None
#     except Exception as e:
#         st.error(f"Error creating loader for {ext} file: {str(e)}")
#         return None

# # Function to process documents with better error handling
# def process_documents(files):
#     if not api_key:
#         st.error("Please enter your OpenRouter API Key in the sidebar.")
#         return False
        
#     documents = []
#     file_names = []
#     error_logs = []
    
#     with st.spinner("Processing documents..."):
#         for file in files:
#             # Create a temporary file
#             temp_dir = tempfile.TemporaryDirectory()
#             temp_filepath = os.path.join(temp_dir.name, file.name)
            
#             try:
#                 # Write the uploaded file to the temporary file
#                 with open(temp_filepath, "wb") as f:
#                     f.write(file.getbuffer())
                
#                 # Get the appropriate loader and load the documents
#                 loader = get_loader(temp_filepath)
#                 if loader:
#                     try:
#                         docs = loader.load()
#                         if docs:
#                             documents.extend(docs)
#                             file_names.append(file.name)
#                             st.success(f"Successfully processed {file.name}")
#                         else:
#                             error_msg = f"No content extracted from {file.name}"
#                             error_logs.append(error_msg)
#                             st.warning(error_msg)
#                     except Exception as e:
#                         error_msg = f"Error processing {file.name}: {str(e)}"
#                         error_logs.append(error_msg)
#                         st.error(error_msg)
#             except Exception as e:
#                 error_msg = f"Error handling {file.name}: {str(e)}"
#                 error_logs.append(error_msg)
#                 st.error(error_msg)
#             finally:
#                 # Clean up
#                 temp_dir.cleanup()
    
#     if documents:
#         total_chars = sum(len(doc.page_content) for doc in documents)
#         st.info(f"Processing {len(documents)} document sections with {total_chars} total characters...")
        
#         if debug_mode:
#             st.write(f"Document count: {len(documents)}")
#             st.write(f"Total characters: {total_chars}")
        
#         # Initialize RAG system with temporary directory for cloud deployment
#         if not st.session_state.rag_system:
#             st.session_state.rag_system = DocumentRAG(
#                 api_key, 
#                 embedding_model,
#                 persist_directory=None  # Use temp directory in the cloud
#             )
        
#         # Process documents with more detailed error info
#         with st.spinner("Creating vector embeddings... This may take a few minutes."):
#             try:
#                 success = st.session_state.rag_system.process_documents(
#                     documents, 
#                     chunk_size=chunk_size, 
#                     chunk_overlap=chunk_overlap
#                 )
                
#                 if success:
#                     st.session_state.documents_processed = True
#                     st.session_state.file_names = file_names
#                     st.session_state.processing_error = None
#                     return True
#                 else:
#                     error_msg = "Failed to process documents. Check that your files contain enough text content."
#                     st.session_state.processing_error = error_msg
#                     st.error(error_msg)
#                     return False
#             except Exception as e:
#                 import traceback
#                 error_msg = f"Error during document processing: {str(e)}"
#                 if debug_mode:
#                     error_msg += f"\n\nTraceback: {traceback.format_exc()}"
#                 st.session_state.processing_error = error_msg
#                 st.error(error_msg)
#                 return False
#     else:
#         error_msg = "No documents were successfully processed. Please check your files."
#         if error_logs:
#             error_msg += "\n\nErrors encountered:"
#             for log in error_logs:
#                 error_msg += f"\n- {log}"
#         st.session_state.processing_error = error_msg
#         st.error(error_msg)
#         return False

# # File uploading section with size warnings
# upload_container = st.container()
# with upload_container:
#     st.subheader("Upload Documents")
    
#     # Add file size warning
#     st.info("📌 Note: For cloud deployment, keep your files small (ideally < 5MB total) to avoid memory issues.")
    
#     uploaded_files = st.file_uploader(
#         "Choose files (PDF, DOCX, TXT, CSV, XLS, XLSX)", 
#         accept_multiple_files=True,
#         type=["pdf", "docx", "txt", "csv", "xls", "xlsx"]
#     )
    
#     if uploaded_files:
#         total_size_mb = sum(file.size for file in uploaded_files) / (1024 * 1024)
#         if total_size_mb > 10:
#             st.warning(f"Total file size is {total_size_mb:.1f}MB. Large files may cause memory issues on the cloud deployment.")
        
#         if st.button("Process Documents"):
#             success = process_documents(uploaded_files)
#             if success:
#                 st.session_state.documents_processed = True

# # Display processed files and errors
# if st.session_state.documents_processed:
#     st.subheader("Processed Documents")
#     for file_name in st.session_state.file_names:
#         st.write(f"- {file_name}")
# elif st.session_state.processing_error and debug_mode:
#     st.subheader("Processing Error Details")
#     st.code(st.session_state.processing_error)

# # Query section
# query_container = st.container()
# with query_container:
#     st.subheader("Ask a Question")
#     query = st.text_input("Enter your question", placeholder="What is...")
    
#     if not api_key and query:
#         st.warning("Please enter your OpenRouter API Key in the sidebar.")
    
#     if st.session_state.documents_processed and query and api_key:
#         with st.spinner("Searching for relevant information and generating answer..."):
#             try:
#                 # Update the model name if needed
#                 if hasattr(st.session_state.rag_system, 'model_name') and or_model:
#                     st.session_state.rag_system.model_name = or_model
                
#                 # Query the RAG system
#                 response, docs = st.session_state.rag_system.query(query, k=k_value)
                
#                 # Display retrieved chunks
#                 with st.expander("View Retrieved Chunks"):
#                     for i, doc in enumerate(docs):
#                         st.markdown(f"**Chunk {i+1}:**")
#                         st.markdown(doc.page_content)
#                         st.markdown("---")
                
#                 # Display answer
#                 st.subheader("Answer")
#                 st.markdown(response)
                
#             except Exception as e:
#                 import traceback
#                 error_msg = f"Error generating response: {str(e)}"
#                 if debug_mode:
#                     error_msg += f"\n\nTraceback: {traceback.format_exc()}"
#                 st.error(error_msg)
#     elif query and not st.session_state.documents_processed:
#         st.warning("Please upload and process documents first.")

# # Add explanation and instructions
# with st.sidebar:
#     st.markdown("---")
#     st.subheader("How to use this app")
#     st.markdown("""
#     1. Enter your OpenRouter API key
#     2. Upload document files (PDF, DOCX, TXT, CSV, XLS, XLSX)
#     3. Click 'Process Documents' to analyze and index them
#     4. Ask questions about the content in your documents
#     """)
    
    
#     st.markdown("---")
#     st.markdown("This app uses:")
#     st.markdown("- LangChain for document processing and RAG pipeline")
#     st.markdown("- HuggingFace embeddings for semantic search")
#     st.markdown("- OpenRouter API for generating responses")
#     st.markdown("- Chroma vector database for storing embeddings")



#using FIASS 
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

# Import RAG utilities
from rag import DocumentRAG, OpenRouterLLM 

# Set up page configuration
st.set_page_config(page_title="Namd RAG", page_icon="📚", layout="wide")
st.title("📚 Namd RAG")
st.subheader("Upload documents and ask questions about them")

# Create a lightweight requirements.txt file with dependencies needed
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("""
langchain>=0.1.0
langchain-community>=0.0.11
streamlit>=1.24.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
pydantic>=2.5.0
unstructured>=0.10.30
pdf2image>=1.16.3
python-docx>=0.8.11
python-dotenv>=1.0.0
openpyxl>=3.1.2
docx2txt>=0.8
requests>=2.31.0
""")

# Initialize session state with better error handling
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'file_names' not in st.session_state:
    st.session_state.file_names = []
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None

# Add debug mode toggle in sidebar
with st.sidebar:
    st.header("Configuration")
    debug_mode = st.checkbox("Debug Mode", value=False)
    api_key = st.text_input("OpenRouter API Key", type="password")
    
    # Persistence options
    persistence_path = st.text_input("Persistence Directory (optional)", value="", 
                                    help="Leave empty to use a temporary directory or enter a path to save your FAISS index for future use")
    
    st.subheader("Document Processing")
    chunk_size = st.slider("Chunk Size", 100, 2000, 500, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, 10)
    
    st.subheader("Retrieval Settings")
    k_value = st.slider("Number of chunks to retrieve (k)", 1, 10, 4)
    
    # Add embedding model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-mpnet-base-v2"
        ],
        index=0
    )
    
    # Add OpenRouter model selection
    or_model = st.selectbox(
        "OpenRouter Model",
        ["deepseek/deepseek-r1-zero:free"],
        index=0
    )

# Function to identify document loader based on file extension
def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
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
    except Exception as e:
        st.error(f"Error creating loader for {ext} file: {str(e)}")
        return None

# Function to process documents with better error handling
def process_documents(files):
    if not api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar.")
        return False
        
    documents = []
    file_names = []
    error_logs = []
    
    with st.spinner("Processing documents..."):
        for file in files:
            # Create a temporary file
            temp_dir = tempfile.TemporaryDirectory()
            temp_filepath = os.path.join(temp_dir.name, file.name)
            
            try:
                # Write the uploaded file to the temporary file
                with open(temp_filepath, "wb") as f:
                    f.write(file.getbuffer())
                
                # Get the appropriate loader and load the documents
                loader = get_loader(temp_filepath)
                if loader:
                    try:
                        docs = loader.load()
                        if docs:
                            documents.extend(docs)
                            file_names.append(file.name)
                            st.success(f"Successfully processed {file.name}")
                        else:
                            error_msg = f"No content extracted from {file.name}"
                            error_logs.append(error_msg)
                            st.warning(error_msg)
                    except Exception as e:
                        error_msg = f"Error processing {file.name}: {str(e)}"
                        error_logs.append(error_msg)
                        st.error(error_msg)
            except Exception as e:
                error_msg = f"Error handling {file.name}: {str(e)}"
                error_logs.append(error_msg)
                st.error(error_msg)
            finally:
                # Clean up
                temp_dir.cleanup()
    
    if documents:
        total_chars = sum(len(doc.page_content) for doc in documents)
        st.info(f"Processing {len(documents)} document sections with {total_chars} total characters...")
        
        if debug_mode:
            st.write(f"Document count: {len(documents)}")
            st.write(f"Total characters: {total_chars}")
        
        # Initialize RAG system with persistence directory if provided
        if not st.session_state.rag_system:
            # Use provided persistence path or a temp directory
            persist_dir = persistence_path if persistence_path else None
            st.session_state.rag_system = DocumentRAG(
                api_key, 
                embedding_model,
                persist_directory=persist_dir
            )
        
        # Process documents with more detailed error info
        with st.spinner("Creating FAISS vector embeddings... This may take a few minutes."):
            try:
                success = st.session_state.rag_system.process_documents(
                    documents, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                
                if success:
                    st.session_state.documents_processed = True
                    st.session_state.file_names = file_names
                    st.session_state.processing_error = None
                    # Show persistence info if path was provided
                    if persistence_path:
                        st.success(f"Document embeddings saved to {persistence_path} for future use")
                    return True
                else:
                    error_msg = "Failed to process documents. Check that your files contain enough text content."
                    st.session_state.processing_error = error_msg
                    st.error(error_msg)
                    return False
            except Exception as e:
                import traceback
                error_msg = f"Error during document processing: {str(e)}"
                if debug_mode:
                    error_msg += f"\n\nTraceback: {traceback.format_exc()}"
                st.session_state.processing_error = error_msg
                st.error(error_msg)
                return False
    else:
        error_msg = "No documents were successfully processed. Please check your files."
        if error_logs:
            error_msg += "\n\nErrors encountered:"
            for log in error_logs:
                error_msg += f"\n- {log}"
        st.session_state.processing_error = error_msg
        st.error(error_msg)
        return False

# Check for existing index
if persistence_path and not st.session_state.rag_system and os.path.exists(os.path.join(persistence_path, "faiss_index")):
    st.info(f"Found existing FAISS index at {persistence_path}. Loading...")
    try:
        st.session_state.rag_system = DocumentRAG(
            api_key="" if not api_key else api_key,  # Will be updated when actually querying
            embedding_model=embedding_model,
            persist_directory=persistence_path
        )
        st.session_state.documents_processed = True
        st.success("Existing FAISS index loaded successfully. You can now ask questions.")
    except Exception as e:
        st.error(f"Error loading existing index: {str(e)}")

# File uploading section with size warnings
upload_container = st.container()
with upload_container:
    st.subheader("Upload Documents")
    
    # Add file size warning
    # st.info("📌 Note: For cloud deployment, keep your files small (ideally < 5MB total) to avoid memory issues.")
    
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT, CSV, XLS, XLSX)", 
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xls", "xlsx"]
    )
    
    if uploaded_files:
        total_size_mb = sum(file.size for file in uploaded_files) / (1024 * 1024)
        if total_size_mb > 10:
            st.warning(f"Total file size is {total_size_mb:.1f}MB. Large files may cause memory issues on the cloud deployment.")
        
        if st.button("Process Documents"):
            success = process_documents(uploaded_files)
            if success:
                st.session_state.documents_processed = True

# Display processed files and errors
if st.session_state.documents_processed:
    st.subheader("Processed Documents")
    for file_name in st.session_state.file_names:
        st.write(f"- {file_name}")
elif st.session_state.processing_error and debug_mode:
    st.subheader("Processing Error Details")
    st.code(st.session_state.processing_error)

# Query section
query_container = st.container()
with query_container:
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question", placeholder="What is...")
    
    if not api_key and query:
        st.warning("Please enter your OpenRouter API Key in the sidebar.")
    
    # Update API key in case it was changed
    if st.session_state.rag_system and api_key:
        st.session_state.rag_system.api_key = api_key
    
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
                import traceback
                error_msg = f"Error generating response: {str(e)}"
                if debug_mode:
                    error_msg += f"\n\nTraceback: {traceback.format_exc()}"
                st.error(error_msg)
    elif query and not st.session_state.documents_processed:
        st.warning("Please upload and process documents first.")

# Add explanation and instructions
with st.sidebar:
    st.markdown("---")
    st.subheader("How to use this app")
    st.markdown("""
    1. Enter your OpenRouter API key
    2. (Optional) Specify a persistence directory to save your FAISS index
    3. Upload document files (PDF, DOCX, TXT, CSV, XLS, XLSX)
    4. Click 'Process Documents' to analyze and index them
    5. Ask questions about the content in your documents
    """)
    
    st.markdown("---")
    st.markdown("This app uses:")
    st.markdown("- LangChain for document processing and RAG pipeline")
    st.markdown("- FAISS for efficient similarity search")
    st.markdown("- HuggingFace embeddings for semantic search")
    st.markdown("- OpenRouter API for generating responses")