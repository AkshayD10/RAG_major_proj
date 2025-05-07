# import os
# import tempfile
# import streamlit as st
# from typing import List, Optional, Any
# from langchain_community.document_loaders import (
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     CSVLoader,
#     UnstructuredExcelLoader # Ensure unstructured[xlsx] is installed
# )
# from langchain.docstore.document import Document # Import Document for type hinting

# # Import RAG utilities
# # Ensure rag.py is in the same directory or Python path
# try:
#     from rag import DocumentRAG
# except ImportError:
#     st.error("Error: Could not import DocumentRAG from rag.py. Make sure the file exists and is in the correct path.")
#     # Add dummy class to prevent further errors during rendering
#     class DocumentRAG:
#         def __init__(self, *args, **kwargs): pass
#         def process_documents(self, *args, **kwargs): return False
#         def query(self, *args, **kwargs): return "RAG system not loaded.", []

# # Set up page configuration
# st.set_page_config(page_title="Namd RAG Enhanced", page_icon="📚", layout="wide")
# st.title("📚 Namd RAG - Enhanced")
# st.subheader("Hybrid Retrieval, Dynamic Thresholding, and Confidence Scoring")

# # Initialize session state with better error handling
# if 'rag_system' not in st.session_state:
#     st.session_state.rag_system = None
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'processed_files_info' not in st.session_state:
#     # Store info about processed files (name, number of splits)
#     st.session_state.processed_files_info = {}
# if 'processing_error' not in st.session_state:
#     st.session_state.processing_error = None
# if 'index_loaded_from_path' not in st.session_state:
#     st.session_state.index_loaded_from_path = False


# # --- Sidebar Configuration ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     api_key = st.text_input("OpenRouter API Key", type="password", help="Required for generating answers.")

#     # Debug Mode
#     debug_mode = st.checkbox("🐛 Debug Mode", value=False, help="Show more detailed logs and errors.")

#     # Persistence
#     st.subheader("💾 Persistence")
#     persistence_path = st.text_input("Persistence Directory (Optional)", value="",
#                                      help="Enter a path to save/load the FAISS index and document splits. Leave empty to use a temporary directory (index lost on app restart).")

#     # LLM Selection
#     st.subheader("🧠 Language Model")
#     or_model = st.selectbox(
#         "OpenRouter Model",
#         # Add more models as needed from OpenRouter
#         ["deepseek/deepseek-r1-zero:free", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct", "google/gemini-flash-1.5"],
#         index=0, # Default to deepseek-r1-zero:free
#         help="Select the model for answer generation."
#     )
#     llm_temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.05, help="Controls randomness. Lower is more focused, higher is more creative.")


#     # Document Processing Settings
#     st.subheader("📄 Document Processing")
#     chunk_size = st.slider("Chunk Size", 100, 4000, 1000, 100, help="Size of text chunks for embedding.")
#     chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 150, 10, help="Overlap between chunks.")

#     # Retrieval Settings
#     st.subheader("🔍 Retrieval Settings")
#     k_value = st.slider("Initial K", 1, 20, 10, 1, help="Retrieve top K docs *each* from dense (FAISS) and sparse (BM25) search initially.")
#     threshold_percentile = st.slider("Confidence Threshold Percentile", 0, 100, 25, 5, help="Keep documents with scores above this percentile of retrieved scores (lower = keep more).")
#     max_final_docs = st.slider("Max Docs for LLM", 1, 15, 4, 1, help="Maximum number of filtered documents to send to the LLM.")


#     # Embedding Model (Consider making this less prominent if only one is well-tested)
#     st.subheader("🔡 Embedding Model")
#     embedding_model = st.selectbox(
#         "Embedding Model",
#         ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"],
#         index=0,
#         help="Model used to create vector embeddings for dense search."
#     )

#     st.markdown("---")
#     st.subheader("ℹ️ How to Use")
#     st.markdown("""
#     1.  **Enter API Key:** Add your OpenRouter key.
#     2.  **(Optional) Persistence:** Set a directory to save the index for reuse.
#     3.  **Upload:** Add PDF, DOCX, TXT, CSV, XLS, or XLSX files.
#     4.  **Process:** Click 'Process Documents'. This creates embeddings and initializes retrievers.
#     5.  **Ask:** Enter your question and get an answer based on the documents.
#     """)
#     st.markdown("---")
#     st.info("Using temporary storage by default. Set a Persistence Directory to save your index.")


# # --- Core Logic Functions ---

# # Function to identify document loader based on file extension
# def get_loader(file_path: str, file_ext: str) -> Optional[Any]:
#     """Gets the appropriate Langchain document loader."""
#     try:
#         if file_ext == ".pdf":
#             return PyPDFLoader(file_path)
#         elif file_ext == ".docx":
#             return Docx2txtLoader(file_path)
#         elif file_ext == ".txt":
#             return TextLoader(file_path, autodetect_encoding=True) # Add encoding detection
#         elif file_ext == ".csv":
#             # Consider adding CSVLoader options if needed (e.g., column selection)
#             return CSVLoader(file_path, autodetect_encoding=True)
#         elif file_ext in [".xls", ".xlsx"]:
#             # Ensure 'unstructured' and 'openpyxl'/'pypdf' are installed
#             # May require additional dependencies like 'libreoffice-calc' on Linux for older XLS
#             return UnstructuredExcelLoader(file_path, mode="elements") # "elements" mode often works better
#         else:
#             st.error(f"Unsupported file format: {file_ext}")
#             return None
#     except ImportError as ie:
#          if "unstructured" in str(ie) and file_ext in [".xls", ".xlsx"]:
#               st.error(f"Please install 'unstructured[xlsx]' to process Excel files: pip install unstructured[xlsx]")
#          else:
#               st.error(f"Missing dependency for {file_ext} files: {str(ie)}")
#          return None
#     except Exception as e:
#         st.error(f"Error creating loader for {file_path}: {str(e)}")
#         return None

# # Function to process documents with better error handling
# def process_uploaded_documents(files: List[st.runtime.uploaded_file_manager.UploadedFile]):
#     """Loads, splits, and indexes the uploaded documents."""
#     if not api_key:
#         st.error("🚨 Please enter your OpenRouter API Key in the sidebar.")
#         return False
#     if not files:
#         st.warning("No files selected for processing.")
#         return False

#     st.session_state.processing_error = None # Reset error state
#     st.session_state.documents_processed = False # Reset processed state
#     st.session_state.processed_files_info = {} # Reset file info

#     # Initialize RAG system (if not already loaded from persistence)
#     # Crucially, pass the selected LLM model and temperature here
#     if not st.session_state.rag_system:
#         persist_dir = persistence_path if persistence_path else None
#         try:
#             st.session_state.rag_system = DocumentRAG(
#                 openrouter_api_key=api_key,
#                 embedding_model=embedding_model,
#                 persist_directory=persist_dir,
#                 llm_model_name=or_model,
#                 llm_temperature=llm_temperature
#             )
#             # Check if it loaded something (covers the case where persist_dir was set but empty/invalid)
#             if st.session_state.rag_system.vectorstore and st.session_state.rag_system.bm25_retriever:
#                  st.session_state.index_loaded_from_path = True # Mark as loaded
#                  st.info("Existing index loaded. Adding new documents to it.")
#             else:
#                  st.session_state.index_loaded_from_path = False

#         except Exception as e:
#             st.error(f"🚨 Failed to initialize RAG system: {e}")
#             st.session_state.rag_system = None # Ensure it's None on failure
#             return False

#     all_docs: List[Document] = []
#     processed_file_names = []
#     error_logs = []

#     with st.spinner(f"Loading {len(files)} file(s)..."):
#         for file in files:
#             file_ext = os.path.splitext(file.name)[1].lower()
#             # Use a robust temp file handling approach
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                     tmp_file.write(file.getvalue())
#                     tmp_filepath = tmp_file.name

#                 loader = get_loader(tmp_filepath, file_ext)
#                 if loader:
#                     try:
#                         docs = loader.load() # This might return multiple docs for one file
#                         if docs:
#                             # Add source metadata BEFORE splitting
#                             for doc in docs:
#                                 doc.metadata["source"] = file.name
#                             all_docs.extend(docs)
#                             processed_file_names.append(file.name)
#                             st.write(f"📄 Loaded {file.name} ({len(docs)} section(s))")
#                         else:
#                             msg = f"⚠️ No content extracted from {file.name}"
#                             st.warning(msg)
#                             error_logs.append(msg)
#                     except Exception as e:
#                         msg = f"❌ Error processing {file.name}: {str(e)}"
#                         st.error(msg)
#                         error_logs.append(msg)
#                         if debug_mode:
#                             st.exception(e)
#                 os.unlink(tmp_filepath) # Clean up the temp file

#             except Exception as e:
#                  msg = f"❌ Error handling file {file.name}: {str(e)}"
#                  st.error(msg)
#                  error_logs.append(msg)
#                  if debug_mode:
#                      st.exception(e)
#                  # Ensure cleanup if tempfile creation failed mid-way
#                  if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
#                      os.unlink(tmp_filepath)


#     if not all_docs:
#         error_msg = "No documents could be loaded or extracted. Please check file formats and content."
#         st.error(error_msg)
#         st.session_state.processing_error = error_msg + "\n" + "\n".join(error_logs)
#         return False

#     st.info(f"Loaded {len(all_docs)} document sections from {len(processed_file_names)} files. Now processing...")

#     # Now process the loaded documents using the RAG system instance
#     with st.spinner("🧠 Creating embeddings and indexing... (This might take a while for large documents)"):
#         try:
#             # Pass processing parameters from UI
#             success = st.session_state.rag_system.process_documents(
#                 all_docs,
#                 chunk_size=chunk_size,
#                 chunk_overlap=chunk_overlap
#             )

#             if success:
#                 st.success("✅ Documents processed and indexed successfully!")
#                 st.session_state.documents_processed = True
#                 # Store info about processed files and split counts
#                 st.session_state.processed_files_info = {name: "Processed" for name in processed_file_names}
#                 if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
#                      st.session_state.processed_files_info["total_splits"] = len(st.session_state.rag_system.splits)
#                      st.info(f"Total splits created: {len(st.session_state.rag_system.splits)}")

#                 if persistence_path and not st.session_state.rag_system._using_temp_dir: # Check if saving happened
#                      st.success(f"✅ Index and splits saved to {persistence_path}")
#                 return True
#             else:
#                 error_msg = "⚠️ Failed to process documents into the index. Check logs above."
#                 st.error(error_msg)
#                 st.session_state.processing_error = error_msg + "\n" + "\n".join(error_logs)
#                 return False
#         except Exception as e:
#             import traceback
#             error_msg = f"🚨 Critical error during document processing: {str(e)}"
#             st.error(error_msg)
#             if debug_mode:
#                  st.code(traceback.format_exc())
#             st.session_state.processing_error = error_msg + "\n" + traceback.format_exc()
#             return False


# # --- Attempt to load index on startup if persistence path is set ---
# if persistence_path and not st.session_state.rag_system and not st.session_state.get('attempted_load', False):
#     st.session_state.attempted_load = True # Ensure this runs only once per session start
#     st.info(f"Checking for existing index in {persistence_path}...")
#     # Need API key even for loading FAISS with HuggingFaceEmbeddings
#     if not api_key:
#          st.warning("API Key needed to initialize embedding model for loading index. Please enter it.")
#     else:
#         try:
#             # Initialize RAG system to trigger loading logic
#             st.session_state.rag_system = DocumentRAG(
#                 openrouter_api_key=api_key,
#                 embedding_model=embedding_model,
#                 persist_directory=persistence_path,
#                 llm_model_name=or_model, # Pass current UI selection
#                 llm_temperature=llm_temperature
#             )
#             # _load_index_and_splits is called in __init__
#             if st.session_state.rag_system.vectorstore and st.session_state.rag_system.bm25_retriever:
#                 st.session_state.documents_processed = True
#                 st.session_state.index_loaded_from_path = True
#                 st.success(f"✅ Existing index and splits loaded successfully from {persistence_path}!")
#                 # Display info about loaded splits
#                 if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
#                     st.info(f"Loaded {len(st.session_state.rag_system.splits)} document splits.")
#             else:
#                 st.info("No valid existing index found at the specified path, or loading failed. Ready for new processing.")
#                 st.session_state.rag_system = None # Reset if loading didn't fully succeed

#         except Exception as e:
#             st.error(f"🚨 Error loading existing index: {str(e)}")
#             if debug_mode:
#                 st.exception(e)
#             st.session_state.rag_system = None # Ensure reset on error


# # --- File Upload Section ---
# upload_container = st.container()
# with upload_container:
#     st.subheader("📂 Upload Documents")
#     uploaded_files = st.file_uploader(
#         "Choose files (PDF, DOCX, TXT, CSV, XLS, XLSX)",
#         accept_multiple_files=True,
#         type=["pdf", "docx", "txt", "csv", "xls", "xlsx"],
#         key="file_uploader" # Add a key for stability
#     )

#     if uploaded_files:
#         total_size_mb = sum(file.size for file in uploaded_files) / (1024 * 1024)
#         if total_size_mb > 50: # Increased warning threshold
#             st.warning(f"Total file size is {total_size_mb:.1f}MB. Processing very large files might be slow or hit memory limits.")

#         # Use columns for better layout
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             if st.button("🚀 Process Documents", key="process_button"):
#                 process_uploaded_documents(uploaded_files)
#         with col2:
#              # Show status message related to processing
#              if st.session_state.documents_processed:
#                  st.success("Ready to answer questions.")
#              elif st.session_state.processing_error:
#                  st.error("Processing failed. See error details.")


# # Display processed files and errors
# status_container = st.container()
# with status_container:
#     if st.session_state.index_loaded_from_path and not uploaded_files:
#          st.info("✅ Index loaded from disk. You can upload more files to add them, or start asking questions.")

#     if st.session_state.processed_files_info and "total_splits" in st.session_state.processed_files_info:
#         # Display info about processed files if available
#         with st.expander("Processed Files & Splits", expanded=False):
#             st.write(f"Total document splits indexed: {st.session_state.processed_files_info['total_splits']}")
#             for file_name, status in st.session_state.processed_files_info.items():
#                 if file_name != "total_splits":
#                     st.write(f"- {file_name}: {status}")

#     if st.session_state.processing_error:
#         with st.expander("🚨 Processing Error Details", expanded=True):
#             st.error(st.session_state.processing_error)


# # --- Query Section ---
# query_container = st.container()
# with query_container:
#     st.subheader("❓ Ask a Question")
#     query = st.text_input("Enter your question about the documents:", placeholder="e.g., What is the main topic of document X?", key="query_input")

#     if query:
#         if not api_key:
#             st.warning("⚠️ Please enter your OpenRouter API Key in the sidebar to get an answer.")
#         elif not st.session_state.documents_processed and not st.session_state.index_loaded_from_path:
#             st.warning("⚠️ Please upload and process documents first, or load an existing index.")
#         elif st.session_state.rag_system:
#             # Update RAG system instance with potentially changed settings from sidebar
#             st.session_state.rag_system.api_key = api_key
#             st.session_state.rag_system.llm_model_name = or_model
#             st.session_state.rag_system.llm_temperature = llm_temperature

#             with st.spinner("🧠 Thinking... Searching documents and generating answer..."):
#                 try:
#                     # Call the enhanced query method with UI parameters
#                     response, final_docs = st.session_state.rag_system.query(
#                         query,
#                         k=k_value, # Initial K for hybrid retrieval
#                         threshold_percentile=threshold_percentile, # Confidence threshold percentile
#                         max_final_docs=max_final_docs # Max docs to send to LLM
#                     )

#                     st.subheader("💬 Answer")
#                     st.markdown(response) # Display the LLM's answer

#                     # Display the final set of documents used for context
#                     if final_docs:
#                          with st.expander(f"📚 View {len(final_docs)} Context Documents (passed confidence threshold)", expanded=False):
#                             for i, doc in enumerate(final_docs):
#                                 st.markdown(f"--- **Chunk {i+1}** ---")
#                                 # Try to display source nicely
#                                 source = doc.metadata.get('source', 'Unknown Source')
#                                 st.caption(f"Source: {source}")
#                                 st.markdown(doc.page_content)
#                     elif "couldn't find relevant information" not in response.lower(): # Avoid showing if response already indicated no docs found
#                          st.info("No documents met the confidence threshold to provide context for this query.")


#                 except Exception as e:
#                     import traceback
#                     error_msg = f"🚨 Error during query processing: {str(e)}"
#                     st.error(error_msg)
#                     if debug_mode:
#                          st.code(traceback.format_exc())
#         else:
#              st.error("RAG System not available. Please process documents or check for initialization errors.")















# ver--2
# import os
# import tempfile
# import streamlit as st
# from typing import List, Optional, Any, Dict
# import traceback # Import traceback for better error logging

# from langchain_community.document_loaders import (
#     PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
# )
# from langchain.docstore.document import Document

# # Import RAG utilities
# try:
#     from rag import DocumentRAG
# except ImportError:
#     st.error("Error: Could not import DocumentRAG from rag.py. Make sure the file exists and is in the correct path.")
#     st.stop() # Stop execution if core component is missing

# # Set up page configuration
# st.set_page_config(page_title="Conversational RAG", page_icon="💬", layout="wide")
# st.title("💬 Conversational RAG")
# st.subheader("Chat with your documents")

# # --- Session State Initialization ---
# # (Keep this section as it was)
# if "chat_history" not in st.session_state: st.session_state.chat_history = []
# if 'rag_system' not in st.session_state: st.session_state.rag_system = None
# if "last_used_docs" not in st.session_state: st.session_state.last_used_docs = None
# if 'documents_processed' not in st.session_state: st.session_state.documents_processed = False
# if 'processed_files_info' not in st.session_state: st.session_state.processed_files_info = {}
# if 'processing_error' not in st.session_state: st.session_state.processing_error = None
# if 'index_loaded_from_path' not in st.session_state: st.session_state.index_loaded_from_path = False
# if 'attempted_load' not in st.session_state: st.session_state.attempted_load = False
# if "api_key" not in st.session_state: st.session_state.api_key = ""


# # --- Helper Functions (Moved to the top) ---
# # (Keep get_loader and process_uploaded_documents functions exactly as they were)
# def get_loader(file_path: str, file_ext: str) -> Optional[Any]:
#     """Gets the appropriate Langchain document loader."""
#     try:
#         if file_ext == ".pdf": return PyPDFLoader(file_path)
#         elif file_ext == ".docx": return Docx2txtLoader(file_path)
#         elif file_ext == ".txt": return TextLoader(file_path, autodetect_encoding=True)
#         elif file_ext == ".csv": return CSVLoader(file_path, autodetect_encoding=True)
#         elif file_ext in [".xls", ".xlsx"]: return UnstructuredExcelLoader(file_path, mode="elements")
#         else: st.error(f"Unsupported file format: {file_ext}"); return None
#     except ImportError as ie:
#          if "unstructured" in str(ie) and file_ext in [".xls", ".xlsx"]: st.error(f"Install 'unstructured[xlsx]': pip install \"unstructured[xlsx]\"")
#          elif "pypdf" in str(ie) and file_ext == ".pdf": st.error("Install 'pypdf': pip install pypdf")
#          else: st.error(f"Missing dependency for {file_ext}: {str(ie)}")
#          return None
#     except Exception as e: st.error(f"Error creating loader for {file_path}: {str(e)}"); return None

# def process_uploaded_documents(files: List[st.runtime.uploaded_file_manager.UploadedFile],
#                                api_key: str, # Pass API key directly
#                                embedding_model: str,
#                                persistence_path: str,
#                                or_model: str,
#                                llm_temperature: float,
#                                chunk_size: int,
#                                chunk_overlap: int,
#                                debug_mode: bool # Pass debug mode
#                                ):
#     """Loads, splits, and indexes the uploaded documents."""
#     if not api_key:
#         st.session_state.processing_error = "🚨 Please enter your OpenRouter API Key in the sidebar."
#         return False
#     if not files:
#         st.session_state.processing_error = "No files selected for processing."
#         return False

#     st.session_state.processing_error = None
#     # Check if RAG system exists and is valid, otherwise initialize
#     if not st.session_state.rag_system or not isinstance(st.session_state.rag_system, DocumentRAG):
#         persist_dir = persistence_path if persistence_path else None
#         with st.spinner("Initializing RAG system..."):
#             try:
#                 st.session_state.rag_system = DocumentRAG(
#                     openrouter_api_key=api_key, embedding_model=embedding_model,
#                     persist_directory=persist_dir, llm_model_name=or_model, llm_temperature=llm_temperature
#                 )
#                 if hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore:
#                      st.session_state.index_loaded_from_path = True
#                      print("Initialized with existing index. New documents will be added.")
#                 else:
#                      st.session_state.index_loaded_from_path = False
#                      print("Initialized RAG system without loading existing index.")

#             except Exception as e:
#                 st.session_state.processing_error = f"🚨 Failed to initialize RAG system: {e}"
#                 st.session_state.rag_system = None
#                 if debug_mode: print(traceback.format_exc()) # Print to console/logs
#                 return False
#     elif hasattr(st.session_state.rag_system, 'api_key') and st.session_state.rag_system.api_key != api_key:
#          # Update API key if it changed in the UI
#          st.session_state.rag_system.api_key = api_key
#          os.environ['OPENROUTER_API_KEY'] = api_key # Keep env var updated too
#          print("Updated RAG system API key.")


#     all_docs: List[Document] = []; processed_file_names = []; error_logs = []
#     with st.spinner(f"Loading {len(files)} file(s)..."):
#         for file in files:
#             file_ext = os.path.splitext(file.name)[1].lower(); tmp_filepath = None
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                     tmp_file.write(file.getvalue()); tmp_filepath = tmp_file.name
#                 if tmp_filepath:
#                     loader = get_loader(tmp_filepath, file_ext)
#                     if loader:
#                         try:
#                             docs = loader.load()
#                             if docs:
#                                 for doc in docs:
#                                     if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict): doc.metadata = {}
#                                     doc.metadata["source"] = file.name
#                                 all_docs.extend(docs); processed_file_names.append(file.name)
#                                 print(f"Loaded {file.name} ({len(docs)} section(s))")
#                             else: error_logs.append(f"No content extracted from {file.name}")
#                         except Exception as e: error_logs.append(f"Error processing {file.name}: {str(e)}")
#             except Exception as e: error_logs.append(f"Error handling file {file.name}: {str(e)}")
#             finally:
#                  if tmp_filepath and os.path.exists(tmp_filepath):
#                      try: os.unlink(tmp_filepath)
#                      except Exception as e_unlink: print(f"Error removing temp file {tmp_filepath}: {e_unlink}")

#     if not all_docs:
#         st.session_state.processing_error = "No new documents loaded." + "\n".join(error_logs)
#         return False

#     st.info(f"Loaded {len(all_docs)} new sections. Processing...")
#     with st.spinner("🧠 Indexing..."):
#         try:
#             # Ensure rag_system is available before calling process_documents
#             if not st.session_state.rag_system:
#                  st.session_state.processing_error = "RAG system not initialized before processing."
#                  return False

#             success = st.session_state.rag_system.process_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#             if success:
#                 st.session_state.documents_processed = True
#                 current_info = st.session_state.processed_files_info
#                 for name in processed_file_names: current_info[name] = "Processed"
#                 if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
#                      current_info["total_splits"] = len(st.session_state.rag_system.splits)
#                 st.session_state.processed_files_info = current_info
#                 if persistence_path and hasattr(st.session_state.rag_system, '_using_temp_dir') and not st.session_state.rag_system._using_temp_dir:
#                      print(f"Index saved to {persistence_path}")
#                 # Rerun only if processing was successful and files were uploaded
#                 if files: st.rerun()
#                 return True
#             else:
#                 # If process_documents returns False, it should have printed errors in rag.py
#                 st.session_state.processing_error = "Failed to process new documents (check console logs for details)." + "\n".join(error_logs)
#                 return False
#         except Exception as e:
#             st.session_state.processing_error = f"Critical processing error: {str(e)}\n{traceback.format_exc()}"
#             if debug_mode: print(traceback.format_exc())
#             return False


# # --- Sidebar Definition (Define all widgets first) ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     # Define API Key widget and store in session state
#     api_key_input_val = st.text_input("OpenRouter API Key", type="password", key="api_key_input", value=st.session_state.get("api_key", ""))
#     if api_key_input_val: st.session_state.api_key = api_key_input_val # Update session state if value changes

#     # Define other config widgets and store their values in variables
#     debug_mode_val = st.checkbox("🐛 Debug Mode", value=False, key="debug_mode_check")
#     persistence_path_val = st.text_input("Persistence Directory (Optional)", value="",
#                                      help="Path to save/load index. Uses temp dir if empty.", key="persistence_path_input")
#     or_model_val = st.selectbox("OpenRouter Model", ["deepseek/deepseek-r1-zero:free", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct", "google/gemini-flash-1.5"], index=0, key="or_model_select")
#     llm_temperature_val = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.05, key="llm_temp_slider")
#     chunk_size_val = st.slider("Chunk Size", 100, 4000, 1000, 100, key="chunk_size_slider")
#     chunk_overlap_val = st.slider("Chunk Overlap", 0, 1000, 150, 10, key="chunk_overlap_slider")
#     k_value_val = st.slider("Initial K", 1, 20, 10, 1, key="k_value_slider")
#     threshold_percentile_val = st.slider("Confidence Threshold Percentile", 0, 100, 25, 5, key="thresh_pct_slider")
#     max_final_docs_val = st.slider("Max Docs for LLM", 1, 15, 4, 1, key="max_docs_slider")
#     embedding_model_val = st.selectbox("Embedding Model", ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"], index=0, key="embed_model_select")

#     # --- File Upload Section (Still inside Sidebar) ---
#     st.subheader("📂 Upload Documents")
#     uploaded_files = st.file_uploader(
#         "Choose files (PDF, DOCX, TXT, CSV, XLS, XLSX)",
#         accept_multiple_files=True, type=["pdf", "docx", "txt", "csv", "xls", "xlsx"],
#         key="file_uploader"
#     )
#     process_button_text = "➕ Process & Add" if st.session_state.index_loaded_from_path else "🚀 Process Docs"
#     process_pressed = st.button(process_button_text, key="process_button")

#     # Process button logic (Calls the function defined earlier)
#     if process_pressed:
#         if uploaded_files:
#              # Pass current values from variables defined above
#              process_uploaded_documents(
#                  uploaded_files,
#                  st.session_state.api_key, # Use session state API key
#                  embedding_model_val,
#                  persistence_path_val,
#                  or_model_val,
#                  llm_temperature_val,
#                  chunk_size_val,
#                  chunk_overlap_val,
#                  debug_mode_val
#                  )
#         else:
#              st.warning("Please upload files before processing.")

#     # Display processing status/errors in sidebar
#     if st.session_state.processing_error and process_pressed: # Only show error if processing was just attempted
#         st.error(f"Processing Error: {st.session_state.processing_error}")
#     elif st.session_state.documents_processed:
#          total_splits = st.session_state.processed_files_info.get('total_splits', 'N/A')
#          st.success(f"✅ Ready. Index has {total_splits} splits.")
#     elif st.session_state.index_loaded_from_path:
#          total_splits = st.session_state.processed_files_info.get('total_splits', 'N/A')
#          st.success(f"✅ Index loaded ({total_splits} splits). Ready.")
#     else:
#          # Only show "Upload docs" if neither processed nor loaded
#          if not st.session_state.documents_processed and not st.session_state.index_loaded_from_path:
#             st.info("Upload documents to begin.")

#     st.markdown("---")
#     # Button to clear chat history
#     if st.button("🧹 Clear Chat History"):
#         st.session_state.chat_history = []
#         st.session_state.last_used_docs = None
#         st.rerun() # Rerun the app to reflect the cleared history


# # --- Attempt to load index on startup (Moved AFTER sidebar definition) ---
# # Now this block can safely use variables like persistence_path_val
# if persistence_path_val and not st.session_state.rag_system and not st.session_state.attempted_load:
#     st.session_state.attempted_load = True
#     st.info(f"Checking for existing index in {persistence_path_val}...")
#     api_key_on_load = st.session_state.get("api_key") # Use API key from session state
#     if not api_key_on_load:
#          st.warning("API Key needed to initialize embedding model for loading index. Please enter it in the sidebar.")
#     else:
#         with st.spinner("Loading existing index..."):
#             try:
#                 # Pass current UI settings from sidebar variables during init
#                 st.session_state.rag_system = DocumentRAG(
#                     openrouter_api_key=api_key_on_load,
#                     embedding_model=embedding_model_val, # Use variable from widget
#                     persist_directory=persistence_path_val, # Use variable from widget
#                     llm_model_name=or_model_val, # Use variable from widget
#                     llm_temperature=llm_temperature_val # Use variable from widget
#                 )
#                 # Check if loading was successful
#                 if hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore and \
#                    hasattr(st.session_state.rag_system, 'bm25_retriever') and st.session_state.rag_system.bm25_retriever:
#                     st.session_state.documents_processed = True
#                     st.session_state.index_loaded_from_path = True
#                     if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
#                         st.session_state.processed_files_info = {"total_splits": len(st.session_state.rag_system.splits), "status": "Loaded from disk"}
#                     print(f"Existing index loaded successfully from {persistence_path_val} with {st.session_state.processed_files_info.get('total_splits', 'N/A')} splits.")
#                     st.rerun() # Rerun to update UI after successful load
#                 elif hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore:
#                      st.warning("Loaded FAISS index, but failed to load splits/BM25. Sparse search may be unavailable.")
#                      st.session_state.documents_processed = True; st.session_state.index_loaded_from_path = True
#                      st.rerun()
#                 else:
#                      print("No valid index found at path, or loading failed.")
#                      st.info("No valid index found at path. Ready for new processing.")
#                      st.session_state.rag_system = None # Reset if loading didn't fully succeed
#             except Exception as e:
#                 st.error(f"🚨 Error loading existing index: {str(e)}")
#                 if debug_mode_val: st.exception(e) # Use debug mode variable
#                 st.session_state.rag_system = None


# # --- Main Chat Interface ---

# st.header("Chat Window")

# # Display chat messages from history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# prompt = st.chat_input("Ask a question about the documents...", key="chat_input")

# # Process chat input only if a prompt was submitted in this run
# if prompt:
#     # Check if RAG system is ready and API key is available
#     api_key_available = st.session_state.get("api_key")
#     system_ready = bool(st.session_state.rag_system and (st.session_state.documents_processed or st.session_state.index_loaded_from_path))

#     # Display user message immediately and add to history
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.chat_history.append({"role": "user", "content": prompt})

#     # Check prerequisites *after* showing the user message
#     if not api_key_available:
#         st.warning("⚠️ Please enter your OpenRouter API Key in the sidebar.")
#         st.stop() # Stop before trying to generate assistant response
#     if not system_ready:
#         st.warning("⚠️ Please process documents or load an index using the sidebar first.")
#         st.stop() # Stop before trying to generate assistant response

#     # --- Generate Assistant Response ---
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         message_placeholder.markdown("Thinking...")

#         try:
#             # Update RAG system settings from sidebar variables (important for LLM model/temp)
#             st.session_state.rag_system.api_key = api_key_available # Already checked
#             st.session_state.rag_system.llm_model_name = or_model_val # Use variable
#             st.session_state.rag_system.llm_temperature = llm_temperature_val # Use variable

#             # Decide if follow-up
#             is_follow_up = bool(len(st.session_state.chat_history) > 1 and st.session_state.last_used_docs is not None)

#             # Prepare arguments
#             query_args = {
#                 "query_text": prompt, # Use the prompt submitted in this run
#                 "k": k_value_val, # Use variable
#                 "threshold_percentile": threshold_percentile_val, # Use variable
#                 "max_final_docs": max_final_docs_val, # Use variable
#             }
#             if is_follow_up:
#                 print("--- Calling RAG Query (Follow-up) ---")
#                 query_args["chat_history"] = st.session_state.chat_history[:-1] # Exclude current prompt
#                 query_args["use_docs_from_history"] = st.session_state.last_used_docs
#             else:
#                  print("--- Calling RAG Query (Initial) ---")
#                  st.session_state.last_used_docs = None # Clear context for initial query

#             # Call the query function
#             response, final_docs_used = st.session_state.rag_system.query(**query_args)
#             if not response or response.strip() == "":
#                 print("LLM returned an empty or whitespace-only response.")
#                 # Provide a user-friendly default message
#                 response = "[The model could not provide further explanation based on the retrieved documents.]"

#             # Display the response (potentially the default message)
#             message_placeholder.markdown(response)

#             # Add assistant response to chat history (potentially the default message)
#             st.session_state.chat_history.append({"role": "assistant", "content": response})
#             # Store context if it was initial query
#             if not is_follow_up:
#                  st.session_state.last_used_docs = final_docs_used
#                  if debug_mode_val: print(f"Stored {len(final_docs_used)} docs for next turn context.")

#             # Display the response
#             message_placeholder.markdown(response)

#             # Add assistant response to chat history
#             st.session_state.chat_history.append({"role": "assistant", "content": response})

#             # Optional debug display
#             if final_docs_used and debug_mode_val:
#                 with st.expander("Context Docs Used for This Answer"):
#                      for i, doc in enumerate(final_docs_used):
#                          st.markdown(f"--- **Chunk {i+1}** ---")
#                          source = doc.metadata.get('source', 'Unknown Source')
#                          st.caption(f"Source: {source}")
#                          st.markdown(doc.page_content, unsafe_allow_html=False)

#         except Exception as e:
#             error_msg = f"🚨 Error generating response: {str(e)}"
#             message_placeholder.error(error_msg)
#             if debug_mode_val: st.code(traceback.format_exc())
#             # Add error message to history
#             st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

#         # Don't use rerun here, let Streamlit handle the update after processing is done.
#         # Using rerun inside the "if prompt:" block was causing issues.
#         # The UI should update automatically when the message_placeholder is filled.




































































# --- app.py ---

# import os
# import tempfile
# import streamlit as st
# from typing import List, Optional, Any, Dict
# import traceback # Import traceback for better error logging

# from langchain_community.document_loaders import (
#     PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
# )
# from langchain.docstore.document import Document

# # Import RAG utilities
# try:
#     from rag import DocumentRAG
# except ImportError:
#     st.error("Error: Could not import DocumentRAG from rag.py. Make sure the file exists and is in the correct path.")
#     st.stop() # Stop execution if core component is missing

# # Set up page configuration
# st.set_page_config(page_title="Conversational RAG", page_icon="💬", layout="wide")
# st.title("💬 Conversational RAG")
# st.subheader("Chat with your documents")

# # --- Session State Initialization ---
# # (Keep this section as it was)
# if "chat_history" not in st.session_state: st.session_state.chat_history = []
# if 'rag_system' not in st.session_state: st.session_state.rag_system = None
# if "last_used_docs" not in st.session_state: st.session_state.last_used_docs = None
# if 'documents_processed' not in st.session_state: st.session_state.documents_processed = False
# if 'processed_files_info' not in st.session_state: st.session_state.processed_files_info = {}
# if 'processing_error' not in st.session_state: st.session_state.processing_error = None
# if 'index_loaded_from_path' not in st.session_state: st.session_state.index_loaded_from_path = False
# if 'attempted_load' not in st.session_state: st.session_state.attempted_load = False
# if "api_key" not in st.session_state: st.session_state.api_key = ""


# # --- Helper Functions (Moved to the top) ---
# # (Keep get_loader and process_uploaded_documents functions exactly as they were)
# def get_loader(file_path: str, file_ext: str) -> Optional[Any]:
#     """Gets the appropriate Langchain document loader."""
#     try:
#         if file_ext == ".pdf": return PyPDFLoader(file_path)
#         elif file_ext == ".docx": return Docx2txtLoader(file_path)
#         elif file_ext == ".txt": return TextLoader(file_path, autodetect_encoding=True)
#         elif file_ext == ".csv": return CSVLoader(file_path, autodetect_encoding=True)
#         elif file_ext in [".xls", ".xlsx"]: return UnstructuredExcelLoader(file_path, mode="elements")
#         else: st.error(f"Unsupported file format: {file_ext}"); return None
#     except ImportError as ie:
#          if "unstructured" in str(ie) and file_ext in [".xls", ".xlsx"]: st.error(f"Install 'unstructured[xlsx]': pip install \"unstructured[xlsx]\"")
#          elif "pypdf" in str(ie) and file_ext == ".pdf": st.error("Install 'pypdf': pip install pypdf")
#          else: st.error(f"Missing dependency for {file_ext}: {str(ie)}")
#          return None
#     except Exception as e: st.error(f"Error creating loader for {file_path}: {str(e)}"); return None

# def process_uploaded_documents(files: List[st.runtime.uploaded_file_manager.UploadedFile],
#                                api_key: str, # Pass API key directly
#                                embedding_model: str,
#                                persistence_path: str,
#                                or_model: str,
#                                llm_temperature: float,
#                                chunk_size: int,
#                                chunk_overlap: int,
#                                debug_mode: bool # Pass debug mode
#                                ):
#     """Loads, splits, and indexes the uploaded documents."""
#     if not api_key:
#         st.session_state.processing_error = "🚨 Please enter your OpenRouter API Key in the sidebar."
#         return False
#     if not files:
#         st.session_state.processing_error = "No files selected for processing."
#         return False

#     st.session_state.processing_error = None
#     # Check if RAG system exists and is valid, otherwise initialize
#     if not st.session_state.rag_system or not isinstance(st.session_state.rag_system, DocumentRAG):
#         persist_dir = persistence_path if persistence_path else None
#         with st.spinner("Initializing RAG system..."):
#             try:
#                 st.session_state.rag_system = DocumentRAG(
#                     openrouter_api_key=api_key, embedding_model=embedding_model,
#                     persist_directory=persist_dir, llm_model_name=or_model, llm_temperature=llm_temperature
#                 )
#                 if hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore:
#                      st.session_state.index_loaded_from_path = True
#                      print("Initialized with existing index. New documents will be added.")
#                 else:
#                      st.session_state.index_loaded_from_path = False
#                      print("Initialized RAG system without loading existing index.")

#             except Exception as e:
#                 st.session_state.processing_error = f"🚨 Failed to initialize RAG system: {e}"
#                 st.session_state.rag_system = None
#                 if debug_mode: print(traceback.format_exc()) # Print to console/logs
#                 return False
#     elif hasattr(st.session_state.rag_system, 'api_key') and st.session_state.rag_system.api_key != api_key:
#          # Update API key if it changed in the UI
#          st.session_state.rag_system.api_key = api_key
#          os.environ['OPENROUTER_API_KEY'] = api_key # Keep env var updated too
#          print("Updated RAG system API key.")


#     all_docs: List[Document] = []; processed_file_names = []; error_logs = []
#     with st.spinner(f"Loading {len(files)} file(s)..."):
#         for file in files:
#             file_ext = os.path.splitext(file.name)[1].lower(); tmp_filepath = None
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                     tmp_file.write(file.getvalue()); tmp_filepath = tmp_file.name
#                 if tmp_filepath:
#                     loader = get_loader(tmp_filepath, file_ext)
#                     if loader:
#                         try:
#                             docs = loader.load()
#                             if docs:
#                                 for doc in docs:
#                                     if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict): doc.metadata = {}
#                                     doc.metadata["source"] = file.name
#                                 all_docs.extend(docs); processed_file_names.append(file.name)
#                                 print(f"Loaded {file.name} ({len(docs)} section(s))")
#                             else: error_logs.append(f"No content extracted from {file.name}")
#                         except Exception as e: error_logs.append(f"Error processing {file.name}: {str(e)}")
#             except Exception as e: error_logs.append(f"Error handling file {file.name}: {str(e)}")
#             finally:
#                  if tmp_filepath and os.path.exists(tmp_filepath):
#                      try: os.unlink(tmp_filepath)
#                      except Exception as e_unlink: print(f"Error removing temp file {tmp_filepath}: {e_unlink}")

#     if not all_docs:
#         st.session_state.processing_error = "No new documents loaded." + "\n".join(error_logs)
#         return False

#     st.info(f"Loaded {len(all_docs)} new sections. Processing...")
#     with st.spinner("🧠 Indexing..."):
#         try:
#             # Ensure rag_system is available before calling process_documents
#             if not st.session_state.rag_system:
#                  st.session_state.processing_error = "RAG system not initialized before processing."
#                  return False

#             success = st.session_state.rag_system.process_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#             if success:
#                 st.session_state.documents_processed = True
#                 current_info = st.session_state.processed_files_info
#                 for name in processed_file_names: current_info[name] = "Processed"
#                 if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
#                      current_info["total_splits"] = len(st.session_state.rag_system.splits)
#                 st.session_state.processed_files_info = current_info
#                 if persistence_path and hasattr(st.session_state.rag_system, '_using_temp_dir') and not st.session_state.rag_system._using_temp_dir:
#                      print(f"Index saved to {persistence_path}")
#                 # Rerun after successful processing to update sidebar status cleanly
#                 # *** Remove rerun from here ***
#                 # if files: st.rerun()
#                 return True
#             else:
#                 st.session_state.processing_error = "Failed to process new documents (check console logs for details)." + "\n".join(error_logs)
#                 return False
#         except Exception as e:
#             st.session_state.processing_error = f"Critical processing error: {str(e)}\n{traceback.format_exc()}"
#             if debug_mode: print(traceback.format_exc())
#             return False


# # --- Sidebar Definition ---
# # (Define all widgets first, as before)
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     api_key_input_val = st.text_input("OpenRouter API Key", type="password", key="api_key_input", value=st.session_state.get("api_key", ""))
#     if api_key_input_val: st.session_state.api_key = api_key_input_val
#     debug_mode_val = st.checkbox("🐛 Debug Mode", value=False, key="debug_mode_check")
#     persistence_path_val = st.text_input("Persistence Directory (Optional)", value="", help="Path to save/load index. Uses temp dir if empty.", key="persistence_path_input")
#     or_model_val = st.selectbox("OpenRouter Model", ["deepseek/deepseek-r1-zero:free", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct", "google/gemini-flash-1.5"], index=0, key="or_model_select")
#     llm_temperature_val = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.05, key="llm_temp_slider")
#     chunk_size_val = st.slider("Chunk Size", 100, 4000, 1000, 100, key="chunk_size_slider")
#     chunk_overlap_val = st.slider("Chunk Overlap", 0, 1000, 150, 10, key="chunk_overlap_slider")
#     k_value_val = st.slider("Initial K", 1, 20, 10, 1, key="k_value_slider")
#     threshold_percentile_val = st.slider("Confidence Threshold Percentile", 0, 100, 25, 5, key="thresh_pct_slider")
#     max_final_docs_val = st.slider("Max Docs for LLM", 1, 15, 4, 1, key="max_docs_slider")
#     embedding_model_val = st.selectbox("Embedding Model", ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"], index=0, key="embed_model_select")

#     st.subheader("📂 Upload Documents")
#     uploaded_files = st.file_uploader(
#         "Choose files (PDF, DOCX, TXT, CSV, XLS, XLSX)",
#         accept_multiple_files=True, type=["pdf", "docx", "txt", "csv", "xls", "xlsx"],
#         key="file_uploader"
#     )
#     process_button_text = "➕ Process & Add" if st.session_state.index_loaded_from_path else "🚀 Process Docs"
#     process_pressed = st.button(process_button_text, key="process_button")

#     if process_pressed:
#         if uploaded_files:
#              processing_success = process_uploaded_documents( # Store success status
#                  uploaded_files,
#                  st.session_state.api_key, embedding_model_val, persistence_path_val,
#                  or_model_val, llm_temperature_val, chunk_size_val,
#                  chunk_overlap_val, debug_mode_val
#                  )
#              if processing_success:
#                   st.rerun() # Rerun ONLY after successful processing to update status
#         else:
#              st.warning("Please upload files before processing.")

#     # Display status/errors (remains the same)
#     if st.session_state.processing_error and process_pressed:
#         st.error(f"Processing Error: {st.session_state.processing_error}")
#     elif st.session_state.documents_processed:
#          total_splits = st.session_state.processed_files_info.get('total_splits', 'N/A')
#          st.success(f"✅ Ready. Index has {total_splits} splits.")
#     elif st.session_state.index_loaded_from_path:
#          total_splits = st.session_state.processed_files_info.get('total_splits', 'N/A')
#          st.success(f"✅ Index loaded ({total_splits} splits). Ready.")
#     else:
#          if not st.session_state.documents_processed and not st.session_state.index_loaded_from_path:
#             st.info("Upload documents to begin.")

#     st.markdown("---")
#     if st.button("🧹 Clear Chat History"):
#         st.session_state.chat_history = []
#         st.session_state.last_used_docs = None
#         st.rerun()


# # --- Attempt to load index on startup ---
# # (Keep this block exactly as it was in the previous fix, AFTER the sidebar)
# if persistence_path_val and not st.session_state.rag_system and not st.session_state.attempted_load:
#     st.session_state.attempted_load = True
#     st.info(f"Checking for existing index in {persistence_path_val}...")
#     api_key_on_load = st.session_state.get("api_key")
#     if not api_key_on_load:
#          st.warning("API Key needed to initialize embedding model for loading index. Please enter it in the sidebar.")
#     else:
#         with st.spinner("Loading existing index..."):
#             try:
#                 st.session_state.rag_system = DocumentRAG(
#                     openrouter_api_key=api_key_on_load, embedding_model=embedding_model_val,
#                     persist_directory=persistence_path_val, llm_model_name=or_model_val,
#                     llm_temperature=llm_temperature_val
#                 )
#                 if hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore and \
#                    hasattr(st.session_state.rag_system, 'bm25_retriever') and st.session_state.rag_system.bm25_retriever:
#                     st.session_state.documents_processed = True
#                     st.session_state.index_loaded_from_path = True
#                     if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
#                         st.session_state.processed_files_info = {"total_splits": len(st.session_state.rag_system.splits), "status": "Loaded from disk"}
#                     print(f"Existing index loaded successfully from {persistence_path_val} with {st.session_state.processed_files_info.get('total_splits', 'N/A')} splits.")
#                     st.rerun()
#                 elif hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore:
#                      st.warning("Loaded FAISS index, but failed to load splits/BM25. Sparse search may be unavailable.")
#                      st.session_state.documents_processed = True; st.session_state.index_loaded_from_path = True
#                      st.rerun()
#                 else:
#                      print("No valid index found at path, or loading failed.")
#                      st.info("No valid index found at path. Ready for new processing.")
#                      st.session_state.rag_system = None
#             except Exception as e:
#                 st.error(f"🚨 Error loading existing index: {str(e)}")
#                 if debug_mode_val: st.exception(e)
#                 st.session_state.rag_system = None


# # --- Main Chat Interface ---
# st.header("Chat Window")

# # Display chat messages from history ON EVERY RUN
# for i, message in enumerate(st.session_state.chat_history):
#     with st.chat_message(message["role"]): # No key needed
#         st.markdown(message["content"])

# # Use the assignment expression feature (walrus operator) := for cleaner input handling
# if prompt := st.chat_input("Ask a question about the documents...", key="chat_input"):
#     # 1. Append user prompt to history IMMEDIATELY
#     st.session_state.chat_history.append({"role": "user", "content": prompt})

#     # 2. Check prerequisites (API Key, System Ready)
#     api_key_available = st.session_state.get("api_key")
#     system_ready = bool(st.session_state.rag_system and (st.session_state.documents_processed or st.session_state.index_loaded_from_path))

#     if not api_key_available or not system_ready:
#         # Add warning message(s) to history and rerun
#         if not api_key_available:
#             st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ Please enter your OpenRouter API Key in the sidebar."})
#         if not system_ready:
#             st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ Please process documents or load an index using the sidebar first."})
#         st.rerun()
#     else:
#         # 3. System is ready, proceed to generate assistant response

#         # Display "Thinking..." placeholder
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             message_placeholder.markdown("Thinking...")

#             try:
#                 # Update RAG system settings from sidebar variables
#                 # Ensure variables are fetched correctly (using .get for safety)
#                 current_api_key = st.session_state.get("api_key")
#                 current_or_model = st.session_state.get("or_model_select", "deepseek/deepseek-r1-zero:free") # Match key used in sidebar widget
#                 current_llm_temp = st.session_state.get("llm_temp_slider", 0.7) # Match key used in sidebar widget
#                 current_k_val = st.session_state.get("k_value_slider", 10)
#                 current_thresh_pct = st.session_state.get("thresh_pct_slider", 25)
#                 current_max_docs = st.session_state.get("max_docs_slider", 4)
#                 current_debug_mode = st.session_state.get("debug_mode_check", False)

#                 # Ensure rag_system exists and update its settings
#                 if not st.session_state.rag_system:
#                      raise ValueError("RAG system is not initialized.") # Should be caught by system_ready check, but belts & braces
#                 st.session_state.rag_system.api_key = current_api_key
#                 st.session_state.rag_system.llm_model_name = current_or_model
#                 st.session_state.rag_system.llm_temperature = current_llm_temp

#                 # Prepare arguments for the REVISED query method
#                 # REMOVED the is_follow_up logic and use_docs_from_history
#                 query_args = {
#                     "query_text": prompt, # Use the current prompt
#                     "k": current_k_val,
#                     "threshold_percentile": current_thresh_pct,
#                     "max_final_docs": current_max_docs,
#                     # ALWAYS pass the history up to the point BEFORE the current user prompt
#                     "chat_history": st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else None
#                 }
#                 print(f"--- Calling RAG Query (Args: { {k: v for k, v in query_args.items() if k != 'chat_history'} }, History Len: {len(query_args.get('chat_history') or [])}) ---")


#                 # Call the revised query function (no longer needs is_follow_up distinction here)
#                 response, final_docs_used = st.session_state.rag_system.query(**query_args)

#                 # Clean the response (Keep the cleaning logic from previous step)
#                 cleaned_response = response.strip()
#                 if cleaned_response.startswith("\\boxed{") and cleaned_response.endswith("}"):
#                     cleaned_response = cleaned_response[7:-1].strip()

#                 # Handle potential empty response AFTER cleaning
#                 if not cleaned_response or cleaned_response.isspace():
#                     print("LLM returned an empty or whitespace-only response.")
#                     cleaned_response = "[The model could not provide an answer based on the retrieved documents.]"

#                 # Update the placeholder with the actual cleaned response
#                 message_placeholder.markdown(cleaned_response)

#                 # Add the final assistant response to chat history
#                 st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})

#                 # ** REMOVED: No need to store last_used_docs anymore **

#                 # Optional debug display (shows docs retrieved for THIS turn)
#                 if final_docs_used and current_debug_mode:
#                     with st.expander("Context Docs Used for This Answer"):
#                          for i, doc in enumerate(final_docs_used):
#                              st.markdown(f"--- **Chunk {i+1}** ---")
#                              source = doc.metadata.get('source', 'Unknown Source')
#                              st.caption(f"Source: {source}")
#                              st.markdown(doc.page_content, unsafe_allow_html=False)

#             except Exception as e:
#                 error_msg = f"🚨 Error generating response: {str(e)}"
#                 message_placeholder.error(error_msg)
#                 if current_debug_mode: st.code(traceback.format_exc())
#                 # Add error message to history state
#                 st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

#             # Trigger final rerun AFTER processing
#             st.rerun()









# --- app.py ---

import os
import tempfile
import streamlit as st
from typing import List, Optional, Any, Dict
import traceback # Import traceback for better error logging
import json # Ensure json is imported

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
)
# Ensure you have this or the newer langchain_huggingface import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Import RAG utilities
try:
    from rag import DocumentRAG
except ImportError:
    st.error("Error: Could not import DocumentRAG from rag.py. Make sure the file exists and is in the correct path.")
    st.stop() # Stop execution if core component is missing

# Set up page configuration
st.set_page_config(page_title="Conversational RAG", page_icon="💬", layout="wide")
st.title("💬 Conversational RAG")
st.subheader("Chat with your documents")

# --- Session State Initialization ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if 'rag_system' not in st.session_state: st.session_state.rag_system = None
if 'documents_processed' not in st.session_state: st.session_state.documents_processed = False
if 'processed_files_info' not in st.session_state: st.session_state.processed_files_info = {}
if 'processing_error' not in st.session_state: st.session_state.processing_error = None
if 'index_loaded_from_path' not in st.session_state: st.session_state.index_loaded_from_path = False
if 'attempted_load' not in st.session_state: st.session_state.attempted_load = False
if "api_key" not in st.session_state: st.session_state.api_key = ""
# Initialize sidebar widget values in session state if not already set
# This helps retain selections across reruns more reliably
default_values = {
    "debug_mode_check": False,
    "persistence_path_input": "",
    "or_model_select": "deepseek/deepseek-r1-zero:free",
    "llm_temp_slider": 0.7,
    "chunk_size_slider": 1000,
    "chunk_overlap_slider": 150,
    "k_value_slider": 10,
    "thresh_pct_slider": 25,
    "max_docs_slider": 4,
    "embed_model_select": "sentence-transformers/all-mpnet-base-v2"
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Helper Functions ---
def get_loader(file_path: str, file_ext: str) -> Optional[Any]:
    """Gets the appropriate Langchain document loader."""
    try:
        if file_ext == ".pdf": return PyPDFLoader(file_path) # Or PyMuPDFLoader if installed
        elif file_ext == ".docx": return Docx2txtLoader(file_path)
        elif file_ext == ".txt": return TextLoader(file_path, autodetect_encoding=True)
        elif file_ext == ".csv": return CSVLoader(file_path, autodetect_encoding=True)
        elif file_ext in [".xls", ".xlsx"]: return UnstructuredExcelLoader(file_path, mode="elements")
        else: st.error(f"Unsupported file format: {file_ext}"); return None
    except ImportError as ie:
         if "unstructured" in str(ie) and file_ext in [".xls", ".xlsx"]: st.error(f"Install 'unstructured[xlsx]': pip install \"unstructured[xlsx]\"")
         elif "pypdf" in str(ie) and file_ext == ".pdf": st.error("Install 'pypdf': pip install pypdf")
         else: st.error(f"Missing dependency for {file_ext}: {str(ie)}")
         return None
    except Exception as e: st.error(f"Error creating loader for {file_path}: {str(e)}"); return None

def process_uploaded_documents(files: List[st.runtime.uploaded_file_manager.UploadedFile],
                               api_key: str,
                               embedding_model: str,
                               persistence_path: str,
                               or_model: str,
                               llm_temperature: float,
                               chunk_size: int,
                               chunk_overlap: int,
                               debug_mode: bool
                               ):
    """Loads, splits, and indexes the uploaded documents."""
    if not api_key:
        st.session_state.processing_error = "🚨 Please enter your OpenRouter API Key in the sidebar."
        return False
    if not files:
        st.session_state.processing_error = "No files selected for processing."
        return False

    st.session_state.processing_error = None
    if not st.session_state.rag_system or not isinstance(st.session_state.rag_system, DocumentRAG):
        persist_dir = persistence_path if persistence_path else None
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.rag_system = DocumentRAG(
                    openrouter_api_key=api_key, embedding_model=embedding_model,
                    persist_directory=persist_dir, llm_model_name=or_model, llm_temperature=llm_temperature
                )
                if hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore:
                     st.session_state.index_loaded_from_path = True
                     print("Initialized RAG system, potentially with existing index.")
                else:
                     st.session_state.index_loaded_from_path = False
                     print("Initialized RAG system without loading existing index.")
            except Exception as e:
                st.session_state.processing_error = f"🚨 Failed to initialize RAG system: {e}"
                st.session_state.rag_system = None
                if debug_mode: print(traceback.format_exc())
                return False
    elif hasattr(st.session_state.rag_system, 'api_key') and st.session_state.rag_system.api_key != api_key:
         st.session_state.rag_system.api_key = api_key
         os.environ['OPENROUTER_API_KEY'] = api_key
         print("Updated RAG system API key.")

    all_docs: List[Document] = []; processed_file_names = []; error_logs = []
    with st.spinner(f"Loading {len(files)} file(s)..."):
        # (Loading logic remains the same)
        for file in files:
            file_ext = os.path.splitext(file.name)[1].lower(); tmp_filepath = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(file.getvalue()); tmp_filepath = tmp_file.name
                if tmp_filepath:
                    loader = get_loader(tmp_filepath, file_ext)
                    if loader:
                        try:
                            docs = loader.load()
                            if docs:
                                for doc in docs:
                                    if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict): doc.metadata = {}
                                    doc.metadata["source"] = file.name
                                all_docs.extend(docs); processed_file_names.append(file.name)
                                print(f"Loaded {file.name} ({len(docs)} section(s))")
                            else: error_logs.append(f"No content extracted from {file.name}")
                        except Exception as e: error_logs.append(f"Error processing {file.name}: {str(e)}")
            except Exception as e: error_logs.append(f"Error handling file {file.name}: {str(e)}")
            finally:
                 if tmp_filepath and os.path.exists(tmp_filepath):
                     try: os.unlink(tmp_filepath)
                     except Exception as e_unlink: print(f"Error removing temp file {tmp_filepath}: {e_unlink}")


    if not all_docs:
        st.session_state.processing_error = "No new documents loaded." + "\n".join(error_logs)
        return False

    st.info(f"Loaded {len(all_docs)} new sections. Processing...")
    with st.spinner("🧠 Indexing..."):
        try:
            if not st.session_state.rag_system:
                 st.session_state.processing_error = "RAG system not initialized before processing."
                 return False
            success = st.session_state.rag_system.process_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if success:
                st.session_state.documents_processed = True
                current_info = st.session_state.processed_files_info
                for name in processed_file_names: current_info[name] = "Processed"
                if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
                     current_info["total_splits"] = len(st.session_state.rag_system.splits)
                st.session_state.processed_files_info = current_info
                if persistence_path and hasattr(st.session_state.rag_system, '_using_temp_dir') and not st.session_state.rag_system._using_temp_dir:
                     print(f"Index saved to {persistence_path}")
                return True
            else:
                st.session_state.processing_error = "Failed to process new documents." + "\n".join(error_logs)
                return False
        except Exception as e:
            st.session_state.processing_error = f"Critical processing error: {str(e)}\n{traceback.format_exc()}"
            if debug_mode: print(traceback.format_exc())
            return False


# --- Sidebar Definition ---
# Use session state keys directly for widget values/defaults
with st.sidebar:
    st.header("⚙️ Configuration")
    st.text_input("OpenRouter API Key", type="password", key="api_key") # Bind directly to session state key
    st.checkbox("🐛 Debug Mode", key="debug_mode_check")
    st.text_input("Persistence Directory (Optional)", help="Path to save/load index.", key="persistence_path_input")
    st.selectbox("OpenRouter Model", ["deepseek/deepseek-r1-zero:free", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct", "google/gemini-flash-1.5"], key="or_model_select")
    st.slider("LLM Temperature", 0.0, 1.0, key="llm_temp_slider")
    st.slider("Chunk Size", 100, 4000, key="chunk_size_slider")
    st.slider("Chunk Overlap", 0, 1000, key="chunk_overlap_slider")
    st.slider("Initial K", 1, 20, key="k_value_slider")
    st.slider("Confidence Threshold Percentile", 0, 100, key="thresh_pct_slider")
    st.slider("Max Docs for LLM", 1, 15, key="max_docs_slider")
    st.selectbox("Embedding Model", ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"], key="embed_model_select")

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT, CSV, XLS, XLSX)",
        accept_multiple_files=True, type=["pdf", "docx", "txt", "csv", "xls", "xlsx"],
        key="file_uploader"
    )
    process_button_text = "➕ Process & Add" if st.session_state.index_loaded_from_path else "🚀 Process Docs"
    process_pressed = st.button(process_button_text, key="process_button")

    if process_pressed:
        if uploaded_files:
             # Pass values directly from session state
             processing_success = process_uploaded_documents(
                 uploaded_files,
                 st.session_state.api_key, st.session_state.embed_model_select,
                 st.session_state.persistence_path_input, st.session_state.or_model_select,
                 st.session_state.llm_temp_slider, st.session_state.chunk_size_slider,
                 st.session_state.chunk_overlap_slider, st.session_state.debug_mode_check
                 )
             if processing_success:
                  st.rerun() # Rerun after successful processing
        else:
             st.warning("Please upload files before processing.")

    # Display status/errors
    if st.session_state.processing_error and process_pressed:
        st.error(f"Processing Error: {st.session_state.processing_error}")
    elif st.session_state.documents_processed:
         total_splits = st.session_state.processed_files_info.get('total_splits', 'N/A')
         st.success(f"✅ Ready. Index has {total_splits} splits.")
    elif st.session_state.index_loaded_from_path:
         total_splits = st.session_state.processed_files_info.get('total_splits', 'N/A')
         st.success(f"✅ Index loaded ({total_splits} splits). Ready.")
    else:
         if not st.session_state.documents_processed and not st.session_state.index_loaded_from_path:
            st.info("Upload documents to begin.")

    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# --- Attempt to load index on startup ---
if st.session_state.persistence_path_input and not st.session_state.rag_system and not st.session_state.attempted_load:
    st.session_state.attempted_load = True
    st.info(f"Checking for existing index in {st.session_state.persistence_path_input}...")
    if not st.session_state.api_key:
         st.warning("API Key needed for loading index. Please enter it.")
    else:
        with st.spinner("Loading existing index..."):
            try:
                st.session_state.rag_system = DocumentRAG(
                    openrouter_api_key=st.session_state.api_key,
                    embedding_model=st.session_state.embed_model_select, # Use state key
                    persist_directory=st.session_state.persistence_path_input, # Use state key
                    llm_model_name=st.session_state.or_model_select, # Use state key
                    llm_temperature=st.session_state.llm_temp_slider # Use state key
                )
                if hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore and \
                   hasattr(st.session_state.rag_system, 'bm25_retriever') and st.session_state.rag_system.bm25_retriever:
                    st.session_state.documents_processed = True; st.session_state.index_loaded_from_path = True
                    if hasattr(st.session_state.rag_system, 'splits') and st.session_state.rag_system.splits:
                        st.session_state.processed_files_info = {"total_splits": len(st.session_state.rag_system.splits), "status": "Loaded from disk"}
                    print(f"Existing index loaded successfully from {st.session_state.persistence_path_input}.")
                    st.rerun()
                elif hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore:
                     st.warning("Loaded FAISS index, but failed to load splits/BM25.")
                     st.session_state.documents_processed = True; st.session_state.index_loaded_from_path = True
                     st.rerun()
                else:
                     print("No valid index found at path, or loading failed.")
                     st.info("No valid index found at path. Ready for new processing.")
                     st.session_state.rag_system = None
            except Exception as e:
                st.error(f"🚨 Error loading existing index: {str(e)}")
                if st.session_state.debug_mode_check: st.exception(e)
                st.session_state.rag_system = None


# --- Main Chat Interface ---
st.header("Chat Window")

# Display chat messages from history FIRST
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question...", key="chat_input_main"):
    # Append user message immediately for display on next implicit rerun
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    # Trigger rerun to show the user message while processing happens
    st.rerun()

# --- Process the LAST message in history if it's from the user ---
# This block runs on the rerun triggered by the user input above OR by final rerun below
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    # Get the latest user prompt
    latest_prompt = st.session_state.chat_history[-1]["content"]

    # Check prerequisites (API Key, System Ready)
    api_key_available = st.session_state.get("api_key")
    system_ready = bool(st.session_state.rag_system and (st.session_state.documents_processed or st.session_state.index_loaded_from_path))

    if not api_key_available or not system_ready:
        # If prerequisites fail *after* user prompt added, add warning and stop
        if not api_key_available:
            st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ API Key missing."})
        if not system_ready:
            st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ RAG system not ready."})
        # Don't rerun here, let the history display loop show the error
    else:
        # System is ready, generate response
        with st.chat_message("assistant"): # Prepare placeholder for assistant msg
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                # Ensure rag_system exists and update settings
                if not st.session_state.rag_system:
                    raise ValueError("RAG system became unavailable.")
                st.session_state.rag_system.api_key = st.session_state.api_key
                st.session_state.rag_system.llm_model_name = st.session_state.or_model_select
                st.session_state.rag_system.llm_temperature = st.session_state.llm_temp_slider

                query_args = {
                    "query_text": latest_prompt,
                    "k": st.session_state.k_value_slider,
                    "threshold_percentile": st.session_state.thresh_pct_slider,
                    "max_final_docs": st.session_state.max_docs_slider,
                    # Pass history *before* the latest user prompt
                    "chat_history": st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else None
                }
                print(f"--- Calling RAG Query (Args: { {k: v for k, v in query_args.items() if k != 'chat_history'} }, History Len: {len(query_args.get('chat_history') or [])}) ---")

                response, final_docs_used = st.session_state.rag_system.query(**query_args)

                # --- ENHANCED CLEANING STEP ---
                cleaned_response = response.strip()
                is_json_response_attempted = False
                if (cleaned_response.startswith("{") and cleaned_response.endswith("}")) or \
                   (cleaned_response.startswith("[") and cleaned_response.endswith("]")):
                    is_json_response_attempted = True
                    try:
                        data = json.loads(cleaned_response)
                        if isinstance(data, dict):
                            extracted = False
                            for key in ["response", "answer", "text", "summary"]: # Check common keys
                                if key in data and isinstance(data[key], str) and data[key].strip():
                                    cleaned_response = data[key]
                                    extracted = True
                                    break
                            if not extracted: # Fallback dict formatting
                                temp_lines = []
                                for key, value in data.items():
                                    if isinstance(value, str): temp_lines.append(f"**{key.replace('_', ' ').title()}**: {value.strip()}")
                                    elif isinstance(value, list) and all(isinstance(i,str) for i in value): temp_lines.append(f"**{key.replace('_', ' ').title()}**: \n" + "\n".join([f"- {v_item}" for v_item in value]))
                                if temp_lines: cleaned_response = "\n\n".join(temp_lines)
                                else: cleaned_response = response.strip() # Revert if formatting fails
                        elif isinstance(data, list): # Fallback list formatting
                            temp_items = []
                            for item in data:
                                if isinstance(item, str): temp_items.append(f"- {item.strip()}")
                                elif isinstance(item, dict):
                                    item_parts = [f"{k.replace('_',' ').title()}: {v}" for k, v in item.items() if isinstance(v,str)]
                                    if item_parts: temp_items.append(f"- {'; '.join(item_parts)}")
                            if temp_items: cleaned_response = "\n".join(temp_items)
                            else: cleaned_response = response.strip() # Revert if formatting fails
                        else: cleaned_response = response.strip() # Revert if not dict/list
                    except json.JSONDecodeError:
                        print("Attempted JSON parse but failed."); cleaned_response = response.strip()

                # General cleaning
                if cleaned_response.startswith("\\boxed{") and cleaned_response.endswith("}"): cleaned_response = cleaned_response[7:-1].strip()
                if cleaned_response.startswith('"') and cleaned_response.endswith('"') and len(cleaned_response) > 1: cleaned_response = cleaned_response[1:-1]
                cleaned_response = cleaned_response.replace("\\n", "\n").replace("\\\"", "\"")

                # Final empty check
                if not cleaned_response or cleaned_response.isspace():
                    print("LLM response empty/whitespace after cleaning.")
                    if is_json_response_attempted: cleaned_response = "[Model provided unusable structured data.]"
                    else: cleaned_response = "[Model could not provide answer.]"

                # Display response and add to history
                message_placeholder.markdown(cleaned_response)
                st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})

                # Optional debug display
                if final_docs_used and st.session_state.debug_mode_check:
                    with st.expander("Context Docs Used for This Answer"):
                         for i, doc in enumerate(final_docs_used):
                             st.markdown(f"--- **Chunk {i+1}** ---")
                             source = doc.metadata.get('source', 'Unknown Source')
                             st.caption(f"Source: {source}")
                             st.markdown(doc.page_content, unsafe_allow_html=False)

            except Exception as e:
                error_msg = f"🚨 Error generating response: {str(e)}"
                message_placeholder.error(error_msg)
                if st.session_state.debug_mode_check: st.code(traceback.format_exc())
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            # ** REMOVED final rerun from here - let Streamlit manage update after block finishes **

# --- End of script ---