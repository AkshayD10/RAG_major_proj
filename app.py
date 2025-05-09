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