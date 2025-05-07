# import os
# import requests
# import json
# import tempfile
# import pickle
# from typing import List, Optional
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.language_models import BaseLLM
# from langchain_core.callbacks import CallbackManagerForLLMRun
# from langchain_core.outputs import LLMResult, Generation

# # Custom OpenRouter LLM class
# from pydantic import BaseModel, Field

# class OpenRouterLLM(BaseLLM, BaseModel):
#     api_key: str = Field(..., description="OpenRouter API key")
#     model_name: str = "deepseek/deepseek-r1-zero:free"
#     api_url: str = "https://openrouter.ai/api/v1/chat/completions"

#     def __init__(self, api_key: str, **kwargs):
#         super().__init__(api_key=api_key, **kwargs)

#     def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": self.model_name,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             **kwargs
#         }
#         response = requests.post(self.api_url, headers=headers, json=payload)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"]
#         else:
#             raise Exception(f"Error: {response.status_code}, {response.text}")

#     def _generate(
#         self,
#         prompts: List[str],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs
#     ) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
#             generations.append([Generation(text=text)])
#         return LLMResult(generations=generations)

#     @property
#     def _llm_type(self) -> str:
#         return "openrouter-deepseek-r1-zero"

# class DocumentRAG:
#     def __init__(self, openrouter_api_key, embedding_model="sentence-transformers/all-mpnet-base-v2", persist_directory=None):
#         """
#         Initialize the RAG system
        
#         Args:
#             openrouter_api_key (str): API key for OpenRouter
#             embedding_model (str): HuggingFace embedding model name
#             persist_directory (str): Directory to store FAISS index and documents
#         """
#         self.api_key = openrouter_api_key
#         os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
#         self.embedding_model = embedding_model
        
#         # Create embeddings with a smaller model for faster processing
#         self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
#         # Use temporary directory if no persist_directory is provided
#         self.persist_directory = persist_directory or tempfile.mkdtemp()
#         os.makedirs(self.persist_directory, exist_ok=True)
        
#         self.index_path = os.path.join(self.persist_directory, "faiss_index")
#         self.documents_path = os.path.join(self.persist_directory, "documents.pkl")
        
#         self.vectorstore = None
        
#         # Try to load existing index if it exists
#         self._load_index()

#     def _load_index(self):
#         """Load FAISS index if it exists"""
#         try:
#             if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
#                 print(f"Loading existing FAISS index from {self.index_path}")
#                 self.vectorstore = FAISS.load_local(
#                     self.index_path,
#                     self.embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#                 print("FAISS index loaded successfully")
#                 return True
#         except Exception as e:
#             print(f"Error loading existing FAISS index: {e}")
#         return False

#     def _save_index(self):
#         """Save FAISS index and document mapping"""
#         if self.vectorstore:
#             try:
#                 # Save FAISS index
#                 self.vectorstore.save_local(self.index_path)
#                 print(f"FAISS index saved to {self.index_path}")
#                 return True
#             except Exception as e:
#                 print(f"Error saving FAISS index: {e}")
#                 import traceback
#                 traceback.print_exc()
#         return False

#     def process_documents(self, documents, chunk_size=500, chunk_overlap=50):
#         """
#         Process documents and create a vectorstore
        
#         Args:
#             documents (list): List of Document objects
#             chunk_size (int): Size of each text chunk
#             chunk_overlap (int): Overlap between chunks
            
#         Returns:
#             bool: Whether processing was successful
#         """
#         if not documents:
#             return False
            
#         try:
#             # Split documents
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=chunk_size, 
#                 chunk_overlap=chunk_overlap
#             )
#             splits = text_splitter.split_documents(documents)
            
#             if not splits:
#                 print("No splits were created from the documents")
#                 return False
                
#             print(f"Created {len(splits)} splits from documents")
                
#             # Create FAISS vectorstore
#             self.vectorstore = FAISS.from_documents(
#                 documents=splits, 
#                 embedding=self.embeddings
#             )
            
#             # Save the index for persistence
#             success = self._save_index()
#             if not success:
#                 print("Warning: Failed to save FAISS index")
            
#             return True
#         except Exception as e:
#             print(f"Error creating FAISS index: {e}")
#             import traceback
#             traceback.print_exc()
#             return False
    
#     def get_retriever(self, k=4):
#         """
#         Get a retriever from the vectorstore
        
#         Args:
#             k (int): Number of documents to retrieve
            
#         Returns:
#             Retriever: A retriever object or None
#         """
#         if not self.vectorstore:
#             return None
            
#         return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
#     def query(self, query_text, k=4):
#         """
#         Query the RAG system
        
#         Args:
#             query_text (str): The query text
#             k (int): Number of documents to retrieve
            
#         Returns:
#             tuple: (answer, retrieved_docs)
#         """
#         if not self.vectorstore:
#             return "No documents have been processed yet.", []
            
#         # Get retriever
#         retriever = self.get_retriever(k)
        
#         # Get relevant documents
#         docs = retriever.get_relevant_documents(query_text)
        
#         # Format docs
#         def format_docs(docs):
#             return "\n\n".join(doc.page_content for doc in docs)
            
#         # Create prompt
#         template = """You are a helpful AI assistant that provides accurate information based only on the given context.
        
#         Context:
#         {context}
        
#         Question: {question}
        
#         Instructions:
#         1. Answer based ONLY on the information provided in the context
#         3. Be concise and clear
#         4. If the context does not explicitly contain the information needed to answer the question, make a logical inference based on the available information.
#         5. If appropriate, cite which specific document(s) your information comes from
        
#         Answer:
#         """
#         prompt = ChatPromptTemplate.from_template(template)
        
#         # Create LLM with the custom OpenRouterLLM class
#         llm = OpenRouterLLM(api_key=self.api_key, model_name=getattr(self, 'model_name', 'deepseek/deepseek-r1-zero:free'))
        
#         # Create RAG chain
#         rag_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
        
#         # Execute chain
#         response = rag_chain.invoke(query_text)
        
#         return response, docs
    
    
# #     --------------------------
    
    
    
    
    
    
    
    
import os
import requests
import json
import tempfile
import pickle
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder # Added MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Added for history

# Custom OpenRouter LLM class (remains the same)
from pydantic import BaseModel, Field

class OpenRouterLLM(BaseLLM, BaseModel):
    api_key: str = Field(..., description="OpenRouter API key")
    model_name: str = "deepseek/deepseek-r1-zero:free"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = 0.7 # Optional: Add temperature control

    class Config:
        extra = 'allow' # Allow extra fields like temperature

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        # Simplified: Assuming prompt is the user message for single turn
        # For history, _generate should handle message formatting
        return self._generate([prompt], stop, run_manager, **kwargs).generations[0][0].text


    def _generate(
        self,
        prompts: List[Dict[str, str]], # Prompts might be single strings or formatted history
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        generations = []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = prompts

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
             **kwargs
        }

        api_response_text = "No response" # Default value
        response_obj = None # Initialize response object

        try:
            print(f"\n--- Inside LLM _generate ---") # Debug print
            print(f"Sending request to OpenRouter API for model: {self.model_name}")
            # print(f"Payload messages structure: {json.dumps(messages, indent=2)}") # Uncomment for deep debug

            response_obj = requests.post(self.api_url, headers=headers, json=payload, timeout=90) # Increased timeout
            api_response_text = response_obj.text # Store response text regardless of status

            # Check for HTTP errors first
            response_obj.raise_for_status()

            # Parse JSON *after* checking HTTP status
            response_json = response_obj.json()

            # Check for application-level errors in the JSON response
            if "error" in response_json:
                print(f"OpenRouter API returned an application error: {response_json['error']}")
                # Extract error details if possible
                error_message = response_json['error'].get('message', 'Unknown API error')
                error_code = response_json['error'].get('code', 'N/A')
                # Create a generation with the error message
                generations.append([Generation(text=f"Error from API: {error_message} (Code: {error_code})")])

            # Check if 'choices' exists and is valid
            elif "choices" in response_json and isinstance(response_json["choices"], list) and len(response_json["choices"]) > 0:
                first_choice = response_json["choices"][0]
                if "message" in first_choice and "content" in first_choice["message"]:
                    text = first_choice["message"]["content"]
                    generations.append([Generation(text=text)])
                    print(f"Successfully received and parsed response from {self.model_name}.")
                else:
                    print("Error: API response format invalid (missing message/content).")
                    generations.append([Generation(text="Error: Invalid response format from API.")])
            else:
                 print("Error: API response format invalid (missing choices).")
                 generations.append([Generation(text="Error: Invalid response format from API (no choices).")])


        except requests.exceptions.Timeout:
            print("Error: API request timed out.")
            generations.append([Generation(text="Error: The request to the language model timed out.")])
        except requests.exceptions.RequestException as e:
            # Includes HTTP errors caught by raise_for_status
            print(f"API Request Error: {e}, Status Code: {response_obj.status_code if response_obj else 'N/A'}, Response: {api_response_text}")
            generations.append([Generation(text=f"Error: API request failed. Status: {response_obj.status_code if response_obj else 'N/A'}. Check console logs.")])
        except Exception as e: # Catch other errors like JSONDecodeError, KeyError
            print(f"LLM Generation Error: {e}, Status: {response_obj.status_code if response_obj else 'N/A'}, Response: {api_response_text}")
            import traceback
            print(traceback.format_exc()) # Print full traceback for unexpected errors
            generations.append([Generation(text=f"Error: Failed to process LLM response. Check console logs.")])

        # Ensure we always return a valid LLMResult, even if empty
        if not generations:
             # This case should ideally be covered by the error handling above,
             # but as a fallback, append an error generation.
             print("Error: No generations were created.")
             generations.append([Generation(text="Error: Unknown error occurred during generation.")])

        return LLMResult(generations=generations)


        # Handle the case where 'prompts' is a list of message dicts (for history)
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                 **kwargs
            }
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            generations.append([Generation(text=text)])
        except requests.exceptions.RequestException as e:
            error_content = response.text if 'response' in locals() else "No response"
            print(f"API Error: {e}, Response: {error_content}")
            generations.append([Generation(text=f"Error: API request failed. {e}")])
        except (KeyError, IndexError, Exception) as e:
            print(f"LLM Generation Error: {e}, Response: {response.text if 'response' in locals() else 'N/A'}")
            generations.append([Generation(text=f"Error: Failed to process LLM response. {e}")])

        return LLMResult(generations=generations)


    @property
    def _llm_type(self) -> str:
        return f"openrouter-{self.model_name.replace('/', '-')}"


class DocumentRAG:
    # __init__, __del__, _load_index_and_splits, _save_index_and_splits,
    # _initialize_bm25_retriever, process_documents, hybrid_retrieve,
    # _calculate_dynamic_threshold, _filter_and_rerank_docs
    # remain the SAME as before.

    # --- Start Previous Code (Keep as is) ---
    def __init__(self,
                 openrouter_api_key: str,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 persist_directory: Optional[str] = None,
                 llm_model_name: str = "deepseek/deepseek-r1-zero:free",
                 llm_temperature: float = 0.7):
        self.api_key = openrouter_api_key
        os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        if persist_directory:
            self.persist_directory = persist_directory
            os.makedirs(self.persist_directory, exist_ok=True)
            self._using_temp_dir = False
        else:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.persist_directory = self._temp_dir.name
            self._using_temp_dir = True
            print(f"Using temporary directory for persistence: {self.persist_directory}")
        self.index_path = os.path.join(self.persist_directory, "faiss_index")
        self.splits_path = os.path.join(self.persist_directory, "splits.pkl")
        self.vectorstore: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.splits: List[Document] = []
        self._load_index_and_splits()

    def __del__(self):
        if hasattr(self, '_using_temp_dir') and self._using_temp_dir and hasattr(self, '_temp_dir'):
            print(f"Cleaning up temporary directory: {self.persist_directory}")
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")

    def _load_index_and_splits(self):
        loaded_faiss = False
        loaded_splits = False
        try:
            if os.path.exists(self.index_path):
                print(f"Loading existing FAISS index from {self.index_path}")
                self.vectorstore = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                print("FAISS index loaded successfully.")
                loaded_faiss = True
            else: print(f"FAISS index file not found at {self.index_path}")
            if os.path.exists(self.splits_path):
                print(f"Loading existing document splits from {self.splits_path}")
                try:
                    with open(self.splits_path, "rb") as f: self.splits = pickle.load(f)
                    if self.splits and isinstance(self.splits, list):
                        self._initialize_bm25_retriever()
                        print(f"Document splits loaded and BM25 retriever initialized ({len(self.splits)} splits).")
                        loaded_splits = True
                    else:
                        print("Warning: Loaded splits file was empty or invalid."); self.splits = []
                except (EOFError, pickle.UnpicklingError) as pe:
                     print(f"Error: Could not load splits from {self.splits_path}: {pe}"); self.splits = []
            else: print(f"Splits file not found at {self.splits_path}")
        except Exception as e:
            print(f"Error loading existing FAISS index or splits: {e}"); import traceback; print(traceback.format_exc())
            self.vectorstore = None; self.splits = []; self.bm25_retriever = None
        return loaded_faiss and loaded_splits

    def _save_index_and_splits(self):
        saved_faiss = False; saved_splits = False
        if self.vectorstore:
            try: self.vectorstore.save_local(self.index_path); print(f"FAISS index saved to {self.index_path}"); saved_faiss = True
            except Exception as e: print(f"Error saving FAISS index: {e}"); import traceback; traceback.print_exc()
        if self.splits:
            try:
                with open(self.splits_path, "wb") as f: pickle.dump(self.splits, f)
                print(f"Document splits saved to {self.splits_path}"); saved_splits = True
            except Exception as e: print(f"Error saving document splits: {e}"); import traceback; traceback.print_exc()
        else: print("Warning: No splits available to save.")
        return saved_faiss and saved_splits

    def _initialize_bm25_retriever(self):
        if not self.splits: print("Warning: Cannot initialize BM25 retriever, no splits available."); self.bm25_retriever = None; return
        print(f"Initializing BM25 retriever with {len(self.splits)} document splits...")
        try:
            if all(isinstance(doc, Document) for doc in self.splits): docs_for_bm25 = self.splits
            else:
                 print("Warning: Splits are not all Document objects. Extracting page_content."); docs_for_bm25 = [doc.page_content for doc in self.splits if hasattr(doc, 'page_content')]
                 if not docs_for_bm25: raise ValueError("Could not extract text content from splits for BM25.")
            self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
            self.bm25_doc_lookup = {i: doc for i, doc in enumerate(self.splits)} if docs_for_bm25 != self.splits else None
            print("BM25 retriever initialized successfully.")
        except Exception as e: print(f"Error initializing BM25 retriever: {e}"); self.bm25_retriever = None

    def process_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> bool:
        if not documents: print("Error: No documents provided for processing."); return False
        print("Processing text-based content...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False)
            new_splits = text_splitter.split_documents(documents)
            if not new_splits: print("Error: No text splits were created."); return False
            print(f"Created {len(new_splits)} new splits from {len(documents)} documents.")
            if self.splits: print(f"Adding {len(new_splits)} new splits to existing {len(self.splits)} splits."); self.splits.extend(new_splits)
            else: self.splits = new_splits
            print("Creating/Updating FAISS vector store...")
            if self.vectorstore:
                ids = self.vectorstore.add_documents(new_splits); print(f"Added {len(ids)} new documents to the existing FAISS index.")
            else: self.vectorstore = FAISS.from_documents(documents=self.splits, embedding=self.embeddings); print("FAISS vector store created.")
            self._initialize_bm25_retriever()
            if not self._using_temp_dir:
               success = self._save_index_and_splits()
               if not success: print("Warning: Failed to save index/splits.")
            else: print("Index and splits updated in temporary directory.")
            return True
        except Exception as e: print(f"Error during document processing: {e}"); import traceback; traceback.print_exc(); return False

    def hybrid_retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.vectorstore: print("Error: FAISS vectorstore not initialized."); return []
        if not self.bm25_retriever:
             print("Warning: BM25 retriever not initialized. Performing dense search only.")
             dense_results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
             dense_results = [(doc, 1.0 / (1.0 + score)) for doc, score in dense_results_with_scores]
             return sorted(dense_results, key=lambda item: item[1], reverse=True)
        dense_results: List[Tuple[Document, float]] = []; sparse_results_docs: List[Document] = []
        try:
            dense_results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            dense_results = [(doc, 1.0 / (1.0 + max(score, 1e-6))) for doc, score in dense_results_with_scores] # Avoid division by zero if score is -1.0
            print(f"Dense search returned {len(dense_results)} results.")
        except Exception as e: print(f"Error during FAISS search: {e}")
        try:
             sparse_results_docs = self.bm25_retriever.invoke(query, k=k)
             print(f"Sparse search returned {len(sparse_results_docs)} results.")
        except Exception as e: print(f"Error during BM25 search: {e}")
        combined_results: Dict[str, Tuple[Document, float]] = {}
        for doc, score in dense_results:
            content_key = doc.page_content
            if content_key not in combined_results or score > combined_results[content_key][1]: combined_results[content_key] = (doc, score)
        default_sparse_score = 0.1; sparse_boost = 0.05
        for doc in sparse_results_docs:
            content_key = doc.page_content
            if content_key in combined_results:
                 existing_doc, existing_score = combined_results[content_key]
                 combined_results[content_key] = (existing_doc, existing_score + sparse_boost)
            else: combined_results[content_key] = (doc, default_sparse_score)
        sorted_results = sorted(combined_results.values(), key=lambda item: item[1], reverse=True)
        print(f"Hybrid search combined to {len(sorted_results)} unique results.")
        return sorted_results

    def _calculate_dynamic_threshold(self, scores: List[float], method: str = "percentile", percentile: int = 25) -> float:
        if not scores or len(scores) < 2: return -1.0
        scores_array = np.array(scores); threshold = -1.0
        try:
            if method == "percentile": safe_percentile = max(0, min(100, percentile)); threshold = np.percentile(scores_array, safe_percentile)
            elif method == "mean_std":
                mean_score = np.mean(scores_array); std_score = np.std(scores_array)
                threshold = max(mean_score - 1.0 * std_score, np.min(scores_array) * 0.5)
            else: print(f"Warning: Unknown threshold method '{method}'. Using 25th percentile."); threshold = np.percentile(scores_array, 25)
            threshold = max(threshold, 0.0)
        except Exception as e: print(f"Error calculating dynamic threshold: {e}. Returning low threshold.")
        print(f"Calculated dynamic threshold: {threshold:.4f} using {method} (percentile={percentile if method=='percentile' else 'N/A'})")
        return threshold

    def _filter_and_rerank_docs(self, retrieved_docs_with_scores: List[Tuple[Document, float]], threshold_method: str = "percentile", threshold_percentile: int = 25, max_final_docs: int = 5) -> List[Document]:
        if not retrieved_docs_with_scores: return []
        scores = [score for _, score in retrieved_docs_with_scores]
        dynamic_threshold = self._calculate_dynamic_threshold(scores, method=threshold_method, percentile=threshold_percentile)
        filtered_docs_with_scores = [(doc, score) for doc, score in retrieved_docs_with_scores if score >= dynamic_threshold]
        print(f"Filtered docs: {len(retrieved_docs_with_scores)} -> {len(filtered_docs_with_scores)} using threshold {dynamic_threshold:.4f}")
        final_docs_with_scores = filtered_docs_with_scores[:max_final_docs]
        print(f"Returning top {len(final_docs_with_scores)} documents after filtering and ranking.")
        final_docs = [doc for doc, score in final_docs_with_scores]
        return final_docs
    # --- End Previous Code ---


    # --- MODIFIED query method ---
    def query(self,
              query_text: str,
              k: int = 5,
              threshold_percentile: int = 25,
              max_final_docs: int = 4,
              chat_history: Optional[List[Dict[str, str]]] = None # Accept simple history list
             ) -> Tuple[str, List[Document]]: # Return value is still (answer, docs_used_for_context)
        """
        Query the RAG system, performing retrieval and using chat history for context.

        Args:
            query_text (str): The user's latest query.
            k (int): Initial number of documents to retrieve.
            threshold_percentile (int): Percentile for dynamic thresholding.
            max_final_docs (int): Max number of documents to pass to LLM context.
            chat_history (Optional[List[Dict[str, str]]]): List of {"role": ..., "content": ...} dicts.

        Returns:
            tuple: (answer_text, final_docs_used) - final_docs_used are the ones passed to the LLM.
        """
        print(f"\n--- Inside RAG Query Method (Always Retrieves) ---")
        print(f"Received query: '{query_text[:50]}...'")
        print(f"Received chat_history: {'Yes' if chat_history else 'No'} (Length: {len(chat_history) if chat_history else 0})")

        # 1. Always Perform Retrieval and Filtering
        if not self.vectorstore:
            msg = "Vector store not initialized. Please process documents first."
            print(f"Query Error: {msg}")
            # Basic check for persistence path existence
            persist_dir = getattr(self, 'persist_directory', None)
            index_exists = persist_dir and os.path.exists(os.path.join(persist_dir, "faiss_index"))
            splits_exist = persist_dir and os.path.exists(os.path.join(persist_dir, "splits.pkl"))
            if persist_dir and not index_exists and not splits_exist:
                 msg += " No existing index/splits found in the persistence directory."
            return msg, []

        # Perform hybrid retrieval based on the current query_text
        retrieved_docs_with_scores = self.hybrid_retrieve(query_text, k=k)

        # Filter the retrieved documents
        final_docs = self._filter_and_rerank_docs(
            retrieved_docs_with_scores,
            threshold_method="percentile",
            threshold_percentile=threshold_percentile,
            max_final_docs=max_final_docs
        )

        if not final_docs:
            print("No documents passed the confidence threshold for this query.")
            return "I couldn't find relevant information in the documents for this specific question.", []

        # 2. Prepare Context String from Newly Retrieved Docs
        def format_docs(docs: List[Document]):
            return "\n\n".join(f"---\nSource: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
                               if doc.metadata else f"---\nContent: {doc.page_content}"
                               for doc in docs)

        context_string = format_docs(final_docs)
        print(f"Using {len(final_docs)} retrieved documents for context.")

        # 3. Prepare for LLM
        llm = OpenRouterLLM(
            api_key=self.api_key,
            model_name=self.llm_model_name,
            temperature=self.llm_temperature
        )

        # 4. Construct Messages for LLM (including history if available)
        llm_messages = []
        current_user_question = query_text # The latest question from the user

        # Determine if this is a follow-up based on history presence
        is_follow_up_based_on_history = bool(chat_history and len(chat_history) > 0)

        if is_follow_up_based_on_history:
            # System prompt for FOLLOW-UP questions - emphasize explanation
            system_prompt = f"""You are an AI assistant having a conversation. The user is asking a follow-up question: "{current_user_question}".
            Your previous response is in the 'Chat History' below.
            Use the 'Context Documents' provided to explain or elaborate on your previous answer in a natural, conversational, plain text format.
            **Do NOT use JSON or other code-like structures unless explicitly asked to generate code.**
            Focus on providing the reasoning or details requested.
            Answer ONLY based on the provided 'Context Documents' and 'Chat History'. If the answer isn't found, say so clearly.

            Context Documents:
            {context_string}
            ---
            """
            llm_messages.append({"role": "system", "content": system_prompt})
            llm_messages.extend(chat_history) # Add previous Q&A
            llm_messages.append({"role": "user", "content": current_user_question}) # Add current question
        else:
            # System prompt for INITIAL questions
            system_prompt = f"""You are an AI assistant. Answer the user's question: "{current_user_question}"
            Use ONLY the "Context Documents" below.
            **Provide your answer in plain text, as a natural language response.**
            If the question implies a list (e.g., "list the skills", "what are the top questions"), present the items clearly, perhaps using bullet points or numbered lists in plain text, but AVOID JSON or code block formatting unless specifically asked to generate code.
            If the answer isn't found in the documents, say so clearly.

            Context Documents:
            {context_string}
            ---
            """
            llm_messages.append({"role": "system", "content": system_prompt})
            llm_messages.append({"role": "user", "content": current_user_question})
        # 5. Execute Generation
        print(f"Generating response using {self.llm_model_name} with {len(final_docs)} context docs. Is follow-up: {is_follow_up_based_on_history}")
        try:
            llm_result = llm._generate(prompts=llm_messages)

            if llm_result.generations and llm_result.generations[0]:
                response = llm_result.generations[0][0].text or "[LLM returned empty response]"
            else:
                response = "[Error: No generation found in LLM result]"

            return response, final_docs # Return the response and the docs used FOR THIS TURN'S context

        except ConnectionError as e:
             print(f"LLM API Error: {e}")
             return f"Error: Could not connect to the language model API. {e}", final_docs
        except ValueError as e:
             print(f"LLM Response Error: {e}")
             return f"Error: Invalid response from the language model. {e}", final_docs
        except Exception as e:
             print(f"LLM Generation Error: {e}")
             import traceback
             print(traceback.format_exc())
             return f"An unexpected error occurred during response generation: {e}", final_docs