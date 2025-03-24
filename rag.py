
# import os
# import requests
# import json
# import tempfile
# import pickle
# from typing import List, Optional
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# # from langchain_community.vectorstores import FIASS
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
#             persist_directory (str): Directory to store ChromaDB data
#         """
#         self.api_key = openrouter_api_key
#         os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
#         self.embedding_model = embedding_model
        
#         # Create embeddings with a smaller model for faster processing
#         self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
#         # Use temporary directory if no persist_directory is provided
#         # This will work better on Streamlit Cloud
#         self.persist_directory = persist_directory or tempfile.mkdtemp()
#         self.vectorstore = None


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
                
#             # Create vectorstore with persist_directory
#             self.vectorstore = Chroma.from_documents(
#                 documents=splits, 
#                 embedding=self.embeddings,
#                 persist_directory=self.persist_directory
#             )
            
#             # No need to call persist() as it's automatically done when using persist_directory
            
#             return True
#         except Exception as e:
#             print(f"Error creating Chroma collection: {e}")
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

#using FIASS

import os
import requests
import json
import tempfile
import pickle
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

# Custom OpenRouter LLM class
from pydantic import BaseModel, Field

class OpenRouterLLM(BaseLLM, BaseModel):
    api_key: str = Field(..., description="OpenRouter API key")
    model_name: str = "deepseek/deepseek-r1-zero:free"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **kwargs
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "openrouter-deepseek-r1-zero"

class DocumentRAG:
    def __init__(self, openrouter_api_key, embedding_model="sentence-transformers/all-mpnet-base-v2", persist_directory=None):
        """
        Initialize the RAG system
        
        Args:
            openrouter_api_key (str): API key for OpenRouter
            embedding_model (str): HuggingFace embedding model name
            persist_directory (str): Directory to store FAISS index and documents
        """
        self.api_key = openrouter_api_key
        os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
        self.embedding_model = embedding_model
        
        # Create embeddings with a smaller model for faster processing
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Use temporary directory if no persist_directory is provided
        self.persist_directory = persist_directory or tempfile.mkdtemp()
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.index_path = os.path.join(self.persist_directory, "faiss_index")
        self.documents_path = os.path.join(self.persist_directory, "documents.pkl")
        
        self.vectorstore = None
        
        # Try to load existing index if it exists
        self._load_index()

    def _load_index(self):
        """Load FAISS index if it exists"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
                print(f"Loading existing FAISS index from {self.index_path}")
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("FAISS index loaded successfully")
                return True
        except Exception as e:
            print(f"Error loading existing FAISS index: {e}")
        return False

    def _save_index(self):
        """Save FAISS index and document mapping"""
        if self.vectorstore:
            try:
                # Save FAISS index
                self.vectorstore.save_local(self.index_path)
                print(f"FAISS index saved to {self.index_path}")
                return True
            except Exception as e:
                print(f"Error saving FAISS index: {e}")
                import traceback
                traceback.print_exc()
        return False

    def process_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """
        Process documents and create a vectorstore
        
        Args:
            documents (list): List of Document objects
            chunk_size (int): Size of each text chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            bool: Whether processing was successful
        """
        if not documents:
            return False
            
        try:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            splits = text_splitter.split_documents(documents)
            
            if not splits:
                print("No splits were created from the documents")
                return False
                
            print(f"Created {len(splits)} splits from documents")
                
            # Create FAISS vectorstore
            self.vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=self.embeddings
            )
            
            # Save the index for persistence
            success = self._save_index()
            if not success:
                print("Warning: Failed to save FAISS index")
            
            return True
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_retriever(self, k=4):
        """
        Get a retriever from the vectorstore
        
        Args:
            k (int): Number of documents to retrieve
            
        Returns:
            Retriever: A retriever object or None
        """
        if not self.vectorstore:
            return None
            
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def query(self, query_text, k=4):
        """
        Query the RAG system
        
        Args:
            query_text (str): The query text
            k (int): Number of documents to retrieve
            
        Returns:
            tuple: (answer, retrieved_docs)
        """
        if not self.vectorstore:
            return "No documents have been processed yet.", []
            
        # Get retriever
        retriever = self.get_retriever(k)
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query_text)
        
        # Format docs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # Create prompt
        template = """You are a helpful AI assistant that provides accurate information based only on the given context.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Answer based ONLY on the information provided in the context
        3. Be concise and clear
        4. If the context does not explicitly contain the information needed to answer the question, make a logical inference based on the available information.
        5. If appropriate, cite which specific document(s) your information comes from
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create LLM with the custom OpenRouterLLM class
        llm = OpenRouterLLM(api_key=self.api_key, model_name=getattr(self, 'model_name', 'deepseek/deepseek-r1-zero:free'))
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Execute chain
        response = rag_chain.invoke(query_text)
        
        return response, docs