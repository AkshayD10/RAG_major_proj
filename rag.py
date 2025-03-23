# import os
# import requests
# import json
# from typing import List, Optional
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.language_models import BaseLLM
# from langchain_core.callbacks import CallbackManagerForLLMRun
# from langchain_core.outputs import LLMResult, Generation

# # Custom OpenRouter LLM class
# class OpenRouterLLM(BaseLLM):
#     api_key: str
#     model_name: str = "deepseek/deepseek-r1-zero:free"  # Model ID for DeepSeek-R1-Zero
#     api_url: str = "https://openrouter.ai/api/v1/chat/completions"  # OpenRouter API endpoint

#     def __init__(self, api_key=None, **kwargs):
#         super().__init__(**kwargs)
#         self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

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
#     def __init__(self, openrouter_api_key, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
#         """
#         Initialize the RAG system
        
#         Args:
#             openrouter_api_key (str): API key for OpenRouter
#             embedding_model (str): HuggingFace embedding model name
#         """
#         self.api_key = openrouter_api_key
#         os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
#         self.embedding_model = embedding_model
#         self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
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
            
#         # Split documents
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size, 
#             chunk_overlap=chunk_overlap
#         )
#         splits = text_splitter.split_documents(documents)
        
#         if not splits:
#             return False
            
#         # Create vectorstore
#         self.vectorstore = Chroma.from_documents(
#             documents=splits, 
#             embedding=self.embeddings
#         )
        
#         return True
    
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
#         2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
#         3. Be concise and clear
#         4. If appropriate, cite which specific document(s) your information comes from
        
#         Answer:
#         """
#         prompt = ChatPromptTemplate.from_template(template)
        
#         # Create LLM with the custom OpenRouterLLM class
#         llm = OpenRouterLLM(api_key=self.api_key)
        
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

#comment comment

import os
import requests
import json
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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
    api_key: str = Field(..., description="OpenRouter API key")  # Marked as required
    model_name: str = "deepseek/deepseek-r1-zero:free"  # Model ID for DeepSeek-R1-Zero
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"  # OpenRouter API endpoint

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)  # Ensure api_key is passed to Pydantic

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
    def __init__(self, openrouter_api_key, embedding_model="entence-transformers/all-mpnet-base-v2", persist_directory="./chroma_db"):
        """
        Initialize the RAG system
        
        Args:
            openrouter_api_key (str): API key for OpenRouter
            embedding_model (str): HuggingFace embedding model name
            persist_directory (str): Directory to store ChromaDB data
        """
        self.api_key = openrouter_api_key
        os.environ['OPENROUTER_API_KEY'] = openrouter_api_key
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.persist_directory = persist_directory

        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)


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
                return False
                
            # Create vectorstore with persist_directory
            self.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the vectorstore
            self.vectorstore.persist()
            
            return True
        except Exception as e:
            print(f"Error creating Chroma collection: {e}")
            return False


    # def process_documents(self, documents, chunk_size=500, chunk_overlap=50):
    #     """
    #     Process documents and create a vectorstore
        
    #     Args:
    #         documents (list): List of Document objects
    #         chunk_size (int): Size of each text chunk
    #         chunk_overlap (int): Overlap between chunks
            
    #     Returns:
    #         bool: Whether processing was successful
    #     """
    #     if not documents:
    #         return False
            
    #     # Split documents
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size, 
    #         chunk_overlap=chunk_overlap
    #     )
    #     splits = text_splitter.split_documents(documents)
        
    #     if not splits:
    #         return False
            
    #     # Create vectorstore with persist_directory
    #     self.vectorstore = Chroma.from_documents(
    #         documents=splits, 
    #         embedding=self.embeddings,
    #         persist_directory=self.persist_directory  # Add this line
    #     )
        
    #     # Persist the vectorstore
    #     self.vectorstore.persist()
        
    #     return True
    
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
        llm = OpenRouterLLM(api_key=self.api_key)
        
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