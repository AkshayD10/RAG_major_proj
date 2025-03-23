# reset_and_reinitialize.py
import shutil
from rag import DocumentRAG

# Step 1: Delete the Chroma collection directory
shutil.rmtree("./chroma_db", ignore_errors=True)
print("Chroma collection deleted.")

# Step 2: Reinitialize the RAG system
rag = DocumentRAG(openrouter_api_key="your_openrouter_api_key", embedding_model="sentence-transformers/all-mpnet-base-v2")

# Step 3: Process documents to create a new collection
documents = [...]  # Your list of documents
rag.process_documents(documents)
print("New Chroma collection created.")