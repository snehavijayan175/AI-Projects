import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, vector_store: FaissVectorStore, groq_api_key: str):
        self.vector_store = vector_store
        self.groq_client = ChatGroq(api_key=groq_api_key)
        print("[INFO] RAG Search initialized with Groq client and FAISS vector store.")

    def retrieve(self, query: str, top_k: int = 3):
        print(f"[INFO] Retrieving top {top_k} documents for the query: {query}")
        results = self.vector_store.query_vector(query, top_k=top_k)
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Distance: {result['distance']}):\n{result['metadata']['text'][:500]}")  # Print first 500 characters
        return results

