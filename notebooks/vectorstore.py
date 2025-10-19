import os
import faiss
import numpy as np
from typing import List, Dict, Any
import pickle
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline
from langchain.vectorstores import FAISS


class FAISSVectorStore:
    """
    A class to handle FAISS vector store operations.
    """
    def __init__(self, persist_directory: str = 'faiss_store', embedding_model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_directory
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.metadata = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Initialized FAISSVectorStore with embedding model: {embedding_model_name}")

    """
    def build_index(self, chunks: List[Dict[Any]]):
        
        #Build a FAISS index from document chunks.
        
        print(f"[INFO] Building FAISS index from {len(chunks)} chunks...")
        chunk_texts = [doc['content'] for doc in chunks]
        print(f"[INFO] Embedding {len(chunk_texts)} chunks for FAISS index...")
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.documents = chunks
        """

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} embeddings to FAISS index. Total is now {self.index.ntotal}.")
        self.save_index()                                                                                                      

    def save_index(self):
        """
        Save the FAISS index embeddings and metadata to disk.
        """
        index_embeddings_path = os.path.join(self.persist_dir, 'faiss_index_embeddings.idx')
        metadata_path = os.path.join(self.persist_dir, 'faiss_metadata.pkl')
        faiss.write_index(self.index, index_embeddings_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index to {index_embeddings_path} and metadata to {metadata_path}.")

    def load_index(self):
        """
        Load the FAISS index embeddings and metadata from disk.
        """
        index_embeddings_path = os.path.join(self.persist_dir, 'faiss_index_embeddings.idx')
        metadata_path = os.path.join(self.persist_dir, 'faiss_metadata.pkl')
        self.index = faiss.read_index(index_embeddings_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded FAISS index Embeddings and Metadata from {self.persist_dir}.")
        print(f"[INFO] Loaded FAISS index from {index_embeddings_path} with {self.index.ntotal} vectors and metadata from {metadata_path}.")

    def search(self, query_embeddings: np.ndarray, top_k: int = 5):
        """
        Search the FAISS index for the most similar document chunks to the query.
        """
        distances, indices = self.index.search(query_embeddings, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
         return results

    def query_vector(self, query_text: str, top_k: int = 5):
        """   
        Query the FAISS index with its text input converted to embedding.
        """
        print(f"[INFO] Querying FAISS index for: {query_text}")
        query_embedding = self.embedding_model.encode([query_text]).astype(np.float32)
        return self.search(query_embedding, top_k=top_k)