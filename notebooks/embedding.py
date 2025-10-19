from typing import List, Dict, Any
from langchain_community.text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents


class EmbeddingPipeline:
    """
    A pipeline for processing and embedding documents.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Dict[Any]]) -> List[Dict[Any]]:
        """
        Chunk documents into smaller pieces.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunked_docs = text_splitter.split_documents(documents)
        return chunked_docs # returns a list of dictionaries
    
        print(f"[INFO] Chunked documents into {len(chunked_docs)} chunk pieces.")
        return chunked_docs
    
    def embed_chunks(self, chunks: List[Dict[Any]]) -> np.ndarray:
        """
        Embed a list of documents.
        """
        chunk_texts = [doc['content'] for doc in chunks]
        print(f"[INFO] Now Embedding {len(chunk_texts)} chunks..")
        embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
        return embeddings
