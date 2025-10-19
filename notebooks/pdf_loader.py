## RAG Pipelines - Data Ingestion to Vector DB Pipeline

# Importing the necessary libraries
from msilib.schema import Class
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import torch


# Read all the files inside the directory and sub-directories   
def process_all_pdfs(directory_path):
    """Process all PDF files in the given directory and return a list of documents."""
    all_documents = []
    pdf_files = list(Path(directory_path).rglob("*.pdf"))  # Recursively find all PDF files

    print(f"Found {len(pdf_files)} PDF files.")

#Batch PDF Document Loading and Metadata Enrichment using PyMuPDFLoader
    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            #Add source information also to the document metadata
            for doc in documents:
                doc.metadata["source"] = str(pdf_file)
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages from {pdf_file}")

        except Exception as e:
            print(f"Following error occurred while loading {pdf_file} with PyMuPDFLoader: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")

    return all_documents

all_pdf_documents = process_all_pdfs("data/pdf/")
#print(all_pdf_documents) # Print the first document to verify


# Text Splitting documents into chunks

def text_splitting(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunk for better RAG Performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    try:
        splits = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(splits)} chunks.")
    except Exception as e:
        print(f"Error splitting document {documents.metadata.get('source', 'unknown')}: {e}")

    if splits:
        print(f"\nFirst Chunk as Example Chunk:")
        print(f"First chunk preview: {splits[0].page_content[:200]}...")  # Print first 200 characters of the first chunk
        print(f"First chunk metadata: {splits[0].metadata}")
        print(f"First Chunk Content: {splits[0].page_content}")
    return splits

split_chunks = text_splitting(all_pdf_documents, chunk_size=1000, chunk_overlap=100)
print(split_chunks[0]) # Print the first chunk to verify


# Embedding & VectorStore DB

import numpy as np
from sentence_transformers import SentenceTransformer ## Embedding Model
#from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingManager:
    """The class manages embedding generation using a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the Sentence Transformer Embedding model.

        Args:
            model_name (str): HuggingFace model name for sentence embeddings.
        """
        self.model = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the Sentence Transformer embedding model. """
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name} successfully. Embedding Dimension: {self.model.get_sentence_embedding_dimension()}")

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for list of texts.

        Args:
            text : list of texts to generate embeddings for.

        Returns :
        numpy array of embeddings with shape (len(text), embedding_dimension)
        """

        if self.model is None:
            raise ValueError("Embedding model is not loaded.")
        
        print(f"Generating embeddings for {len(text)} number of texts...")
        embeddings = self.model.encode(text, show_progress_bar=True, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings

    """
    def embedding_dimension(self) -> int:
        # Return the dimension of the embeddings
        if self.model is None:
            raise ValueError("Embedding model is not loaded.")
        return self.model.get_sentence_embedding_dimension() # Not necessary to have a separate function created to check the embedding dimension and print it out. As you can see its already included under _load_model() function.
    """

# Initialize Embedding Manager
embedding_manager = EmbeddingManager()
embedding_manager


# VECTOR STORE - ChromaDB

class VectorStoreManager:
    """Manages ChromaDB vector store operations including adding, querying, and deleting documents."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/VectorStore/chroma_db"):
        """
        Initialize ChromaDB client and collection.

        Args:
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory to persist the vector store or ChromaDB data.
        """
        self.client = None
        self.collection = None
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._initialize_store()  # Connect to or create the index / storage layer where your RAG data lives.
    
    def _initialize_store(self):
        """Initialize the ChromaDB Client and Collection."""
        try:
            # Create persistent ChromaDB Client
            os.makedirs(self.persist_directory, exist_ok=True)  # Create directory if it doesn't exist
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Documents Embeddings For RAG"}
            )

            print(f"✅ Initialized ChromaDB vector store: {self.collection_name}")
            print(f"📦 Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"❌ Error initializing ChromaDB vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and embeddings to the ChromaDB collection.

        Args:
            documents (List[Any]): List of documents with 'page_content' and 'metadata'.
            embeddings (np.ndarray): Corresponding embeddings for list of LangChain documents.
        """
        if not documents:
            print("⚠️ No documents to add.")
            return
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")
        
        print(f"🧩 Adding {len(documents)} documents to the vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())  # Generate a unique ID for each document. # Generate a unique ID for each document # can try out another unique ID creation as per Krish Naik's RG LangChain Video also required
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata) if hasattr(doc, "metadata") else {}
            metadata["chunk_index"] = i
            metadata["content_length"] = len(getattr(doc, "page_content", ""))
            metadatas.append(metadata)

            # Prepare document text
            documents_text.append(getattr(doc, "page_content", ""))

            # Prepare embeddings
            embeddings_list.append(emb.tolist())  # Convert numpy array to list

        # Add to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            
            print(f"✅ Added {len(documents)} documents to the vector store.")
            print(f"📈 Total documents in collection now: {self.collection.count()}")

        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            raise

# Instantiate and verify initialization
vector_store_manager = VectorStoreManager()
vector_store_manager


# Convert Text to Embeddings

text = [doc.page_content for doc in split_chunks]
embeddings = embedding_manager.generate_embedding(text)
print(f"Embeddings shape: {embeddings.shape}")

# Store the embeddings in the chromadb collections
vector_store_manager.add_documents(split_chunks, embeddings)


## RAG Pipelines - Retriever Pipeline From Vector Store
# 
class RetrieverManager:
    """Manages retrieval of relevant documents from ChromaDB based on query embeddings."""

    def __init__(self, vector_store_manager: VectorStoreManager, embedding_manager: EmbeddingManager):
        """
        Initialize with existing VectorStoreManager and EmbeddingManager instances.

        Args:
            vector_store_manager (VectorStoreManager): Instance managing the ChromaDB vector store.
            embedding_manager (EmbeddingManager): Instance managing the embedding generation.
        """    

        self.vector_store_manager = vector_store_manager
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve top_k relevant documents from the vector store based on the query.

        Args:
            query (str): The input query string.
            top_k (int): Number of top relevant documents to retrieve.
            score_threshold (float): Minimum similarity score threshold for retrieved documents.
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing retrieved documents with metadata.
        """
        if not query:
            print("⚠️ Query is empty.")
            return []

        print(f"🔍 Retrieving top {top_k} documents for the query: '{query}'")

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embedding([query])[0].reshape(1, -1)

        # Fetch all embeddings and metadata from the collection
        try:
            results = self.vector_store_manager.collection.get(
                include=["embeddings", "metadatas", "documents"]
            )
            all_embeddings = np.array(results["embeddings"])
            all_metadatas = results["metadatas"]
            all_documents = results["documents"]

            if all_embeddings.size == 0:
                print("⚠️ No embeddings found in the vector store.")
                return []

            # Compute cosine similarities
            similarities = cosine_similarity(query_embedding, all_embeddings)[0]

            # Get top_k indices
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]

            # Prepare retrieved documents
            retrieved_docs = []
            for idx in top_k_indices:
                retrieved_docs.append({
                    "document": all_documents[idx],
                    "metadata": all_metadatas[idx],
                    "similarity": float(similarities[idx])
                })

            print(f"✅ Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs

        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []