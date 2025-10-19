## RAG Pipelines - Data Ingestion to Vector DB Pipeline

# Importing the necessary libraries
from msilib.schema import Class
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import torch


print(f"\n\n")


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

all_pdf_documents = process_all_pdfs("/data/pdf/")
#print(f"all_pdf_documents) # Print the first document to verify



print(f"\n\n")


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
        print(f"\nFirst chunk preview: {splits[0].page_content[:200]}...")  # Print first 200 characters of the first chunk
        print(f"First chunk metadata: {splits[0].metadata}")
        print(f"First Chunk Content: {splits[0].page_content}")
    return splits

split_chunks = text_splitting(all_pdf_documents, chunk_size=1000, chunk_overlap=100)
print(f"{split_chunks[0]}") # Print the first chunk to verify



print(f"\n\n")



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

    def generate_embedding(self, chunk_text: str) -> np.ndarray:
        """Generate embedding for list of chunk texts.

        Args:
            chunk_text : list of texts to generate embeddings for.

        Returns :
        numpy array of embeddings with shape (len(chunk_text), embedding_dimension)
        """

        if self.model is None:
            raise ValueError("Embedding model is not loaded.")
        
        print(f"Generating embeddings for {len(chunk_text)} number of chunk texts...")
        #embeddings = self.model.encode(chunk_text, show_progress_bar=True, batch_size=16, convert_to_numpy=True, device="cuda" if torch.cuda.is_available() else "cpu")
        embeddings = self.model.encode(chunk_text, show_progress_bar=True)

        print(f"embeddings created")
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



print(f"\n\n")



# VECTOR STORE - ChromaDB

class VectorStoreManager:
    """Manages ChromaDB vector store operations including adding, querying, and deleting documents."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "/data/VectorStore/chroma_db"):
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
        print(1)
        self._initialize_store()  # Connect to or create the index / storage layer where your RAG data lives.
    
    def _initialize_store(self):
        """Initialize the ChromaDB Client and Collection."""
        try:
            # Create persistent ChromaDB Client
            os.makedirs(self.persist_directory, exist_ok=True)  # Create directory if it doesn't exist
            print(2)

            try:
                self.client = chromadb.PersistentClient(path=self.persist_directory,)
                print(3)
            except Exception as e:
                print(f" Error creating Chromadb Client :{e}")

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Documents Embeddings For RAG"}
            )
            print(4)

            print(f"✅ Initialized ChromaDB vector store: {self.collection_name}")
            print(f"📦 Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"❌ Error initializing ChromaDB vector store: {e}")
            raise

    def add_documents(self, chunks: List[Any], embeddings: np.ndarray):
        """
        Add chunks document and embeddings to the ChromaDB collection.

        Args:
            chunks (List[Any]): List of chunks with 'page_content' and 'metadata'.
            embeddings (np.ndarray): Corresponding embeddings for list of LangChain chunks.
        """
        if not chunks:
            print("⚠️ No chunks to add as documents.")
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")
        
        print(f"🧩 Adding {len(chunks)} chunks as documents to the vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        chunks_text_doc = []
        embeddings_list = []

        print(3)

        for i, (doc, emb) in enumerate(zip(chunks, embeddings)):
            print(4)
            doc_id = str(uuid.uuid4())  # Generate a unique ID for each chunk document. # Generate a unique ID for each chunk document # can try out another unique ID creation as per Krish Naik's RG LangChain Video also required
            ids.append(doc_id)
            print(5)

            # Prepare metadata
            metadata = dict(doc.metadata) if hasattr(doc, "metadata") else {}
            print(6)

            metadata["chunk_index"] = i
            metadata["content_length"] = len(getattr(doc, "page_content", ""))
            print(7)

            metadatas.append(metadata)
            print(8)


            # Prepare document text
            chunks_text_doc.append(getattr(doc, "page_content", ""))
            print(9)


            # Prepare embeddings
            embeddings_list.append(emb.tolist())  # Convert numpy array to list
            print(10)
            
            print (f"count: {i}")

        
        # Add to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=chunks_text_doc
            )
            
            print(f"✅ Added {len(documents)} chunk documents to the vector store.")
            print(f"📈 Total chunk documents in collection now: {self.collection.count()}")

        except Exception as e:
            print(f"❌ Error adding chunk documents to vector store: {e}")
            raise

# Instantiate and verify initialization
vector_store_manager = VectorStoreManager()
vector_store_manager


print(f"\n\n")


# Convert Text to Embeddings

chunk_text = [doc.page_content for doc in split_chunks]
print(1)
print(f"{max(len(t) for t in chunk_text)}")
embeddings = embedding_manager.generate_embedding(chunk_text[:5])
print(2)
print(f"Embeddings shape: {embeddings.shape}")

# Store the embeddings in the chromadb collections
vector_store_manager.add_documents(split_chunks[:5], embeddings)
print(11)


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

        print(f"🔍 Retrieving top {top_k} documents for the query: '{query}' and score_threshold: {score_threshold}")

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        # Fetch all embeddings and metadata from the collection
        try:
            results = self.vector_store_manager.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=top_k
            )

            # Process Results

            retrieved_docs = []
            if results['documents'] and results["documents"][0]:
                all_metadatas = results["metadatas"][0]
                all_documents = results["documents"][]
                all_distances = results["distances"][0]
                all_ids = results["ids"][0]

                for i, (doc, meta, dist, doc_id) in enumerate(zip(all_documents, all_metadatas, all_distances, all_ids)):
                    similarity = 1 - dist  # Convert distance to similarity
                    if similarity >= score_threshold:
                        retrieved_docs.append({
                            "document": doc,
                            "metadata": meta,
                            "similarity": similarity,
                            "id": doc_id,
                            "distance": dist
                            "rank": i + 1
                        })
                print(f"✅ Retrieved {len(retrieved_docs)} documents after filtering with score threshold set to {score_threshold}.")
            else:
                print("⚠️ No documents found in the vector store.")
            return retrieved_docs
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return []
        
 # Instantiate Retriever Manager
RAG_retriever_manager = RetrieverManager(vector_store_manager, embedding_manager)
RAG_retriever_manager

RAG_retriever_manager.retrieve("What are Raw NMEA encoded AIS messages?")

# NOTE: 
# The Query Retrieval Pipeline we just built is the 'retrieval' part of RAG, which provides the external knowledge that augments the generative process.  
# The next step is to build the generative part of RAG — the LLM model — and then combine both pipelines to form a complete RAG system.  
# The retrieved documents (embeddings + metadata) form the 'context', which, along with a carefully designed 'prompt' instructing the LLM how to respond, is passed to the LLM to generate the final answer that is Augmented by the external knowledge to create the augmented response.

## Integration of VectorDB Context Pipeline with LLM Model output for Generative Q&A

# Simple RAG Pipeline with Groq LLM Model
from langchain import PromptTemplate, LLMChain
from langchain.llms import Groq 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever


load_dotenv()

# Initialize Groq LLM and set the Groq API Key in environment.
groq_api_key = os.getenv("GROQ_API_KEY", "your_groq_api_key")
llm = ChatGroq(model="gemma2_9b_it", api_key=groq_api_key, temperature=0.1, max_tokens=1024)  #model="groq-3.5-turbo"

# Simple RAG Function : Retrieve Context + Generate Answer
def rag_simple(query: str, retriever: BaseRetriever, llm_model, top_k=3):
    #Retrieve relevant documents from the vector store (Retrieve Context)
    relevant_docs = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["document"] for doc in relevant_docs]) if relevant_docs else ""
    print(f"\nContext Retrieved:\n{context}\n")
    if not context:
        return "No relevant context found for the query"
        

    # Generate answer using the Groq LLM model (Generate Answer)
    prompt = f"""Use the following context to answer the question concisely.
    \n\nContext: {context}
    \n\nQuestion: {query}
    \n\nAnswer:"""
    answer = llm_model.generate_answer(query, context=relevant_docs)

    response = llm_model.invoke(prompt.format(context=context, question=query))
    return response.content

# Example Query
answer = rag_simple("Explain the different types of AIS messages?", RAG_retriever_manager, llm, top_k=3)
print(f"\nFinal Answer:\n{answer}\n")


# Enhanced RAG Pipeline Features

def rag_enhanced(query: str, retriever: BaseRetriever, llm_model, top_k=3, score_threshold=0.1):

    """
    RAG pipeline with extra features : - Returns Answers, Sources, confidence score, and optionally full context.
    """

    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {'answer': "No relevant context found for the query", 'sources': [], 'confidence': 0.0, 'context': ""}
    
    # Prepare context and sources
    context = "\n\n".join([doc["document"] for doc in results])
    sources = [{'source': doc['metadata'].get('source_file', doc["metadata"].get("source", "unknown")),
                'page': doc ['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview':doc['content'][:300]+ '...'} for doc in results]
    

    confidence = max(doc['similarity_score'] for doc in results)


    # Generate answer using the Groq LLM model (Generate Answer)
    prompt = f"""Use the following context to answer the question concisely.
    \n\nContext: {context}
    \n\nQuestion: {query}
    \n\nAnswer:"""

    response = llm_model.invoke(prompt.format(context=context, question=query))

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }

    if return_context:
        output['context'] = context
    return output

Example Usage :
result = rag_enhanced("Explain the different types of AIS messages?", RAG_retriever_manager, llm, top_k=3, score_threshold=0.1, return_context=True)
print(f"\nFinal Answer:\n{result['answer']}\n")
print(f"Sources:\n{result['sources']}\n")
print(f"Confidence Score: {result['confidence']}\n")
print(f"Context Preview:\n{result.get('context', '')}\n")





