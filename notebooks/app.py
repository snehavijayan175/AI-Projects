from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

#Example Usage

if _name_=="_main__":
    data_directory = "./data"
    documents = load_all_documents(data_directory)
    print(f"Final Number of documents loaded: {len(documents)}")

    for i, doc in enumerate(documents[:5]):  # Print first 5 documents as a sample
        print(f"\nDocument {i+1}:\n{doc.page_content[:500]}")  # Print first 500 characters of each document
        print(f"\nMetadata: {doc.metadata}")

    chunks = EmbeddingPipeline().chunk_documents(documents)
    print(f"Total number of chunks created: {len(chunks)}")    
    chunk_vectors = EmbeddingPipeline().embed_chunks(chunks)
    print(f"Shape of chunk embeddings: {chunk_vectors.shape}")
    print(f"Sample Embedding Vector for first chunk:\n{chunk_vectors[0]}")

    metadatas = [{"text":chunk.page_content} for chunk in chunks]
    vector_store = FaissVectorStore()
    vector_store.add_embeddings(np.array(chunk_vectors).astype(np.float32), metadatas)
    print("FAISS Vector Store created and embeddings added.")
    vector_store.load_index()
    query = "What are Raw NMEA encoded AIS messages?"
    results = vector_store.query_vector(query, top_k=3)
    print(f"Top 3 results for the query '{query}':")




