from src.config import settings
from src.core.ingestion import FileProcessor
from src.core.embedder import Embedder
from src.core.vector_db import VectorStore

def run_indexing_pipeline() :

    # 1. Initialize the system
    print("--- Initializing Local RAG Indexer ---")
    settings.ensure_directories()

    processorc = FileProcessor()
    embedder = Embedder()
    vector_db = VectorStore()

    # 2. Ingestion: Find and Chunk Files
    # This automatically uses the 'is_dirty' check we built
    print("\n[1/3] Scanning for new/modified documents...")
    new_chunks = processorc.process_all()

    if not new_chunks :
        print("Done. No new data to process.")
        return
    
    # 3. Embedding: Turn text into Math
    print(f"\n[2/3] Converting {len(new_chunks)} chunks to vectors...")
    chunks_with_embeddings = embedder.embed_chunks(new_chunks)

    # 4. Storage: Save to the Vector DB
    print("\n[3/3] Saving to local Vector Database...")
    vector_db.upsert_chunks(chunks_with_embeddings)

    print("\n--- Indexing Complete! Your knowledge base is ready. ---")

if __name__ == "__main__":
    run_indexing_pipeline()