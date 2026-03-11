import chromadb
from chromadb.config import Settings
from typing import List, Optional
from src.schema import DocumentChunk, RetrievalResult
from src.config import settings

class VectorStore:

    def __init__(self) :
        # 1. Initialize the persistent client (saves to your data/ folder)
        self.client = chromadb.PersistentClient(path=str(settings.DB_PATH))

        # 2. Get or create a "Collection" (like a table in a database)
        self.collection = self.client.get_or_create_collection(
            name="local_knowledge_base",
            metadata={"hsnw:space": "conise"} # Use Cosine Similarity for the math
        )
    
    def upsert_chunks(self, chunks: List[DocumentChunk]) :
        """Saves chunks and their embeddings to the database."""
        if not chunks :
            return []
        
        # Prepare data in the format ChromaDB expects
        ids = [c.id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # 'Upsert' means: If the ID exists, update it. If not, insert it.
        self.collection.upsert(
            id=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Successfully stored {len(chunks)} chunks in the Vector DB.")

    def search(self, query_embedding: List[float], top_k: int=3) -> List[RetrievalResult] :
        """Finds the most similar chunks for a given query vector."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieval_results = []
        # ChromaDB returns nested lists; we flatten them into our Schema
        for i in range(len(results["ids"][0])):
            chunk = DocumentChunk(
                id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            )
            # 'distances' represent how 'far' the result is; we convert to a score
            score = results["distances"][0][i]
            retrieval_results.append(RetrievalResult(chunk=chunk, score=score))

        return retrieval_results