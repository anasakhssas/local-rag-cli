from typing import List
from src.core.embedder import Embedder
from src.core.vector_db import VectorStore
from src.schema import RetrievalResult

class Retriever :

    def __init__(self, embedder: Embedder, vector_db: VectorStore) :
        self.embedder = embedder
        self.vector_db = vector_db
    
    def retrieve(self, query: str, top_k: int=3) -> List[RetrievalResult] :
        """
        The core retrieval logic:
        1. Embed the query
        2. Search the Vector DB
        3. Return the best context fragments
        """
        print(f"Searching for : '{query}'...")

        # Step 1: Turn the query into math
        query_vector = self.embedder.embed_text(query)

        # Step 2: Query the database
        results = self.vector_db.search(query_vector, top_k=top_k)

        # Optional: Log the quality of the match (the 'score')
        for i, res in enumerate(results):
            print(f"  Match {i+1}: Score {res.score:.4f} (from {res.chunk.metadata['source']})")
            
        return results
        