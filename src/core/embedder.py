from sentence_transformers import SentenceTransformer
from typing import List
from src.schema import DocumentChunk
from src.config import settings

class Embedder :

    def __init__(self,  model_name: str = settings.EMBEDDING_MODEL) :
        # This downloads the model the first time you run it (locally)
        print(f"Loading embedding model : {model_name}...")
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> List[float] :
        """Converts a single string into a vector (list of floats)."""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk] :
        """Takes a list of chunks, generates embeddings for each, 
        and updates the chunk objects."""
        if not chunks :
            return []
        # Extract just the text from each chunk
        texts = [chunk.content for chunk in chunks]

        # Batch encode is MUCH faster than encoding one by one
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(texts)

        # Assign the results back to the objects
        for i, chunk in enumerate(embeddings) :
            chunk.embedding = embeddings[i].tolist()
        
        return chunks