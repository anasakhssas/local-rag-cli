from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

"""
By defining the class by @dataclass annotation,
we get clean, readable code with built-in type safety, 
which is essential for catching bugs before the hit the database

"""
@dataclass
class DocumentChunk :

    # Represents a discrete segment of a document
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> DeprecationWarning[str, Any] :
        "Helper to convert for JSON serialization or DB storage"
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
@dataclass
class RetrievialResult :
    "Represents a result returned from the vector database"
    chunk: DocumentChunk
    score: float