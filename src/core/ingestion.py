import os
from pathlib import Path
from typing import List, Generator
from src.schema import DocumentChunk
from src.config import settings
import hashlib

class FileProcessor :
    
    def __init__(self, directory:Path = settings.DATA_DIR):
        self.directory = directory
    
    def getfiles(self, extenions: List[str] = [".txt", ".md", ".py"]) -> Generator[Path, None, None] :
        """Recursively yields files with the specified extensions."""
        for path in self.directory.rglob("*") :
            if path.is_file() and path.suffix in extenions :
                yield path

    def read_file(self, file_path: Path) -> str :
        """Reads the content of a file."""
        try :
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def process_all(self) -> List[DocumentChunk] :
        """Orchestrates the ingestion of all files."""
        all_chunks = []
        for file_path in self.get_files() :
            content = self.read_file(file_path)
            if content :
                # We will delegate the chunking logic to a separate method
                chunks = self.chunk_text(content, str(file_path))
                all_chunks.extend(chunks)
        return all_chunks
    
    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """ Splits text into overlapping chunks based on Config settings."""
        chunks = []
        text_len = len(text)

        # Calculate the step : how far the window moves each iteration
        step = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP

        # Hanldle edge case : text shorter than chunk size
        if text_len <= settings.CHUNK_SIZE :
            return [self._create_chunk(text, source, 0)]
        for i in range(0, text_len, step):
            # Grab a slice of text
            chunk_content = text[i : i + settings.CHUNK_SIZE]
            
            # Create the DocumentChunk object
            new_chunk = self._create_chunk(chunk_content, source, i)
            chunks.append(new_chunk)
            
            # Break if we've reached the end of the text
            if i + settings.CHUNK_SIZE >= text_len:
                break
                
        return chunks

    def _create_chunk(self, content: str, source: str, start_index: int) -> DocumentChunk:
        """Helper to build a DocumentChunk with metadata and a unique ID."""
        # Create a unique ID based on source path and start position
        # This prevents duplicate chunks if you re-run the ingestion
        unique_string = f"{source}_{start_index}"
        chunk_id = hashlib.md5(unique_string.encode()).hexdigest()
        
        metadata = {
            "source": source,
            "start_index": start_index,
            "char_count": len(content)
        }
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata
        )       
