import os
from pathlib import Path
from typing import List, Generator
from src.schema import DocumentChunk
from src.config import settings
import hashlib
import json

class FileProcessor :
    
    def __init__(self, directory:Path = settings.DATA_DIR):
        self.directory = directory
        self.state_path = settings.DATA_DIR / "ingestion_state.json"
        self.state = self._load_state()
    
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
        """Orchestrates the ingestion (of all files) with a dirty check."""
        all_chunks = []
        files_processed = 0

        for file_path in self.get_files() :
            content = self.read_file(file_path)
            if content :
                # We will delegate the chunking logic to a separate method
                chunks = self.chunk_text(content, str(file_path))
                all_chunks.extend(chunks)
                files_processed += 1
            else :
                # In a real system, you might load existing chunks from the DB here
                continue
        
        if files_processed > 0 :
            self._save_state()
            print(f"Ingested {files_processed} new or modified files.")
        else:
            print("No changes detected. Skipping ingestion.")

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
    
    def _load_state(self) -> dict:
        """Loads the last known modification times from a JSON file."""
        if self.state_path.exists():
            with open(self.state_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        """Persists the current state to disk."""
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=4)
    
    def is_dirty(self, file_path: Path) -> bool:
        """
        Checks if the file has been modified since the last ingestion.
        Returns True if it's new or changed.
        """
        mtime = file_path.stat().st_mtime
        last_mtime = self.state.get(str(file_path))
        
        if last_mtime is None or mtime > last_mtime:
            # Update the state in memory
            self.state[str(file_path)] = mtime
            return True
        return False
    

