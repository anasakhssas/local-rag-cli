import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

class Config :
    # --- Paths ---
    DATA_DIR = BASE_DIR / "data"
    DB_PATH = DATA_DIR / "vector_store"

    # --- Ingestion Settings ---
    # Define these clearly so you can tune your "Search Accuracy" later
    CHUNK_SIZE = 500      # Characters per chunk
    CHUNK_OVERLAP = 50    # Characters to overlap to preserve context

    # --- Model Settings ---
    # We define these here so you can easily swap models for testing
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # --- System ---
    LOG_LEVEL = "INFO"

    @classmethod
    def ensure_directories(cls) :
        """Ensure critical directories exist before the app starts."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        if not cls.DB_PATH.exists():
            cls.DB_PATH.mkdir(exist_ok=True)
    
# Instantiate for easy import
settings = Config()