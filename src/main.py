import argparse
import sys
from src.config import settings
from src.core.ingestion import FileProcessor
from src.core.embedder import Embedder
from src.core.vector_db import VectorStore
from src.core.retriever import Retriever
from src.core.generator import Generator

def main():
    parser = argparse.ArgumentParser(description="Local Knowledge RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ingest
    subparsers.add_parser("ingest", help="Scan local files and update the vector database")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question based on your local data")
    ask_parser.add_argument("query", type=str, help="The question you want to ask")

    args = parser.parse_args()

    # Ensure system folders exist
    settings.ensure_directories()

    # Initialize core components
    embedder = Embedder()
    vector_db = VectorStore()

    if args.command == "ingest":
        print("--- Running Ingestion Pipeline ---")
        processor = FileProcessor()
        new_chunks = processor.process_all()
        
        if new_chunks:
            chunks_with_embeddings = embedder.embed_chunks(new_chunks)
            vector_db.upsert_chunks(chunks_with_embeddings)
            print("Successfully updated your knowledge base.")
        else:
            print("No new changes detected.")

    elif args.command == "ask":
        # 1. Setup Retrieval
        retriever = Retriever(embedder, vector_db)
        # 2. Setup Generation
        generator = Generator()
        
        # 3. The RAG Flow
        results = retriever.retrieve(args.query)
        
        if not results:
            print("No relevant context found in your local files.")
            return

        prompt = generator.format_prompt(args.query, results)
        generator.generate_answer_stream(prompt)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()