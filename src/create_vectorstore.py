#!/usr/bin/env python3
"""
Script to create vector store from documents
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Create vector store from documents")
    parser.add_argument("--input_dir", 
                      default=Config.DEFAULT_INPUT_DIR,
                      help="Directory containing documents to process")
    parser.add_argument("--output_dir", 
                      default=Config.DEFAULT_VECTORSTORE_DIR,
                      help="Directory to save vector store")
    parser.add_argument("--chunk_size", 
                      type=int, 
                      default=Config.CHUNK_SIZE,
                      help="Size of text chunks")
    parser.add_argument("--chunk_overlap", 
                      type=int, 
                      default=Config.CHUNK_OVERLAP,
                      help="Overlap between chunks")
    
    args = parser.parse_args()
    
    try:
        # Validate configuration
        Config.validate()
        
        print("=== Creating Vector Store ===")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Chunk overlap: {args.chunk_overlap}")
        print()
        
        # Process documents
        print("1. Processing documents...")
        processor = DocumentProcessor()
        processor.chunk_size = args.chunk_size
        processor.chunk_overlap = args.chunk_overlap
        
        documents = processor.process_directory(args.input_dir)
        
        if not documents:
            print("No documents found to process!")
            return
        
        print(f"Processed {len(documents)} document chunks")
        print()
        
        # Create vector store
        print("2. Creating vector store...")
        vector_store = VectorStore()
        vector_store.create_from_documents(documents)
        
        # Save vector store
        print("3. Saving vector store...")
        vector_store.save(args.output_dir)
        
        # Print statistics
        stats = vector_store.get_stats()
        print("\n=== Vector Store Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print(f"\n✅ Vector store created successfully!")
        print(f"📁 Saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
