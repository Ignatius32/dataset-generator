import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from pathlib import Path
from .embeddings_api import EmbeddingsAPI
from .config import Config

class VectorStore:
    def __init__(self, dimension: int = None):
        self.dimension = dimension or Config.VECTOR_DIMENSION
        self.index = None
        self.documents = []
        self.embeddings_api = EmbeddingsAPI()
        
    def create_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Create vector store from processed documents
        
        Args:
            documents: List of document chunks with metadata
        """
        print(f"Creating vector store from {len(documents)} documents...")
        
        # Extract texts for embedding
        texts = [doc['text'] for doc in documents]
        
        # Get embeddings
        print("Getting embeddings...")
        embeddings = self.embeddings_api.get_embeddings(texts)
        
        if not embeddings:
            raise ValueError("Failed to get embeddings")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Update dimension if needed
        if embeddings_array.shape[1] != self.dimension:
            self.dimension = embeddings_array.shape[1]
            print(f"Updated dimension to {self.dimension}")
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        # Store documents
        self.documents = documents
        
        print(f"Vector store created with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if not self.index or not self.documents:
            raise ValueError("Vector store not initialized")
        
        # Get query embedding
        query_embedding = self.embeddings_api.get_single_embedding(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save(self, output_dir: str) -> None:
        """
        Save vector store to disk
        
        Args:
            output_dir: Directory to save the vector store
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, str(output_path / "index.faiss"))
        
        # Save documents and metadata
        with open(output_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save configuration
        config = {
            'dimension': self.dimension,
            'num_documents': len(self.documents),
            'model': Config.EMBEDDING_MODEL
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Vector store saved to {output_dir}")
    
    def load(self, input_dir: str) -> None:
        """
        Load vector store from disk
        
        Args:
            input_dir: Directory containing the vector store
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Vector store directory not found: {input_dir}")
        
        # Load configuration
        with open(input_path / "config.json", 'r') as f:
            config = json.load(f)
        
        self.dimension = config['dimension']
        
        # Load FAISS index
        index_path = input_path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load documents
        with open(input_path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"Vector store loaded from {input_dir}")
        print(f"Loaded {len(self.documents)} documents with dimension {self.dimension}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.documents:
            return {}
        
        total_chars = sum(doc['char_count'] for doc in self.documents)
        total_tokens = sum(doc.get('token_count', 0) for doc in self.documents)
        
        doc_types = {}
        for doc in self.documents:
            doc_type = doc.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            'num_documents': len(self.documents),
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'average_chars_per_doc': total_chars / len(self.documents),
            'document_types': doc_types,
            'vector_dimension': self.dimension,
            'index_size': self.index.ntotal if self.index else 0
        }
