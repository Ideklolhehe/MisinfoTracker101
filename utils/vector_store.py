import numpy as np
import os
import pickle
import logging
import faiss
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStore:
    """A simple vector database implementation using FAISS."""
    
    def __init__(self, dimension: int = 768, index_path: Optional[str] = None):
        """Initialize the vector store.
        
        Args:
            dimension: Dimension of the vectors to store
            index_path: Path to load an existing index from
        """
        self.dimension = dimension
        self.index = None
        self.metadata = {}  # Maps vector ID to metadata
        self.next_id = 0
        
        # Create or load index
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)
        else:
            self._create_index()
            
        logger.info(f"VectorStore initialized with dimension {dimension}")
    
    def _create_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance (Euclidean)
        self.metadata = {}
        self.next_id = 0
        logger.debug("Created new FAISS index")
        
    def _load_index(self, path: str):
        """Load index from disk."""
        try:
            # Load the index
            self.index = faiss.read_index(f"{path}.index")
            
            # Load metadata
            with open(f"{path}.metadata", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.next_id = data['next_id']
                
            logger.info(f"Loaded index from {path} with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            self._create_index()
    
    def save(self, path: str):
        """Save index to disk."""
        try:
            # Save the index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save metadata
            with open(f"{path}.metadata", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'next_id': self.next_id,
                    'timestamp': datetime.utcnow().isoformat()
                }, f)
                
            logger.info(f"Saved index to {path} with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to save index to {path}: {e}")
            return False
    
    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any] = None) -> str:
        """Add a vector to the index with optional metadata.
        
        Args:
            vector: The vector to add (1D numpy array)
            metadata: Optional metadata to associate with the vector
            
        Returns:
            vector_id: String ID of the added vector
        """
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError(f"Vector must be 1D with dimension {self.dimension}")
        
        # Reshape for FAISS (expects 2D array)
        vector_reshaped = np.array([vector], dtype=np.float32)
        
        # Create vector ID
        vector_id = str(self.next_id)
        self.next_id += 1
        
        # Add to index
        self.index.add(vector_reshaped)
        
        # Store metadata
        self.metadata[vector_id] = metadata or {}
        
        return vector_id
    
    def add_vectors(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add multiple vectors to the index.
        
        Args:
            vectors: 2D numpy array of shape (n_vectors, dimension)
            metadatas: List of metadata dictionaries, one per vector
            
        Returns:
            vector_ids: List of string IDs for the added vectors
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Vectors must be 2D with second dimension {self.dimension}")
        
        n_vectors = vectors.shape[0]
        
        # Ensure vectors are float32 for FAISS
        vectors = vectors.astype(np.float32)
        
        # Add to index
        self.index.add(vectors)
        
        # Create and store vector IDs with metadata
        vector_ids = []
        for i in range(n_vectors):
            vector_id = str(self.next_id)
            self.next_id += 1
            vector_ids.append(vector_id)
            
            # Store metadata if provided
            if metadatas and i < len(metadatas):
                self.metadata[vector_id] = metadatas[i]
            else:
                self.metadata[vector_id] = {}
        
        return vector_ids
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: The query vector (1D numpy array)
            k: Number of results to return
            
        Returns:
            results: List of dictionaries containing vector_id, score, and metadata
        """
        if self.index.ntotal == 0:
            return []
            
        if query_vector.ndim != 1 or query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector must be 1D with dimension {self.dimension}")
        
        # Reshape for FAISS
        query_reshaped = np.array([query_vector], dtype=np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_reshaped, min(k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # -1 indicates no result found
                vector_id = str(idx)
                results.append({
                    'vector_id': vector_id,
                    'score': float(distance),
                    'metadata': self.metadata.get(vector_id, {})
                })
        
        return results
    
    def delete(self, vector_id: str) -> bool:
        """Delete a vector from the index.
        
        Note: FAISS doesn't support simple deletion. To properly implement this,
        we would need to rebuild the index. This is a simplified version.
        
        Args:
            vector_id: String ID of the vector to delete
            
        Returns:
            success: Whether the deletion was successful
        """
        # For now, just remove the metadata
        if vector_id in self.metadata:
            self.metadata.pop(vector_id)
            logger.debug(f"Removed metadata for vector {vector_id}")
            # Note: The vector still exists in the FAISS index
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'vector_count': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata),
            'index_type': type(self.index).__name__
        }
