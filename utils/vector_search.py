"""
Vector search utilities using FAISS for efficient similarity search.
Provides high-performance vector indexing and retrieval capabilities.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import faiss

from utils.environment import ENABLE_FAISS, NARRATIVE_EMBEDDING_DIM
from utils.concurrency import ResourceLock

# Configure module logger
logger = logging.getLogger(__name__)


class VectorIndex:
    """
    FAISS-powered vector index for efficient similarity search.
    Provides thread-safe access to the index.
    """
    
    def __init__(
        self, 
        dimension: int = NARRATIVE_EMBEDDING_DIM,
        index_type: str = "flat",
        use_gpu: bool = False,
        normalize: bool = True,
        distance_metric: str = "cosine"
    ):
        """
        Initialize a vector search index.
        
        Args:
            dimension: Vector dimension
            index_type: FAISS index type ('flat', 'ivf', 'hnsw', etc.)
            use_gpu: Whether to use GPU acceleration if available
            normalize: Whether to normalize vectors (needed for cosine similarity)
            distance_metric: Distance metric to use ('cosine', 'l2', 'ip')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.distance_metric = distance_metric
        
        # Create the appropriate FAISS index based on configuration
        if distance_metric == "cosine" or distance_metric == "ip":
            # Inner product is used for cosine similarity with normalized vectors
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # L2 distance is used for Euclidean distance
            self.index = faiss.IndexFlatL2(dimension)
            
        # For more advanced index types
        if index_type == "ivf":
            # IVF index with 100 centroids for faster search
            nlist = 100
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, 
                                           faiss.METRIC_L2 if distance_metric == "l2" else faiss.METRIC_INNER_PRODUCT)
            # IVF indices need training
            self.trained = False
        elif index_type == "hnsw":
            # HNSW index for even faster search with good recall
            self.index = faiss.IndexHNSWFlat(dimension, 32,
                                            faiss.METRIC_L2 if distance_metric == "l2" else faiss.METRIC_INNER_PRODUCT)
            self.trained = True
        else:
            # Flat index doesn't need training
            self.trained = True
            
        # Try to use GPU if requested
        self.is_gpu_index = False
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.is_gpu_index = True
                logger.info("Using GPU acceleration for vector search")
            except Exception as e:
                logger.warning(f"GPU acceleration requested but not available: {e}")
                
        # Thread-safe access to the index
        self.index_lock = ResourceLock(self.index)
        
        # ID mapping from FAISS sequential IDs to application IDs
        self.id_map = {}
        self.rev_id_map = {}
        self.next_id = 0
        
        logger.info(f"Created {index_type} vector index with dimension {dimension}")
        
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            vectors: Matrix of vectors to add (n_vectors x dimension)
            ids: List of string IDs corresponding to the vectors
            
        Returns:
            List of assigned FAISS IDs
        """
        if not ENABLE_FAISS:
            logger.warning("FAISS is disabled, skipping vector indexing")
            return list(range(len(ids)))
            
        if len(vectors) != len(ids):
            raise ValueError(f"Number of vectors ({len(vectors)}) must match number of IDs ({len(ids)})")
            
        if len(vectors) == 0:
            return []
            
        # Ensure vectors are in the correct shape
        if len(vectors.shape) == 1:
            # Single vector
            vectors = vectors.reshape(1, -1)
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")
            
        # Copy vectors to avoid modifying originals
        vectors_copy = vectors.copy().astype(np.float32)
        
        # Normalize if needed
        if self.normalize and self.distance_metric == "cosine":
            faiss.normalize_L2(vectors_copy)
            
        # Train if needed
        with self.index_lock.write() as index:
            if not self.trained and index.ntotal == 0 and len(vectors) > 0:
                if hasattr(index, 'train'):
                    index.train(vectors_copy)
                self.trained = True
                
            # Assign FAISS IDs
            faiss_ids = []
            for i, id_str in enumerate(ids):
                if id_str in self.id_map:
                    # Update existing vector
                    faiss_id = self.id_map[id_str]
                    # Currently, FAISS doesn't support direct updates, so we'll remove and re-add
                    if hasattr(index, 'remove_ids'):
                        index.remove_ids(np.array([faiss_id], dtype=np.int64))
                    else:
                        logger.warning(f"Index type {self.index_type} doesn't support updates")
                else:
                    # Add new vector
                    faiss_id = self.next_id
                    self.next_id += 1
                    self.id_map[id_str] = faiss_id
                    self.rev_id_map[faiss_id] = id_str
                    
                faiss_ids.append(faiss_id)
                
            # Add vectors to index
            if hasattr(index, 'add_with_ids'):
                index.add_with_ids(vectors_copy, np.array(faiss_ids, dtype=np.int64))
            else:
                # Simpler index just uses sequential IDs
                index.add(vectors_copy)
                
        return faiss_ids
        
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        return_distances: bool = True
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            return_distances: Whether to return distances
            
        Returns:
            List of IDs or tuple of (IDs, distances)
        """
        if not ENABLE_FAISS:
            logger.warning("FAISS is disabled, returning empty search results")
            if return_distances:
                return [], []
            return []
            
        # Reshape if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_vector.shape[1]} does not match index dimension {self.dimension}")
            
        # Copy to avoid modifying original
        query_copy = query_vector.copy().astype(np.float32)
        
        # Normalize if needed
        if self.normalize and self.distance_metric == "cosine":
            faiss.normalize_L2(query_copy)
            
        # Search
        with self.index_lock.read() as index:
            if index.ntotal == 0:
                if return_distances:
                    return [], []
                return []
                
            D, I = index.search(query_copy, min(k, index.ntotal))
            
        # Map FAISS IDs to string IDs
        results = []
        distances = []
        
        for i in range(len(I[0])):
            if I[0][i] < 0:  # Invalid ID
                continue
                
            faiss_id = int(I[0][i])
            if faiss_id in self.rev_id_map:
                results.append(self.rev_id_map[faiss_id])
                distances.append(float(D[0][i]))
                
        if return_distances:
            return results, distances
            
        return results
        
    def remove(self, ids: List[str]) -> int:
        """
        Remove vectors from the index.
        
        Args:
            ids: List of string IDs to remove
            
        Returns:
            Number of vectors removed
        """
        if not ENABLE_FAISS:
            logger.warning("FAISS is disabled, skipping vector removal")
            return 0
            
        removed = 0
        
        with self.index_lock.write() as index:
            # Convert string IDs to FAISS IDs
            faiss_ids = []
            for id_str in ids:
                if id_str in self.id_map:
                    faiss_ids.append(self.id_map[id_str])
                    
            if not faiss_ids:
                return 0
                
            # Remove from index if supported
            if hasattr(index, 'remove_ids'):
                index.remove_ids(np.array(faiss_ids, dtype=np.int64))
                removed = len(faiss_ids)
            else:
                logger.warning(f"Index type {self.index_type} doesn't support removal")
                
            # Update ID mappings
            for id_str in ids:
                if id_str in self.id_map:
                    faiss_id = self.id_map[id_str]
                    del self.id_map[id_str]
                    if faiss_id in self.rev_id_map:
                        del self.rev_id_map[faiss_id]
                    removed += 1
                    
        return removed
        
    def clear(self) -> None:
        """Clear the index."""
        if not ENABLE_FAISS:
            logger.warning("FAISS is disabled, skipping index clearing")
            return
            
        with self.index_lock.write() as index:
            # FAISS doesn't have a clear method, so we need to reset the index
            if self.index_type == "flat":
                if self.distance_metric == "cosine" or self.distance_metric == "ip":
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                nlist = 100
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist,
                                               faiss.METRIC_L2 if self.distance_metric == "l2" else faiss.METRIC_INNER_PRODUCT)
                self.trained = False
            elif self.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32,
                                                faiss.METRIC_L2 if self.distance_metric == "l2" else faiss.METRIC_INNER_PRODUCT)
                
            # If was GPU index, transfer back to GPU
            if self.is_gpu_index:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    logger.warning(f"GPU transfer failed during clear: {e}")
                    self.is_gpu_index = False
                    
            # Update lock resource
            self.index_lock = ResourceLock(self.index)
            
            # Clear ID mappings
            self.id_map = {}
            self.rev_id_map = {}
            self.next_id = 0
            
    def get_count(self) -> int:
        """
        Get the number of vectors in the index.
        
        Returns:
            Number of indexed vectors
        """
        if not ENABLE_FAISS:
            logger.warning("FAISS is disabled, returning 0 count")
            return 0
            
        with self.index_lock.read() as index:
            return index.ntotal
            
    def save(self, file_path: str) -> bool:
        """
        Save the index to a file.
        
        Args:
            file_path: Path to save the index
            
        Returns:
            True if successful
        """
        if not ENABLE_FAISS:
            logger.warning("FAISS is disabled, skipping index save")
            return False
            
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert to CPU index if needed
            with self.index_lock.read() as index:
                if self.is_gpu_index:
                    cpu_index = faiss.index_gpu_to_cpu(index)
                    faiss.write_index(cpu_index, file_path)
                else:
                    faiss.write_index(index, file_path)
                    
            # Save ID mappings
            with open(f"{file_path}.meta", "wb") as f:
                pickle.dump({
                    "id_map": self.id_map,
                    "rev_id_map": self.rev_id_map,
                    "next_id": self.next_id,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "normalize": self.normalize,
                    "distance_metric": self.distance_metric
                }, f)
                
            logger.info(f"Saved vector index to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
            return False
            
    @classmethod
    def load(cls, file_path: str, use_gpu: bool = False) -> Optional['VectorIndex']:
        """
        Load an index from a file.
        
        Args:
            file_path: Path to load the index from
            use_gpu: Whether to transfer the index to GPU
            
        Returns:
            Loaded index or None if failed
        """
        if not ENABLE_FAISS:
            logger.warning("FAISS is disabled, skipping index load")
            return None
            
        try:
            # Load index
            index = faiss.read_index(file_path)
            
            # Load metadata
            with open(f"{file_path}.meta", "rb") as f:
                meta = pickle.load(f)
                
            # Create instance
            instance = cls(
                dimension=meta["dimension"],
                index_type=meta["index_type"],
                use_gpu=use_gpu,
                normalize=meta["normalize"],
                distance_metric=meta["distance_metric"]
            )
            
            # Replace index
            instance.index = index
            instance.id_map = meta["id_map"]
            instance.rev_id_map = meta["rev_id_map"]
            instance.next_id = meta["next_id"]
            instance.trained = True
            
            # Transfer to GPU if requested
            if use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
                    instance.is_gpu_index = True
                    logger.info("Transferred loaded index to GPU")
                except Exception as e:
                    logger.warning(f"GPU transfer failed: {e}")
                    
            # Update lock resource
            instance.index_lock = ResourceLock(instance.index)
            
            logger.info(f"Loaded vector index from {file_path} with {instance.get_count()} vectors")
            return instance
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return None


# Create default index
default_index = VectorIndex()