"""
SECLEDS (Sequence Clustering in Evolving Data Streams via Multiple Medoids and Medoid Voting)
implementation for the CIVILIAN system.

This module provides a Python implementation of the SECLEDS algorithm for streaming
sequence clustering with concept drift adaptation via multiple medoids per cluster.

Based on the paper: "SECLEDS: Sequence Clustering in Evolving Data Streams via Multiple
Medoids and Medoid Voting" (ECML 2022)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import threading
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class MedoidCluster:
    """
    A cluster with multiple medoids, supporting voting for adaptation to concept drift.
    """
    def __init__(self, cluster_id: int, max_medoids: int = 3):
        """
        Initialize a new medoid cluster.
        
        Args:
            cluster_id: Unique identifier for this cluster
            max_medoids: Maximum number of medoids to maintain per cluster
        """
        self.cluster_id = cluster_id
        self.max_medoids = max_medoids
        self.medoids = []  # List of (vector, timestamp, weight) tuples
        self.members = {}  # Dict of {member_id: (vector, timestamp, distance_to_medoid)}
        self.last_updated = time.time()
    
    def add_medoid(self, medoid_vector: np.ndarray, timestamp: Optional[datetime] = None) -> None:
        """
        Add a new medoid to the cluster.
        
        Args:
            medoid_vector: Vector representation of the medoid
            timestamp: Timestamp for the medoid (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        # Add medoid with initial weight
        self.medoids.append((medoid_vector, timestamp, 1.0))
        
        # If we have too many medoids, remove the one with lowest weight
        if len(self.medoids) > self.max_medoids:
            self.medoids.sort(key=lambda x: x[2], reverse=True)  # Sort by weight descending
            self.medoids = self.medoids[:self.max_medoids]
            
        self.last_updated = time.time()
    
    def update_medoid_weights(self, decay_factor: float = 0.05) -> None:
        """
        Update the weights of medoids based on age and member assignments.
        
        Args:
            decay_factor: Factor by which to decay old medoids
        """
        # Apply time-based decay to all medoids
        for i in range(len(self.medoids)):
            vector, timestamp, weight = self.medoids[i]
            age = (datetime.utcnow() - timestamp).total_seconds() / 86400.0  # Age in days
            decayed_weight = weight * np.exp(-decay_factor * age)
            self.medoids[i] = (vector, timestamp, decayed_weight)
        
        # Sort by weight descending
        self.medoids.sort(key=lambda x: x[2], reverse=True)
        
        self.last_updated = time.time()
    
    def add_member(self, member_id: str, vector: np.ndarray, 
                  timestamp: Optional[datetime] = None) -> Tuple[float, int]:
        """
        Add a member to the cluster.
        
        Args:
            member_id: Unique identifier for the member
            vector: Vector representation of the member
            timestamp: Timestamp for the member (defaults to current time)
            
        Returns:
            Tuple of (distance_to_medoid, medoid_index)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        # Find the closest medoid
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (medoid_vector, _, _) in enumerate(self.medoids):
            dist = calculate_distance(vector, medoid_vector)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Store the member
        self.members[member_id] = (vector, timestamp, min_dist)
        
        # Update last access time
        self.last_updated = time.time()
        
        return min_dist, closest_idx
    
    def vote_for_assignment(self, vector: np.ndarray) -> Tuple[float, float]:
        """
        Have all medoids vote on the assignment of a vector.
        
        Args:
            vector: Vector to assign
            
        Returns:
            Tuple of (average_distance, confidence)
        """
        if not self.medoids:
            return float('inf'), 0.0
            
        # Calculate distance to each medoid
        distances = []
        total_weight = 0.0
        
        for medoid_vector, _, weight in self.medoids:
            dist = calculate_distance(vector, medoid_vector)
            distances.append(dist * weight)  # Weight the distance
            total_weight += weight
        
        # Calculate weighted average distance
        if total_weight > 0:
            avg_distance = sum(distances) / total_weight
        else:
            avg_distance = float('inf')
        
        # Calculate confidence based on agreement between medoids
        if len(distances) > 1:
            # Use standard deviation of distances as a measure of agreement
            # Lower std_dev means higher confidence
            std_dev = np.std(distances)
            max_dist = max(distances)
            if max_dist > 0:
                confidence = 1.0 - (std_dev / max_dist)
            else:
                confidence = 1.0
        else:
            confidence = 1.0  # With only one medoid, confidence is 100%
        
        return avg_distance, confidence


def calculate_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Distance between vectors
    """
    # Ensure vectors are 1D
    vec1 = np.ravel(vec1)
    vec2 = np.ravel(vec2)
    
    # Use cosine distance (1 - similarity) for text embeddings
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1.0 - similarity


class SECLEDS:
    """
    Implementation of the SECLEDS algorithm for sequence clustering in evolving data streams.
    """
    def __init__(self, k: int = 5, p: int = 3, distance_threshold: float = 0.3, 
                decay: float = 0.01, drift: bool = True):
        """
        Initialize the SECLEDS algorithm.
        
        Args:
            k: Number of clusters to maintain
            p: Number of medoids per cluster
            distance_threshold: Threshold for considering a point part of a cluster
            decay: Decay factor for medoid weights
            drift: Whether to adapt to concept drift
        """
        self.k = k
        self.p = p
        self.distance_threshold = distance_threshold
        self.decay = decay
        self.drift = drift
        
        self.clusters = {}  # Dict of {cluster_id: MedoidCluster}
        self.next_cluster_id = 0
        self.items_processed = 0
        
        # Thread lock for concurrency safety
        self.lock = threading.Lock()
        
        logger.info(f"SECLEDS initialized with k={k}, p={p}, decay={decay}")
    
    def partial_fit(self, sequence: np.ndarray, sequence_id: Optional[str] = None, 
                   timestamp: Optional[datetime] = None) -> None:
        """
        Update the model with a new sequence.
        
        Args:
            sequence: Vector representation of a sequence
            sequence_id: Optional unique identifier for the sequence
            timestamp: Optional timestamp for the sequence
        """
        with self.lock:
            self.items_processed += 1
            
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            if sequence_id is None:
                sequence_id = f"seq_{self.items_processed}"
            
            # If no clusters yet, initialize the first one
            if not self.clusters:
                new_cluster = MedoidCluster(self.next_cluster_id, max_medoids=self.p)
                new_cluster.add_medoid(sequence, timestamp)
                self.clusters[self.next_cluster_id] = new_cluster
                self.next_cluster_id += 1
                return
            
            # Find the best cluster for this sequence
            best_cluster_id, min_distance, confidence = self._find_best_cluster(sequence)
            
            # If the distance is below threshold, add to the cluster
            if min_distance <= self.distance_threshold:
                cluster = self.clusters[best_cluster_id]
                cluster.add_member(sequence_id, sequence, timestamp)
                
                # Check if we need to update medoids to adapt to drift
                if self.drift and self.items_processed % 10 == 0:  # Update every 10 items
                    cluster.update_medoid_weights(self.decay)
            else:
                # Create a new cluster if below k clusters
                if len(self.clusters) < self.k:
                    new_cluster = MedoidCluster(self.next_cluster_id, max_medoids=self.p)
                    new_cluster.add_medoid(sequence, timestamp)
                    self.clusters[self.next_cluster_id] = new_cluster
                    self.next_cluster_id += 1
                else:
                    # Replace the least recently used cluster
                    oldest_id = min(self.clusters, key=lambda c: self.clusters[c].last_updated)
                    new_cluster = MedoidCluster(oldest_id, max_medoids=self.p)
                    new_cluster.add_medoid(sequence, timestamp)
                    self.clusters[oldest_id] = new_cluster
    
    def predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict the cluster for a sequence.
        
        Args:
            sequence: Vector representation of a sequence
            
        Returns:
            Tuple of (cluster_id, confidence)
        """
        with self.lock:
            # If no clusters, return noise
            if not self.clusters:
                return -1, 0.0
            
            # Find best cluster
            best_cluster_id, min_distance, confidence = self._find_best_cluster(sequence)
            
            # If distance above threshold, consider it noise
            if min_distance > self.distance_threshold:
                return -1, confidence
            
            return best_cluster_id, confidence
    
    def _find_best_cluster(self, sequence: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best cluster for a sequence.
        
        Args:
            sequence: Vector to cluster
            
        Returns:
            Tuple of (cluster_id, distance, confidence)
        """
        min_distance = float('inf')
        best_cluster_id = -1
        best_confidence = 0.0
        
        for cluster_id, cluster in self.clusters.items():
            avg_distance, confidence = cluster.vote_for_assignment(sequence)
            
            if avg_distance < min_distance:
                min_distance = avg_distance
                best_cluster_id = cluster_id
                best_confidence = confidence
        
        return best_cluster_id, min_distance, best_confidence
    
    def get_cluster_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics about all clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        with self.lock:
            stats = {}
            
            for cluster_id, cluster in self.clusters.items():
                stats[cluster_id] = {
                    "medoid_count": len(cluster.medoids),
                    "member_count": len(cluster.members),
                    "last_updated": cluster.last_updated,
                }
            
            return stats