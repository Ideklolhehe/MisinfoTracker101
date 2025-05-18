"""
Enhanced SECLEDS implementation for the CIVILIAN system.

This module provides an enhanced version of the SECLEDS algorithm with:
1. Adaptive training updates from analyst feedback
2. Feature importance visualization
3. Novel narrative detection
4. Confidence-based escalation
5. Cross-algorithm feedback integration

Based on the original SECLEDS algorithm but extended with CIVILIAN-specific features.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta
import time
import json

# Configure logging
logger = logging.getLogger(__name__)

class FeatureWeightTracker:
    """
    Tracks the importance of different features in classification decisions.
    """
    def __init__(self, feature_dim: int = 768):
        """
        Initialize the feature weight tracker.
        
        Args:
            feature_dim: Dimension of the feature vectors
        """
        self.feature_dim = feature_dim
        self.feature_weights = np.ones(feature_dim)  # Start with equal weights
        self.feature_importance = np.zeros(feature_dim)  # Accumulated importance
        self.update_count = 0
        
    def update_weights(self, vectors: List[np.ndarray], labels: List[int]) -> None:
        """
        Update feature weights based on classification results.
        
        Args:
            vectors: List of vector representations
            labels: Corresponding cluster labels
        """
        if not vectors or not labels or len(vectors) != len(labels):
            return
            
        # Convert to arrays
        X = np.array(vectors)
        y = np.array(labels)
        
        # Skip if all labels are the same
        if len(set(labels)) < 2:
            return
            
        try:
            # Train a simple model to get feature importance
            from sklearn.ensemble import RandomForestClassifier
            
            # Create and train model
            model = RandomForestClassifier(n_estimators=10, max_depth=5)
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Update accumulated importance with exponential decay
            decay = 0.9  # Retain 90% of previous importance
            self.feature_importance = decay * self.feature_importance + (1 - decay) * importance
            
            # Normalize weights (add small epsilon to avoid division by zero)
            epsilon = 1e-10
            self.feature_weights = self.feature_importance + epsilon
            self.feature_weights = self.feature_weights / np.sum(self.feature_weights)
            
            self.update_count += 1
            
        except Exception as e:
            logger.error(f"Error updating feature weights: {e}")
            
    def get_top_features(self, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get the top most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_index, importance) tuples
        """
        indices = np.argsort(self.feature_importance)[-n:][::-1]
        return [(int(idx), float(self.feature_importance[idx])) for idx in indices]
        
    def apply_weights(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply feature weights to a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Weighted vector
        """
        return vector * self.feature_weights
        
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizing feature importance.
        
        Returns:
            Dictionary with visualization data
        """
        top_features = self.get_top_features(20)
        
        return {
            "top_features": top_features,
            "update_count": self.update_count,
            "weight_distribution": {
                "min": float(np.min(self.feature_weights)),
                "max": float(np.max(self.feature_weights)),
                "mean": float(np.mean(self.feature_weights)),
                "std": float(np.std(self.feature_weights))
            }
        }


class EnhancedMedoidCluster:
    """
    Enhanced cluster with multiple medoids and feedback integration.
    """
    def __init__(self, cluster_id: int, max_medoids: int = 3, tracking_window: int = 100):
        """
        Initialize a new medoid cluster.
        
        Args:
            cluster_id: Unique identifier for this cluster
            max_medoids: Maximum number of medoids to maintain per cluster
            tracking_window: Number of recent members to track for drift detection
        """
        self.cluster_id = cluster_id
        self.max_medoids = max_medoids
        self.tracking_window = tracking_window
        
        # Core structures
        self.medoids = []  # List of (vector, timestamp, weight) tuples
        self.members = {}  # Dict of {member_id: (vector, timestamp, distance_to_medoid)}
        
        # Enhanced tracking
        self.last_updated = time.time()
        self.created_at = datetime.utcnow()
        self.recent_members = deque(maxlen=tracking_window)  # For drift detection
        self.feature_weights = None  # Feature weights
        
        # Feedback integration
        self.feedback_scores = []  # List of (timestamp, score, source) tuples
        self.cross_algorithm_evidence = {}  # Dict of {algorithm: confidence}
        self.escalation_status = "normal"  # One of: normal, review, escalated
        self.novelty_score = 0.0  # How novel/unusual is this cluster
        
    def add_medoid(self, medoid_vector: np.ndarray, timestamp: Optional[datetime] = None,
                 weight: float = 1.0) -> None:
        """
        Add a new medoid to the cluster.
        
        Args:
            medoid_vector: Vector representation of the medoid
            timestamp: Timestamp for the medoid (defaults to current time)
            weight: Initial weight for the medoid
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        # Add medoid with initial weight
        self.medoids.append((medoid_vector, timestamp, weight))
        
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
            
        # Apply feature weights if available
        weighted_vector = vector
        if self.feature_weights is not None:
            weighted_vector = vector * self.feature_weights
            
        # Find the closest medoid
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (medoid_vector, _, _) in enumerate(self.medoids):
            # Apply same feature weights to medoid
            weighted_medoid = medoid_vector
            if self.feature_weights is not None:
                weighted_medoid = medoid_vector * self.feature_weights
                
            dist = calculate_distance(weighted_vector, weighted_medoid)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Store the member
        self.members[member_id] = (vector, timestamp, min_dist)
        
        # Add to recent members for drift detection
        self.recent_members.append((vector, timestamp, member_id))
        
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
            
        # Apply feature weights if available
        weighted_vector = vector
        if self.feature_weights is not None:
            weighted_vector = vector * self.feature_weights
            
        # Calculate distance to each medoid
        distances = []
        total_weight = 0.0
        
        for medoid_vector, _, weight in self.medoids:
            # Apply same feature weights to medoid
            weighted_medoid = medoid_vector
            if self.feature_weights is not None:
                weighted_medoid = medoid_vector * self.feature_weights
                
            dist = calculate_distance(weighted_vector, weighted_medoid)
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
        
    def update_feature_weights(self, weights: np.ndarray) -> None:
        """
        Update feature weights for adaptive classification.
        
        Args:
            weights: New feature weights
        """
        self.feature_weights = weights
        
    def add_feedback(self, score: float, source: str, timestamp: Optional[datetime] = None) -> None:
        """
        Add feedback about this cluster from analysts or other sources.
        
        Args:
            score: Feedback score (0-1, higher is better)
            source: Source of the feedback
            timestamp: Timestamp for the feedback (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.feedback_scores.append((timestamp, score, source))
        
        # Keep only recent feedback (last 50)
        if len(self.feedback_scores) > 50:
            self.feedback_scores = self.feedback_scores[-50:]
            
    def update_cross_algorithm_evidence(self, algorithm: str, confidence: float) -> None:
        """
        Update evidence from another clustering algorithm.
        
        Args:
            algorithm: Name of the algorithm
            confidence: Confidence score (0-1)
        """
        self.cross_algorithm_evidence[algorithm] = confidence
        
    def check_novelty(self, global_vectors: List[np.ndarray]) -> float:
        """
        Check if this cluster represents a novel pattern.
        
        Args:
            global_vectors: Representative vectors from all clusters
            
        Returns:
            Novelty score (0-1, higher means more novel)
        """
        if not self.medoids or not global_vectors:
            return 0.0
            
        # Get primary medoid
        medoid = self.medoids[0][0]
        
        # Calculate distances to all global vectors
        distances = []
        for vec in global_vectors:
            dist = calculate_distance(medoid, vec)
            distances.append(dist)
            
        # Average distance to other clusters
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 range (using sigmoid)
        novelty = 1.0 / (1.0 + np.exp(-5 * (avg_distance - 0.5)))
        
        self.novelty_score = novelty
        return novelty
        
    def check_drift(self) -> Tuple[bool, float]:
        """
        Check if this cluster is experiencing concept drift.
        
        Returns:
            Tuple of (is_drifting, drift_magnitude)
        """
        if len(self.recent_members) < self.tracking_window // 2:
            return False, 0.0
            
        # Split recent members into two halves
        midpoint = len(self.recent_members) // 2
        first_half = [m[0] for m in list(self.recent_members)[:midpoint]]
        second_half = [m[0] for m in list(self.recent_members)[midpoint:]]
        
        if not first_half or not second_half:
            return False, 0.0
            
        # Calculate centroids
        first_centroid = np.mean(first_half, axis=0)
        second_centroid = np.mean(second_half, axis=0)
        
        # Calculate drift as distance between centroids
        drift = calculate_distance(first_centroid, second_centroid)
        
        # Determine if drift is significant
        is_drifting = drift > 0.3  # Threshold for significant drift
        
        return is_drifting, drift
        
    def get_confidence_score(self) -> float:
        """
        Calculate overall confidence in this cluster based on multiple factors.
        
        Returns:
            Confidence score (0-1)
        """
        # Start with average member distance
        if not self.members:
            return 0.5
            
        avg_dist = np.mean([d for _, _, d in self.members.values()])
        dist_score = max(0, 1.0 - avg_dist)
        
        # Factor in feedback if available
        feedback_score = 0.5
        if self.feedback_scores:
            recent_scores = [s for _, s, _ in self.feedback_scores[-10:]]
            feedback_score = np.mean(recent_scores)
            
        # Factor in cross-algorithm evidence
        algo_score = 0.5
        if self.cross_algorithm_evidence:
            algo_score = np.mean(list(self.cross_algorithm_evidence.values()))
            
        # Weight the different factors
        confidence = 0.5 * dist_score + 0.3 * feedback_score + 0.2 * algo_score
        
        return confidence
        
    def should_escalate(self) -> bool:
        """
        Determine if this cluster should be escalated for human review.
        
        Returns:
            True if escalation is recommended
        """
        # Check confidence
        confidence = self.get_confidence_score()
        
        # Check novelty
        novelty = self.novelty_score
        
        # Check drift
        is_drifting, drift_magnitude = self.check_drift()
        
        # Escalate if confidence is low
        if confidence < 0.4:
            self.escalation_status = "review"
            return True
            
        # Escalate if novelty is high
        if novelty > 0.7:
            self.escalation_status = "review"
            return True
            
        # Escalate if significant drift
        if is_drifting and drift_magnitude > 0.4:
            self.escalation_status = "review"
            return True
            
        return False


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


class EnhancedSECLEDS:
    """
    Enhanced implementation of the SECLEDS algorithm with additional features.
    """
    def __init__(self, k: int = 5, p: int = 3, distance_threshold: float = 0.3, 
                decay: float = 0.01, drift: bool = True, feature_dim: int = 768,
                feedback_interval: int = 100, escalation_threshold: float = 0.4,
                novelty_detection: bool = True):
        """
        Initialize the EnhancedSECLEDS algorithm.
        
        Args:
            k: Number of clusters to maintain
            p: Number of medoids per cluster
            distance_threshold: Threshold for considering a point part of a cluster
            decay: Decay factor for medoid weights
            drift: Whether to adapt to concept drift
            feature_dim: Dimension of feature vectors
            feedback_interval: Number of items between feature weight updates
            escalation_threshold: Threshold for escalating low-confidence results
            novelty_detection: Whether to detect novel narratives
        """
        self.k = k
        self.p = p
        self.distance_threshold = distance_threshold
        self.decay = decay
        self.drift = drift
        self.feature_dim = feature_dim
        self.feedback_interval = feedback_interval
        self.escalation_threshold = escalation_threshold
        self.novelty_detection = novelty_detection
        
        # Core structures
        self.clusters = {}  # Dict of {cluster_id: EnhancedMedoidCluster}
        self.next_cluster_id = 0
        self.items_processed = 0
        
        # Enhanced features
        self.feature_weight_tracker = FeatureWeightTracker(feature_dim)
        self.training_buffer = []  # Buffer for adaptive training: [(vector, cluster_id)]
        self.feedback_registry = {}  # Dict of {narrative_id: (cluster_id, feedback)}
        self.escalated_items = []  # List of escalated items: [(narrative_id, timestamp, reason)]
        self.novel_narratives = []  # List of potentially novel narratives: [(narrative_id, vector, score)]
        
        # Cross-algorithm integration
        self.external_evidence = {}  # Dict of {narrative_id: {algorithm: cluster_id}}
        
        # Thread lock for concurrency safety
        self.lock = threading.Lock()
        
        logger.info(f"EnhancedSECLEDS initialized with k={k}, p={p}, drift={drift}")
        
    def partial_fit(self, sequence: np.ndarray, sequence_id: Optional[str] = None, 
                   timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update the model with a new sequence.
        
        Args:
            sequence: Vector representation of a sequence
            sequence_id: Optional unique identifier for the sequence
            timestamp: Optional timestamp for the sequence
            
        Returns:
            Dictionary with fitting results
        """
        with self.lock:
            self.items_processed += 1
            
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            if sequence_id is None:
                sequence_id = f"seq_{self.items_processed}"
            
            result = {
                "status": "success",
                "narrative_id": sequence_id,
                "cluster_id": -1,
                "confidence": 0.0,
                "is_novel": False,
                "is_escalated": False,
                "timestamp": timestamp.isoformat()
            }
            
            # If no clusters yet, initialize the first one
            if not self.clusters:
                new_cluster = EnhancedMedoidCluster(self.next_cluster_id, max_medoids=self.p)
                new_cluster.add_medoid(sequence, timestamp)
                self.clusters[self.next_cluster_id] = new_cluster
                result["cluster_id"] = self.next_cluster_id
                result["confidence"] = 1.0
                self.next_cluster_id += 1
                return result
            
            # Find the best cluster for this sequence
            best_cluster_id, min_distance, confidence = self._find_best_cluster(sequence)
            
            # If the distance is below threshold, add to the cluster
            if min_distance <= self.distance_threshold:
                cluster = self.clusters[best_cluster_id]
                cluster.add_member(sequence_id, sequence, timestamp)
                
                # Add to training buffer
                self.training_buffer.append((sequence, best_cluster_id))
                if len(self.training_buffer) > 1000:
                    self.training_buffer = self.training_buffer[-1000:]
                
                # Update result
                result["cluster_id"] = best_cluster_id
                result["confidence"] = confidence
                
                # Check if cluster needs escalation
                if cluster.should_escalate():
                    result["is_escalated"] = True
                    self.escalated_items.append(
                        (sequence_id, timestamp, f"Low confidence: {confidence:.2f}")
                    )
                
                # Check if we need to update medoids to adapt to drift
                if self.drift and self.items_processed % 10 == 0:  # Update every 10 items
                    cluster.update_medoid_weights(self.decay)
            else:
                # Check for novel narrative pattern
                is_novel = False
                novelty_score = 0.0
                
                if self.novelty_detection:
                    # Get representative vectors from all clusters
                    rep_vectors = []
                    for c_id, c in self.clusters.items():
                        if c.medoids:
                            rep_vectors.append(c.medoids[0][0])
                            
                    # Calculate average distance to all cluster centers
                    if rep_vectors:
                        distances = []
                        for vec in rep_vectors:
                            distances.append(calculate_distance(sequence, vec))
                            
                        avg_distance = np.mean(distances)
                        # Higher avg_distance means more novel
                        novelty_score = min(1.0, avg_distance / 0.7)  # Normalize to 0-1
                        
                        # Consider it novel if sufficiently different from all clusters
                        is_novel = novelty_score > 0.6
                
                # Create a new cluster if below k clusters or if novel
                if len(self.clusters) < self.k or is_novel:
                    new_cluster = EnhancedMedoidCluster(self.next_cluster_id, max_medoids=self.p)
                    new_cluster.add_medoid(sequence, timestamp)
                    new_cluster.novelty_score = novelty_score
                    self.clusters[self.next_cluster_id] = new_cluster
                    
                    # Update result
                    result["cluster_id"] = self.next_cluster_id
                    result["confidence"] = 1.0  # New cluster is perfect match for its first member
                    result["is_novel"] = is_novel
                    
                    if is_novel:
                        self.novel_narratives.append((sequence_id, sequence, novelty_score))
                        # Escalate novel narratives
                        result["is_escalated"] = True
                        self.escalated_items.append(
                            (sequence_id, timestamp, f"Novel pattern: {novelty_score:.2f}")
                        )
                        
                    self.next_cluster_id += 1
                else:
                    # Replace the least recently used cluster
                    oldest_id = min(self.clusters, key=lambda c: self.clusters[c].last_updated)
                    new_cluster = EnhancedMedoidCluster(oldest_id, max_medoids=self.p)
                    new_cluster.add_medoid(sequence, timestamp)
                    self.clusters[oldest_id] = new_cluster
                    
                    # Update result
                    result["cluster_id"] = oldest_id
                    result["confidence"] = 1.0  # New cluster is perfect match for its first member
            
            # Periodically update feature weights
            if self.items_processed % self.feedback_interval == 0 and len(self.training_buffer) >= 10:
                X = [item[0] for item in self.training_buffer]
                y = [item[1] for item in self.training_buffer]
                self.feature_weight_tracker.update_weights(X, y)
                
                # Apply new weights to all clusters
                weights = self.feature_weight_tracker.feature_weights
                for cluster in self.clusters.values():
                    cluster.update_feature_weights(weights)
                    
            return result
            
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
        
        # Apply feature weights if available
        if self.feature_weight_tracker.update_count > 0:
            weighted_sequence = self.feature_weight_tracker.apply_weights(sequence)
        else:
            weighted_sequence = sequence
        
        for cluster_id, cluster in self.clusters.items():
            avg_distance, confidence = cluster.vote_for_assignment(weighted_sequence)
            
            if avg_distance < min_distance:
                min_distance = avg_distance
                best_cluster_id = cluster_id
                best_confidence = confidence
        
        return best_cluster_id, min_distance, best_confidence
        
    def add_feedback(self, narrative_id: str, cluster_id: int, 
                    feedback_score: float, source: str = "analyst") -> None:
        """
        Add analyst feedback about a clustering decision.
        
        Args:
            narrative_id: ID of the narrative
            cluster_id: Assigned cluster ID
            feedback_score: Feedback score (0-1, higher is better)
            source: Source of the feedback
        """
        with self.lock:
            self.feedback_registry[narrative_id] = (cluster_id, feedback_score, source)
            
            # If cluster exists, add feedback
            if cluster_id in self.clusters:
                self.clusters[cluster_id].add_feedback(feedback_score, source)
                
            # Use feedback to retrain on next update
            
    def add_cross_algorithm_evidence(self, narrative_id: str, 
                                    algorithm: str, cluster_id: int, 
                                    confidence: float) -> None:
        """
        Add evidence from another clustering algorithm.
        
        Args:
            narrative_id: ID of the narrative
            algorithm: Name of the algorithm
            cluster_id: Cluster ID assigned by that algorithm
            confidence: Confidence of the assignment
        """
        with self.lock:
            if narrative_id not in self.external_evidence:
                self.external_evidence[narrative_id] = {}
                
            self.external_evidence[narrative_id][algorithm] = (cluster_id, confidence)
            
            # Find which cluster this narrative was assigned to in SECLEDS
            secleds_cluster_id = None
            for c_id, cluster in self.clusters.items():
                if narrative_id in cluster.members:
                    secleds_cluster_id = c_id
                    break
                    
            if secleds_cluster_id is not None:
                # Update the cluster with this evidence
                self.clusters[secleds_cluster_id].update_cross_algorithm_evidence(
                    algorithm, confidence
                )
                
    def get_escalated_items(self) -> List[Dict[str, Any]]:
        """
        Get items that need human review.
        
        Returns:
            List of escalated items
        """
        with self.lock:
            result = []
            
            for narrative_id, timestamp, reason in self.escalated_items:
                result.append({
                    "narrative_id": narrative_id,
                    "timestamp": timestamp.isoformat(),
                    "reason": reason,
                    "status": "pending"
                })
                
            return result
            
    def get_novel_narratives(self) -> List[Dict[str, Any]]:
        """
        Get detected novel narratives.
        
        Returns:
            List of novel narratives
        """
        with self.lock:
            result = []
            
            for narrative_id, vector, score in self.novel_narratives:
                result.append({
                    "narrative_id": narrative_id,
                    "novelty_score": score,
                    "vector": vector.tolist() if isinstance(vector, np.ndarray) else None
                })
                
            return result
            
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance visualization data.
        
        Returns:
            Dictionary with visualization data
        """
        with self.lock:
            return self.feature_weight_tracker.get_visualization_data()
            
    def get_cluster_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics about all clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        with self.lock:
            stats = {}
            
            for cluster_id, cluster in self.clusters.items():
                # Check drift
                is_drifting, drift_magnitude = cluster.check_drift()
                
                # Get confidence
                confidence = cluster.get_confidence_score()
                
                stats[cluster_id] = {
                    "medoid_count": len(cluster.medoids),
                    "member_count": len(cluster.members),
                    "last_updated": cluster.last_updated,
                    "created_at": cluster.created_at.isoformat(),
                    "is_drifting": is_drifting,
                    "drift_magnitude": drift_magnitude,
                    "confidence": confidence,
                    "novelty_score": cluster.novelty_score,
                    "escalation_status": cluster.escalation_status,
                    "cross_algorithm_evidence": cluster.cross_algorithm_evidence
                }
            
            return stats