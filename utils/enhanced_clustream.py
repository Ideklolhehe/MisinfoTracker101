"""
Enhanced CluStream implementation for the CIVILIAN system.

This module provides an enhanced version of the CluStream algorithm with:
1. Real-time macro-clustering
2. Dynamic temporal snapshots based on narrative lifecycle
3. Cluster evolution tracking
4. Cluster reassignment mechanism for unassigned narratives

Based on the original CluStream algorithm but extended with CIVILIAN-specific features.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from datetime import datetime, timedelta
import threading
import heapq

# Configure logging
logger = logging.getLogger(__name__)

class TemporalWindow:
    """
    A temporal window representing a specific time range for clustering.
    """
    def __init__(self, start_time: datetime, end_time: datetime, level: int = 0):
        """
        Initialize a temporal window.
        
        Args:
            start_time: Start of the time window
            end_time: End of the time window
            level: Granularity level (0 = finest, higher = coarser)
        """
        self.start_time = start_time
        self.end_time = end_time
        self.level = level
        self.points = []  # List of (vector, timestamp, narrative_id) tuples
        self.cluster_id = -1  # ID of the cluster this window belongs to
        self.is_active = True  # Whether this window is still accepting new points
        
    def add_point(self, vector: np.ndarray, timestamp: datetime, 
                 narrative_id: Optional[str] = None) -> None:
        """
        Add a point to this window.
        
        Args:
            vector: Vector representation of the point
            timestamp: Timestamp of the point
            narrative_id: Optional narrative ID
        """
        if not self.is_active:
            return
            
        if timestamp < self.start_time or timestamp > self.end_time:
            return
            
        self.points.append((vector, timestamp, narrative_id))
        
    def get_centroid(self) -> np.ndarray:
        """
        Calculate the centroid of points in this window.
        
        Returns:
            Centroid vector
        """
        if not self.points:
            return None
            
        vectors = [p[0] for p in self.points]
        return np.mean(vectors, axis=0)
        
    def get_narrative_ids(self) -> List[str]:
        """
        Get all narrative IDs in this window.
        
        Returns:
            List of narrative IDs
        """
        return [p[2] for p in self.points if p[2] is not None]
        
    def get_size(self) -> int:
        """
        Get the number of points in this window.
        
        Returns:
            Number of points
        """
        return len(self.points)
        
    def close(self) -> None:
        """Mark this window as inactive (no more points can be added)."""
        self.is_active = False


class EnhancedMicroCluster:
    """
    Micro-cluster representation for the CluStream algorithm.
    """
    def __init__(self, center: np.ndarray, radius: float = 0.0, creation_time: datetime = None,
                weight: float = 1.0, id: int = -1):
        """
        Initialize a micro-cluster.
        
        Args:
            center: Center of the micro-cluster
            radius: Radius of the micro-cluster
            creation_time: Creation time (defaults to now)
            weight: Weight of the micro-cluster
            id: ID of the micro-cluster
        """
        self.center = center
        self.radius = radius
        self.creation_time = creation_time or datetime.utcnow()
        self.last_updated = self.creation_time
        self.weight = weight
        self.id = id
        
        # Member tracking
        self.member_ids = set()  # Set of narrative IDs in this MC
        self.member_vectors = {}  # Dict of {id: vector}
        self.member_timestamps = {}  # Dict of {id: timestamp}
        
        # Evolution tracking
        self.parent_id = None  # ID of parent MC if this was split from another
        self.child_ids = set()  # IDs of MCs that were split from this one
        self.has_evolved = False  # Whether this MC has significantly evolved
        
    def add_point(self, vector: np.ndarray, timestamp: datetime, 
                narrative_id: Optional[str] = None) -> bool:
        """
        Add a point to this micro-cluster.
        
        Args:
            vector: Vector to add
            timestamp: Timestamp of the vector
            narrative_id: Optional narrative ID
            
        Returns:
            True if the point was added, False otherwise
        """
        dist = np.linalg.norm(vector - self.center)
        
        # Check if point is within radius
        if dist > self.radius * 3 and self.radius > 0:
            return False
            
        # Update center (weighted average)
        self.center = (self.center * self.weight + vector) / (self.weight + 1)
        
        # Update radius
        if self.radius == 0:
            self.radius = dist
        else:
            self.radius = (self.radius * self.weight + dist) / (self.weight + 1)
            
        # Update weight and timestamps
        self.weight += 1
        self.last_updated = timestamp
        
        # Add member if ID provided
        if narrative_id:
            self.member_ids.add(narrative_id)
            self.member_vectors[narrative_id] = vector
            self.member_timestamps[narrative_id] = timestamp
            
        # Check if cluster has evolved significantly
        if len(self.member_ids) > 10 and not self.has_evolved:
            # Check recency distribution
            now = datetime.utcnow()
            recent_count = sum(1 for ts in self.member_timestamps.values() 
                             if (now - ts).total_seconds() < 86400)  # Last 24 hours
            
            if recent_count > len(self.member_ids) * 0.5:
                self.has_evolved = True
            
        return True
        
    def merge_with(self, other: 'EnhancedMicroCluster') -> None:
        """
        Merge with another micro-cluster.
        
        Args:
            other: Another micro-cluster to merge with
        """
        # Weighted center
        total_weight = self.weight + other.weight
        self.center = (self.center * self.weight + other.center * other.weight) / total_weight
        
        # Weighted radius
        self.radius = (self.radius * self.weight + other.radius * other.weight) / total_weight
        
        # Update weight and timestamp
        self.weight = total_weight
        self.last_updated = max(self.last_updated, other.last_updated)
        
        # Merge members
        self.member_ids.update(other.member_ids)
        self.member_vectors.update(other.member_vectors)
        self.member_timestamps.update(other.member_timestamps)
        
        # Track evolution
        self.child_ids.update(other.child_ids)
        
    def get_age(self) -> float:
        """
        Get the age of this micro-cluster in days.
        
        Returns:
            Age in days
        """
        return (datetime.utcnow() - self.creation_time).total_seconds() / 86400.0
        
    def get_temporal_distribution(self) -> Dict[str, int]:
        """
        Get the temporal distribution of points in this micro-cluster.
        
        Returns:
            Dict mapping time periods to counts
        """
        if not self.member_timestamps:
            return {}
            
        # Group by day
        distribution = {}
        for timestamp in self.member_timestamps.values():
            day_key = timestamp.strftime("%Y-%m-%d")
            distribution[day_key] = distribution.get(day_key, 0) + 1
            
        return distribution


class ClusterEvolutionTracker:
    """
    Tracks the evolution of clusters over time.
    """
    def __init__(self):
        """Initialize the evolution tracker."""
        self.cluster_history = {}  # Dict of {cluster_id: [(timestamp, narratives, center)]}
        self.evolution_events = []  # List of (timestamp, event_type, details) tuples
        
    def record_cluster_state(self, cluster_id: int, timestamp: datetime, 
                           narratives: List[str], center: np.ndarray) -> None:
        """
        Record the state of a cluster at a point in time.
        
        Args:
            cluster_id: ID of the cluster
            timestamp: Current timestamp
            narratives: List of narrative IDs in the cluster
            center: Center of the cluster
        """
        if cluster_id not in self.cluster_history:
            self.cluster_history[cluster_id] = []
            
        self.cluster_history[cluster_id].append((timestamp, narratives, center))
        
    def record_event(self, timestamp: datetime, event_type: str, details: Dict[str, Any]) -> None:
        """
        Record an evolution event.
        
        Args:
            timestamp: When the event occurred
            event_type: Type of event (e.g., 'merge', 'split', 'grow', 'shrink')
            details: Details about the event
        """
        self.evolution_events.append((timestamp, event_type, details))
        
    def get_cluster_evolution(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get the evolution history of a cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            List of evolution stages
        """
        if cluster_id not in self.cluster_history:
            return []
            
        history = self.cluster_history[cluster_id]
        
        # Calculate growth rates, center shifts, etc.
        evolution = []
        for i in range(len(history)):
            stage = {
                "timestamp": history[i][0],
                "size": len(history[i][1]),
                "center": history[i][2]
            }
            
            if i > 0:
                # Calculate growth rate
                prev_size = len(history[i-1][1])
                curr_size = len(history[i][1])
                stage["growth_rate"] = (curr_size - prev_size) / max(1, prev_size)
                
                # Calculate center shift
                prev_center = history[i-1][2]
                curr_center = history[i][2]
                stage["center_shift"] = np.linalg.norm(curr_center - prev_center)
                
                # Calculate churn (narratives added/removed)
                prev_narratives = set(history[i-1][1])
                curr_narratives = set(history[i][1])
                added = curr_narratives - prev_narratives
                removed = prev_narratives - curr_narratives
                stage["narratives_added"] = len(added)
                stage["narratives_removed"] = len(removed)
                stage["churn_rate"] = (len(added) + len(removed)) / max(1, prev_size)
            
            evolution.append(stage)
            
        return evolution
        
    def get_cluster_events(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get events related to a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            List of events
        """
        relevant_events = []
        
        for timestamp, event_type, details in self.evolution_events:
            if 'cluster_id' in details and details['cluster_id'] == cluster_id:
                relevant_events.append({
                    "timestamp": timestamp,
                    "type": event_type,
                    "details": details
                })
            elif 'source_cluster_id' in details and details['source_cluster_id'] == cluster_id:
                relevant_events.append({
                    "timestamp": timestamp,
                    "type": event_type,
                    "details": details
                })
            elif 'target_cluster_id' in details and details['target_cluster_id'] == cluster_id:
                relevant_events.append({
                    "timestamp": timestamp,
                    "type": event_type,
                    "details": details
                })
                
        return relevant_events


class EnhancedCluStream:
    """
    Enhanced CluStream implementation with additional features for the CIVILIAN system.
    """
    
    def __init__(self, max_micro_clusters: int = 100, 
                time_window: float = 1.0, 
                time_decay: float = 0.05, 
                snapshot_interval: int = 1000,
                epsilon: float = 0.3,
                min_macro_clusters: int = 5,
                max_macro_clusters: int = 20,
                min_points_for_macro: int = 10,
                reassignment_interval: int = 50):
        """
        Initialize the EnhancedCluStream algorithm.
        
        Args:
            max_micro_clusters: Maximum number of micro-clusters
            time_window: Window length in days (float, can be fractional)
            time_decay: Decay rate for old micro-clusters
            snapshot_interval: Number of points between snapshots
            epsilon: Maximum distance for micro-cluster assignment
            min_macro_clusters: Minimum number of macro-clusters
            max_macro_clusters: Maximum number of macro-clusters
            min_points_for_macro: Minimum points needed for macro-clustering
            reassignment_interval: How often to try reassigning noise points
        """
        self.max_micro_clusters = max_micro_clusters
        self.time_window = time_window
        self.time_decay = time_decay
        self.snapshot_interval = snapshot_interval
        self.epsilon = epsilon
        self.min_macro_clusters = min_macro_clusters
        self.max_macro_clusters = max_macro_clusters
        self.min_points_for_macro = min_points_for_macro
        self.reassignment_interval = reassignment_interval
        
        # Clusters
        self.micro_clusters = []  # List of EnhancedMicroCluster objects
        self.next_mc_id = 0
        
        # Temporal structures
        self.current_window = None
        self.windows = {}  # Dict of {level: [TemporalWindow]}
        self.snapshots = []  # List of (timestamp, mc_centers, mc_weights, mc_radiuses) tuples
        
        # Unassigned points for later reassignment
        self.unassigned_points = []  # List of (vector, timestamp, narrative_id) tuples
        
        # Tracking
        self.items_processed = 0
        self.last_timestamp = None
        
        # Evolution tracking
        self.evolution_tracker = ClusterEvolutionTracker()
        
        # Current macro-clusters
        self.macro_clusters = {}  # Dict of {cluster_id: [micro_cluster_ids]}
        self.next_macro_id = 0
        
        # Thread lock for concurrency
        self.lock = threading.Lock()
        
        logger.info(f"EnhancedCluStream initialized with {max_micro_clusters} micro-clusters")
        
    def _initialize_current_window(self, timestamp: datetime) -> None:
        """
        Initialize the current temporal window.
        
        Args:
            timestamp: Current timestamp
        """
        if self.current_window is None or not self.current_window.is_active:
            # Calculate window duration in seconds
            window_seconds = int(self.time_window * 86400)  # days to seconds
            
            # Create new window
            start_time = timestamp
            end_time = timestamp + timedelta(seconds=window_seconds)
            self.current_window = TemporalWindow(start_time, end_time)
            
            # Add to level 0 windows
            if 0 not in self.windows:
                self.windows[0] = []
                
            self.windows[0].append(self.current_window)
            
    def _create_pyramidal_time_frame(self, timestamp: datetime, 
                                    vector: np.ndarray, narrative_id: Optional[str] = None) -> None:
        """
        Create pyramidal time frame structure and assign point to appropriate windows.
        
        Args:
            timestamp: Current timestamp
            vector: Vector to add
            narrative_id: Optional narrative ID
        """
        # Add to current window at level 0
        self._initialize_current_window(timestamp)
        self.current_window.add_point(vector, timestamp, narrative_id)
        
        # Check if current window is expired
        if timestamp > self.current_window.end_time:
            self.current_window.close()
            self._initialize_current_window(timestamp)
            self.current_window.add_point(vector, timestamp, narrative_id)
            
        # Add to higher level windows
        order = 1
        time_order = 1
        
        while order <= 10:  # Up to 10 levels
            frame_size = 2 ** order * int(self.time_window * 86400)  # 2^order times larger window
            snippet_id = int(timestamp.timestamp() / frame_size)
            
            # Get or create window for this level and snippet
            if order not in self.windows:
                self.windows[order] = []
                
            # Find matching window or create new one
            window = None
            for w in self.windows[order]:
                if timestamp >= w.start_time and timestamp <= w.end_time:
                    window = w
                    break
                    
            if window is None:
                # Create new window
                start_time = datetime.fromtimestamp(snippet_id * frame_size)
                end_time = datetime.fromtimestamp((snippet_id + 1) * frame_size)
                window = TemporalWindow(start_time, end_time, level=order)
                self.windows[order].append(window)
                
            # Add point to this window
            window.add_point(vector, timestamp, narrative_id)
            
            order += 1
            
    def _find_closest_micro_cluster(self, vector: np.ndarray) -> Tuple[int, float]:
        """
        Find the closest micro-cluster to a vector.
        
        Args:
            vector: Vector to find closest MC for
            
        Returns:
            Tuple of (index, distance) or (-1, inf) if none found
        """
        min_dist = float('inf')
        closest_idx = -1
        
        for i, mc in enumerate(self.micro_clusters):
            dist = np.linalg.norm(vector - mc.center)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx, min_dist
        
    def _try_add_to_micro_clusters(self, vector: np.ndarray, timestamp: datetime, 
                                  narrative_id: Optional[str] = None) -> int:
        """
        Try to add a point to existing micro-clusters.
        
        Args:
            vector: Vector to add
            timestamp: Timestamp of the vector
            narrative_id: Optional narrative ID
            
        Returns:
            ID of the micro-cluster the point was added to, or -1 if none
        """
        closest_idx, min_dist = self._find_closest_micro_cluster(vector)
        
        # If we found a close enough cluster, add the point
        if closest_idx >= 0 and min_dist <= self.epsilon:
            added = self.micro_clusters[closest_idx].add_point(vector, timestamp, narrative_id)
            if added:
                return self.micro_clusters[closest_idx].id
            
        # If we couldn't add to existing clusters, create a new one
        if len(self.micro_clusters) < self.max_micro_clusters:
            # Create new micro-cluster
            new_mc = EnhancedMicroCluster(vector, 0.0, timestamp, 1.0, self.next_mc_id)
            self.next_mc_id += 1
            
            if narrative_id:
                new_mc.add_point(vector, timestamp, narrative_id)
                
            self.micro_clusters.append(new_mc)
            return new_mc.id
            
        else:
            # Find the oldest micro-cluster
            oldest_idx = min(range(len(self.micro_clusters)), 
                           key=lambda i: self.micro_clusters[i].last_updated)
                           
            # Replace it if it's old enough
            if (timestamp - self.micro_clusters[oldest_idx].last_updated).total_seconds() > self.time_window * 86400:
                new_mc = EnhancedMicroCluster(vector, 0.0, timestamp, 1.0, self.next_mc_id)
                self.next_mc_id += 1
                
                if narrative_id:
                    new_mc.add_point(vector, timestamp, narrative_id)
                    
                self.micro_clusters[oldest_idx] = new_mc
                return new_mc.id
                
        # Couldn't add to any cluster - save for later reassignment
        self.unassigned_points.append((vector, timestamp, narrative_id))
        return -1
        
    def _update_macro_clusters(self) -> None:
        """Update macro clusters based on current micro-clusters."""
        if len(self.micro_clusters) < self.min_points_for_macro:
            return
            
        try:
            # Extract micro-cluster centers and weights
            centers = np.array([mc.center for mc in self.micro_clusters])
            weights = np.array([mc.weight for mc in self.micro_clusters])
            
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Determine optimal number of clusters
            best_k = self.min_macro_clusters
            best_score = -1
            best_labels = None
            
            for k in range(self.min_macro_clusters, min(self.max_macro_clusters + 1, len(centers))):
                kmeans = KMeans(n_clusters=k, n_init=10)
                labels = kmeans.fit_predict(centers)
                
                if len(set(labels)) > 1:  # Ensure we have at least 2 clusters
                    score = silhouette_score(centers, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                        
            # If we have valid clustering results
            if best_labels is not None:
                # Create new macro-clusters
                new_macros = {}
                
                for i, label in enumerate(best_labels):
                    if label not in new_macros:
                        new_macros[label] = []
                        
                    new_macros[label].append(self.micro_clusters[i].id)
                    
                # Record evolution events for significantly changed clusters
                self._record_cluster_evolution(new_macros)
                
                # Update macro-clusters
                self.macro_clusters = {}
                for old_id, mc_ids in new_macros.items():
                    new_id = self.next_macro_id
                    self.next_macro_id += 1
                    self.macro_clusters[new_id] = mc_ids
                    
                    # Update cluster ID in micro-clusters
                    for mc in self.micro_clusters:
                        if mc.id in mc_ids:
                            mc.cluster_id = new_id
                
        except Exception as e:
            logger.error(f"Error updating macro-clusters: {e}")
            
    def _record_cluster_evolution(self, new_macros: Dict[int, List[int]]) -> None:
        """
        Record cluster evolution events.
        
        Args:
            new_macros: New macro-cluster assignments
        """
        now = datetime.utcnow()
        
        # Map micro-cluster IDs to old macro-cluster IDs
        mc_to_old_macro = {}
        for macro_id, mc_ids in self.macro_clusters.items():
            for mc_id in mc_ids:
                mc_to_old_macro[mc_id] = macro_id
                
        # Check for significant events
        for new_label, new_mc_ids in new_macros.items():
            # Get old macro-clusters these MCs belonged to
            old_macros = set()
            for mc_id in new_mc_ids:
                if mc_id in mc_to_old_macro:
                    old_macros.add(mc_to_old_macro[mc_id])
                    
            # If these MCs came from multiple old clusters, it's a merge
            if len(old_macros) > 1:
                self.evolution_tracker.record_event(
                    now, 
                    "merge", 
                    {
                        "source_cluster_ids": list(old_macros),
                        "target_cluster_id": new_label,
                        "micro_cluster_ids": new_mc_ids
                    }
                )
                
        # Check if any old clusters were split
        for old_macro_id, old_mc_ids in self.macro_clusters.items():
            # Where did these MCs go in the new clustering?
            new_destinations = {}
            for mc_id in old_mc_ids:
                found = False
                for new_label, new_mc_ids in new_macros.items():
                    if mc_id in new_mc_ids:
                        if new_label not in new_destinations:
                            new_destinations[new_label] = []
                        new_destinations[new_label].append(mc_id)
                        found = True
                        break
                
            # If MCs from this old cluster went to multiple new clusters, it's a split
            if len(new_destinations) > 1:
                self.evolution_tracker.record_event(
                    now, 
                    "split", 
                    {
                        "source_cluster_id": old_macro_id,
                        "target_cluster_ids": list(new_destinations.keys()),
                        "split_distribution": {k: len(v) for k, v in new_destinations.items()}
                    }
                )
                
    def _try_reassign_unassigned_points(self) -> int:
        """
        Try to reassign previously unassigned points.
        
        Returns:
            Number of points successfully reassigned
        """
        if not self.unassigned_points:
            return 0
            
        reassigned = 0
        still_unassigned = []
        
        for vector, timestamp, narrative_id in self.unassigned_points:
            closest_idx, min_dist = self._find_closest_micro_cluster(vector)
            
            # If we found a close enough cluster, add the point
            if closest_idx >= 0 and min_dist <= self.epsilon:
                added = self.micro_clusters[closest_idx].add_point(vector, timestamp, narrative_id)
                if added:
                    reassigned += 1
                    continue
                    
            # Still can't assign it
            still_unassigned.append((vector, timestamp, narrative_id))
            
        # Update unassigned points list
        self.unassigned_points = still_unassigned
        
        return reassigned
        
    def _take_snapshot(self, timestamp: datetime) -> None:
        """
        Take a snapshot of the current micro-clusters.
        
        Args:
            timestamp: Current timestamp
        """
        if not self.micro_clusters:
            return
            
        centers = [mc.center for mc in self.micro_clusters]
        weights = [mc.weight for mc in self.micro_clusters]
        radiuses = [mc.radius for mc in self.micro_clusters]
        mc_ids = [mc.id for mc in self.micro_clusters]
        
        self.snapshots.append((timestamp, centers, weights, radiuses, mc_ids))
        
        # Limit number of snapshots (keep most recent 100)
        if len(self.snapshots) > 100:
            self.snapshots.pop(0)
            
    def _get_dynamic_window_duration(self) -> float:
        """
        Calculate a dynamic window duration based on narrative lifecycle.
        
        Returns:
            Window duration in days
        """
        if not self.micro_clusters or not self.snapshots:
            return self.time_window
            
        # Analyze recent cluster evolution to determine appropriate window size
        try:
            # Calculate average narrative lifecycle from micro-clusters
            lifecycle_estimates = []
            for mc in self.micro_clusters:
                if len(mc.member_timestamps) < 2:
                    continue
                    
                # Get time span between first and last narrative
                times = list(mc.member_timestamps.values())
                time_span = (max(times) - min(times)).total_seconds() / 86400  # in days
                
                if time_span > 0:
                    lifecycle_estimates.append(time_span)
                    
            if lifecycle_estimates:
                # Use median lifecycle as window duration, with bounds
                median_lifecycle = np.median(lifecycle_estimates)
                return max(0.5, min(30, median_lifecycle))  # Between 12 hours and 30 days
            
        except Exception as e:
            logger.error(f"Error calculating dynamic window duration: {e}")
            
        return self.time_window
        
    def learn_one(self, x: np.ndarray, timestamp: datetime = None, 
                narrative_id: Optional[str] = None) -> int:
        """
        Update the model with a new sample.
        
        Args:
            x: Vector representing the sample
            timestamp: Timestamp of the sample (defaults to now)
            narrative_id: Optional narrative ID
            
        Returns:
            Cluster ID assigned to the sample
        """
        with self.lock:
            if timestamp is None:
                timestamp = datetime.utcnow()
                
            self.items_processed += 1
            self.last_timestamp = timestamp
            
            # Add to temporal windows
            self._create_pyramidal_time_frame(timestamp, x, narrative_id)
            
            # Try to add to micro-clusters
            mc_id = self._try_add_to_micro_clusters(x, timestamp, narrative_id)
            
            # Periodically update macro-clusters in real-time
            if self.items_processed % 10 == 0:
                self._update_macro_clusters()
                
            # Periodically try to reassign unassigned points
            if self.items_processed % self.reassignment_interval == 0:
                reassigned = self._try_reassign_unassigned_points()
                if reassigned > 0:
                    logger.debug(f"Reassigned {reassigned} previously unassigned points")
                    
            # Periodically take snapshots
            if self.items_processed % self.snapshot_interval == 0:
                self._take_snapshot(timestamp)
                
                # Update window duration dynamically
                new_duration = self._get_dynamic_window_duration()
                if abs(new_duration - self.time_window) > 0.1:  # Significant change
                    logger.info(f"Adjusting temporal window from {self.time_window:.2f} to {new_duration:.2f} days")
                    self.time_window = new_duration
                    
            # Map micro-cluster ID to macro-cluster ID
            if mc_id != -1:
                for mc in self.micro_clusters:
                    if mc.id == mc_id:
                        return mc.cluster_id
                        
            return -1  # Unassigned
            
    def predict(self, x: np.ndarray) -> int:
        """
        Predict the cluster for a new sample.
        
        Args:
            x: Vector to classify
            
        Returns:
            Cluster ID
        """
        with self.lock:
            closest_idx, min_dist = self._find_closest_micro_cluster(x)
            
            if closest_idx >= 0 and min_dist <= self.epsilon:
                return self.micro_clusters[closest_idx].cluster_id
                
            return -1  # Noise point
            
    def get_clusters(self) -> Dict[str, Any]:
        """
        Get the current clustering results.
        
        Returns:
            Dictionary with cluster information
        """
        with self.lock:
            clusters = {}
            
            # Reorganize by macro-cluster
            for macro_id, mc_ids in self.macro_clusters.items():
                narratives = set()
                
                # Collect all narratives from component micro-clusters
                for mc in self.micro_clusters:
                    if mc.id in mc_ids:
                        narratives.update(mc.member_ids)
                        
                # Skip empty clusters
                if not narratives:
                    continue
                    
                # Create cluster data
                clusters[str(macro_id)] = {
                    "narratives": list(narratives),
                    "size": len(narratives),
                    "micro_clusters": mc_ids,
                    "evolution": self.evolution_tracker.get_cluster_events(macro_id)
                }
                
            # Add noise cluster for unassigned points
            noise_narratives = set()
            for vector, timestamp, narrative_id in self.unassigned_points:
                if narrative_id:
                    noise_narratives.add(narrative_id)
                    
            if noise_narratives:
                clusters["-1"] = {
                    "narratives": list(noise_narratives),
                    "size": len(noise_narratives),
                    "micro_clusters": [],
                    "evolution": []
                }
                
            return {
                "clusters": clusters,
                "total_micro_clusters": len(self.micro_clusters),
                "total_macro_clusters": len(self.macro_clusters),
                "total_unassigned": len(self.unassigned_points),
                "window_duration": self.time_window,
                "timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None
            }
            
    def get_cluster_evolution(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get evolution data for a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary with evolution information
        """
        with self.lock:
            if cluster_id not in self.macro_clusters:
                return {"error": "Cluster not found"}
                
            mc_ids = self.macro_clusters[cluster_id]
            narratives = set()
            
            # Collect all narratives from component micro-clusters
            for mc in self.micro_clusters:
                if mc.id in mc_ids:
                    narratives.update(mc.member_ids)
                    
            events = self.evolution_tracker.get_cluster_events(cluster_id)
            
            temporal_data = {}
            for mc in self.micro_clusters:
                if mc.id in mc_ids:
                    distrib = mc.get_temporal_distribution()
                    for day, count in distrib.items():
                        if day not in temporal_data:
                            temporal_data[day] = 0
                        temporal_data[day] += count
                        
            return {
                "cluster_id": cluster_id,
                "size": len(narratives),
                "narratives": list(narratives),
                "micro_clusters": mc_ids,
                "events": events,
                "temporal_distribution": [{"date": k, "count": v} for k, v in temporal_data.items()],
                "evolution": "stable" if not events else events[-1]["type"]
            }