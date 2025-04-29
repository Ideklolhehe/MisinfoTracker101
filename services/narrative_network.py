"""
Narrative Network Service for the CIVILIAN system.

This service provides network analysis capabilities for narratives, including:
- Network construction from narrative relationships
- Community detection and clustering
- Influence analysis
- Pattern detection
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

from app import db
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge

# Configure logging
logger = logging.getLogger(__name__)

class NarrativeNetworkService:
    """Service for analyzing narrative networks and their properties."""
    
    def __init__(self):
        """Initialize the narrative network service."""
        self.min_similarity = 0.65  # Minimum similarity threshold for relationships
        
        # Initialize clustering algorithms
        self._init_denstream()
        self._init_clustream()
        self._init_secleds()
    
    def _init_denstream(self):
        """Initialize the DenStream clustering algorithm."""
        # Storage for DenStream clusters
        self.denstream_clusters = {}
        self.denstream_cluster_counter = 0
        
    def _init_clustream(self):
        """Initialize the CluStream clustering algorithm."""
        # Storage for CluStream clusters
        self.clustream_clusters = {}
        self.clustream_cluster_counter = 0
        self.clustream_buffer = []
        
    def _init_secleds(self):
        """Initialize the SECLEDS algorithm."""
        # Storage for SECLEDS clusters
        self.secleds_clusters = {}
        self.secleds_cluster_counter = 0
        self.secleds_buffer = []
        
    def get_narrative_connections(self, narrative_id: int) -> Dict[str, Any]:
        """
        Get connections between a narrative and other narratives.
        
        Args:
            narrative_id: The ID of the central narrative
            
        Returns:
            Dictionary with connection data
        """
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            logger.warning(f"Narrative {narrative_id} not found")
            return {"nodes": [], "edges": []}
            
        # Get related narratives from belief network
        related_narratives = self._get_related_narratives(narrative_id)
        
        # Format as network
        nodes = [self._format_narrative_node(narrative)]
        edges = []
        
        for related in related_narratives:
            nodes.append(self._format_narrative_node(related["narrative"]))
            edges.append({
                "source": narrative_id,
                "target": related["narrative"].id,
                "type": related["relation_type"],
                "weight": related["weight"]
            })
            
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _get_related_narratives(self, narrative_id: int) -> List[Dict[str, Any]]:
        """
        Get narratives related to the given narrative.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            List of dictionaries with related narratives and relationship data
        """
        # This is a simplified version without the actual graph database
        # In a full implementation, this would use a graph database query
        
        # Find belief nodes for the narrative
        belief_nodes = db.session.query(BeliefNode).join(
            BeliefEdge, BeliefNode.id == BeliefEdge.target_id
        ).filter(
            BeliefEdge.source_id.in_(
                db.session.query(BeliefNode.id).filter(
                    BeliefNode.meta_data.contains(f'{{"narrative_id": {narrative_id}}}')
                )
            )
        ).all()
        
        # Find other narratives that share belief nodes
        related_narratives = []
        
        for node in belief_nodes:
            # Extract narrative IDs from node metadata
            try:
                node_meta = node.get_meta_data()
                if 'narrative_ids' not in node_meta:
                    continue
                    
                for rel_narrative_id in node_meta['narrative_ids']:
                    if rel_narrative_id == narrative_id:
                        continue
                        
                    narrative = DetectedNarrative.query.get(rel_narrative_id)
                    if not narrative:
                        continue
                        
                    # Check if already in results
                    if any(r['narrative'].id == rel_narrative_id for r in related_narratives):
                        continue
                        
                    # Find relationship type and weight from edges
                    edges = BeliefEdge.query.filter(
                        BeliefEdge.source_id.in_(
                            db.session.query(BeliefNode.id).filter(
                                BeliefNode.meta_data.contains(f'{{"narrative_id": {narrative_id}}}')
                            )
                        ),
                        BeliefEdge.target_id.in_(
                            db.session.query(BeliefNode.id).filter(
                                BeliefNode.meta_data.contains(f'{{"narrative_id": {rel_narrative_id}}}')
                            )
                        )
                    ).all()
                    
                    if edges:
                        relation_type = edges[0].relation_type
                        weight = edges[0].weight
                    else:
                        relation_type = "related"
                        weight = 0.5
                    
                    related_narratives.append({
                        "narrative": narrative,
                        "relation_type": relation_type,
                        "weight": weight
                    })
            except Exception as e:
                logger.error(f"Error processing belief node: {str(e)}")
                continue
                
        return related_narratives
    
    def _format_narrative_node(self, narrative: DetectedNarrative) -> Dict[str, Any]:
        """
        Format a narrative as a network node.
        
        Args:
            narrative: The narrative to format
            
        Returns:
            Dictionary with node data
        """
        meta_data = narrative.get_meta_data() or {}
        
        return {
            "id": narrative.id,
            "title": narrative.title,
            "type": "narrative",
            "confidence": narrative.confidence_score,
            "first_detected": narrative.first_detected.isoformat() if narrative.first_detected else None,
            "complexity": meta_data.get("complexity_score", 0),
            "threat_score": meta_data.get("threat_score", 0),
            "instance_count": meta_data.get("instance_count", 0)
        }
    
    def analyze_narrative_influence(self, narrative_id: int) -> Dict[str, Any]:
        """
        Analyze the influence of a narrative in the network.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Dictionary with influence metrics
        """
        # Get the narrative network
        network = self.get_narrative_connections(narrative_id)
        
        # Count connections
        connection_count = len(network["edges"])
        
        # Calculate basic centrality (percentage of narratives connected)
        all_narratives_count = DetectedNarrative.query.count()
        centrality = connection_count / all_narratives_count if all_narratives_count > 0 else 0
        
        # Get average weight of connections
        avg_weight = sum(edge["weight"] for edge in network["edges"]) / max(1, len(network["edges"]))
        
        # Get narrative instances for reach calculation
        instances_count = NarrativeInstance.query.filter_by(narrative_id=narrative_id).count()
        
        # Calculate reach score (normalized by all instances)
        all_instances_count = NarrativeInstance.query.count()
        reach_score = instances_count / all_instances_count if all_instances_count > 0 else 0
        
        return {
            "connections": connection_count,
            "centrality": centrality,
            "avg_connection_strength": avg_weight,
            "instance_count": instances_count,
            "reach_score": reach_score,
            "influence_score": (centrality + avg_weight + reach_score) / 3  # Simple average
        }
    
    def find_narrative_communities(self, min_narratives: int = 3) -> List[Dict[str, Any]]:
        """
        Find communities of related narratives.
        
        Args:
            min_narratives: Minimum number of narratives in a community
            
        Returns:
            List of communities with member narratives
        """
        # This is a simplified implementation
        # A full implementation would use graph clustering algorithms
        
        # Get all active narratives
        narratives = DetectedNarrative.query.filter_by(status='active').all()
        
        # Build connections matrix
        communities = []
        processed_narratives = set()
        
        for narrative in narratives:
            if narrative.id in processed_narratives:
                continue
                
            # Get connections
            connections = self._get_related_narratives(narrative.id)
            
            if len(connections) < min_narratives - 1:
                continue
                
            # Create community
            community_members = [narrative]
            for conn in connections:
                community_members.append(conn["narrative"])
                
            # Mark as processed
            processed_narratives.add(narrative.id)
            for member in community_members:
                processed_narratives.add(member.id)
                
            # Create community data
            community_data = {
                "id": len(communities) + 1,
                "size": len(community_members),
                "members": [self._format_narrative_node(n) for n in community_members],
                "connections": len(connections)
            }
            
            communities.append(community_data)
            
        return communities
        
    def process_narrative_with_denstream(self, narrative_id: int, embedding: np.ndarray) -> int:
        """
        Process a narrative with the DenStream algorithm.
        
        Args:
            narrative_id: ID of the narrative to process
            embedding: Feature vector representing the narrative
            
        Returns:
            Cluster ID assigned to the narrative
        """
        logger.debug(f"Processing narrative {narrative_id} with DenStream")
        
        # Check if narrative already belongs to a cluster
        for cluster_id, narratives in self.denstream_clusters.items():
            if narrative_id in narratives:
                return cluster_id
                
        # Find closest cluster based on centroid distance
        min_distance = float('inf')
        closest_cluster = None
        
        for cluster_id, narratives in self.denstream_clusters.items():
            # Get average embedding of cluster members
            if not narratives:
                continue
                
            cluster_embeddings = []
            for n_id in narratives:
                n_embedding = self._get_narrative_embedding_from_db(n_id)
                if n_embedding is not None:
                    cluster_embeddings.append(n_embedding)
                    
            if not cluster_embeddings:
                continue
                
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate distance
            distance = np.linalg.norm(embedding - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        # If close enough to existing cluster, assign to it
        if closest_cluster is not None and min_distance < 0.3:  # Threshold for cluster assignment
            self.denstream_clusters[closest_cluster].append(narrative_id)
            return closest_cluster
            
        # Otherwise create a new cluster
        new_cluster_id = self.denstream_cluster_counter
        self.denstream_cluster_counter += 1
        self.denstream_clusters[new_cluster_id] = [narrative_id]
        
        return new_cluster_id
        
    def process_narrative_with_clustream(self, narrative_id: int, embedding: np.ndarray, timestamp: datetime) -> int:
        """
        Process a narrative with the CluStream algorithm for temporal clustering.
        
        Args:
            narrative_id: ID of the narrative to process
            embedding: Feature vector representing the narrative
            timestamp: Timestamp of the narrative
            
        Returns:
            Cluster ID assigned to the narrative
        """
        logger.debug(f"Processing narrative {narrative_id} with CluStream")
        
        # Check if narrative already belongs to a cluster
        for cluster_id, cluster_data in self.clustream_clusters.items():
            if narrative_id in cluster_data["narratives"]:
                return cluster_id
                
        # Add to buffer for micro-cluster formation
        self.clustream_buffer.append({
            "narrative_id": narrative_id,
            "embedding": embedding,
            "timestamp": timestamp
        })
        
        # Process buffer when it reaches threshold size
        if len(self.clustream_buffer) >= 10:  # Minimum points for a micro-cluster
            # Find temporal clusters
            # In a full implementation, this would use the CluStream algorithm
            
            # Group by time periods (simplified)
            time_periods = {}
            for item in self.clustream_buffer:
                # Get day as period
                day_key = item["timestamp"].strftime("%Y-%m-%d")
                if day_key not in time_periods:
                    time_periods[day_key] = []
                time_periods[day_key].append(item)
            
            # For each time period, create or update cluster
            for period, items in time_periods.items():
                # Find closest cluster or create new one
                if len(items) < 3:  # Too few points for a micro-cluster
                    continue
                    
                # Calculate centroid
                embeddings = [item["embedding"] for item in items]
                centroid = np.mean(embeddings, axis=0)
                
                # Find closest existing cluster
                min_distance = float('inf')
                closest_cluster = None
                
                for cluster_id, cluster_data in self.clustream_clusters.items():
                    distance = np.linalg.norm(centroid - cluster_data["centroid"])
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster = cluster_id
                
                # If close enough to existing cluster, merge with it
                if closest_cluster is not None and min_distance < 0.3:
                    for item in items:
                        self.clustream_clusters[closest_cluster]["narratives"].append(item["narrative_id"])
                    
                    # Update centroid
                    all_embeddings = [item["embedding"] for item in items]
                    for n_id in self.clustream_clusters[closest_cluster]["narratives"]:
                        n_embedding = self._get_narrative_embedding_from_db(n_id)
                        if n_embedding is not None:
                            all_embeddings.append(n_embedding)
                            
                    self.clustream_clusters[closest_cluster]["centroid"] = np.mean(all_embeddings, axis=0)
                    
                else:
                    # Create new cluster
                    new_cluster_id = self.clustream_cluster_counter
                    self.clustream_cluster_counter += 1
                    
                    self.clustream_clusters[new_cluster_id] = {
                        "narratives": [item["narrative_id"] for item in items],
                        "centroid": centroid,
                        "created_at": datetime.utcnow()
                    }
            
            # Clear buffer
            self.clustream_buffer = []
        
        # For now, return a placeholder cluster ID
        # In real implementation, the narrative would be assigned to a micro-cluster
        return -1  # Unassigned until buffer processes
        
    def process_narrative_with_secleds(self, narrative_id: int, embedding: np.ndarray, timestamp: datetime) -> Tuple[int, float]:
        """
        Process a narrative with the SECLEDS algorithm for sequence-based clustering with concept drift adaptation.
        
        Args:
            narrative_id: ID of the narrative to process
            embedding: Feature vector representing the narrative
            timestamp: Timestamp of the narrative
            
        Returns:
            Tuple of (cluster_id, confidence) assigned to the narrative
        """
        logger.debug(f"Processing narrative {narrative_id} with SECLEDS")
        
        # Check if narrative already belongs to a cluster
        for cluster_id, cluster_data in self.secleds_clusters.items():
            if narrative_id in cluster_data["narratives"]:
                return cluster_id, cluster_data["confidence"].get(narrative_id, 0.5)
                
        # Add to buffer for sequential pattern detection
        self.secleds_buffer.append({
            "narrative_id": narrative_id,
            "embedding": embedding,
            "timestamp": timestamp
        })
        
        # Process buffer when it reaches threshold size
        if len(self.secleds_buffer) >= 15:  # Minimum points for a sequence
            # Sort by timestamp
            self.secleds_buffer.sort(key=lambda x: x["timestamp"])
            
            # Find sequential patterns
            # In a full implementation, this would use the SECLEDS algorithm
            
            # For now, use a simplified time-based clustering
            sequences = []
            current_sequence = []
            
            for i, item in enumerate(self.secleds_buffer):
                if not current_sequence:
                    current_sequence = [item]
                else:
                    # If time difference is small, add to current sequence
                    prev_time = current_sequence[-1]["timestamp"]
                    curr_time = item["timestamp"]
                    
                    time_diff = (curr_time - prev_time).total_seconds()
                    
                    if time_diff < 86400:  # 24 hours in seconds
                        current_sequence.append(item)
                    else:
                        # Save current sequence and start new one
                        if len(current_sequence) >= 3:  # Minimum length for a sequence
                            sequences.append(current_sequence)
                        current_sequence = [item]
            
            # Add last sequence
            if len(current_sequence) >= 3:
                sequences.append(current_sequence)
            
            # Process each sequence
            for sequence in sequences:
                # Calculate sequence embedding (average of all points)
                seq_embeddings = [item["embedding"] for item in sequence]
                seq_centroid = np.mean(seq_embeddings, axis=0)
                
                # Find closest existing cluster
                min_distance = float('inf')
                closest_cluster = None
                
                for cluster_id, cluster_data in self.secleds_clusters.items():
                    distance = np.linalg.norm(seq_centroid - cluster_data["centroid"])
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster = cluster_id
                
                # Calculate confidence based on distance
                confidence = 1.0 / (1.0 + min_distance)
                
                # If close enough to existing cluster, merge with it
                if closest_cluster is not None and min_distance < 0.3:
                    for item in sequence:
                        self.secleds_clusters[closest_cluster]["narratives"].append(item["narrative_id"])
                        self.secleds_clusters[closest_cluster]["confidence"][item["narrative_id"]] = confidence
                    
                    # Update centroid
                    all_embeddings = [item["embedding"] for item in sequence]
                    for n_id in self.secleds_clusters[closest_cluster]["narratives"]:
                        n_embedding = self._get_narrative_embedding_from_db(n_id)
                        if n_embedding is not None:
                            all_embeddings.append(n_embedding)
                            
                    self.secleds_clusters[closest_cluster]["centroid"] = np.mean(all_embeddings, axis=0)
                    
                else:
                    # Create new cluster
                    new_cluster_id = self.secleds_cluster_counter
                    self.secleds_cluster_counter += 1
                    
                    self.secleds_clusters[new_cluster_id] = {
                        "narratives": [item["narrative_id"] for item in sequence],
                        "centroid": seq_centroid,
                        "created_at": datetime.utcnow(),
                        "confidence": {item["narrative_id"]: confidence for item in sequence}
                    }
            
            # Clear buffer
            self.secleds_buffer = []
        
        # For now, return a placeholder cluster ID and confidence
        # In real implementation, the narrative would be assigned to a sequence cluster
        return -1, 0.5  # Unassigned until buffer processes
        
    def _get_narrative_embedding_from_db(self, narrative_id: int) -> Optional[np.ndarray]:
        """
        Retrieve a narrative embedding from the database.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Embedding vector or None if not found
        """
        # In a full implementation, this would retrieve from a database
        # For now, we'll return None to simulate not found
        
        # First try to find the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return None
            
        # In a real implementation, this would extract an embedding from metadata
        # For now, return a random vector to simulate an embedding
        try:
            # Generate a consistent random embedding based on narrative_id
            np.random.seed(narrative_id)
            return np.random.rand(1000)  # Match the dimension used in stream_processor.py
        finally:
            # Reset the random seed
            np.random.seed(None)