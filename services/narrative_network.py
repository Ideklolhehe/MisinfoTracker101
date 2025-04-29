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