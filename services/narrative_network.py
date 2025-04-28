"""
Narrative Network Analysis Service for the CIVILIAN system.
Analyzes and visualizes relationships between narratives using graph theory.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import db
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge

logger = logging.getLogger(__name__)

class NarrativeNetworkAnalyzer:
    """
    Service for analyzing relationships between narratives and generating 
    network visualizations using graph theory.
    """
    
    def __init__(self):
        """Initialize the narrative network analyzer."""
        self.graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def build_narrative_network(self, include_archived: bool = False) -> nx.DiGraph:
        """
        Build a network of narratives based on content similarity and relationships.
        
        Args:
            include_archived: Whether to include archived narratives in the network
            
        Returns:
            A directed graph representing the narrative network
        """
        try:
            # Clear existing graph
            self.graph.clear()
            
            # Get narratives
            query = DetectedNarrative.query
            if not include_archived:
                query = query.filter(DetectedNarrative.status != 'archived')
            
            narratives = query.all()
            if not narratives:
                logger.warning("No narratives found for network analysis")
                return self.graph
            
            # Extract titles and descriptions for similarity analysis
            narrative_texts = []
            for narrative in narratives:
                text = f"{narrative.title} {narrative.description or ''}"
                narrative_texts.append(text)
            
            # Calculate similarity matrix
            try:
                text_matrix = self.vectorizer.fit_transform(narrative_texts)
                similarity_matrix = cosine_similarity(text_matrix)
            except Exception as e:
                logger.error(f"Error calculating similarity matrix: {e}")
                similarity_matrix = np.zeros((len(narratives), len(narratives)))
            
            # Add nodes to graph
            for i, narrative in enumerate(narratives):
                # Add node with attributes
                self.graph.add_node(
                    narrative.id,
                    title=narrative.title,
                    status=narrative.status,
                    confidence=float(narrative.confidence_score or 0),
                    first_detected=narrative.first_detected.isoformat(),
                    last_updated=narrative.last_updated.isoformat(),
                    language=narrative.language
                )
            
            # Add edges based on similarity
            similarity_threshold = 0.3  # Minimum similarity to create an edge
            for i in range(len(narratives)):
                for j in range(len(narratives)):
                    if i != j and similarity_matrix[i, j] > similarity_threshold:
                        self.graph.add_edge(
                            narratives[i].id,
                            narratives[j].id,
                            weight=float(similarity_matrix[i, j]),
                            type="similar_content"
                        )
            
            # Add edges from BeliefEdge table where both nodes are narratives
            belief_edges = BeliefEdge.query.all()
            for edge in belief_edges:
                source_node = BeliefNode.query.get(edge.source_id)
                target_node = BeliefNode.query.get(edge.target_id)
                
                if source_node and target_node:
                    meta_data = source_node.get_meta_data() or {}
                    source_narrative_id = meta_data.get('narrative_id')
                    
                    meta_data = target_node.get_meta_data() or {}
                    target_narrative_id = meta_data.get('narrative_id')
                    
                    if source_narrative_id and target_narrative_id:
                        self.graph.add_edge(
                            source_narrative_id,
                            target_narrative_id,
                            weight=float(edge.weight),
                            type=edge.relation_type
                        )
            
            return self.graph
            
        except Exception as e:
            logger.error(f"Error building narrative network: {e}")
            return nx.DiGraph()
    
    def identify_coordinated_campaigns(self, min_narratives: int = 3, min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify potential coordinated misinformation campaigns based on narrative clustering.
        
        Args:
            min_narratives: Minimum number of narratives to consider a campaign
            min_similarity: Minimum similarity threshold for narratives to be considered related
            
        Returns:
            List of potential campaigns with related narratives
        """
        try:
            # Ensure we have a built graph
            if self.graph.number_of_nodes() == 0:
                self.build_narrative_network()
            
            # Create similarity-based subgraph
            similarity_graph = nx.DiGraph()
            
            for u, v, data in self.graph.edges(data=True):
                weight = data.get('weight', 0)
                edge_type = data.get('type', '')
                
                if weight >= min_similarity or edge_type in ['supports', 'relates_to', 'references']:
                    similarity_graph.add_edge(u, v, weight=weight, type=edge_type)
            
            # Find connected components (potential campaigns)
            undirected = similarity_graph.to_undirected()
            connected_components = list(nx.connected_components(undirected))
            
            # Filter and format campaigns
            campaigns = []
            for i, component in enumerate(connected_components):
                if len(component) >= min_narratives:
                    # Get narratives in this component
                    narratives = []
                    for node_id in component:
                        narrative = DetectedNarrative.query.get(node_id)
                        if narrative:
                            narratives.append({
                                'id': narrative.id,
                                'title': narrative.title,
                                'confidence': float(narrative.confidence_score or 0),
                                'status': narrative.status,
                                'first_detected': narrative.first_detected.isoformat()
                            })
                    
                    # Calculate internal density (connectedness)
                    subgraph = similarity_graph.subgraph(component)
                    possible_edges = len(component) * (len(component) - 1)
                    density = subgraph.number_of_edges() / possible_edges if possible_edges > 0 else 0
                    
                    # Get first detection timeframe
                    first_detected_dates = [n['first_detected'] for n in narratives]
                    first_detected_dates.sort()
                    
                    campaigns.append({
                        'id': i + 1,
                        'size': len(component),
                        'density': density,
                        'narrative_ids': list(component),
                        'narratives': narratives,
                        'first_narrative_date': first_detected_dates[0] if first_detected_dates else None,
                        'timespan_days': (datetime.fromisoformat(first_detected_dates[-1]) - 
                                          datetime.fromisoformat(first_detected_dates[0])).days
                                         if len(first_detected_dates) > 1 else 0
                    })
            
            # Sort by size (descending)
            campaigns.sort(key=lambda x: x['size'], reverse=True)
            
            return campaigns
            
        except Exception as e:
            logger.error(f"Error identifying coordinated campaigns: {e}")
            return []
    
    def get_central_narratives(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify the most central narratives in the network based on centrality metrics.
        
        Args:
            top_n: Number of top central narratives to return
            
        Returns:
            List of dictionaries with narrative information and centrality scores
        """
        try:
            # Ensure we have a built graph
            if self.graph.number_of_nodes() == 0:
                self.build_narrative_network()
            
            # Calculate centrality metrics
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            eigenvector_centrality = nx.eigenvector_centrality_numpy(self.graph)
            
            # Combine metrics for each node
            centrality_scores = {}
            for node in self.graph.nodes():
                centrality_scores[node] = {
                    'degree': degree_centrality.get(node, 0),
                    'betweenness': betweenness_centrality.get(node, 0),
                    'eigenvector': eigenvector_centrality.get(node, 0),
                    'combined': (degree_centrality.get(node, 0) + 
                                 betweenness_centrality.get(node, 0) + 
                                 eigenvector_centrality.get(node, 0)) / 3
                }
            
            # Sort nodes by combined centrality
            sorted_nodes = sorted(centrality_scores.items(), 
                                 key=lambda x: x[1]['combined'], 
                                 reverse=True)
            
            # Get top N central narratives
            central_narratives = []
            for node_id, scores in sorted_nodes[:top_n]:
                narrative = DetectedNarrative.query.get(node_id)
                if narrative:
                    central_narratives.append({
                        'id': narrative.id,
                        'title': narrative.title,
                        'status': narrative.status,
                        'confidence': float(narrative.confidence_score or 0),
                        'first_detected': narrative.first_detected.isoformat(),
                        'centrality': scores
                    })
            
            return central_narratives
            
        except Exception as e:
            logger.error(f"Error getting central narratives: {e}")
            return []
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Calculate various statistics about the narrative network.
        
        Returns:
            Dictionary containing network statistics
        """
        try:
            # Ensure we have a built graph
            if self.graph.number_of_nodes() == 0:
                self.build_narrative_network()
            
            # Basic statistics
            nodes = self.graph.number_of_nodes()
            edges = self.graph.number_of_edges()
            
            # Density
            density = nx.density(self.graph)
            
            # Degree distribution
            in_degrees = [d for n, d in self.graph.in_degree()]
            out_degrees = [d for n, d in self.graph.out_degree()]
            
            # Connected components (treats graph as undirected)
            undirected = self.graph.to_undirected()
            connected_components = list(nx.connected_components(undirected))
            
            # Component sizes
            component_sizes = [len(c) for c in connected_components]
            largest_component_size = max(component_sizes) if component_sizes else 0
            
            # Return statistics
            return {
                'node_count': nodes,
                'edge_count': edges,
                'density': density,
                'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
                'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
                'max_in_degree': max(in_degrees) if in_degrees else 0,
                'max_out_degree': max(out_degrees) if out_degrees else 0,
                'component_count': len(connected_components),
                'largest_component_size': largest_component_size,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating network statistics: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def export_network_json(self) -> Dict[str, Any]:
        """
        Export the narrative network as a JSON object for visualization.
        
        Returns:
            JSON-serializable dictionary with nodes and edges
        """
        try:
            # Ensure we have a built graph
            if self.graph.number_of_nodes() == 0:
                self.build_narrative_network()
            
            # Format nodes
            nodes = []
            for node_id, data in self.graph.nodes(data=True):
                node = {
                    'id': node_id,
                    'title': data.get('title', f"Narrative {node_id}"),
                    'status': data.get('status', 'unknown'),
                    'confidence': data.get('confidence', 0)
                }
                nodes.append(node)
            
            # Format edges
            edges = []
            for source, target, data in self.graph.edges(data=True):
                edge = {
                    'source': source,
                    'target': target,
                    'weight': data.get('weight', 1.0),
                    'type': data.get('type', 'unknown')
                }
                edges.append(edge)
            
            # Combine into network object
            network = {
                'nodes': nodes,
                'edges': edges,
                'generated_at': datetime.now().isoformat()
            }
            
            return network
            
        except Exception as e:
            logger.error(f"Error exporting network JSON: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }