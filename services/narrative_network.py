"""
Narrative Network Analysis Service for the CIVILIAN system.
Analyzes and visualizes relationships between narratives using graph theory and
advanced streaming clustering algorithms.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import threading
from collections import defaultdict

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from river import cluster

from app import db
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge

logger = logging.getLogger(__name__)

# Thread lock for concurrency safety with streaming clustering
data_lock = threading.Lock()

class NarrativeNetworkAnalyzer:
    """
    Service for analyzing relationships between narratives and generating 
    network visualizations using graph theory and advanced streaming clustering.
    """
    
    def __init__(self):
        """Initialize the narrative network analyzer."""
        self.graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize streaming clustering models
        self.denstream = cluster.DenStream(
            decaying_factor=0.01,  # decay λ
            beta=0.5,              # potential micro-cluster threshold
            mu=2.5,                # minimum weight
            epsilon=0.5,           # radius multiplier
            n_samples_init=20      # initial buffer size
        )
        
        # Initialize CluStream for temporal analysis
        self.clustream = cluster.CluStream(
            n_macro_clusters=5,    # Number of macro clusters
            time_window=1000,      # Time window
            n_micro_clusters=50,   # Number of micro clusters
            max_micro_clusters=100, # Maximum micro clusters
            seed=42                # Random seed for reproducibility
        )
        
        # Narrative cluster mapping
        self.cluster_assignments = {}
        self.clustream_assignments = {}
        self.cluster_centers = {}
        
        # Streaming data
        self.narrative_embeddings = {}
        self.embedding_buffer = []
        
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
    
    def identify_communities_with_louvain(self) -> List[Dict[str, Any]]:
        """
        Use Louvain community detection to identify communities in the narrative network.
        
        Returns:
            List of communities with narrative information
        """
        try:
            # Ensure we have a built graph
            if self.graph.number_of_nodes() == 0:
                self.build_narrative_network()
            
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Run Louvain community detection
            communities = nx.community.louvain_communities(
                undirected_graph, 
                weight='weight',
                resolution=1.0
            )
            
            # Format communities
            formatted_communities = []
            for i, community in enumerate(communities):
                # Get narratives in this community
                narratives = []
                for node_id in community:
                    narrative = DetectedNarrative.query.get(node_id)
                    if narrative:
                        narratives.append({
                            'id': narrative.id,
                            'title': narrative.title,
                            'confidence': float(narrative.confidence_score or 0),
                            'status': narrative.status,
                            'first_detected': narrative.first_detected.isoformat()
                        })
                
                # Calculate internal cohesion
                subgraph = undirected_graph.subgraph(community)
                internal_edges = subgraph.number_of_edges()
                possible_edges = len(community) * (len(community) - 1) / 2
                cohesion = internal_edges / possible_edges if possible_edges > 0 else 0
                
                formatted_communities.append({
                    'id': i + 1,
                    'size': len(community),
                    'cohesion': cohesion,
                    'narrative_ids': list(community),
                    'narratives': narratives
                })
            
            # Sort by size (descending)
            formatted_communities.sort(key=lambda x: x['size'], reverse=True)
            
            return formatted_communities
            
        except Exception as e:
            logger.error(f"Error identifying communities with Louvain: {e}")
            return []
    
    def process_narrative_with_denstream(self, narrative_id: int, embedding: np.ndarray) -> int:
        """
        Process a narrative with DenStream for streaming clustering.
        
        Args:
            narrative_id: ID of the narrative
            embedding: Vector embedding of the narrative content
            
        Returns:
            Assigned cluster ID
        """
        try:
            with data_lock:
                # Store the embedding
                self.narrative_embeddings[narrative_id] = embedding
                
                # Convert embedding to dict format for River
                embedding_dict = {i: float(val) for i, val in enumerate(embedding)}
                
                # Learn from this example
                self.denstream.learn_one(embedding_dict)
                
                # Get current clustering
                cluster_id = self.denstream.predict_one(embedding_dict)
                
                # Store assignment
                self.cluster_assignments[narrative_id] = cluster_id
                
                logger.info(f"Narrative {narrative_id} assigned to stream cluster {cluster_id}")
                return cluster_id
                
        except Exception as e:
            logger.error(f"Error processing narrative with DenStream: {e}")
            return -1
    
    def process_narrative_with_clustream(self, narrative_id: int, embedding: np.ndarray, timestamp: Optional[datetime] = None) -> int:
        """
        Process a narrative with CluStream for temporal clustering.
        
        Args:
            narrative_id: ID of the narrative
            embedding: Vector embedding of the narrative content
            timestamp: Optional timestamp for the narrative (if None, current time is used)
            
        Returns:
            Assigned cluster ID
        """
        try:
            with data_lock:
                # Use current time if timestamp not provided
                if timestamp is None:
                    timestamp = datetime.utcnow()
                
                # Convert embedding to dict format for River
                embedding_dict = {i: float(val) for i, val in enumerate(embedding)}
                
                # Learn from this example with timestamp
                # Note: River's API changed and CluStream doesn't accept 'time' parameter directly
                # We'll store the timestamp separately
                self.clustream.learn_one(embedding_dict)
                
                # Get current clustering
                cluster_id = self.clustream.predict_one(embedding_dict)
                
                # Store assignment
                self.clustream_assignments[narrative_id] = {
                    'cluster_id': cluster_id, 
                    'timestamp': timestamp.isoformat()
                }
                
                logger.info(f"Narrative {narrative_id} assigned to temporal cluster {cluster_id}")
                return cluster_id
                
        except Exception as e:
            logger.error(f"Error processing narrative with CluStream: {e}")
            return -1
    
    def get_denstream_clusters(self) -> Dict[str, Any]:
        """
        Get the current DenStream clustering results.
        
        Returns:
            Dictionary with cluster information
        """
        try:
            with data_lock:
                # Group narratives by cluster
                clusters = defaultdict(list)
                for narrative_id, cluster_id in self.cluster_assignments.items():
                    clusters[cluster_id].append(narrative_id)
                
                # Format clusters
                formatted_clusters = []
                for cluster_id, narrative_ids in clusters.items():
                    # Skip noise cluster (-1)
                    if cluster_id == -1:
                        continue
                    
                    # Get narratives
                    narratives = []
                    for nid in narrative_ids:
                        narrative = DetectedNarrative.query.get(nid)
                        if narrative:
                            narratives.append({
                                'id': narrative.id,
                                'title': narrative.title,
                                'confidence': float(narrative.confidence_score or 0),
                                'status': narrative.status
                            })
                    
                    formatted_clusters.append({
                        'id': cluster_id,
                        'size': len(narrative_ids),
                        'narratives': narratives
                    })
                
                # Get noise points
                noise_points = []
                if -1 in clusters:
                    for nid in clusters[-1]:
                        narrative = DetectedNarrative.query.get(nid)
                        if narrative:
                            noise_points.append({
                                'id': narrative.id,
                                'title': narrative.title
                            })
                
                return {
                    'clusters': formatted_clusters,
                    'noise_points': noise_points,
                    'total_processed': len(self.narrative_embeddings)
                }
                
        except Exception as e:
            logger.error(f"Error getting DenStream clusters: {e}")
            return {'clusters': [], 'noise_points': [], 'total_processed': 0}

    def get_clustream_clusters(self) -> Dict[str, Any]:
        """
        Get the current CluStream clustering results with temporal information.
        
        Returns:
            Dictionary with temporal cluster information
        """
        try:
            with data_lock:
                # Group narratives by cluster
                clusters = defaultdict(list)
                for narrative_id, data in self.clustream_assignments.items():
                    cluster_id = data['cluster_id']
                    clusters[cluster_id].append((narrative_id, data['timestamp']))
                
                # Format clusters with time information
                formatted_clusters = []
                for cluster_id, narrative_data in clusters.items():
                    # Skip noise cluster (-1)
                    if cluster_id == -1:
                        continue
                    
                    # Get narratives with timestamps
                    narratives = []
                    timestamps = []
                    for nid, timestamp in narrative_data:
                        narrative = DetectedNarrative.query.get(nid)
                        if narrative:
                            narratives.append({
                                'id': narrative.id,
                                'title': narrative.title,
                                'confidence': float(narrative.confidence_score or 0),
                                'status': narrative.status,
                                'timestamp': timestamp
                            })
                            timestamps.append(datetime.fromisoformat(timestamp))
                    
                    # Sort narratives by timestamp
                    narratives.sort(key=lambda x: x['timestamp'])
                    
                    # Calculate temporal statistics
                    if timestamps:
                        earliest = min(timestamps)
                        latest = max(timestamps)
                        duration = (latest - earliest).total_seconds()
                        
                        formatted_clusters.append({
                            'id': cluster_id,
                            'size': len(narratives),
                            'narratives': narratives,
                            'earliest': earliest.isoformat(),
                            'latest': latest.isoformat(),
                            'duration_seconds': duration,
                            'duration_hours': duration / 3600,
                            'duration_days': duration / 86400
                        })
                
                # Get noise points
                noise_points = []
                if -1 in clusters:
                    for nid, timestamp in clusters[-1]:
                        narrative = DetectedNarrative.query.get(nid)
                        if narrative:
                            noise_points.append({
                                'id': narrative.id,
                                'title': narrative.title,
                                'timestamp': timestamp
                            })
                
                return {
                    'clusters': formatted_clusters,
                    'noise_points': noise_points,
                    'total_processed': len(self.clustream_assignments),
                    'generated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting CluStream clusters: {e}")
            return {
                'clusters': [], 
                'noise_points': [], 
                'total_processed': 0,
                'generated_at': datetime.utcnow().isoformat()
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
            
            # Detect communities using Louvain
            try:
                communities = list(nx.community.louvain_communities(
                    self.graph.to_undirected(), 
                    weight='weight'
                ))
                
                # Create mapping of node to community
                node_community = {}
                for i, community in enumerate(communities):
                    for node in community:
                        node_community[node] = i
            except Exception as e:
                logger.warning(f"Error detecting communities: {e}")
                node_community = {}
                communities = []
            
            # Format nodes
            nodes = []
            for node_id, data in self.graph.nodes(data=True):
                node = {
                    'id': node_id,
                    'title': data.get('title', f"Narrative {node_id}"),
                    'status': data.get('status', 'unknown'),
                    'confidence': data.get('confidence', 0),
                    'community': node_community.get(node_id, -1)  # Add community information
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
                'communities': [list(c) for c in communities],
                'generated_at': datetime.now().isoformat()
            }
            
            return network
            
        except Exception as e:
            logger.error(f"Error exporting network JSON: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }