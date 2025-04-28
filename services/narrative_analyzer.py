"""
Narrative analyzer service for misinformation detection.
Analyzes and clusters narratives based on content similarity and patterns.
"""

import json
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from app import db
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge

logger = logging.getLogger(__name__)

class NarrativeAnalyzer:
    """Service for clustering, analyzing, and visualizing relationships between narratives."""
    
    def __init__(self, num_clusters: int = 5, tfidf_max_features: int = 1000):
        """
        Initializes the NarrativeAnalyzer with clustering parameters.
        
        Args:
            num_clusters: The number of clusters to form.
            tfidf_max_features: The maximum number of features to use for TF-IDF vectorization.
        """
        self.num_clusters = num_clusters
        self.tfidf_max_features = tfidf_max_features
        self.vectorizer = TfidfVectorizer(max_features=self.tfidf_max_features, stop_words='english')
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        self.clusters = None
        self.tfidf_matrix = None
    
    def cluster_narratives(self, narratives: List[str]) -> Dict[int, List[str]]:
        """
        Clusters narratives based on their textual similarity using TF-IDF and K-means.
        
        Args:
            narratives: A list of narrative texts.
            
        Returns:
            A dictionary where keys are cluster IDs and values are lists of narratives 
            belonging to that cluster.
        """
        try:
            if not narratives:
                raise ValueError("The input list of narratives cannot be empty.")
            
            self.tfidf_matrix = self.vectorizer.fit_transform(narratives)
            self.kmeans.fit(self.tfidf_matrix)
            
            self.clusters = defaultdict(list)
            for i, label in enumerate(self.kmeans.labels_):
                self.clusters[label].append(narratives[i])
            
            return dict(self.clusters)
            
        except ValueError as e:
            logger.error(f"Error during narrative clustering: {e}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred during clustering: {e}")
            return {}
    
    def visualize_narrative_relationships(self, similarity_threshold: float = 0.7) -> Dict:
        """
        Creates a graph representing the relationships between narratives based on cosine similarity.
        
        Args:
            similarity_threshold: The minimum cosine similarity score for two narratives 
                                to be considered related.
            
        Returns:
            A dictionary representing the graph data in a format suitable for visualization.
        """
        try:
            if self.tfidf_matrix is None:
                raise ValueError("Narratives must be clustered first using cluster_narratives().")
            
            similarity_matrix = cosine_similarity(self.tfidf_matrix)
            graph = nx.Graph()
            
            narratives = []
            for cluster_id, cluster_narratives in self.clusters.items():
                narratives.extend(cluster_narratives)
            
            for i in range(len(narratives)):
                graph.add_node(i, label=f"Narrative {i}")
            
            for i in range(len(narratives)):
                for j in range(i + 1, len(narratives)):
                    if similarity_matrix[i][j] > similarity_threshold:
                        graph.add_edge(i, j, weight=similarity_matrix[i][j])
            
            graph_data = nx.node_link_data(graph)
            return graph_data
            
        except ValueError as e:
            logger.error(f"Error during visualization: {e}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred during visualization: {e}")
            return {}
    
    def detect_coordinated_campaigns(
        self, narratives: List[str], metadata: List[Dict]
    ) -> List[Tuple[int, List[Dict]]]:
        """
        Detects coordinated campaigns based on narrative similarity and metadata.
        
        Args:
            narratives: A list of narrative texts.
            metadata: A list of dictionaries containing metadata for each narrative.
            
        Returns:
            A list of tuples, where each tuple contains a cluster ID and a list of 
            metadata dictionaries for narratives in that cluster.
        """
        try:
            if not narratives or not metadata:
                raise ValueError("Narratives and metadata lists cannot be empty.")
            
            if len(narratives) != len(metadata):
                raise ValueError("The number of narratives and metadata entries must be the same.")
            
            clusters = self.cluster_narratives(narratives)
            coordinated_campaigns = []
            
            for cluster_id, cluster_narratives in clusters.items():
                cluster_metadata = []
                for narrative in cluster_narratives:
                    index = narratives.index(narrative)
                    cluster_metadata.append(metadata[index])
                
                coordinated_campaigns.append((cluster_id, cluster_metadata))
            
            return coordinated_campaigns
            
        except ValueError as e:
            logger.error(f"Error during coordinated campaign detection: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during campaign detection: {e}")
            return []
    
    def dimension_level_similarity(
        self, narrative_dicts: List[Dict], dimensions: List[str]
    ) -> Dict[str, float]:
        """
        Analyzes the similarity between narratives across different dimensions.
        
        Args:
            narrative_dicts: A list of narrative dictionaries with dimension keys.
            dimensions: A list of dimensions to analyze.
            
        Returns:
            A dictionary where keys are dimensions and values are similarity scores.
        """
        try:
            if not narrative_dicts or not dimensions:
                raise ValueError("Narratives and dimensions lists cannot be empty.")
            
            if not all(isinstance(narrative, dict) for narrative in narrative_dicts):
                raise ValueError("Narratives must be dictionaries with dimension keys.")
            
            similarity_scores = {}
            
            for dimension in dimensions:
                dimension_values = [narrative.get(dimension) for narrative in narrative_dicts]
                
                if not all(value is not None for value in dimension_values):
                    logger.warning(f"Dimension '{dimension}' is missing in some narratives. Skipping.")
                    continue
                
                # Calculate similarity scores (exact match)
                scores = []
                for i in range(len(narrative_dicts)):
                    for j in range(i + 1, len(narrative_dicts)):
                        if dimension_values[i] == dimension_values[j]:
                            scores.append(1.0)
                        else:
                            scores.append(0.0)
                
                if scores:
                    similarity_scores[dimension] = sum(scores) / len(scores)
                else:
                    similarity_scores[dimension] = 0.0
            
            return similarity_scores
            
        except ValueError as e:
            logger.error(f"Error during dimension-level similarity analysis: {e}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred during dimension analysis: {e}")
            return {}
    
    def temporal_mapping(
        self, narratives: List[str], timestamps: List[str]
    ) -> Dict[int, List[datetime]]:
        """
        Maps the spread of narratives across clusters over time.
        
        Args:
            narratives: A list of narrative texts.
            timestamps: A list of timestamps (ISO 8601 format) for each narrative.
            
        Returns:
            A dictionary where keys are cluster IDs and values are lists of timestamps.
        """
        try:
            if not narratives or not timestamps:
                raise ValueError("Narratives and timestamps lists cannot be empty.")
            
            if len(narratives) != len(timestamps):
                raise ValueError("The number of narratives and timestamps must be the same.")
            
            clusters = self.cluster_narratives(narratives)
            temporal_data = {}
            
            for cluster_id, cluster_narratives in clusters.items():
                temporal_data[cluster_id] = []
                for narrative in cluster_narratives:
                    index = narratives.index(narrative)
                    timestamp_str = timestamps[index]
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        temporal_data[cluster_id].append(timestamp)
                    except ValueError:
                        logger.warning(f"Invalid timestamp format: {timestamp_str}. Skipping.")
            
            return temporal_data
            
        except ValueError as e:
            logger.error(f"Error during temporal mapping: {e}")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred during temporal mapping: {e}")
            return {}
    
    def cluster_system_narratives(self, days: int = 30, min_instances: int = 2) -> Dict[str, Any]:
        """
        Clusters narratives from the CIVILIAN system database.
        
        Args:
            days: Number of days to look back for active narratives.
            min_instances: Minimum number of instances required for a narrative.
            
        Returns:
            Dictionary with clustering results.
        """
        try:
            # Get active narratives with sufficient instances
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.status == 'active',
                DetectedNarrative.last_updated >= cutoff_date
            ).all()
            
            # Filter narratives with sufficient instances
            valid_narratives = []
            narrative_texts = []
            narrative_metadata = []
            
            for narrative in narratives:
                instance_count = NarrativeInstance.query.filter_by(narrative_id=narrative.id).count()
                
                if instance_count >= min_instances:
                    valid_narratives.append(narrative)
                    narrative_texts.append(narrative.description or narrative.title)
                    
                    # Extract metadata
                    meta_data = {}
                    if narrative.meta_data:
                        try:
                            meta_data = json.loads(narrative.meta_data)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    
                    narrative_metadata.append({
                        'id': narrative.id,
                        'title': narrative.title,
                        'last_updated': narrative.last_updated.isoformat() if narrative.last_updated else None,
                        'first_detected': narrative.first_detected.isoformat() if narrative.first_detected else None,
                        'language': narrative.language,
                        'complexity': meta_data.get('complexity_analysis', {}).get('overall_complexity_score', 0)
                    })
            
            if not narrative_texts:
                return {"error": "No valid narratives found for clustering"}
            
            # Cluster the narratives
            clusters = self.cluster_narratives(narrative_texts)
            
            # Map narrative IDs to clusters
            cluster_mapping = {}
            for cluster_id, texts in clusters.items():
                cluster_mapping[cluster_id] = []
                for text in texts:
                    idx = narrative_texts.index(text)
                    cluster_mapping[cluster_id].append(narrative_metadata[idx])
            
            # Generate graph visualization data
            graph_data = self.visualize_narrative_relationships()
            
            # Prepare result
            result = {
                'narratives_count': len(narrative_texts),
                'cluster_count': len(clusters),
                'clusters': cluster_mapping,
                'graph_data': graph_data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error clustering system narratives: {e}")
            return {"error": str(e)}
    
    def create_belief_network(self) -> Dict[str, Any]:
        """
        Creates a belief network from narrative data in the system.
        
        Returns:
            Dictionary with belief network results.
        """
        try:
            # Get all active narratives
            narratives = DetectedNarrative.query.filter_by(status='active').all()
            
            if not narratives:
                return {"error": "No active narratives found"}
            
            # Extract narrative texts and metadata
            narrative_texts = []
            narrative_data = []
            
            for narrative in narratives:
                narrative_texts.append(narrative.description or narrative.title)
                narrative_data.append({
                    'id': narrative.id,
                    'title': narrative.title
                })
            
            # Cluster the narratives
            self.cluster_narratives(narrative_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
            # Create belief nodes and edges
            nodes_created = 0
            edges_created = 0
            
            # Create nodes for each narrative if they don't exist
            for i, narrative in enumerate(narrative_data):
                # Check if a node already exists for this narrative
                existing_node = BeliefNode.query.filter_by(
                    node_type='narrative',
                    content=f"Narrative ID: {narrative['id']}"
                ).first()
                
                if not existing_node:
                    # Create a new node
                    node = BeliefNode(
                        content=f"Narrative ID: {narrative['id']}",
                        node_type='narrative',
                        meta_data=json.dumps({
                            'narrative_id': narrative['id'],
                            'title': narrative['title']
                        })
                    )
                    db.session.add(node)
                    db.session.flush()  # Get the ID without committing
                    nodes_created += 1
            
            # Commit changes
            db.session.commit()
            
            # Create edges between similar narratives
            for i in range(len(narrative_data)):
                for j in range(i + 1, len(narrative_data)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > 0.5:  # Only connect if similarity is significant
                        # Get the nodes
                        source_node = BeliefNode.query.filter_by(
                            node_type='narrative',
                            content=f"Narrative ID: {narrative_data[i]['id']}"
                        ).first()
                        
                        target_node = BeliefNode.query.filter_by(
                            node_type='narrative',
                            content=f"Narrative ID: {narrative_data[j]['id']}"
                        ).first()
                        
                        if source_node and target_node:
                            # Check if edge already exists
                            existing_edge = BeliefEdge.query.filter(
                                ((BeliefEdge.source_id == source_node.id) & (BeliefEdge.target_id == target_node.id)) |
                                ((BeliefEdge.source_id == target_node.id) & (BeliefEdge.target_id == source_node.id))
                            ).first()
                            
                            if not existing_edge:
                                # Create edge
                                edge = BeliefEdge(
                                    source_id=source_node.id,
                                    target_id=target_node.id,
                                    relation_type='similar',
                                    weight=float(similarity),
                                    meta_data=json.dumps({
                                        'similarity_score': float(similarity),
                                        'cluster_relationship': self.kmeans.labels_[i] == self.kmeans.labels_[j]
                                    })
                                )
                                db.session.add(edge)
                                edges_created += 1
            
            # Commit changes
            db.session.commit()
            
            # Build the network graph
            nodes = BeliefNode.query.filter_by(node_type='narrative').all()
            edges = BeliefEdge.query.filter_by(relation_type='similar').all()
            
            graph = nx.Graph()
            
            # Add nodes
            for node in nodes:
                try:
                    meta_data = json.loads(node.meta_data) if node.meta_data else {}
                    title = meta_data.get('title', f"Node {node.id}")
                    graph.add_node(node.id, label=title)
                except:
                    graph.add_node(node.id, label=f"Node {node.id}")
            
            # Add edges
            for edge in edges:
                graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
            
            # Convert to visualization format
            graph_data = nx.node_link_data(graph)
            
            # Prepare result
            result = {
                'belief_network_updated': True,
                'nodes_created': nodes_created,
                'edges_created': edges_created,
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'graph_data': graph_data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating belief network: {e}")
            db.session.rollback()
            return {"error": str(e)}
"""