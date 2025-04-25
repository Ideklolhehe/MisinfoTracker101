import logging
import networkx as nx
import json
import os
from typing import Dict, Any, Optional, List, Tuple
import time
from datetime import datetime

# Import application components
from models import BeliefNode, BeliefEdge, SystemLog
from app import db

logger = logging.getLogger(__name__)

class GraphStore:
    """Manages storage and operations on the belief graph."""
    
    def __init__(self, export_path: str = None):
        """Initialize the graph store.
        
        Args:
            export_path: Path for exporting graph files
        """
        self.export_path = export_path or os.environ.get('GRAPH_EXPORT_PATH', './graph_exports')
        
        # Create export directory if it doesn't exist
        os.makedirs(self.export_path, exist_ok=True)
        
        logger.info(f"GraphStore initialized with export path: {self.export_path}")
    
    def get_graph(self) -> nx.DiGraph:
        """Get the current belief graph as a NetworkX graph.
        
        Returns:
            graph: NetworkX DiGraph object
        """
        try:
            # Get all nodes and edges from database
            nodes = BeliefNode.query.all()
            edges = BeliefEdge.query.all()
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes with attributes
            for node in nodes:
                metadata = {}
                if node.meta_data:
                    try:
                        metadata = json.loads(node.meta_data)
                    except json.JSONDecodeError:
                        pass
                
                G.add_node(
                    node.id,
                    content=node.content,
                    type=node.node_type,
                    created_at=node.created_at.isoformat(),
                    **metadata
                )
            
            # Add edges with attributes
            for edge in edges:
                G.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relation=edge.relation_type,
                    weight=edge.weight,
                    created_at=edge.created_at.isoformat()
                )
            
            logger.debug(f"Retrieved belief graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error retrieving belief graph: {e}")
            # Return empty graph on error
            return nx.DiGraph()
    
    def add_node(self, content: str, node_type: str, metadata: Dict[str, Any] = None) -> Optional[int]:
        """Add a node to the belief graph.
        
        Args:
            content: Node content text
            node_type: Type of node
            metadata: Optional metadata dictionary
            
        Returns:
            node_id: ID of the created node, or None on error
        """
        try:
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Create new node
            with db.session.begin():
                node = BeliefNode(
                    content=content,
                    node_type=node_type,
                    meta_data=metadata_json,
                    created_at=datetime.utcnow()
                )
                db.session.add(node)
                db.session.flush()
                
                node_id = node.id
            
            logger.debug(f"Added node to belief graph: {node_id} ({node_type})")
            return node_id
            
        except Exception as e:
            logger.error(f"Error adding node to belief graph: {e}")
            self._log_error("add_node", str(e))
            return None
    
    def add_edge(self, source_id: int, target_id: int, relation_type: str, weight: float = 1.0) -> Optional[int]:
        """Add an edge to the belief graph.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation_type: Type of relation
            weight: Edge weight
            
        Returns:
            edge_id: ID of the created edge, or None on error
        """
        try:
            # Check if nodes exist
            source_node = BeliefNode.query.get(source_id)
            target_node = BeliefNode.query.get(target_id)
            
            if not source_node or not target_node:
                logger.warning(f"Cannot add edge: Node not found (source_id={source_id}, target_id={target_id})")
                return None
            
            # Create new edge
            with db.session.begin():
                edge = BeliefEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    weight=weight,
                    created_at=datetime.utcnow()
                )
                db.session.add(edge)
                db.session.flush()
                
                edge_id = edge.id
            
            logger.debug(f"Added edge to belief graph: {edge_id} ({relation_type})")
            return edge_id
            
        except Exception as e:
            logger.error(f"Error adding edge to belief graph: {e}")
            self._log_error("add_edge", str(e))
            return None
    
    def get_node_connections(self, node_id: int, depth: int = 1) -> Dict[str, Any]:
        """Get a node and its connections up to a certain depth.
        
        Args:
            node_id: ID of the node to start from
            depth: Maximum connection depth
            
        Returns:
            result: Dictionary with node and connection data
        """
        try:
            # Get the node
            node = BeliefNode.query.get(node_id)
            if not node:
                logger.warning(f"Node not found: {node_id}")
                return {'error': 'Node not found'}
            
            # Get the full graph
            G = self.get_graph()
            
            # Check if node exists in graph
            if node_id not in G.nodes:
                logger.warning(f"Node {node_id} not in graph")
                return {
                    'node': {
                        'id': node.id,
                        'content': node.content,
                        'type': node.node_type,
                        'created_at': node.created_at.isoformat()
                    },
                    'connections': []
                }
            
            # Get subgraph within depth
            nodes_within_depth = {node_id}
            
            # For directed graphs, we need to look at both incoming and outgoing edges
            current_nodes = {node_id}
            for _ in range(depth):
                next_nodes = set()
                
                # Add outgoing connections
                for n in current_nodes:
                    next_nodes.update(G.successors(n))
                
                # Add incoming connections
                for n in current_nodes:
                    next_nodes.update(G.predecessors(n))
                
                current_nodes = next_nodes - nodes_within_depth
                nodes_within_depth.update(current_nodes)
                
                if not current_nodes:
                    break
            
            # Get subgraph
            subgraph = G.subgraph(nodes_within_depth)
            
            # Format nodes
            formatted_nodes = []
            for n in subgraph.nodes:
                node_data = G.nodes[n]
                formatted_nodes.append({
                    'id': n,
                    'content': node_data.get('content', ''),
                    'type': node_data.get('type', ''),
                    'created_at': node_data.get('created_at', '')
                })
            
            # Format edges
            formatted_edges = []
            for s, t, data in subgraph.edges(data=True):
                formatted_edges.append({
                    'source': s,
                    'target': t,
                    'relation': data.get('relation', ''),
                    'weight': data.get('weight', 1.0)
                })
            
            return {
                'node': {
                    'id': node.id,
                    'content': node.content,
                    'type': node.node_type,
                    'created_at': node.created_at.isoformat()
                },
                'connections': {
                    'nodes': formatted_nodes,
                    'edges': formatted_edges
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting node connections for {node_id}: {e}")
            return {'error': str(e)}
    
    def export_graph(self, format: str = 'gexf') -> Optional[str]:
        """Export the belief graph to a file.
        
        Args:
            format: Export format ('gexf', 'graphml', or 'json')
            
        Returns:
            filepath: Path to the exported file, or None on error
        """
        try:
            # Get the graph
            G = self.get_graph()
            
            # Create filename with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"belief_graph_{timestamp}.{format}"
            filepath = os.path.join(self.export_path, filename)
            
            # Export based on format
            if format == 'gexf':
                nx.write_gexf(G, filepath)
            elif format == 'graphml':
                nx.write_graphml(G, filepath)
            elif format == 'json':
                data = nx.node_link_data(G)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
            
            logger.info(f"Exported belief graph to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting belief graph: {e}")
            self._log_error("export_graph", str(e))
            return None
    
    def get_path_between(self, source_id: int, target_id: int) -> Dict[str, Any]:
        """Find the shortest path between two nodes in the belief graph.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            result: Dictionary with path data
        """
        try:
            # Get the graph
            G = self.get_graph()
            
            # Check if nodes exist
            if source_id not in G.nodes:
                return {'error': f"Source node {source_id} not found in graph"}
            if target_id not in G.nodes:
                return {'error': f"Target node {target_id} not found in graph"}
            
            # Find shortest path
            try:
                path = nx.shortest_path(G, source=source_id, target=target_id)
            except nx.NetworkXNoPath:
                # Try reversed direction
                try:
                    path = nx.shortest_path(G, source=target_id, target=source_id)
                    path.reverse()  # Reverse to match original direction
                except nx.NetworkXNoPath:
                    return {'path_exists': False, 'message': 'No path exists between nodes'}
            
            # Get path edges
            edges = []
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                edge_data = G.get_edge_data(source, target)
                edges.append({
                    'source': source,
                    'target': target,
                    'relation': edge_data.get('relation', ''),
                    'weight': edge_data.get('weight', 1.0)
                })
            
            # Get path nodes
            nodes = []
            for node_id in path:
                node_data = G.nodes[node_id]
                nodes.append({
                    'id': node_id,
                    'content': node_data.get('content', ''),
                    'type': node_data.get('type', '')
                })
            
            return {
                'path_exists': True,
                'path_length': len(path) - 1,
                'path': {
                    'nodes': nodes,
                    'edges': edges
                }
            }
            
        except Exception as e:
            logger.error(f"Error finding path between nodes {source_id} and {target_id}: {e}")
            return {'error': str(e)}
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="graph_store",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
