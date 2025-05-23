"""
Enhanced analyzer agent for the CIVILIAN multi-agent system.
Responsible for analyzing detected narratives to extract patterns and insights.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from datetime import datetime, timedelta

from utils.text_processor import TextProcessor
from utils.app_context import ensure_app_context
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge
from app import db
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing detected misinformation narratives."""
    
    def __init__(self, text_processor: TextProcessor):
        """Initialize the analyzer agent.
        
        Args:
            text_processor: Text processing utility
        """
        super().__init__('analyzer', 600)  # 10 min refresh interval
        self.text_processor = text_processor
        self.batch_size = 100
        
        logger.info("AnalyzerAgent initialized")
    
    def _process_cycle(self):
        """Process a single analysis cycle."""
        # Analyze recent narratives
        self._analyze_recent_narratives()
        
        # Update belief graph metrics
        self._update_belief_graph_metrics()
    
    @ensure_app_context
    def _analyze_recent_narratives(self):
        """Analyze recent narratives to extract patterns and propagation insights."""
        # Get narratives updated in the last day
        cutoff_time = datetime.utcnow() - timedelta(days=1)
        
        try:
            recent_narratives = DetectedNarrative.query.filter(
                DetectedNarrative.last_updated >= cutoff_time
            ).all()
            
            logger.debug(f"Analyzing {len(recent_narratives)} recent narratives")
            
            for narrative in recent_narratives:
                try:
                    # Process each narrative in a separate transaction
                    result = self.analyze_narrative(narrative.id)
                    # Small delay between narratives to avoid overloading the database
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error analyzing narrative {narrative.id}: {e}")
                    # Continue with the next narrative
                    continue
                
        except Exception as e:
            logger.error(f"Error analyzing recent narratives: {e}")
            self._log_error("analyze_recent_narratives", str(e))
    
    @ensure_app_context
    def analyze_narrative(self, narrative_id: int) -> Dict[str, Any]:
        """Analyze a specific narrative.
        
        Args:
            narrative_id: ID of the narrative to analyze
            
        Returns:
            result: Dictionary with analysis results
        """
        try:
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": "Narrative not found"}
                
            # Get all instances of this narrative
            instances = NarrativeInstance.query.filter_by(narrative_id=narrative_id).all()
            
            if not instances:
                return {"narrative_id": narrative_id, "propagation_score": 0, "velocity": 0}
                
            # Calculate basic metrics
            instance_count = len(instances)
            unique_sources = len(set(i.source_id for i in instances if i.source_id))
            
            # Calculate time-based metrics
            if instance_count > 1:
                times = sorted([i.detected_at for i in instances])
                time_span = (times[-1] - times[0]).total_seconds()
                if time_span > 0:
                    velocity = instance_count / (time_span / 3600)  # Instances per hour
                else:
                    velocity = 0
            else:
                velocity = 0
                
            # Categorize content by performing entity analysis
            all_entities = []
            entity_categories = {}
            
            for instance in instances[:min(20, len(instances))]:  # Analyze up to 20 instances
                entities = self.text_processor.extract_entities(instance.content)
                all_entities.extend(entities)
                
            # Group entities by type
            for entity in all_entities:
                category = entity.get('label', 'MISC')
                if category not in entity_categories:
                    entity_categories[category] = []
                entity_categories[category].append(entity['text'])
            
            # Calculate propagation score - higher when spreading fast across sources
            # Basic formula: combines volume, unique sources, and velocity
            propagation_score = (
                0.4 * min(1.0, instance_count / 20) +  # Volume component (caps at 20 instances)
                0.4 * min(1.0, unique_sources / 5) +   # Source diversity (caps at 5 sources)
                0.2 * min(1.0, velocity / 10)          # Velocity (caps at 10 instances/hour)
            )
            
            # Calculate viral threat level (0-5 scale)
            viral_threat = round(propagation_score * 5)
            
            # Store analysis results in metadata
            metadata = {
                "instance_count": instance_count,
                "unique_sources": unique_sources,
                "velocity": velocity,
                "propagation_score": propagation_score,
                "viral_threat": viral_threat,
                "entity_categories": entity_categories,
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
            # Build a graph representation for this narrative outside any transaction
            graph = self._build_narrative_graph(narrative_id)
            if graph:
                # Calculate centrality metrics
                centrality = nx.degree_centrality(graph)
                betweenness = nx.betweenness_centrality(graph)
                
                # Find key nodes
                key_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Add graph metrics to metadata
                metadata["graph_metrics"] = {
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                    "density": nx.density(graph),
                    "key_nodes": [{"id": n, "centrality": c} for n, c in key_nodes]
                }
            
            # Now update the narrative in a single transaction
            try:
                # Get a fresh instance of the narrative
                db.session.expire_all()  # Clear any stale data
                narrative = DetectedNarrative.query.get(narrative_id)
                
                # Update narrative with analysis results
                if hasattr(narrative, 'meta_data') and narrative.meta_data:
                    try:
                        existing_metadata = json.loads(narrative.meta_data)
                        existing_metadata.update(metadata)
                        narrative.meta_data = json.dumps(existing_metadata)
                    except (json.JSONDecodeError, AttributeError):
                        narrative.meta_data = json.dumps(metadata)
                else:
                    narrative.meta_data = json.dumps(metadata)
                    
                # Commit the changes
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error updating narrative metadata: {e}")
                    
            logger.info(f"Analyzed narrative {narrative_id}: propagation={propagation_score:.2f}, threat={viral_threat}")
            return {
                "narrative_id": narrative_id,
                "title": narrative.title,
                "instance_count": instance_count,
                "unique_sources": unique_sources,
                "velocity": velocity,
                "propagation_score": propagation_score,
                "viral_threat": viral_threat,
                "entity_categories": entity_categories
            }
            
        except Exception as e:
            logger.error(f"Error analyzing narrative {narrative_id}: {e}")
            self._log_error("analyze_narrative", f"Error analyzing narrative {narrative_id}: {e}")
            return {"narrative_id": narrative_id, "error": str(e)}
    
    @ensure_app_context
    def _build_narrative_graph(self, narrative_id: int) -> Optional[nx.Graph]:
        """Build a NetworkX graph for a specific narrative.
        
        This constructs a graph representation of how the narrative is spreading.
        """
        try:
            # Get all instances of this narrative
            instances = NarrativeInstance.query.filter_by(narrative_id=narrative_id).all()
            
            if not instances:
                return None
                
            # Create a graph
            G = nx.Graph()
            
            # Add nodes for all sources
            source_ids = set(i.source_id for i in instances if i.source_id)
            for source_id in source_ids:
                G.add_node(f"source_{source_id}", type="source")
            
            # Add narrative as central node
            G.add_node(f"narrative_{narrative_id}", type="narrative")
            
            # Add edges connecting sources to narrative
            for source_id in source_ids:
                # Count instances from this source
                source_count = sum(1 for i in instances if i.source_id == source_id)
                G.add_edge(f"source_{source_id}", f"narrative_{narrative_id}", weight=source_count)
            
            # Add temporal connections between sources
            # Sources that post about the narrative around the same time may be connected
            source_timings = {}
            for instance in instances:
                if instance.source_id:
                    if instance.source_id not in source_timings:
                        source_timings[instance.source_id] = []
                    source_timings[instance.source_id].append(instance.detected_at)
            
            # Find sources with close timing (within 1 hour)
            sources = list(source_timings.keys())
            for i in range(len(sources)):
                for j in range(i+1, len(sources)):
                    source1, source2 = sources[i], sources[j]
                    # Use earliest timestamp for each source
                    time1 = min(source_timings[source1])
                    time2 = min(source_timings[source2])
                    
                    # Check if within 1 hour
                    if abs((time1 - time2).total_seconds()) < 3600:
                        G.add_edge(f"source_{source1}", f"source_{source2}", type="temporal")
            
            return G
            
        except Exception as e:
            logger.error(f"Error building narrative graph for {narrative_id}: {e}")
            return None
    
    @ensure_app_context
    def _update_belief_graph_metrics(self):
        """Update the global belief graph metrics."""
        try:
            # Get all nodes and edges
            nodes = BeliefNode.query.all()
            edges = BeliefEdge.query.all()
            
            if not nodes or not edges:
                logger.debug("Empty belief graph, skipping metrics update")
                return
                
            # Construct a NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in nodes:
                G.add_node(node.id, type=node.node_type, content=node.content)
                
            # Add edges
            for edge in edges:
                G.add_edge(edge.source_id, edge.target_id, 
                          type=edge.relation_type, weight=edge.weight)
            
            # Calculate graph-level metrics
            metrics = {
                "node_count": len(G.nodes),
                "edge_count": len(G.edges),
                "density": nx.density(G),
                "connected_components": nx.number_strongly_connected_components(G),
                "largest_component_size": len(max(nx.strongly_connected_components(G), key=len))
            }
            
            # Calculate node centrality for main node types
            narrative_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'narrative']
            
            if narrative_nodes:
                # Calculate centrality for narrative nodes
                centrality = nx.degree_centrality(G)
                betweenness = nx.betweenness_centrality(G)
                
                # Get top 5 narrative nodes by centrality
                top_narratives = []
                for node_id in narrative_nodes:
                    node_data = next((n for n in nodes if n.id == node_id), None)
                    if node_data:
                        top_narratives.append({
                            "id": node_id,
                            "content": node_data.content[:100],
                            "centrality": centrality.get(node_id, 0),
                            "betweenness": betweenness.get(node_id, 0)
                        })
                
                # Sort by centrality
                top_narratives.sort(key=lambda x: x['centrality'], reverse=True)
                metrics["top_narratives"] = top_narratives[:5]
            
            # Log the metrics
            logger.info(f"Updated belief graph metrics: {len(G.nodes)} nodes, {len(G.edges)} edges")
            
            # Store metrics in a database log entry
            # Make sure there's no active transaction 
            db.session.rollback()
            
            self._log_info("Updated belief graph metrics", metrics)
                
        except Exception as e:
            logger.error(f"Error updating belief graph metrics: {e}")
            self._log_error("update_belief_graph_metrics", str(e))