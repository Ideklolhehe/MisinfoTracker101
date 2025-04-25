import logging
import time
import threading
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime, timedelta

# Import application components
from utils.text_processor import TextProcessor
from utils.app_context import ensure_app_context
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge, SystemLog
from app import db

logger = logging.getLogger(__name__)

class AnalyzerAgent:
    """Agent responsible for analyzing detected misinformation narratives."""
    
    def __init__(self, text_processor: TextProcessor):
        """Initialize the analyzer agent.
        
        Args:
            text_processor: Text processing utility
        """
        self.text_processor = text_processor
        self.running = False
        self.thread = None
        self.refresh_interval = int(os.environ.get('ANALYZER_REFRESH_INTERVAL', 600))  # 10 min default
        self.batch_size = int(os.environ.get('ANALYZER_BATCH_SIZE', 100))
        
        logger.info("AnalyzerAgent initialized")
    
    def start(self):
        """Start the analyzer agent in a background thread."""
        if self.running:
            logger.warning("AnalyzerAgent is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_analysis_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("AnalyzerAgent started")
        
    def stop(self):
        """Stop the analyzer agent."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("AnalyzerAgent stopped")
    
    def _run_analysis_loop(self):
        """Main analysis loop that runs in background thread."""
        while self.running:
            try:
                # Log start of analysis cycle
                logger.debug("Starting analysis cycle")
                
                # Analyze recent narratives
                self._analyze_recent_narratives()
                
                # Update belief graph metrics
                self._update_belief_graph_metrics()
                
                # Wait for next cycle
                logger.debug(f"Analysis cycle complete, sleeping for {self.refresh_interval} seconds")
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                # Log error to database
                self._log_error("analysis_loop", str(e))
                time.sleep(30)  # Short sleep on error
    
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
                self.analyze_narrative(narrative.id)
                
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
            
            # Update narrative with analysis results
            with db.session.begin():
                # Store analysis results in metadata if not already present
                metadata = {
                    "instance_count": instance_count,
                    "unique_sources": unique_sources,
                    "velocity": velocity,
                    "propagation_score": propagation_score,
                    "viral_threat": viral_threat,
                    "entity_categories": entity_categories,
                    "analyzed_at": datetime.utcnow().isoformat()
                }
                
                # Build a graph representation for this narrative
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
            log_entry = SystemLog(
                log_type="info",
                component="analyzer_agent",
                message="Updated belief graph metrics",
                meta_data=json.dumps(metrics)
            )
            with db.session.begin():
                db.session.add(log_entry)
                
        except Exception as e:
            logger.error(f"Error updating belief graph metrics: {e}")
            self._log_error("update_belief_graph_metrics", str(e))
    
    @ensure_app_context
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="analyzer_agent",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
