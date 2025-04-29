"""
Service for counter-narrative generation, strategy evaluation, and effectiveness tracking.
"""

import logging
import json
import os
import time
import threading
import uuid
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import networkx as nx
from transformers import pipeline
from sqlalchemy import desc, func

from app import db
from models import DetectedNarrative, NarrativeInstance, CounterMessage, DataSource
from utils.app_context import with_app_context
from utils.metrics import time_operation, Counter, Gauge
from utils.concurrency import RWLock, ThreadSafeDict

# Configure logger
logger = logging.getLogger(__name__)

# Prometheus metrics
counter_gen_counter = Counter('counter_narratives_generated_total', 'Total counter-narratives generated')
counter_eff_gauge = Gauge('counter_narrative_effectiveness', 'Counter-narrative effectiveness score')
narrative_threat_gauge = Gauge('narrative_threat_level', 'Narrative threat level by ID')

# Lock for thread safety
lock = RWLock()
narratives_cache = ThreadSafeDict()

class CounterNarrativeService:
    """Service for counter-narrative generation and tracking."""
    
    def __init__(self):
        """Initialize the counter-narrative service."""
        self.transformers_model = os.environ.get('COUNTER_NARRATIVE_MODEL', 'facebook/bart-large-mnli')
        self.generator = None
        self.init_generator()
        logger.info(f"Counter-narrative service initialized with model: {self.transformers_model}")
    
    def init_generator(self):
        """Initialize the text generation pipeline."""
        try:
            # Only load the model if it hasn't been loaded yet
            if self.generator is None:
                # Load in a separate thread to avoid blocking
                def load_model():
                    try:
                        self.generator = pipeline('text2text-generation', model=self.transformers_model)
                        logger.info(f"Successfully loaded counter-narrative model: {self.transformers_model}")
                    except Exception as e:
                        logger.error(f"Error loading counter-narrative model: {e}")
                
                thread = threading.Thread(target=load_model)
                thread.daemon = True
                thread.start()
        except Exception as e:
            logger.error(f"Error initializing counter-narrative generator: {e}")
    
    def get_dimension_strategies(self, dimension: str) -> List[str]:
        """Get dimension-specific counter-narrative strategies."""
        # Strategy mapping by dimension
        mapping = {
            "emotional": [
                "Evoke empathy with personal stories",
                "Highlight hopeful outcomes",
                "Use metaphorical language"
            ],
            "cognitive": [
                "Present clear factual corrections",
                "Use data visualizations",
                "Cite authoritative sources"
            ],
            "inoculation": [
                "Pre-empt myths with weakened refutations",
                "Explain manipulation techniques",
                "Offer refutation templates"
            ],
        }
        return mapping.get(dimension, ["General factual rebuttal", "Edge-case inoculation"])
    
    @with_app_context
    def prioritize_clusters(self, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Prioritize narrative clusters for counter-narrative deployment."""
        # Get top threat narratives
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.status != 'debunked'
        ).order_by(
            DetectedNarrative.last_updated.desc()
        ).limit(100).all()
        
        # Calculate threat scores and group by cluster
        clusters = defaultdict(list)
        for narrative in narratives:
            meta = narrative.get_meta_data()
            # Use stream cluster (real-time) as the primary grouping
            cluster_id = meta.get('stream_cluster', 0)
            threat_score = meta.get('threat_score', 0)
            # Update threat gauge
            narrative_threat_gauge.labels(narrative_id=narrative.id).set(threat_score)
            # Add to cluster group
            clusters[cluster_id].append({
                'id': narrative.id,
                'title': narrative.title,
                'threat_score': threat_score,
                'complexity_score': meta.get('complexity_score', 0),
                'propagation_score': meta.get('propagation_score', 0)
            })
        
        # Calculate cluster metrics
        cluster_metrics = []
        for cluster_id, narratives in clusters.items():
            # Calculate average complexity and threat
            avg_threat = sum(n['threat_score'] for n in narratives) / len(narratives) if narratives else 0
            avg_complexity = sum(n['complexity_score'] for n in narratives) / len(narratives) if narratives else 0
            # Calculate the number of high-threat narratives
            high_threat_count = sum(1 for n in narratives if n['threat_score'] >= threshold)
            
            cluster_metrics.append({
                'cluster_id': cluster_id,
                'narrative_count': len(narratives),
                'avg_threat': avg_threat,
                'avg_complexity': avg_complexity,
                'high_threat_count': high_threat_count,
                'narratives': sorted(narratives, key=lambda x: x['threat_score'], reverse=True)
            })
        
        # Sort clusters by average threat score
        return sorted(cluster_metrics, key=lambda x: x['avg_threat'], reverse=True)
    
    @with_app_context
    def generate_counter_narrative(self, narrative_text: str, dimension: str = 'cognitive', strategy: str = 'factual') -> str:
        """Generate a counter-narrative for a given narrative."""
        start_time = time.time()
        
        try:
            # If the generator isn't ready, use a fallback
            if self.generator is None:
                logger.warning("Counter-narrative generator not ready, using fallback")
                return self._fallback_counter_narrative(narrative_text, dimension, strategy)
            
            # Apply dimension-specific framing
            if dimension == 'emotional':
                prompt = f"Generate an empathetic counter-narrative that appeals to shared values while refuting: {narrative_text}"
            elif dimension == 'inoculation':
                prompt = f"Create a pre-emptive counter-narrative that explains manipulation techniques used in: {narrative_text}"
            else:  # cognitive/factual default
                prompt = f"Generate a fact-based counter-narrative with authoritative sources to refute: {narrative_text}"
            
            # Generate counter-narrative
            result = self.generator(prompt, max_length=250, num_return_sequences=1)
            counter_text = result[0]['generated_text']
            
            # Apply strategy-specific formatting
            if strategy == 'story':
                counter_text = f"Here's what actually happened: {counter_text}"
            elif strategy == 'question':
                counter_text = f"Consider this: {counter_text}"
            
            # Record metric
            counter_gen_counter.inc()
            
            # Log generation time
            elapsed = time.time() - start_time
            logger.info(f"Generated counter-narrative in {elapsed:.2f}s")
            
            return counter_text
        
        except Exception as e:
            logger.error(f"Error generating counter-narrative: {e}")
            return self._fallback_counter_narrative(narrative_text, dimension, strategy)
    
    def _fallback_counter_narrative(self, narrative_text: str, dimension: str, strategy: str) -> str:
        """Fallback method when the ML model is unavailable."""
        # Create a deterministic but unique ID for this narrative
        narrative_hash = hash(narrative_text) % 10000
        
        templates = [
            "Evidence contradicts this claim. Reliable sources indicate the opposite is true.",
            "This appears to be misleading. The full context provides a different perspective.",
            "Fact-checkers have disproven this narrative. Here's what verified sources say.",
            "This narrative omits critical context. Consider these additional verified facts.",
            "Research shows this claim is inaccurate. Here's what studies actually demonstrate."
        ]
        
        # Use the hash to select a template deterministically
        template = templates[narrative_hash % len(templates)]
        
        if dimension == 'emotional':
            prefix = "It's understandable to be concerned about this, but "
        elif dimension == 'inoculation':
            prefix = "Be aware of this manipulation technique: "
        else:
            prefix = "The facts show that "
        
        return f"{prefix}{template}"
    
    @with_app_context
    def optimize_sources(self, narrative_id: Optional[int] = None, edges: Optional[List[List[str]]] = None) -> List[Dict[str, Any]]:
        """Optimize sources for counter-narrative targeting."""
        # Create network graph
        G = nx.DiGraph()
        
        if edges:
            # Use provided edges
            for edge in edges:
                if len(edge) >= 2:
                    G.add_edge(edge[0], edge[1])
        elif narrative_id:
            # Build graph from narrative instances
            instances = NarrativeInstance.query.filter_by(narrative_id=narrative_id).all()
            
            for instance in instances:
                meta = instance.get_meta_data()
                # Use source ID as the node
                source_id = instance.source_id
                if source_id:
                    # Add node with attributes
                    G.add_node(str(source_id), type='source')
                    # Connect to other sources if we have that metadata
                    references = meta.get('references', [])
                    for ref in references:
                        G.add_edge(str(source_id), str(ref))
        else:
            # Use general source relationships from database
            sources = DataSource.query.filter_by(is_active=True).all()
            for source in sources:
                G.add_node(str(source.id), type='source', name=source.name)
                
                # Add relationships based on metadata
                meta = source.get_meta_data()
                related = meta.get('related_sources', [])
                for related_id in related:
                    G.add_edge(str(source.id), str(related_id))
        
        # Calculate centrality if we have nodes
        if len(G.nodes) > 0:
            try:
                # Calculate betweenness centrality
                centrality = nx.betweenness_centrality(G)
                # Calculate eigenvector centrality
                eigen_centrality = nx.eigenvector_centrality_numpy(G, max_iter=1000)
                
                # Combine centrality scores
                combined_scores = {}
                for node in G.nodes:
                    combined_scores[node] = centrality.get(node, 0) + eigen_centrality.get(node, 0)
                
                # Get top sources by centrality
                top_sources = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Get source details
                result = []
                for source_id, score in top_sources:
                    try:
                        source = DataSource.query.get(int(source_id))
                        if source:
                            result.append({
                                'id': source.id,
                                'name': source.name,
                                'type': source.source_type,
                                'centrality_score': score
                            })
                    except (ValueError, AttributeError):
                        # If we can't convert to int or source doesn't exist
                        result.append({
                            'id': source_id,
                            'centrality_score': score
                        })
                
                return result
            except Exception as e:
                logger.error(f"Error calculating centrality: {e}")
                return []
        
        return []
    
    @with_app_context
    def track_effectiveness(self, counter_id: int, metrics: Dict[str, Any]) -> bool:
        """Track the effectiveness of a counter-message."""
        try:
            # Get counter-message
            counter_message = CounterMessage.query.get(counter_id)
            if not counter_message:
                logger.error(f"Counter message not found: {counter_id}")
                return False
            
            # Update meta_data with new metrics
            meta = counter_message.get_meta_data() or {}
            
            # Merge existing effectiveness data with new metrics
            effectiveness = meta.get('effectiveness', {})
            effectiveness.update(metrics)
            meta['effectiveness'] = effectiveness
            
            # Save updated meta_data
            counter_message.set_meta_data(meta)
            counter_message.last_updated = func.now()
            db.session.commit()
            
            # Update effectiveness gauge
            if 'score' in metrics:
                counter_eff_gauge.labels(counter_id=counter_id).set(metrics['score'])
            
            return True
        
        except Exception as e:
            logger.error(f"Error tracking counter-message effectiveness: {e}")
            return False