"""
Enhanced Clustering Service for the CIVILIAN system.

This service provides an interface to the enhanced clustering algorithms:
1. EnhancedDenStream - Improved outlier handling and dynamic time-decaying
2. EnhancedCluStream - Real-time macro-clustering and temporal evolution tracking
3. EnhancedSECLEDS - Adaptive training and novel narrative detection
4. Cross-Algorithm Collaboration - Ensemble approach combining all algorithms

This service integrates with the existing CIVILIAN architecture.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from datetime import datetime
import time
import json
import threading

from app import db
from models import DetectedNarrative, NarrativeInstance
from utils.enhanced_denstream import EnhancedDenStream
from utils.enhanced_clustream import EnhancedCluStream
from utils.enhanced_secleds import EnhancedSECLEDS
from utils.cross_algorithm_collaboration import CrossAlgorithmCollaborator

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedClusteringService:
    """
    Service for enhanced narrative clustering with advanced algorithms.
    """
    
    def __init__(self):
        """Initialize the enhanced clustering service."""
        # Initialize cross-algorithm collaborator
        self.collaborator = CrossAlgorithmCollaborator(
            denstream_params={
                'epsilon': 0.3,
                'beta': 0.5,
                'mu': 2.5,
                'base_lambda': 0.01,
                'outlier_reevaluation_interval': 50
            },
            clustream_params={
                'max_micro_clusters': 100,
                'time_window': 1.0,  # 1 day
                'epsilon': 0.3,
                'reassignment_interval': 50
            },
            secleds_params={
                'k': 10,
                'p': 3,
                'distance_threshold': 0.3,
                'decay': 0.01,
                'novelty_detection': True
            }
        )
        
        # Cache for narrative embeddings
        self.embedding_cache = {}
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("EnhancedClusteringService initialized")
        
    def process_narrative(self, narrative_id: int) -> Dict[str, Any]:
        """
        Process a narrative through the enhanced clustering algorithms.
        
        Args:
            narrative_id: ID of the narrative to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Get narrative from database
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                return {"error": f"Narrative with ID {narrative_id} not found"}
                
            # Get or compute embedding
            embedding = self._get_narrative_embedding(narrative_id)
            if embedding is None:
                return {"error": f"Could not get embedding for narrative {narrative_id}"}
                
            # Get metadata
            metadata = {}
            if narrative.meta_data:
                try:
                    metadata = json.loads(narrative.meta_data)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                    
            # Add propagation score from metadata if available
            propagation_score = 0.0
            if 'propagation_analysis' in metadata:
                propagation_score = metadata.get('propagation_analysis', {}).get('score', 0.0)
                
            # Process through the collaborator
            result = self.collaborator.process_narrative(
                vector=embedding,
                narrative_id=str(narrative_id),
                timestamp=narrative.last_updated or datetime.utcnow(),
                metadata={"propagation": propagation_score}
            )
            
            # Update narrative metadata with clustering results
            self._update_narrative_metadata(narrative_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing narrative {narrative_id}: {e}")
            return {"error": str(e)}
            
    def _get_narrative_embedding(self, narrative_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a narrative.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Numpy array with embedding or None if not found
        """
        with self.lock:
            # Check cache first
            if narrative_id in self.embedding_cache:
                return self.embedding_cache[narrative_id]
                
            # Try to get from database
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                return None
                
            # Check if embedding exists in metadata
            embedding = None
            if narrative.meta_data:
                try:
                    metadata = json.loads(narrative.meta_data)
                    if 'embedding' in metadata:
                        embedding = np.array(metadata['embedding'])
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                    
            # If no embedding, generate one
            if embedding is None:
                # Get text content
                text = narrative.description or narrative.title
                if not text:
                    return None
                    
                # Generate embedding
                from services.embedding import get_embedding
                embedding = get_embedding(text)
                
                # Cache the result
                if embedding is not None:
                    self.embedding_cache[narrative_id] = embedding
                    
                    # Store in metadata
                    self._update_embedding_in_metadata(narrative_id, embedding)
                    
            return embedding
            
    def _update_embedding_in_metadata(self, narrative_id: int, embedding: np.ndarray) -> None:
        """
        Update the embedding in narrative metadata.
        
        Args:
            narrative_id: ID of the narrative
            embedding: Embedding vector
        """
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return
            
        # Get existing metadata
        metadata = {}
        if narrative.meta_data:
            try:
                metadata = json.loads(narrative.meta_data)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
                
        # Update embedding
        metadata['embedding'] = embedding.tolist()
        
        # Save back to database
        narrative.meta_data = json.dumps(metadata)
        db.session.commit()
        
    def _update_narrative_metadata(self, narrative_id: int, result: Dict[str, Any]) -> None:
        """
        Update narrative metadata with clustering results.
        
        Args:
            narrative_id: ID of the narrative
            result: Clustering result dictionary
        """
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return
            
        # Get existing metadata
        metadata = {}
        if narrative.meta_data:
            try:
                metadata = json.loads(narrative.meta_data)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
                
        # Update with clustering results
        if 'cluster_analysis' not in metadata:
            metadata['cluster_analysis'] = {}
            
        metadata['cluster_analysis']['enhanced'] = {
            'ensemble_cluster': result.get('ensemble_cluster'),
            'confidence': result.get('confidence'),
            'algorithm_clusters': result.get('algorithm_clusters'),
            'is_novel': result.get('is_novel'),
            'is_outlier': result.get('is_outlier'),
            'escalation_status': result.get('escalation_status'),
            'timestamp': result.get('timestamp'),
            'threat_level': result.get('threat_level')
        }
        
        # Update propagation score if needed
        if 'propagation_analysis' not in metadata:
            metadata['propagation_analysis'] = {}
            
        metadata['propagation_analysis']['score'] = result.get('propagation_score', 0.0)
        
        # Save back to database
        narrative.meta_data = json.dumps(metadata)
        db.session.commit()
        
    def get_cluster_overview(self) -> Dict[str, Any]:
        """
        Get an overview of all narrative clusters.
        
        Returns:
            Dictionary with cluster overview data
        """
        return self.collaborator.get_cluster_overview()
        
    def get_temporal_alerts(self) -> List[Dict[str, Any]]:
        """
        Get temporal alerts based on significant changes in clusters.
        
        Returns:
            List of alert dictionaries
        """
        return self.collaborator.get_temporal_alerts()
        
    def get_narrative_relationship_mapping(self) -> Dict[str, Any]:
        """
        Get a graph of relationships between narratives and clusters.
        
        Returns:
            Dictionary with graph data for visualization
        """
        return self.collaborator.get_narrative_relationship_mapping()
        
    def get_narrative_analysis(self, narrative_id: int) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific narrative.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Dictionary with analysis results
        """
        return self.collaborator.get_narrative_analysis(str(narrative_id))
        
    def add_feedback(self, narrative_id: int, cluster_id: int, 
                    feedback_score: float, source: str = "analyst") -> None:
        """
        Add analyst feedback about a clustering decision.
        
        Args:
            narrative_id: ID of the narrative
            cluster_id: Assigned cluster ID
            feedback_score: Feedback score (0-1, higher is better)
            source: Source of the feedback
        """
        self.collaborator.secleds.add_feedback(
            narrative_id=str(narrative_id),
            cluster_id=cluster_id,
            feedback_score=feedback_score,
            source=source
        )
        
    def process_all_narratives(self, days: int = 30) -> Dict[str, Any]:
        """
        Process all active narratives from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Get active narratives from the last N days
            from datetime import datetime, timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.status == 'active',
                DetectedNarrative.last_updated >= cutoff_date
            ).all()
            
            # Process each narrative
            processed = 0
            skipped = 0
            errors = 0
            
            for narrative in narratives:
                try:
                    result = self.process_narrative(narrative.id)
                    if 'error' in result:
                        errors += 1
                    else:
                        processed += 1
                except Exception as e:
                    logger.error(f"Error processing narrative {narrative.id}: {e}")
                    errors += 1
                    
            return {
                "success": True,
                "total_narratives": len(narratives),
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
                "cluster_overview": self.get_cluster_overview()
            }
            
        except Exception as e:
            logger.error(f"Error in process_all_narratives: {e}")
            return {"error": str(e)}
            
    def process_recent_narratives(self, limit: int = 100) -> Dict[str, Any]:
        """
        Process the most recent narratives.
        
        Args:
            limit: Maximum number of narratives to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Get the most recent narratives
            narratives = DetectedNarrative.query.order_by(
                DetectedNarrative.last_updated.desc()
            ).limit(limit).all()
            
            # Process each narrative
            processed = 0
            errors = 0
            
            for narrative in narratives:
                try:
                    result = self.process_narrative(narrative.id)
                    if 'error' in result:
                        errors += 1
                    else:
                        processed += 1
                except Exception as e:
                    logger.error(f"Error processing narrative {narrative.id}: {e}")
                    errors += 1
                    
            return {
                "success": True,
                "total_narratives": len(narratives),
                "processed": processed,
                "errors": errors,
                "cluster_overview": self.get_cluster_overview()
            }
            
        except Exception as e:
            logger.error(f"Error in process_recent_narratives: {e}")
            return {"error": str(e)}

# Global instance
enhanced_clustering_service = EnhancedClusteringService()