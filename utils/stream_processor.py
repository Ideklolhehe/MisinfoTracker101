"""
Utility module for processing narrative streams.
Connects the analyzer agents with streaming clustering algorithms.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

from app import db
from models import DetectedNarrative
from services.narrative_network import NarrativeNetworkService as NarrativeNetworkAnalyzer

logger = logging.getLogger(__name__)

# Initialize network analyzer as a module-level variable
# to ensure we use the same instance across the application
network_analyzer = NarrativeNetworkAnalyzer()

# TF-IDF vectorizer for creating embeddings
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# To maintain consistency, we need to initialize the vectorizer with some data
# This will be done when the first narrative is processed
vectorizer_initialized = False

def get_narrative_embedding(narrative_id: int) -> Optional[np.ndarray]:
    """
    Get the embedding for a narrative.
    
    Args:
        narrative_id: ID of the narrative
        
    Returns:
        Embedding vector or None if an error occurs
    """
    global vectorizer_initialized
    
    try:
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            logger.warning(f"Narrative {narrative_id} not found for embedding")
            return None
        
        # Combine title and description for better embedding
        content = f"{narrative.title} {narrative.description or ''}"
        
        # If vectorizer not initialized, initialize with some content
        if not vectorizer_initialized:
            # Get a sample of narratives to initialize the vectorizer
            sample_narratives = DetectedNarrative.query.limit(100).all()
            sample_texts = [f"{n.title} {n.description or ''}" for n in sample_narratives]
            
            if sample_texts:
                vectorizer.fit(sample_texts)
                vectorizer_initialized = True
            else:
                # If no narratives, just fit on this content
                vectorizer.fit([content])
                vectorizer_initialized = True
        
        # Transform the content to get embedding
        embedding = vectorizer.transform([content]).toarray()[0]
        return embedding
        
    except Exception as e:
        logger.error(f"Error getting narrative embedding: {e}")
        return None

def process_narrative_with_denstream(narrative_id: int) -> Dict[str, Any]:
    """
    Process a narrative with the DenStream algorithm.
    
    Args:
        narrative_id: ID of the narrative to process
        
    Returns:
        Dictionary containing the processing results
    """
    try:
        # Get narrative embedding
        embedding = get_narrative_embedding(narrative_id)
        if embedding is None:
            return {"narrative_id": narrative_id, "status": "error", "message": "Failed to create embedding"}
        
        # Process with DenStream
        cluster_id = network_analyzer.process_narrative_with_denstream(narrative_id, embedding)
        
        return {
            "narrative_id": narrative_id,
            "status": "success",
            "cluster_id": cluster_id,
            "is_noise": cluster_id == -1
        }
        
    except Exception as e:
        logger.error(f"Error processing narrative with DenStream: {e}")
        return {"narrative_id": narrative_id, "status": "error", "message": str(e)}

def process_narrative_with_clustream(narrative_id: int, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Process a narrative with the CluStream algorithm for temporal clustering.
    
    Args:
        narrative_id: ID of the narrative to process
        timestamp: Optional timestamp for the narrative (if None, current time is used)
        
    Returns:
        Dictionary containing the processing results
    """
    try:
        # Get narrative embedding
        embedding = get_narrative_embedding(narrative_id)
        if embedding is None:
            return {"narrative_id": narrative_id, "status": "error", "message": "Failed to create embedding"}
        
        # Use provided timestamp or narrative's first_detected timestamp or current time
        if timestamp is None:
            narrative = DetectedNarrative.query.get(narrative_id)
            if narrative and narrative.first_detected:
                timestamp = narrative.first_detected
            else:
                timestamp = datetime.utcnow()
        
        # Process with CluStream
        cluster_id = network_analyzer.process_narrative_with_clustream(narrative_id, embedding, timestamp)
        
        return {
            "narrative_id": narrative_id,
            "status": "success",
            "cluster_id": cluster_id,
            "is_noise": cluster_id == -1,
            "timestamp": timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing narrative with CluStream: {e}")
        return {"narrative_id": narrative_id, "status": "error", "message": str(e)}


def process_narrative_with_secleds(narrative_id: int, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Process a narrative with the SECLEDS algorithm for sequence-based clustering with concept drift adaptation.
    
    Args:
        narrative_id: ID of the narrative to process
        timestamp: Optional timestamp for the narrative (if None, current time is used)
        
    Returns:
        Dictionary containing the processing results
    """
    try:
        # Get narrative embedding
        embedding = get_narrative_embedding(narrative_id)
        if embedding is None:
            return {"narrative_id": narrative_id, "status": "error", "message": "Failed to create embedding"}
        
        # Use provided timestamp or narrative's first_detected timestamp or current time
        if timestamp is None:
            narrative = DetectedNarrative.query.get(narrative_id)
            if narrative and narrative.first_detected:
                timestamp = narrative.first_detected
            else:
                timestamp = datetime.utcnow()
        
        # Process with SECLEDS
        cluster_id, confidence = network_analyzer.process_narrative_with_secleds(narrative_id, embedding, timestamp)
        
        return {
            "narrative_id": narrative_id,
            "status": "success",
            "cluster_id": cluster_id,
            "confidence": float(confidence),
            "is_noise": cluster_id == -1,
            "timestamp": timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing narrative with SECLEDS: {e}")
        return {"narrative_id": narrative_id, "status": "error", "message": str(e)}