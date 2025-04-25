import logging
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime, timedelta

# Import application components
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.ai_processor import AIProcessor
from utils.app_context import ensure_app_context
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge, SystemLog
from app import db

logger = logging.getLogger(__name__)

class DetectorAgent:
    """Agent responsible for detecting misinformation in content."""
    
    def __init__(self, text_processor: TextProcessor, vector_store: VectorStore):
        """Initialize the detector agent.
        
        Args:
            text_processor: Text processing utility
            vector_store: Vector storage for embeddings
        """
        self.text_processor = text_processor
        self.vector_store = vector_store
        self.ai_processor = AIProcessor()
        self.running = False
        self.thread = None
        self.refresh_interval = int(os.environ.get('DETECTOR_REFRESH_INTERVAL', 300))  # 5 min default
        self.detection_threshold = float(os.environ.get('DETECTION_THRESHOLD', 0.75))
        self.similarity_threshold = float(os.environ.get('SIMILARITY_THRESHOLD', 0.85))
        
        # Simple misinformation indicators (used as fallback if AI services unavailable)
        self.misinfo_indicators = [
            "fake news",
            "conspiracy",
            "they don't want you to know",
            "mainstream media won't report",
            "scientists are hiding",
            "government cover-up",
            "what they're not telling you",
            "secret cure",
            "miracle treatment",
            "100% guaranteed",
            "shocking truth",
            "everyone is lying",
            "wake up sheeple",
            "do your own research",
            "they're hiding this from the public"
        ]
        
        # Spanish language indicators
        self.es_misinfo_indicators = [
            "noticias falsas",
            "conspiración",
            "no quieren que sepas",
            "los medios no informan",
            "los científicos ocultan",
            "encubrimiento del gobierno",
            "lo que no te están diciendo",
            "cura secreta",
            "tratamiento milagroso",
            "100% garantizado",
            "verdad impactante",
            "todos están mintiendo",
            "despierta",
            "investiga por ti mismo",
            "lo están ocultando del público"
        ]
        
        logger.info("DetectorAgent initialized")
    
    def start(self):
        """Start the detector agent in a background thread."""
        if self.running:
            logger.warning("DetectorAgent is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_detection_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("DetectorAgent started")
        
    def stop(self):
        """Stop the detector agent."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("DetectorAgent stopped")
    
    def _run_detection_loop(self):
        """Main detection loop that runs in background thread."""
        while self.running:
            try:
                # Log start of detection cycle
                logger.debug("Starting detection cycle")
                
                # Process recent content from database
                self._process_recent_content()
                
                # Wait for next cycle
                logger.debug(f"Detection cycle complete, sleeping for {self.refresh_interval} seconds")
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                # Log error to database
                self._log_error("detection_loop", str(e))
                time.sleep(30)  # Short sleep on error
    
    @ensure_app_context
    def _process_recent_content(self):
        """Process recent content from the database."""
        # Get unprocessed content from last hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        try:
            # Query for recent narrative instances not yet associated with a narrative
            with db.session.begin():
                recent_instances = NarrativeInstance.query.filter(
                    NarrativeInstance.narrative_id.is_(None),
                    NarrativeInstance.detected_at >= cutoff_time
                ).all()
                
                logger.debug(f"Found {len(recent_instances)} unprocessed instances")
                
                for instance in recent_instances:
                    self.process_content(instance.content, instance.id, metadata=instance.meta_data)
                    
        except Exception as e:
            logger.error(f"Error processing recent content: {e}")
            self._log_error("process_recent_content", str(e))
            
    @ensure_app_context
    def process_content(self, content: str, content_id: Optional[str] = None, 
                       source: Optional[str] = None, metadata: Optional[str] = None) -> Dict[str, Any]:
        """Process a piece of content to detect misinformation.
        
        Args:
            content: The text content to analyze
            content_id: Optional ID for the content
            source: Optional source identifier
            metadata: Optional metadata as JSON string
            
        Returns:
            result: Dictionary with detection results
        """
        if not content or len(content.strip()) == 0:
            return {"is_misinformation": False, "confidence": 0.0, "narrative_id": None}
            
        try:
            # Detect language
            lang = self.text_processor.detect_language(content)
            
            # Preprocess content
            preprocessed = self.text_processor.preprocess(content, lang)
            
            # Extract potential claims
            claims = self.text_processor.extract_claim_candidates(content, lang)
            
            # Simple rule-based detection
            detection_score = self._calculate_misinfo_score(content, lang)
            
            # If score exceeds threshold, consider it misinformation
            is_misinformation = detection_score >= self.detection_threshold
            
            result = {
                "is_misinformation": is_misinformation,
                "confidence": detection_score,
                "language": lang,
                "claims": claims,
                "narrative_id": None
            }
            
            # If misinformation detected, add to database and link to narratives
            if is_misinformation and content_id:
                narrative_id = self._link_to_narrative(content, content_id, detection_score, lang, metadata)
                result["narrative_id"] = narrative_id
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            self._log_error("process_content", str(e))
            return {"is_misinformation": False, "confidence": 0.0, "narrative_id": None, "error": str(e)}
    
    def _calculate_misinfo_score(self, content: str, lang: str) -> float:
        """Calculate a misinformation score for the content.
        
        Uses AI services when available, falls back to rule-based approach if not.
        """
        # Check if AI processing is available (either OpenAI or Anthropic)
        if self.ai_processor.openai_available or self.ai_processor.anthropic_available:
            try:
                # Use AI to analyze content
                analysis_result = self.ai_processor.analyze_content(content, analysis_type='misinfo_detection')
                
                # Check if analysis contains expected fields
                if 'misinfo_score' in analysis_result and 'confidence' in analysis_result:
                    # Store indicators and type in the instance metadata if needed later
                    if 'indicators' in analysis_result:
                        self.last_indicators = analysis_result.get('indicators', [])
                    if 'type' in analysis_result:
                        self.last_misinfo_type = analysis_result.get('type', 'Unknown')
                    
                    # Return the AI-generated score
                    logger.info(f"AI analysis detected misinformation score: {analysis_result['misinfo_score']}")
                    return float(analysis_result['misinfo_score'])
                else:
                    logger.warning("AI analysis result did not contain expected fields, falling back to rule-based detection")
            except Exception as e:
                logger.error(f"Error during AI misinformation analysis: {e}")
                logger.info("Falling back to rule-based detection")
        
        # Fallback to rule-based detection if AI fails or is unavailable
        logger.debug("Using rule-based misinformation detection")
        content_lower = content.lower()
        
        # Choose appropriate indicators based on language
        indicators = self.misinfo_indicators
        if lang == 'es':
            indicators = self.es_misinfo_indicators
            
        # Count matches with misinformation indicators
        indicator_count = sum(1 for indicator in indicators if indicator in content_lower)
        
        # Normalize score to 0-1 range
        base_score = min(1.0, indicator_count / 5)  # Max out at 5 indicators
        
        # Adjust score based on other heuristics
        # Length penalty (very short content is less likely to be substantial misinfo)
        length_factor = min(1.0, len(content) / 500)  # Cap at content length of 500
        
        # Claim presence boosts score
        claims = self.text_processor.extract_claim_candidates(content, lang)
        claim_factor = min(1.0, len(claims) / 3)  # Cap at 3 claims
        
        # Combine factors
        combined_score = 0.7 * base_score + 0.15 * length_factor + 0.15 * claim_factor
        
        return combined_score
    
    @ensure_app_context
    def _link_to_narrative(self, content: str, content_id: str, confidence: float, 
                          lang: str, metadata: Optional[str]) -> Optional[int]:
        """Link detected misinformation to existing narratives or create new ones."""
        try:
            # Get the instance from database
            instance = NarrativeInstance.query.get(content_id)
            if not instance:
                logger.warning(f"Instance {content_id} not found in database")
                return None
                
            # Extract key phrases for narrative matching
            key_phrases = self.text_processor.extract_key_phrases(content)
            
            # Get recent narratives for similarity checking
            recent_narratives = DetectedNarrative.query.filter_by(language=lang).order_by(
                DetectedNarrative.last_updated.desc()
            ).limit(10).all()
            
            # Find most similar narrative
            best_match = None
            best_score = 0
            
            for narrative in recent_narratives:
                similarity = self.text_processor.calculate_similarity(content, narrative.description)
                if similarity > self.similarity_threshold and similarity > best_score:
                    best_match = narrative
                    best_score = similarity
            
            with db.session.begin():
                # If no match found, create new narrative
                if not best_match:
                    # Create title from first claim or content subset
                    claims = self.text_processor.extract_claim_candidates(content, lang)
                    title = claims[0][:100] if claims else content[:100]
                    
                    new_narrative = DetectedNarrative(
                        title=title,
                        description=content[:500],  # First 500 chars as description
                        confidence_score=confidence,
                        language=lang,
                        first_detected=datetime.utcnow(),
                        last_updated=datetime.utcnow()
                    )
                    db.session.add(new_narrative)
                    db.session.flush()  # To get the ID
                    
                    # Update instance with narrative ID
                    instance.narrative_id = new_narrative.id
                    
                    # Add to belief graph as a new node
                    belief_node = BeliefNode(
                        content=title,
                        node_type='narrative',
                        meta_data=metadata
                    )
                    db.session.add(belief_node)
                    
                    logger.info(f"Created new narrative: {new_narrative.id} - {title}")
                    return new_narrative.id
                    
                else:
                    # Update existing narrative
                    best_match.last_updated = datetime.utcnow()
                    best_match.confidence_score = max(best_match.confidence_score, confidence)
                    
                    # Update instance with narrative ID
                    instance.narrative_id = best_match.id
                    
                    # Add to belief graph - connect to existing narrative node
                    # First, find node for the narrative
                    narrative_node = BeliefNode.query.filter_by(
                        content=best_match.title, 
                        node_type='narrative'
                    ).first()
                    
                    if not narrative_node:
                        # Create node if doesn't exist
                        narrative_node = BeliefNode(
                            content=best_match.title,
                            node_type='narrative'
                        )
                        db.session.add(narrative_node)
                        db.session.flush()
                    
                    # Create node for this instance
                    instance_node = BeliefNode(
                        content=content[:200],
                        node_type='instance',
                        meta_data=metadata
                    )
                    db.session.add(instance_node)
                    db.session.flush()
                    
                    # Create edge connecting instance to narrative
                    edge = BeliefEdge(
                        source_id=instance_node.id,
                        target_id=narrative_node.id,
                        relation_type='supports',
                        weight=best_score
                    )
                    db.session.add(edge)
                    
                    logger.info(f"Linked to existing narrative: {best_match.id} - {best_match.title}")
                    return best_match.id
                    
        except Exception as e:
            logger.error(f"Error linking to narrative: {e}")
            self._log_error("link_to_narrative", str(e))
            return None
    
    @ensure_app_context
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="detector_agent",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
