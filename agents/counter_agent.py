import logging
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime, timedelta

# Import application components
from utils.text_processor import TextProcessor
from utils.ai_processor import AIProcessor
from utils.app_context import ensure_app_context
from models import DetectedNarrative, CounterMessage, NarrativeInstance, SystemLog
from app import db

logger = logging.getLogger(__name__)

class CounterAgent:
    """Agent responsible for generating counter-messaging for misinformation."""
    
    def __init__(self, text_processor: TextProcessor):
        """Initialize the counter agent.
        
        Args:
            text_processor: Text processing utility
        """
        self.text_processor = text_processor
        self.ai_processor = AIProcessor()
        self.running = False
        self.thread = None
        self.refresh_interval = int(os.environ.get('COUNTER_REFRESH_INTERVAL', 900))  # 15 min default
        
        logger.info("CounterAgent initialized")
    
    def start(self):
        """Start the counter agent in a background thread."""
        if self.running:
            logger.warning("CounterAgent is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_counter_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("CounterAgent started")
        
    def stop(self):
        """Stop the counter agent."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("CounterAgent stopped")
    
    def _run_counter_loop(self):
        """Main counter-messaging loop that runs in background thread."""
        while self.running:
            try:
                # Log start of counter cycle
                logger.debug("Starting counter-messaging cycle")
                
                # Generate counter-messages for emerging threats
                self._generate_counter_messages()
                
                # Wait for next cycle
                logger.debug(f"Counter-messaging cycle complete, sleeping for {self.refresh_interval} seconds")
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in counter-messaging loop: {e}")
                # Log error to database
                self._log_error("counter_loop", str(e))
                time.sleep(60)  # Short sleep on error
    
    @ensure_app_context
    def _generate_counter_messages(self):
        """Generate counter-messages for high-priority narratives."""
        try:
            # Get active narratives from the last day
            cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            recent_narratives = DetectedNarrative.query.filter(
                DetectedNarrative.last_updated >= cutoff_time,
                DetectedNarrative.status == 'active'
            ).all()
            
            # Sort by threat level if available in metadata
            prioritized_narratives = []
            for narrative in recent_narratives:
                threat_level = 0
                try:
                    if narrative.meta_data:
                        metadata = json.loads(narrative.meta_data)
                        threat_level = metadata.get('viral_threat', 0)
                except (json.JSONDecodeError, AttributeError):
                    pass
                
                prioritized_narratives.append((narrative, threat_level))
            
            # Sort by threat level (descending)
            prioritized_narratives.sort(key=lambda x: x[1], reverse=True)
            
            # Process top threats first (limit to 5 per cycle)
            logger.debug(f"Found {len(prioritized_narratives)} narratives, processing top threats")
            
            for narrative, threat_level in prioritized_narratives[:5]:
                # Check if we already have a counter-message
                existing_counter = CounterMessage.query.filter_by(
                    narrative_id=narrative.id, 
                    status='approved'
                ).first()
                
                if existing_counter:
                    logger.debug(f"Counter-message already exists for narrative {narrative.id}")
                    continue
                
                # Generate counter-message for this narrative
                logger.info(f"Generating counter-message for narrative {narrative.id} (threat level: {threat_level})")
                self.generate_counter_message(narrative.id)
                
        except Exception as e:
            logger.error(f"Error generating counter-messages: {e}")
            self._log_error("generate_counter_messages", str(e))
    
    @ensure_app_context
    def generate_counter_message(self, narrative_id: int) -> Dict[str, Any]:
        """Generate a counter-message for a specific narrative.
        
        Args:
            narrative_id: ID of the narrative to counter
            
        Returns:
            result: Dictionary with counter-message results
        """
        try:
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": "Narrative not found"}
            
            # Get the narrative content
            narrative_text = narrative.description or narrative.title
            
            # Gather evidence points from associated instances for better counter-messaging
            evidence_points = []
            instances = NarrativeInstance.query.filter_by(narrative_id=narrative_id).limit(5).all()
            for instance in instances:
                evidence_points.append(instance.content)
            
            # Generate counter-message using AI if available, otherwise fall back to templates
            if self.ai_processor.openai_available or self.ai_processor.anthropic_available:
                try:
                    # Use AI to generate counter-message
                    logger.info(f"Using AI to generate counter-message for narrative {narrative_id}")
                    ai_result = self.ai_processor.generate_counter_message(narrative_text, evidence_points)
                    
                    # Check if we got a valid counter-message
                    if 'counter_message' in ai_result:
                        counter_message = ai_result['counter_message']
                        strategy = ai_result.get('strategy', 'ai_generated')
                        logger.info(f"AI generated counter-message using strategy: {strategy}")
                    else:
                        # Fall back to template if AI result is not as expected
                        logger.warning("AI counter-message generation did not return expected format, falling back to template")
                        counter_message = self._generate_template_counter(narrative_text, narrative.language)
                        strategy = 'factual_correction'
                except Exception as e:
                    logger.error(f"Error during AI counter-message generation: {e}")
                    logger.info("Falling back to template-based counter-message")
                    counter_message = self._generate_template_counter(narrative_text, narrative.language)
                    strategy = 'factual_correction'
            else:
                # Use template-based generation if AI is not available
                logger.debug("Using template-based counter-message generation")
                counter_message = self._generate_template_counter(narrative_text, narrative.language)
                strategy = 'factual_correction'
            
            # Create counter-message in database
            try:
                # Try with more robust transaction handling
                counter = CounterMessage(
                    narrative_id=narrative_id,
                    content=counter_message,
                    strategy=strategy,
                    status='draft',  # Requires human approval
                    created_at=datetime.utcnow()
                )
                
                # First try adding directly without transaction management
                db.session.add(counter)
                # Flush to get the ID (if we're not in a transaction)
                db.session.flush()
                # Try to commit explicitly to ensure the counter message is saved
                db.session.commit()
                counter_id = counter.id
            except Exception as e:
                if 'transaction is already begun' in str(e):
                    # If we're in a transaction, add but don't force flush
                    counter = CounterMessage(
                        narrative_id=narrative_id,
                        content=counter_message,
                        strategy=strategy,
                        status='draft',  # Requires human approval
                        created_at=datetime.utcnow()
                    )
                    db.session.add(counter)
                    try:
                        # Try to force a commit to make sure the counter message is saved
                        db.session.commit()
                        counter_id = counter.id
                        logger.info(f"Committed counter message in existing transaction, ID: {counter_id}")
                    except Exception as commit_error:
                        # If commit fails, we're still in a parent transaction
                        logger.warning(f"Could not commit counter message in existing transaction: {commit_error}")
                        counter_id = None
                        logger.warning(f"Added counter message in existing transaction, ID not yet available")
                else:
                    # Re-raise other errors
                    raise
            
            logger.info(f"Generated counter-message for narrative {narrative_id}: ID {counter_id}")
            
            return {
                "narrative_id": narrative_id,
                "counter_id": counter_id,
                "content": counter_message,
                "status": "draft"
            }
            
        except Exception as e:
            logger.error(f"Error generating counter-message for narrative {narrative_id}: {e}")
            self._log_error("generate_counter_message", f"Error for narrative {narrative_id}: {e}")
            return {"narrative_id": narrative_id, "error": str(e)}
    
    def _generate_template_counter(self, narrative_text: str, language: str = 'en') -> str:
        """Generate a template-based counter-message.
        
        This is a simple template-based approach. In a real system, this would
        use an LLM with RAG to generate more sophisticated counter-messaging.
        """
        # Extract entities and keywords
        entities = self.text_processor.extract_entities(narrative_text, language)
        key_phrases = self.text_processor.extract_key_phrases(narrative_text, 3)
        
        # Get entity texts
        entity_texts = [e['text'] for e in entities]
        
        # Basic templates
        templates = {
            'en': [
                "Fact check: Claims about {entities} are misleading. Research shows {key_phrases} are not supported by evidence.",
                "Verification needed: Information regarding {entities} requires additional context. Experts clarify that {key_phrases} should be carefully assessed.",
                "Context matters: Recent claims about {entities} omit critical information. The full story includes important nuance about {key_phrases}.",
                "Expert analysis: Statements concerning {entities} have been evaluated by specialists. Their research indicates {key_phrases} may be presented without proper context."
            ],
            'es': [
                "Verificación de hechos: Las afirmaciones sobre {entities} son engañosas. La investigación muestra que {key_phrases} no están respaldadas por evidencia.",
                "Se necesita verificación: La información sobre {entities} requiere contexto adicional. Los expertos aclaran que {key_phrases} deben evaluarse cuidadosamente.",
                "El contexto importa: Las afirmaciones recientes sobre {entities} omiten información crítica. La historia completa incluye matices importantes sobre {key_phrases}.",
                "Análisis de expertos: Las declaraciones sobre {entities} han sido evaluadas por especialistas. Su investigación indica que {key_phrases} pueden presentarse sin el contexto adecuado."
            ]
        }
        
        # Default to English if language not supported
        if language not in templates:
            language = 'en'
            
        # Select a template
        import random
        template = random.choice(templates[language])
        
        # Format with extracted information
        entities_str = ", ".join(entity_texts[:3]) if entity_texts else "this topic"
        keyphrases_str = ", ".join(key_phrases) if key_phrases else "these claims"
        
        counter_message = template.format(
            entities=entities_str,
            key_phrases=keyphrases_str
        )
        
        # Add factual closing
        if language == 'en':
            counter_message += "\n\nWe recommend consulting multiple reliable sources before sharing this information."
        else:  # Spanish
            counter_message += "\n\nRecomendamos consultar múltiples fuentes confiables antes de compartir esta información."
        
        return counter_message
    
    @ensure_app_context
    def approve_counter_message(self, counter_id: int, user_id: int) -> Dict[str, Any]:
        """Approve a counter-message for deployment.
        
        Args:
            counter_id: ID of the counter-message to approve
            user_id: ID of the user approving the message
            
        Returns:
            result: Dictionary with approval results
        """
        try:
            counter = CounterMessage.query.get(counter_id)
            if not counter:
                logger.warning(f"Counter-message {counter_id} not found")
                return {"error": "Counter-message not found"}
                
            if counter.status != 'draft':
                return {"error": f"Counter-message has status '{counter.status}', not 'draft'"}
                
            # Update counter-message status
            with db.session.begin():
                counter.status = 'approved'
                counter.approved_by = user_id
                
            logger.info(f"Counter-message {counter_id} approved by user {user_id}")
            
            return {
                "counter_id": counter_id,
                "narrative_id": counter.narrative_id,
                "status": "approved",
                "approved_by": user_id
            }
            
        except Exception as e:
            logger.error(f"Error approving counter-message {counter_id}: {e}")
            self._log_error("approve_counter_message", f"Error for counter {counter_id}: {e}")
            return {"counter_id": counter_id, "error": str(e)}
    
    @ensure_app_context
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="counter_agent",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
