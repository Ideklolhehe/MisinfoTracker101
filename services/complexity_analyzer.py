import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from app import db
from models import DetectedNarrative
from services.external_api_initializer import get_openai_client

logger = logging.getLogger(__name__)

class ComplexityAnalyzer:
    """
    Analyzes the complexity of misinformation narratives using AI.
    
    This service evaluates the complexity of misinformation narratives by analyzing
    their linguistic structure, logical construction, rhetorical techniques, and
    emotional manipulation strategies. Each analysis produces a complexity score
    and detailed observations for each dimension.
    """
    
    def __init__(self):
        """Initialize the complexity analyzer."""
        self.openai_client = None
        try:
            self.openai_client = get_openai_client()
            logger.info("Complexity analyzer initialized with OpenAI client")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    def analyze_narrative(self, narrative_id: int) -> Dict[str, Any]:
        """
        Analyze a specific narrative and update its complexity metadata.
        
        Args:
            narrative_id: The ID of the narrative to analyze
            
        Returns:
            Dict containing analysis results or error information
        """
        try:
            # Ensure OpenAI client is available
            if not self.openai_client:
                try:
                    self.openai_client = get_openai_client()
                except Exception as e:
                    logger.error(f"Cannot analyze narrative: OpenAI client unavailable - {e}")
                    return {"error": "OpenAI API unavailable for complexity analysis"}
            
            # Get the narrative from the database
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.error(f"Narrative with ID {narrative_id} not found")
                return {"error": f"Narrative with ID {narrative_id} not found"}
            
            # Prepare input data for analysis
            input_data = {
                "id": narrative.id,
                "title": narrative.title,
                "description": narrative.description,
                "content_text": narrative.content_text if hasattr(narrative, 'content_text') else "",
                "evidence_points": self._extract_evidence_points(narrative)
            }
            
            # Call OpenAI API for complexity analysis
            analysis_result = self._call_openai_analysis(input_data)
            
            # Update narrative metadata with analysis results
            self._update_narrative_metadata(narrative, analysis_result)
            
            logger.info(f"Successfully analyzed complexity for narrative {narrative_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing narrative {narrative_id}: {e}")
            logger.exception(e)
            return {"error": f"Analysis failed: {str(e)}"}
    
    def batch_analyze_recent_narratives(self, days: int = 7, limit: int = 20) -> Dict[str, Any]:
        """
        Analyze multiple recent narratives in batch.
        
        Args:
            days: Number of days back to look for narratives
            limit: Maximum number of narratives to analyze
            
        Returns:
            Dict with analysis summary
        """
        try:
            # Ensure OpenAI client is available
            if not self.openai_client:
                try:
                    self.openai_client = get_openai_client()
                except Exception as e:
                    logger.error(f"Cannot run batch analysis: OpenAI client unavailable - {e}")
                    return {"error": "OpenAI API unavailable for complexity analysis"}
            
            # Get recent narratives
            cutoff_date = datetime.now() - timedelta(days=days)
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.status == 'active',
                DetectedNarrative.created_at >= cutoff_date
            ).order_by(
                DetectedNarrative.created_at.desc()
            ).limit(limit).all()
            
            if not narratives:
                return {"error": "No recent active narratives found for analysis"}
            
            # Process each narrative
            successful = 0
            failed = 0
            
            for narrative in narratives:
                try:
                    logger.info(f"Batch analyzing narrative {narrative.id}")
                    result = self.analyze_narrative(narrative.id)
                    
                    if "error" in result:
                        logger.warning(f"Failed to analyze narrative {narrative.id}: {result['error']}")
                        failed += 1
                    else:
                        successful += 1
                        
                    # Add a delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in batch analysis for narrative {narrative.id}: {e}")
                    failed += 1
            
            return {
                "total_analyzed": successful + failed,
                "successful": successful,
                "failed": failed,
                "days_back": days,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            logger.exception(e)
            return {"error": f"Batch analysis failed: {str(e)}"}
    
    def _extract_evidence_points(self, narrative: DetectedNarrative) -> List[str]:
        """
        Extract evidence points from a narrative.
        
        Args:
            narrative: The narrative object
            
        Returns:
            List of evidence point strings
        """
        evidence_points = []
        
        # Try to extract evidence from metadata
        if narrative.meta_data:
            try:
                metadata = json.loads(narrative.meta_data)
                if isinstance(metadata, dict):
                    # Check for evidence in metadata
                    if 'evidence' in metadata and isinstance(metadata['evidence'], list):
                        for evidence in metadata['evidence']:
                            if isinstance(evidence, dict) and 'text' in evidence:
                                evidence_points.append(evidence['text'])
                            elif isinstance(evidence, str):
                                evidence_points.append(evidence)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse metadata for narrative {narrative.id}")
        
        # If we couldn't find evidence points, use the description
        if not evidence_points and narrative.description:
            evidence_points.append(narrative.description)
        
        return evidence_points
    
    def _call_openai_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call OpenAI API to analyze narrative complexity.
        
        Args:
            input_data: Narrative data for analysis
            
        Returns:
            Dict containing analysis results
        """
        # Prepare the system prompt for complexity analysis
        system_prompt = """
        You are an expert in analyzing misinformation narratives for their complexity and persuasiveness. 
        Analyze the narrative for the following dimensions:
        
        1. Linguistic Complexity: Assess vocabulary, syntax complexity, specialized terminology, and textual structure.
        2. Logical Structure: Evaluate the coherence, use of logical fallacies, causal reasoning patterns, and argument structure.
        3. Rhetorical Techniques: Identify persuasive devices, framing strategies, metaphors, and narrative patterns.
        4. Emotional Manipulation: Analyze emotional appeals, fear/outrage triggers, and identity-based messaging.
        
        For each dimension, provide:
        - A score from 1-10 (where 10 is extremely complex/sophisticated)
        - Specific observations with examples from the text
        
        Also include an overall complexity score (1-10) and a brief summary of why this narrative is or isn't persuasive.
        If applicable, note potential impact on different audiences.
        
        Format your analysis as JSON with the following structure:
        {
            "overall_complexity_score": number,
            "summary": "Overall analysis of the narrative's complexity and persuasiveness",
            "potential_impact": "Assessment of potential impact on audiences",
            "linguistic_complexity": {
                "score": number,
                "observations": "Analysis of linguistic features",
                "examples": "Specific examples from the text"
            },
            "logical_structure": {
                "score": number,
                "observations": "Analysis of logical construction",
                "examples": "Specific examples from the text"
            },
            "rhetorical_techniques": {
                "score": number,
                "observations": "Analysis of persuasive devices used",
                "examples": "Specific examples from the text"
            },
            "emotional_manipulation": {
                "score": number,
                "observations": "Analysis of emotional appeals",
                "examples": "Specific examples from the text"
            },
            "analyzed_at": timestamp
        }
        """
        
        # Prepare the user message with narrative data
        user_message = f"""
        Analyze this misinformation narrative for its complexity:
        
        Title: {input_data['title']}
        
        Description: {input_data['description']}
        
        Content: {input_data['content_text']}
        
        Evidence Points:
        {self._format_evidence_points(input_data['evidence_points'])}
        
        Analyze the complexity of this narrative by evaluating its linguistic structure, 
        logical construction, rhetorical techniques, and emotional manipulation.
        """
        
        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse analysis result
            analysis_text = response.choices[0].message.content
            analysis_result = json.loads(analysis_text)
            
            # Add timestamp if not present
            if "analyzed_at" not in analysis_result:
                analysis_result["analyzed_at"] = int(time.time())
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def _format_evidence_points(self, evidence_points: List[str]) -> str:
        """
        Format evidence points for inclusion in the API prompt.
        
        Args:
            evidence_points: List of evidence point strings
            
        Returns:
            Formatted string of evidence points
        """
        formatted = ""
        for i, point in enumerate(evidence_points):
            formatted += f"{i+1}. {point}\n"
        return formatted
    
    def _update_narrative_metadata(self, narrative: DetectedNarrative, analysis_result: Dict[str, Any]) -> None:
        """
        Update narrative metadata with complexity analysis results.
        
        Args:
            narrative: The narrative object to update
            analysis_result: Analysis results to store
        """
        try:
            # Get existing metadata or initialize empty dict
            meta_data = {}
            if narrative.meta_data:
                try:
                    meta_data = json.loads(narrative.meta_data)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse existing metadata for narrative {narrative.id}, initializing new metadata")
                    meta_data = {}
            
            # Add complexity analysis to metadata
            meta_data['complexity_analysis'] = analysis_result
            
            # Update narrative metadata
            narrative.meta_data = json.dumps(meta_data)
            narrative.last_updated = datetime.now()
            
            # Save to database
            db.session.commit()
            
            logger.info(f"Updated metadata with complexity analysis for narrative {narrative.id}")
            
        except Exception as e:
            logger.error(f"Error updating narrative metadata: {e}")
            db.session.rollback()
            raise