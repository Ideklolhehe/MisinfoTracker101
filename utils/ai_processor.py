import logging
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIProcessor:
    """Processor for AI model interactions using OpenAI and Anthropic."""
    
    def __init__(self):
        """Initialize the AI processor."""
        # Check for API keys
        self.openai_available = os.environ.get('OPENAI_API_KEY') is not None
        self.anthropic_available = os.environ.get('ANTHROPIC_API_KEY') is not None
        
        # Initialize clients if available
        self.openai_client = None
        self.anthropic_client = None
        
        if self.openai_available:
            try:
                self.openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                self.openai_available = False
        
        if self.anthropic_available:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}")
                self.anthropic_available = False
        
        # Log availability
        if not self.openai_available and not self.anthropic_available:
            logger.warning("No AI providers available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable AI capabilities.")
        
        logger.info("AIProcessor initialized")
    
    def analyze_content(self, content: str, analysis_type: str = 'misinfo_detection') -> Dict[str, Any]:
        """Analyze content using the best available AI model.
        
        Args:
            content: Text content to analyze
            analysis_type: Type of analysis to perform (misinfo_detection, sentiment, etc.)
            
        Returns:
            result: Dictionary with analysis results
        """
        # Determine which client to use (prefer OpenAI if both available)
        if self.openai_available:
            return self._analyze_with_openai(content, analysis_type)
        elif self.anthropic_available:
            return self._analyze_with_anthropic(content, analysis_type)
        else:
            logger.warning("No AI clients available for content analysis")
            return self._fallback_analysis(content, analysis_type)
    
    def _analyze_with_openai(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze content using OpenAI models.
        
        Args:
            content: Text content to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            result: Dictionary with analysis results
        """
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            if analysis_type == 'misinfo_detection':
                system_prompt = (
                    "You are an expert misinformation analyst tasked with detecting and analyzing "
                    "potential misinformation in content. Provide a structured analysis with: "
                    "1) A misinformation score from 0.0 to 1.0 where 1.0 is definitely misinformation "
                    "2) A confidence score for your analysis from 0.0 to 1.0 "
                    "3) Up to 5 key indicators of misinformation in the content, if any "
                    "4) The apparent type/category of misinformation if present "
                    "5) A summary of the claim or narrative being presented "
                    "Format response as JSON with keys: 'misinfo_score', 'confidence', 'indicators', 'type', 'summary'"
                )
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
            elif analysis_type == 'counter_message':
                system_prompt = (
                    "You are an expert in countering misinformation campaigns. Generate an effective counter-message that: "
                    "1) Is factually accurate "
                    "2) Addresses the key false claims without repeating or reinforcing them "
                    "3) Provides 2-3 credible sources to support the counter-message "
                    "4) Uses neutral, non-polarizing language "
                    "5) Is concise and clear (200 words or less) "
                    "Format response as JSON with keys: 'counter_message', 'sources', 'strategy'"
                )
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
            elif analysis_type == 'narrative_analysis':
                system_prompt = (
                    "You are an expert in narrative analysis and pattern detection. Analyze the provided "
                    "text to extract key narrative patterns, themes, and frames. Provide: "
                    "1) The central narrative or claim being made "
                    "2) 3-5 key rhetorical devices or techniques employed "
                    "3) The apparent target audience "
                    "4) Entities mentioned and their characterization (positive/negative/neutral) "
                    "Format response as JSON with keys: 'central_narrative', 'rhetorical_techniques', 'target_audience', 'entities'"
                )
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
            
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return {"error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            logger.error(f"Error analyzing content with OpenAI: {e}")
            return {"error": str(e)}
    
    def _analyze_with_anthropic(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze content using Anthropic models.
        
        Args:
            content: Text content to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            result: Dictionary with analysis results
        """
        try:
            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            # do not change this unless explicitly requested by the user
            if analysis_type == 'misinfo_detection':
                system_prompt = (
                    "You are an expert misinformation analyst tasked with detecting and analyzing "
                    "potential misinformation in content. Provide a structured analysis with: "
                    "1) A misinformation score from 0.0 to 1.0 where 1.0 is definitely misinformation "
                    "2) A confidence score for your analysis from 0.0 to 1.0 "
                    "3) Up to 5 key indicators of misinformation in the content, if any "
                    "4) The apparent type/category of misinformation if present "
                    "5) A summary of the claim or narrative being presented "
                    "Format response as JSON with keys: 'misinfo_score', 'confidence', 'indicators', 'type', 'summary'"
                )
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    system=system_prompt,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": content}
                    ]
                )
                
                # Extract the JSON response
                text_response = response.content[0].text
                # Clean potential text before/after JSON
                try:
                    # Find JSON pattern
                    json_start = text_response.find('{')
                    json_end = text_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = text_response[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        # Simple approach - just try to parse the whole thing
                        result = json.loads(text_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return a text-based error
                    logger.warning("Failed to parse JSON from Anthropic response, returning raw response")
                    return {
                        "error": "Failed to parse structured response",
                        "raw_response": text_response
                    }
                
                return result
                
            elif analysis_type == 'counter_message':
                system_prompt = (
                    "You are an expert in countering misinformation campaigns. Generate an effective counter-message that: "
                    "1) Is factually accurate "
                    "2) Addresses the key false claims without repeating or reinforcing them "
                    "3) Provides 2-3 credible sources to support the counter-message "
                    "4) Uses neutral, non-polarizing language "
                    "5) Is concise and clear (200 words or less) "
                    "Format response as JSON with keys: 'counter_message', 'sources', 'strategy'"
                )
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    system=system_prompt,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": content}
                    ]
                )
                
                # Extract the JSON response
                text_response = response.content[0].text
                # Clean potential text before/after JSON
                try:
                    # Find JSON pattern
                    json_start = text_response.find('{')
                    json_end = text_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = text_response[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        # Simple approach - just try to parse the whole thing
                        result = json.loads(text_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return a text-based error
                    logger.warning("Failed to parse JSON from Anthropic response, returning raw response")
                    return {
                        "error": "Failed to parse structured response",
                        "raw_response": text_response
                    }
                
                return result
                
            elif analysis_type == 'narrative_analysis':
                system_prompt = (
                    "You are an expert in narrative analysis and pattern detection. Analyze the provided "
                    "text to extract key narrative patterns, themes, and frames. Provide: "
                    "1) The central narrative or claim being made "
                    "2) 3-5 key rhetorical devices or techniques employed "
                    "3) The apparent target audience "
                    "4) Entities mentioned and their characterization (positive/negative/neutral) "
                    "Format response as JSON with keys: 'central_narrative', 'rhetorical_techniques', 'target_audience', 'entities'"
                )
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    system=system_prompt,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": content}
                    ]
                )
                
                # Extract the JSON response
                text_response = response.content[0].text
                # Clean potential text before/after JSON
                try:
                    # Find JSON pattern
                    json_start = text_response.find('{')
                    json_end = text_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = text_response[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        # Simple approach - just try to parse the whole thing
                        result = json.loads(text_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return a text-based error
                    logger.warning("Failed to parse JSON from Anthropic response, returning raw response")
                    return {
                        "error": "Failed to parse structured response",
                        "raw_response": text_response
                    }
                
                return result
            
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return {"error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            logger.error(f"Error analyzing content with Anthropic: {e}")
            return {"error": str(e)}
    
    def _fallback_analysis(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Perform a basic analysis when no AI services are available.
        
        Args:
            content: Text content to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            result: Dictionary with analysis results
        """
        logger.warning("Using fallback analysis - no AI services available")
        
        if analysis_type == 'misinfo_detection':
            # Simple keyword-based approach
            misinfo_indicators = [
                "fake news", "conspiracy", "they don't want you to know",
                "mainstream media won't report", "scientists are hiding",
                "government cover-up", "what they're not telling you",
                "secret cure", "miracle treatment", "100% guaranteed",
                "shocking truth", "everyone is lying", "wake up sheeple",
                "do your own research", "doctors hate this"
            ]
            
            # Count indicators
            found_indicators = []
            for indicator in misinfo_indicators:
                if indicator.lower() in content.lower():
                    found_indicators.append(indicator)
            
            # Calculate simple score based on indicators present
            score = min(1.0, len(found_indicators) / 5) if found_indicators else 0.0
            
            return {
                "misinfo_score": score,
                "confidence": 0.3,  # Low confidence
                "indicators": found_indicators[:5],
                "type": "Unknown - requires AI analysis",
                "summary": "Unable to generate summary without AI service",
                "note": "This is a fallback analysis without AI capabilities"
            }
            
        elif analysis_type == 'counter_message':
            return {
                "counter_message": "To generate an effective counter-message, please enable AI capabilities.",
                "sources": [],
                "strategy": "Unable to generate counter-message without AI service",
                "note": "This is a fallback response without AI capabilities"
            }
            
        elif analysis_type == 'narrative_analysis':
            return {
                "central_narrative": "Unable to extract narrative without AI service",
                "rhetorical_techniques": [],
                "target_audience": "Unknown - requires AI analysis",
                "entities": {},
                "note": "This is a fallback analysis without AI capabilities"
            }
        
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def generate_counter_message(self, narrative: str, evidence: List[str]) -> Dict[str, Any]:
        """Generate a counter-message for a misinformation narrative.
        
        Args:
            narrative: The misinformation narrative
            evidence: List of evidence points to counter the narrative
            
        Returns:
            result: Dictionary with counter-message results
        """
        # Combine input for processing
        prompt_content = f"Misinformation narrative: {narrative}\n\nEvidence points:\n"
        for i, point in enumerate(evidence, 1):
            prompt_content += f"{i}. {point}\n"
        
        return self.analyze_content(prompt_content, analysis_type='counter_message')
    
    def analyze_narrative_pattern(self, narratives: List[str]) -> Dict[str, Any]:
        """Analyze patterns across multiple related narratives.
        
        Args:
            narratives: List of related narrative texts
            
        Returns:
            result: Dictionary with pattern analysis results
        """
        # Combine narratives for processing
        prompt_content = "Analyze the following related misinformation narratives for patterns:\n\n"
        for i, narrative in enumerate(narratives, 1):
            prompt_content += f"Narrative {i}: {narrative}\n\n"
        
        return self.analyze_content(prompt_content, analysis_type='narrative_analysis')