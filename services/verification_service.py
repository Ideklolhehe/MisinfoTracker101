"""
Verification service for analyzing user-submitted content.
This service checks content for misinformation, AI-generated content, and verifies authenticity.
"""

import os
import json
import logging
import base64
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI
from sqlalchemy.exc import SQLAlchemyError

from app import db
from models import UserSubmission, VerificationResult, VerificationType, VerificationStatus

logger = logging.getLogger(__name__)

class VerificationService:
    """Service for verifying user-submitted content."""
    
    def __init__(self):
        """Initialize the verification service."""
        self.openai_client = None
        
        # Try to initialize the OpenAI client
        try:
            self._initialize_openai_client()
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
    
    def _initialize_openai_client(self):
        """Initialize the OpenAI client with API key."""
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    
    def verify_submission(self, submission_id: int) -> Dict[str, Any]:
        """Verify a user submission by running all verification types.
        
        Args:
            submission_id: ID of the submission to verify
            
        Returns:
            result: Dictionary with verification results and status
        """
        try:
            # Get the submission from the database
            submission = UserSubmission.query.get(submission_id)
            if not submission:
                return {"success": False, "error": f"Submission with ID {submission_id} not found"}
            
            # Run each verification type
            results = {}
            
            # Check for misinformation
            misinfo_result = self.verify_misinformation(submission)
            results["misinformation"] = misinfo_result
            
            # Check if AI-generated
            ai_gen_result = self.verify_ai_generated(submission)
            results["ai_generated"] = ai_gen_result
            
            # Check authenticity
            auth_result = self.verify_authenticity(submission)
            results["authenticity"] = auth_result
            
            # Check factual accuracy
            fact_check_result = self.verify_factual_accuracy(submission)
            results["factual_accuracy"] = fact_check_result
            
            # Return the combined results
            return {
                "success": True,
                "submission_id": submission_id,
                "results": results
            }
        
        except Exception as e:
            logger.error(f"Error verifying submission {submission_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def verify_misinformation(self, submission: UserSubmission) -> Dict[str, Any]:
        """Verify if the content contains misinformation.
        
        Args:
            submission: UserSubmission object
            
        Returns:
            result: Dictionary with verification results
        """
        try:
            # Create verification record
            verification = VerificationResult(
                submission_id=submission.id,
                verification_type=VerificationType.MISINFORMATION.value,
                status=VerificationStatus.PROCESSING.value,
                started_at=datetime.utcnow()
            )
            db.session.add(verification)
            db.session.flush()  # Get ID without committing
            
            # Process based on content type
            content_to_analyze = self._prepare_content_for_analysis(submission)
            
            # Analyze with OpenAI for misinformation
            response = self._analyze_with_openai(content_to_analyze, "misinformation")
            
            # Parse and store results
            verification.status = VerificationStatus.COMPLETED.value
            verification.completed_at = datetime.utcnow()
            verification.confidence_score = response.get("confidence", 0.5)
            verification.result_summary = response.get("summary", "")
            verification.evidence = response.get("evidence", "")
            verification.set_meta_data(response)
            
            db.session.commit()
            
            return {
                "success": True,
                "verification_id": verification.id,
                "confidence": verification.confidence_score,
                "summary": verification.result_summary,
                "evidence": verification.evidence
            }
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error verifying misinformation: {e}")
            
            # If verification record was created, update it
            if 'verification' in locals() and verification.id:
                try:
                    verification.status = VerificationStatus.FAILED.value
                    verification.completed_at = datetime.utcnow()
                    verification.result_summary = f"Verification failed: {str(e)}"
                    db.session.commit()
                except SQLAlchemyError:
                    db.session.rollback()
            
            return {"success": False, "error": str(e)}
    
    def verify_ai_generated(self, submission: UserSubmission) -> Dict[str, Any]:
        """Verify if the content was AI-generated.
        
        Args:
            submission: UserSubmission object
            
        Returns:
            result: Dictionary with verification results
        """
        try:
            # Create verification record
            verification = VerificationResult(
                submission_id=submission.id,
                verification_type=VerificationType.AI_GENERATED.value,
                status=VerificationStatus.PROCESSING.value,
                started_at=datetime.utcnow()
            )
            db.session.add(verification)
            db.session.flush()  # Get ID without committing
            
            # Process based on content type
            content_to_analyze = self._prepare_content_for_analysis(submission)
            
            # Analyze with OpenAI for AI generation
            response = self._analyze_with_openai(content_to_analyze, "ai_generated")
            
            # Parse and store results
            verification.status = VerificationStatus.COMPLETED.value
            verification.completed_at = datetime.utcnow()
            verification.confidence_score = response.get("confidence", 0.5)
            verification.result_summary = response.get("summary", "")
            verification.evidence = response.get("evidence", "")
            verification.set_meta_data(response)
            
            db.session.commit()
            
            return {
                "success": True,
                "verification_id": verification.id,
                "confidence": verification.confidence_score,
                "summary": verification.result_summary,
                "evidence": verification.evidence
            }
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error verifying AI generation: {e}")
            
            # If verification record was created, update it
            if 'verification' in locals() and verification.id:
                try:
                    verification.status = VerificationStatus.FAILED.value
                    verification.completed_at = datetime.utcnow()
                    verification.result_summary = f"Verification failed: {str(e)}"
                    db.session.commit()
                except SQLAlchemyError:
                    db.session.rollback()
            
            return {"success": False, "error": str(e)}
    
    def verify_authenticity(self, submission: UserSubmission) -> Dict[str, Any]:
        """Verify the authenticity of the content (manipulated, fake, etc.).
        
        Args:
            submission: UserSubmission object
            
        Returns:
            result: Dictionary with verification results
        """
        try:
            # Create verification record
            verification = VerificationResult(
                submission_id=submission.id,
                verification_type=VerificationType.AUTHENTICITY.value,
                status=VerificationStatus.PROCESSING.value,
                started_at=datetime.utcnow()
            )
            db.session.add(verification)
            db.session.flush()  # Get ID without committing
            
            # Process based on content type
            content_to_analyze = self._prepare_content_for_analysis(submission)
            
            # Analyze with OpenAI for authenticity
            response = self._analyze_with_openai(content_to_analyze, "authenticity")
            
            # Parse and store results
            verification.status = VerificationStatus.COMPLETED.value
            verification.completed_at = datetime.utcnow()
            verification.confidence_score = response.get("confidence", 0.5)
            verification.result_summary = response.get("summary", "")
            verification.evidence = response.get("evidence", "")
            verification.set_meta_data(response)
            
            db.session.commit()
            
            return {
                "success": True,
                "verification_id": verification.id,
                "confidence": verification.confidence_score,
                "summary": verification.result_summary,
                "evidence": verification.evidence
            }
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error verifying authenticity: {e}")
            
            # If verification record was created, update it
            if 'verification' in locals() and verification.id:
                try:
                    verification.status = VerificationStatus.FAILED.value
                    verification.completed_at = datetime.utcnow()
                    verification.result_summary = f"Verification failed: {str(e)}"
                    db.session.commit()
                except SQLAlchemyError:
                    db.session.rollback()
            
            return {"success": False, "error": str(e)}
    
    def verify_factual_accuracy(self, submission: UserSubmission) -> Dict[str, Any]:
        """Verify the factual accuracy of the content.
        
        Args:
            submission: UserSubmission object
            
        Returns:
            result: Dictionary with verification results
        """
        try:
            # Create verification record
            verification = VerificationResult(
                submission_id=submission.id,
                verification_type=VerificationType.FACTUAL_ACCURACY.value,
                status=VerificationStatus.PROCESSING.value,
                started_at=datetime.utcnow()
            )
            db.session.add(verification)
            db.session.flush()  # Get ID without committing
            
            # Process based on content type
            content_to_analyze = self._prepare_content_for_analysis(submission)
            
            # Analyze with OpenAI for factual accuracy
            response = self._analyze_with_openai(content_to_analyze, "factual_accuracy")
            
            # Parse and store results
            verification.status = VerificationStatus.COMPLETED.value
            verification.completed_at = datetime.utcnow()
            verification.confidence_score = response.get("confidence", 0.5)
            verification.result_summary = response.get("summary", "")
            verification.evidence = response.get("evidence", "")
            verification.set_meta_data(response)
            
            db.session.commit()
            
            return {
                "success": True,
                "verification_id": verification.id,
                "confidence": verification.confidence_score,
                "summary": verification.result_summary,
                "evidence": verification.evidence
            }
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error verifying factual accuracy: {e}")
            
            # If verification record was created, update it
            if 'verification' in locals() and verification.id:
                try:
                    verification.status = VerificationStatus.FAILED.value
                    verification.completed_at = datetime.utcnow()
                    verification.result_summary = f"Verification failed: {str(e)}"
                    db.session.commit()
                except SQLAlchemyError:
                    db.session.rollback()
            
            return {"success": False, "error": str(e)}
    
    def _prepare_content_for_analysis(self, submission: UserSubmission) -> Dict[str, Any]:
        """Prepare the submission content for analysis based on type.
        
        Args:
            submission: UserSubmission object
            
        Returns:
            prepared_content: Dictionary with content prepared for analysis
        """
        content = {
            "type": submission.content_type,
            "title": submission.title or "",
            "description": submission.description or "",
            "source_url": submission.source_url or ""
        }
        
        # Add content based on type
        if submission.content_type in ["text", "text_image", "text_video"]:
            content["text"] = submission.text_content or ""
        
        if submission.content_type in ["image", "text_image"]:
            if submission.media_path:
                try:
                    image_path = os.path.join(os.getcwd(), submission.media_path)
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as img_file:
                            content["image_base64"] = base64.b64encode(img_file.read()).decode("utf-8")
                    else:
                        logger.warning(f"Image file not found: {image_path}")
                except Exception as e:
                    logger.error(f"Error reading image file: {e}")
        
        if submission.content_type in ["video", "text_video"]:
            if submission.media_path:
                try:
                    video_path = os.path.join(os.getcwd(), submission.media_path)
                    # Note: We don't base64 encode videos as they're too large
                    # Instead, we just provide the path for processing
                    content["video_path"] = video_path if os.path.exists(video_path) else None
                except Exception as e:
                    logger.error(f"Error with video file: {e}")
        
        return content
    
    def _analyze_with_openai(self, content: Dict[str, Any], verification_type: str) -> Dict[str, Any]:
        """Analyze content using OpenAI's API.
        
        Args:
            content: Dictionary with prepared content
            verification_type: Type of verification to perform
            
        Returns:
            response: Dictionary with analysis results
        """
        if not self.openai_client:
            try:
                self._initialize_openai_client()
            except Exception as e:
                raise ValueError(f"OpenAI client not available: {e}")
        
        # Construct the request based on verification type and content type
        messages = self._build_openai_messages(content, verification_type)
        
        try:
            # Use multimodal capabilities if image is present
            if "image_base64" in content and content["image_base64"]:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
            else:
                # Text-only analysis
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Ensure consistent result format
            result.setdefault("confidence", 0.5)
            result.setdefault("summary", "Analysis performed, but no summary was provided.")
            result.setdefault("evidence", "")
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing with OpenAI: {e}")
            raise
    
    def _build_openai_messages(self, content: Dict[str, Any], verification_type: str) -> List[Dict[str, Any]]:
        """Build the messages array for the OpenAI API request.
        
        Args:
            content: Dictionary with prepared content
            verification_type: Type of verification to perform
            
        Returns:
            messages: List of message dictionaries for the API request
        """
        # Base system message with instructions
        system_message = {
            "role": "system",
            "content": f"""You are an expert verification system for the CIVILIAN platform, which detects and combats misinformation. 
Your task is to analyze the provided content and determine if it contains {verification_type}.

Respond with a JSON object containing:
1. "is_problematic": boolean value (true if the content is likely {verification_type}, false if it appears legitimate)
2. "confidence": a float between 0 and 1 indicating your confidence in this assessment
3. "summary": a concise explanation of your findings (200 words max)
4. "evidence": specific elements from the content that support your conclusion
5. "recommendations": suggestions for users who encounter this content

Be thorough in your analysis and consider all aspects of the content."""
        }
        
        # User message containing the content to analyze
        user_content = f"Please analyze the following content:\n\n"
        
        if content.get("title"):
            user_content += f"Title: {content['title']}\n\n"
        
        if content.get("description"):
            user_content += f"Description: {content['description']}\n\n"
        
        if content.get("text"):
            user_content += f"Text content: {content['text']}\n\n"
        
        if content.get("source_url"):
            user_content += f"Source URL: {content['source_url']}\n\n"
        
        # Create the messages array
        messages = [system_message]
        
        # Handle different content types
        if "image_base64" in content and content["image_base64"]:
            # For image content, use multimodal input
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{content['image_base64']}"
                        }
                    }
                ]
            })
        else:
            # For text-only content
            messages.append({
                "role": "user",
                "content": user_content
            })
        
        # Add specific instructions based on verification type
        if verification_type == "misinformation":
            messages.append({
                "role": "user",
                "content": """Specific considerations for misinformation detection:
1. Check for factual inaccuracies or unsubstantiated claims
2. Identify logical fallacies or misrepresentations
3. Look for emotional manipulation or inflammatory language
4. Consider if context is missing that would change the interpretation
5. Determine if the content contradicts established scientific consensus"""
            })
        elif verification_type == "ai_generated":
            messages.append({
                "role": "user",
                "content": """Specific considerations for AI-generated content detection:
1. Look for patterns typical of language model outputs
2. Check for inconsistencies or repetitive phrasing
3. Examine any unusual or unnatural elements
4. For images, look for telltale signs of AI generation (unusual hands, eyes, artifacts)
5. Identify any elements that seem implausible or physically impossible"""
            })
        elif verification_type == "authenticity":
            messages.append({
                "role": "user",
                "content": """Specific considerations for authenticity verification:
1. Check for signs of manipulation or editing
2. Look for inconsistencies with known facts or contexts
3. Identify if the content is presented in a misleading way
4. Examine if the content is genuine but used out of context
5. Consider if timestamps, locations, or attributions are accurate"""
            })
        elif verification_type == "factual_accuracy":
            messages.append({
                "role": "user",
                "content": """Specific considerations for factual accuracy verification:
1. Identify specific claims that can be verified
2. Check if the information aligns with established knowledge
3. Look for misleading statistics or cherry-picked data
4. Consider if information is outdated or no longer applicable
5. Examine if sources cited are credible and properly represented"""
            })
        
        return messages