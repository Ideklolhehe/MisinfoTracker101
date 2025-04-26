"""
Adversarial content service for the CIVILIAN system.

This service manages the generation, storage, and evaluation of adversarial misinformation
content used for system training and evaluation.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from sqlalchemy.orm.exc import NoResultFound
from app import db
from models import AdversarialContent, AdversarialEvaluation, User
from utils.adversarial_generator import AdversarialGenerator

logger = logging.getLogger(__name__)

class AdversarialService:
    """Service for managing adversarial misinformation content."""
    
    def __init__(self):
        """Initialize the adversarial service with necessary components."""
        self.generator = AdversarialGenerator()
        self.output_dir = os.path.join(os.getcwd(), "training", "adversarial")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_training_content(self, topic: str, misinfo_type: str, 
                                real_content: Optional[str] = None) -> AdversarialContent:
        """Generate and store a piece of adversarial training content.
        
        Args:
            topic: Topic area for the misinformation
            misinfo_type: Type of misinformation pattern to use
            real_content: Optional real content to base the misinfo on
            
        Returns:
            Created AdversarialContent instance
        """
        try:
            # Generate content with the adversarial generator
            generated = self.generator.generate_with_ai(topic, misinfo_type, real_content)
            
            # Create database record
            content = AdversarialContent(
                title=generated.get("title", "Untitled adversarial content"),
                content=generated.get("content", ""),
                topic=topic,
                misinfo_type=misinfo_type,
                generation_method=generated.get("generation_method", "ai"),
                generated_at=datetime.utcnow()
            )
            
            # Store metadata
            metadata = {
                "tactics": generated.get("metadata", {}).get("tactics", []),
                "training_label": "TRUE",
                "generation_context": {
                    "based_on_real_content": bool(real_content),
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            content.set_meta_data(metadata)
            
            # Save to database
            db.session.add(content)
            db.session.commit()
            
            # Also save to filesystem for backup/training purposes
            filename = f"adv_{content.id}_{topic}_{misinfo_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.output_dir, filename)
            self.generator.store_generated_content(generated, filepath)
            
            logger.info(f"Created adversarial content id={content.id}, topic={topic}, type={misinfo_type}")
            return content
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to generate adversarial content: {e}")
            raise
    
    def generate_content_batch(self, batch_size: int, topics: Optional[List[str]] = None, 
                              types: Optional[List[str]] = None) -> List[AdversarialContent]:
        """Generate a batch of adversarial training content.
        
        Args:
            batch_size: Number of content pieces to generate
            topics: Optional list of topics to use
            types: Optional list of misinfo types to use
            
        Returns:
            List of created AdversarialContent instances
        """
        results = []
        generated_batch = self.generator.generate_batch(batch_size, topics, types)
        
        for generated in generated_batch:
            try:
                content = AdversarialContent(
                    title=generated.get("title", "Untitled batch content"),
                    content=generated.get("content", ""),
                    topic=generated.get("metadata", {}).get("topic", "general"),
                    misinfo_type=generated.get("metadata", {}).get("misinfo_type", "unknown"),
                    generation_method=generated.get("generation_method", "ai"),
                    generated_at=datetime.utcnow()
                )
                
                # Store metadata
                metadata = {
                    "tactics": generated.get("metadata", {}).get("tactics", []),
                    "training_label": "TRUE",
                    "batch_generation": True
                }
                content.set_meta_data(metadata)
                
                # Save to database
                db.session.add(content)
                results.append(content)
                
            except Exception as e:
                logger.error(f"Failed to process batch item: {e}")
                continue
        
        db.session.commit()
        logger.info(f"Generated batch of {len(results)} adversarial content items")
        return results
    
    def generate_variants(self, content_id: int, num_variants: int = 3) -> List[AdversarialContent]:
        """Generate variations of an existing piece of adversarial content.
        
        Args:
            content_id: ID of the content to create variants for
            num_variants: Number of variants to generate
            
        Returns:
            List of variant AdversarialContent instances
        """
        try:
            # Retrieve the original content
            original = db.session.query(AdversarialContent).filter_by(id=content_id).one()
            
            # Prepare content dict for the generator
            content_dict = {
                "id": original.id,
                "title": original.title,
                "content": original.content,
                "metadata": {
                    "topic": original.topic,
                    "misinfo_type": original.misinfo_type,
                    **original.get_meta_data()
                }
            }
            
            # Generate variants
            variant_dicts = self.generator.generate_variants(content_dict, num_variants)
            variant_models = []
            
            for variant_dict in variant_dicts:
                variant = AdversarialContent(
                    title=variant_dict.get("title", f"Variant of {original.title}"),
                    content=variant_dict.get("content", ""),
                    topic=original.topic,
                    misinfo_type=original.misinfo_type,
                    generation_method=variant_dict.get("generation_method", "ai"),
                    variant_of_id=original.id,
                    generated_at=datetime.utcnow()
                )
                
                # Store metadata
                metadata = original.get_meta_data()
                metadata.update({
                    "is_variant": True,
                    "variant_number": variant_dict.get("metadata", {}).get("variant_number", 0),
                    "variant_creation_date": datetime.utcnow().isoformat()
                })
                variant.set_meta_data(metadata)
                
                db.session.add(variant)
                variant_models.append(variant)
            
            db.session.commit()
            logger.info(f"Generated {len(variant_models)} variants of content id={content_id}")
            return variant_models
            
        except NoResultFound:
            logger.error(f"Content with id={content_id} not found")
            return []
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to generate variants: {e}")
            return []
    
    def evaluate_content(self, content_id: int, detector_version: str, 
                        correct_detection: bool, confidence_score: float,
                        user_id: Optional[int] = None, notes: Optional[str] = None) -> AdversarialEvaluation:
        """Record an evaluation of adversarial content.
        
        Args:
            content_id: ID of the content being evaluated
            detector_version: Version of the detector used
            correct_detection: Whether it was correctly identified as misinfo
            confidence_score: Detection confidence score
            user_id: Optional ID of the user doing the evaluation
            notes: Optional evaluation notes
            
        Returns:
            Created AdversarialEvaluation instance
        """
        try:
            evaluation = AdversarialEvaluation(
                content_id=content_id,
                detector_version=detector_version,
                correct_detection=correct_detection,
                confidence_score=confidence_score,
                evaluated_by=user_id,
                notes=notes,
                evaluation_date=datetime.utcnow()
            )
            
            # Store additional metadata
            metadata = {
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "detector_details": {
                    "version": detector_version,
                    "confidence_threshold": 0.75  # System default
                }
            }
            evaluation.set_meta_data(metadata)
            
            db.session.add(evaluation)
            db.session.commit()
            
            logger.info(f"Recorded evaluation for content id={content_id}, "
                      f"correct={correct_detection}, score={confidence_score}")
            return evaluation
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to create content evaluation: {e}")
            raise
    
    def get_content_by_id(self, content_id: int) -> Optional[AdversarialContent]:
        """Retrieve adversarial content by ID.
        
        Args:
            content_id: ID of the content to retrieve
            
        Returns:
            AdversarialContent instance or None if not found
        """
        try:
            return db.session.query(AdversarialContent).filter_by(id=content_id).one()
        except NoResultFound:
            logger.warning(f"Content with id={content_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error retrieving content id={content_id}: {e}")
            return None
    
    def get_content_by_topic(self, topic: str, limit: int = 10) -> List[AdversarialContent]:
        """Retrieve adversarial content by topic.
        
        Args:
            topic: Topic to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of AdversarialContent instances
        """
        try:
            return db.session.query(AdversarialContent)\
                .filter_by(topic=topic)\
                .order_by(AdversarialContent.generated_at.desc())\
                .limit(limit)\
                .all()
        except Exception as e:
            logger.error(f"Error retrieving content for topic={topic}: {e}")
            return []
    
    def get_content_for_training(self, limit: int = 100) -> List[AdversarialContent]:
        """Retrieve adversarial content for system training.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of AdversarialContent instances
        """
        try:
            return db.session.query(AdversarialContent)\
                .filter_by(is_active=True)\
                .order_by(AdversarialContent.generated_at.desc())\
                .limit(limit)\
                .all()
        except Exception as e:
            logger.error(f"Error retrieving training content: {e}")
            return []
    
    def deactivate_content(self, content_id: int) -> bool:
        """Mark adversarial content as inactive (don't use for training).
        
        Args:
            content_id: ID of the content to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            content = db.session.query(AdversarialContent).filter_by(id=content_id).one()
            content.is_active = False
            db.session.commit()
            logger.info(f"Deactivated content id={content_id}")
            return True
        except NoResultFound:
            logger.warning(f"Content with id={content_id} not found for deactivation")
            return False
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to deactivate content id={content_id}: {e}")
            return False
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get overall statistics on adversarial content evaluation.
        
        Returns:
            Dictionary of evaluation statistics
        """
        try:
            total_content = db.session.query(AdversarialContent).count()
            total_evaluations = db.session.query(AdversarialEvaluation).count()
            
            # Get correct detection rate
            correct_detections = db.session.query(AdversarialEvaluation)\
                .filter_by(correct_detection=True)\
                .count()
            
            detection_rate = correct_detections / total_evaluations if total_evaluations > 0 else 0
            
            # Get stats by misinfo type
            type_stats = {}
            misinfo_types = db.session.query(AdversarialContent.misinfo_type)\
                .distinct()\
                .all()
            
            for (misinfo_type,) in misinfo_types:
                type_content_count = db.session.query(AdversarialContent)\
                    .filter_by(misinfo_type=misinfo_type)\
                    .count()
                
                # Get evaluations for this type
                type_evaluation_subquery = db.session.query(AdversarialEvaluation.id)\
                    .join(AdversarialContent, AdversarialEvaluation.content_id == AdversarialContent.id)\
                    .filter(AdversarialContent.misinfo_type == misinfo_type)\
                    .subquery()
                
                type_evaluation_count = db.session.query(type_evaluation_subquery).count()
                
                type_stats[misinfo_type] = {
                    "content_count": type_content_count,
                    "evaluation_count": type_evaluation_count
                }
            
            return {
                "total_content": total_content,
                "total_evaluations": total_evaluations,
                "detection_rate": detection_rate,
                "by_misinfo_type": type_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation stats: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }