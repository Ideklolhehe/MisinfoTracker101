"""
Adversarial misinformation generator for CIVILIAN system training.

This module generates realistic but false content that mimics common misinformation
patterns, tactics, and narratives to train and test the system's detection capabilities.
"""

import logging
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

class AdversarialGenerator:
    """Generator for adversarial misinformation content for system training."""
    
    def __init__(self):
        """Initialize the adversarial generator with necessary models and resources."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found, adversarial generation will be limited")
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.misinfo_types = [
            "conspiracy_theory", 
            "misleading_statistics", 
            "out_of_context", 
            "false_attribution",
            "fabricated_content",
            "manipulated_media",
            "impersonation",
            "emotional_manipulation",
            "oversimplification",
            "false_equivalence"
        ]
        
        # Common misinformation topics for training purposes
        self.topic_areas = [
            "health",
            "science",
            "politics",
            "economics",
            "environment",
            "technology",
            "international_relations",
            "security"
        ]
        
        # Load templates for rule-based generation (fallback if AI unavailable)
        self._load_templates()
    
    def _load_templates(self):
        """Load templates for rule-based generation."""
        self.templates = {
            "conspiracy_theory": [
                "SHOCKING: {authority} discovered that {entity} is secretly planning to {harmful_action}.",
                "What {authority} doesn't want you to know: {entity} has been {harmful_action} for decades.",
                "LEAKED DOCUMENTS reveal how {entity} is controlling {target} through {mechanism}."
            ],
            "misleading_statistics": [
                "STUDY SHOWS: {percentage}% increase in {negative_outcome} linked to {cause}.",
                "BREAKING: New data shows {percentage}% of {group} affected by {problem}.",
                "Scientists alarmed by {percentage}% rise in {problem} - {cause} suspected."
            ],
            "fabricated_content": [
                "{authority} ADMITS: '{false_statement}'",
                "EXCLUSIVE: {entity} insider reveals '{false_revelation}'",
                "BANNED VIDEO shows {entity} {false_action} - SHARE BEFORE DELETED!"
            ]
        }
        
        self.template_variables = {
            "authority": ["Scientists", "Experts", "Researchers", "Insiders", "Government officials", "Whistleblowers"],
            "entity": ["Big Pharma", "The government", "Tech companies", "Foreign nations", "Global organizations"],
            "harmful_action": ["control the population", "hide the truth", "manipulate public opinion", "implement surveillance"],
            "target": ["citizens", "children", "vulnerable populations", "the economy", "elections"],
            "mechanism": ["secret technology", "chemical agents", "social media algorithms", "financial systems"],
            "percentage": ["50", "65", "78", "92", "37", "88"],
            "negative_outcome": ["disease", "adverse effects", "mental health issues", "economic collapse"],
            "cause": ["common products", "government policies", "new technologies", "environmental factors"],
            "group": ["children", "adults", "seniors", "workers", "students"],
            "problem": ["health issues", "privacy violations", "financial losses", "social disruption"],
            "false_statement": [
                "We've been covering up the truth for years", 
                "The public was never supposed to find out about this", 
                "We can no longer hide the real effects"
            ],
            "false_revelation": [
                "Everything the public knows is orchestrated", 
                "The real agenda is far more sinister", 
                "We've been manipulating data since the beginning"
            ],
            "false_action": [
                "hiding evidence", 
                "silencing critics", 
                "manipulating test results", 
                "destroying crucial documents"
            ]
        }
    
    def generate_with_ai(self, topic: str, misinfo_type: str, 
                        real_content: Optional[str] = None) -> Dict[str, Any]:
        """Generate adversarial misinformation using AI.
        
        Args:
            topic: Topic area for misinformation (health, politics, etc.)
            misinfo_type: Type of misinformation to generate
            real_content: Optional real content to base the misinfo on
            
        Returns:
            Dictionary with generated content and metadata
        """
        if not self.api_key:
            logger.warning("OpenAI API key not found, falling back to template generation")
            return self.generate_with_template(topic, misinfo_type)
        
        try:
            # Create prompt for the AI model
            system_prompt = """
            You are a tool helping to train an anti-misinformation system. Generate realistic-looking but FALSE information 
            that exhibits specific misinformation tactics. This content will be used ONLY for training detection systems.
            
            IMPORTANT: The content must be clearly false but should use the rhetorical patterns, emotional appeals, 
            and presentation style common in real misinformation. Label your response as TRAINING CONTENT 
            and include metadata that identifies the tactics used.
            """
            
            user_prompt = f"""
            Generate adversarial misinformation training content with the following parameters:
            
            Topic area: {topic}
            Misinformation type: {misinfo_type}
            
            {"Use this real content as inspiration, but create FALSE information: " + real_content if real_content else ""}
            
            Format your response as a JSON object with these fields:
            1. "title": An attention-grabbing headline
            2. "content": The false content (250-500 words)
            3. "metadata": Information about the tactics used
            4. "training_label": "TRUE" (you must include this exact label to mark this as training data)
            """
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add generation metadata
            result["generated_at"] = datetime.utcnow().isoformat()
            result["generation_method"] = "ai"
            result["purpose"] = "system_training"
            
            return result
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return self.generate_with_template(topic, misinfo_type)
    
    def generate_with_template(self, topic: str, misinfo_type: str) -> Dict[str, Any]:
        """Generate adversarial misinformation using templates (fallback method).
        
        Args:
            topic: Topic area for misinformation
            misinfo_type: Type of misinformation to generate
            
        Returns:
            Dictionary with generated content and metadata
        """
        # Default to conspiracy_theory if requested type not in templates
        template_type = misinfo_type if misinfo_type in self.templates else "conspiracy_theory"
        
        # Select random template
        template = random.choice(self.templates[template_type])
        
        # Fill in variables
        for var_name, var_values in self.template_variables.items():
            if "{" + var_name + "}" in template:
                template = template.replace("{" + var_name + "}", random.choice(var_values))
        
        # Create a short content piece based on the template
        content = f"""
        {template}
        
        This {topic} issue is being deliberately hidden from the public. Multiple sources have 
        confirmed these findings, despite official denials. The evidence is clear and concerning.
        
        [This is automatically generated TRAINING CONTENT for system testing purposes only]
        """
        
        return {
            "title": template[:50] + "...",
            "content": content.strip(),
            "metadata": {
                "misinfo_type": misinfo_type,
                "topic": topic,
                "tactics": ["emotional language", "appeal to authority", "false urgency"]
            },
            "training_label": "TRUE",
            "generated_at": datetime.utcnow().isoformat(),
            "generation_method": "template",
            "purpose": "system_training"
        }
    
    def generate_batch(self, batch_size: int, topics: Optional[List[str]] = None, 
                      types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate a batch of adversarial examples.
        
        Args:
            batch_size: Number of examples to generate
            topics: Optional list of topics to use (defaults to all)
            types: Optional list of misinfo types to use (defaults to all)
            
        Returns:
            List of generated content dictionaries
        """
        results = []
        available_topics = topics if topics else self.topic_areas
        available_types = types if types else self.misinfo_types
        
        for _ in range(batch_size):
            topic = random.choice(available_topics)
            misinfo_type = random.choice(available_types)
            
            # Use AI for 80% of generation if available, templates for 20%
            if self.api_key and random.random() < 0.8:
                result = self.generate_with_ai(topic, misinfo_type)
            else:
                result = self.generate_with_template(topic, misinfo_type)
                
            results.append(result)
            
        return results
    
    def generate_variants(self, content: Dict[str, Any], num_variants: int = 3) -> List[Dict[str, Any]]:
        """Generate variations of a piece of misinformation.
        
        Args:
            content: Original misinfo content dictionary
            num_variants: Number of variants to generate
            
        Returns:
            List of variant content dictionaries
        """
        variants = []
        
        # Extract core information
        topic = content["metadata"]["topic"]
        misinfo_type = content["metadata"]["misinfo_type"]
        original_text = content["content"]
        
        for i in range(num_variants):
            try:
                if self.api_key:
                    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                    # do not change this unless explicitly requested by the user
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a tool helping to train an anti-misinformation system. Create variations of FALSE training content that maintain the same false claims but change the presentation."},
                            {"role": "user", "content": f"Create a variation of this false training content that uses the same core claim but changes the wording, style and presentation. Keep it clearly FALSE but make it look like it's coming from a different source or platform.\n\nORIGINAL: {original_text}\n\nVariation #{i+1}:"}
                        ],
                        temperature=0.8
                    )
                    
                    variant_text = response.choices[0].message.content
                    variant = content.copy()
                    variant["content"] = variant_text
                    variant["title"] = f"Variant {i+1}: {content['title']}"
                    variant["metadata"]["variant_of"] = content.get("id", "original")
                    variant["metadata"]["variant_number"] = i+1
                    variants.append(variant)
                else:
                    # Fallback to simple template variant
                    variant = self.generate_with_template(topic, misinfo_type)
                    variant["metadata"]["variant_of"] = content.get("id", "original")
                    variant["metadata"]["variant_number"] = i+1
                    variants.append(variant)
            except Exception as e:
                logger.error(f"Failed to generate variant {i}: {e}")
                continue
                
        return variants
    
    def store_generated_content(self, content: Dict[str, Any], filepath: str):
        """Store generated adversarial content for future use.
        
        Args:
            content: Generated content dictionary
            filepath: Path to save the content
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(content, f, indent=2)
            logger.info(f"Stored adversarial training content to {filepath}")
        except Exception as e:
            logger.error(f"Failed to store adversarial content: {e}")
    
    def load_generated_content(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load previously generated adversarial content.
        
        Args:
            filepath: Path to the content file
            
        Returns:
            Content dictionary or None if loading fails
        """
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
            return content
        except Exception as e:
            logger.error(f"Failed to load adversarial content: {e}")
            return None