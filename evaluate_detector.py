"""
Script to evaluate the CIVILIAN system's detection of adversarial content.
"""

import argparse
import os
import sys
import json
from services.adversarial_service import AdversarialService
from agents.detector_agent import DetectorAgent
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from app import app

def main():
    parser = argparse.ArgumentParser(description='Evaluate detector against adversarial content')
    parser.add_argument('--topic', type=str, 
                      help='Filter content by topic')
    
    parser.add_argument('--type', type=str,
                      help='Filter content by misinformation type')
    
    parser.add_argument('--limit', type=int, default=10, 
                      help='Maximum number of content items to evaluate')
    
    parser.add_argument('--output-file', type=str, default='detector_evaluation.json',
                      help='File to save evaluation results')
    
    args = parser.parse_args()
    
    with app.app_context():
        # Get the adversarial content to test
        service = AdversarialService()
        
        if args.topic:
            print(f"Filtering content by topic: {args.topic}")
            content_items = service.get_content_by_topic(args.topic, args.limit)
        else:
            print(f"Getting up to {args.limit} content items for testing")
            content_items = service.get_content_for_training(args.limit)
        
        if not content_items:
            print("No adversarial content found for evaluation. Generate some first.")
            sys.exit(1)
        
        # Initialize the detector agent
        text_processor = TextProcessor()
        vector_store = VectorStore()
        detector = DetectorAgent(text_processor, vector_store)
        
        print(f"Evaluating detector on {len(content_items)} adversarial content items...")
        
        results = []
        correct_count = 0
        
        # Evaluate each content item
        for i, content in enumerate(content_items, 1):
            print(f"Testing item {i}/{len(content_items)}: {content.title[:50]}...")
            
            # Process the content with the detector
            detection_result = detector.process_content(
                content=content.content,
                content_id=f"adv-{content.id}",
                source="adversarial",
                metadata=json.dumps({"topic": content.topic, "type": content.misinfo_type})
            )
            
            # A high score means detection as misinformation (correct)
            misinfo_score = detection_result.get('confidence', 0)
            threshold = 0.75  # Standard system threshold
            correct_detection = misinfo_score >= threshold
            
            if correct_detection:
                correct_count += 1
                
            # Record evaluation in the database
            service.evaluate_content(
                content_id=content.id,
                detector_version="current",
                correct_detection=correct_detection,
                confidence_score=misinfo_score,
                user_id=None,
                notes=f"Automatic evaluation: {'Correctly' if correct_detection else 'Incorrectly'} detected"
            )
            
            # Record result for the report
            results.append({
                "content_id": content.id,
                "title": content.title,
                "topic": content.topic,
                "misinfo_type": content.misinfo_type,
                "confidence_score": misinfo_score,
                "correct_detection": correct_detection,
                "threshold": threshold
            })
        
        # Calculate accuracy
        accuracy = correct_count / len(content_items) if content_items else 0
        
        # Save evaluation results
        evaluation = {
            "total_items": len(content_items),
            "correct_detections": correct_count,
            "accuracy": accuracy,
            "results": results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
            
        print(f"\nEvaluation complete!")
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(content_items)})")
        print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
