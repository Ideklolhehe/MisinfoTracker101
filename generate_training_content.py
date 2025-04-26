"""
Script to generate adversarial misinformation content for training the CIVILIAN system.
"""

import argparse
import os
import sys
from services.adversarial_service import AdversarialService
from app import app

def main():
    parser = argparse.ArgumentParser(description='Generate adversarial training content')
    parser.add_argument('--topic', type=str, choices=[
        'health', 'science', 'politics', 'economics', 'environment', 
        'technology', 'international_relations', 'security'
    ], required=True, help='Topic area for the misinformation')
    
    parser.add_argument('--type', type=str, choices=[
        'conspiracy_theory', 'misleading_statistics', 'out_of_context', 
        'false_attribution', 'fabricated_content', 'manipulated_media',
        'impersonation', 'emotional_manipulation', 'oversimplification',
        'false_equivalence'
    ], required=True, help='Type of misinformation to generate')
    
    parser.add_argument('--batch', type=int, default=1, 
                      help='Number of examples to generate (1-10)')
    
    parser.add_argument('--input-file', type=str, 
                      help='Optional file with real content to base misinformation on')
    
    args = parser.parse_args()
    
    if args.batch < 1 or args.batch > 10:
        print("Error: Batch size must be between 1 and 10")
        sys.exit(1)
    
    real_content = None
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} not found")
            sys.exit(1)
        with open(args.input_file, 'r') as f:
            real_content = f.read()
    
    with app.app_context():
        service = AdversarialService()
        
        if args.batch == 1:
            # Generate a single example
            print(f"Generating adversarial content for topic '{args.topic}', type '{args.type}'...")
            content = service.generate_training_content(
                topic=args.topic,
                misinfo_type=args.type,
                real_content=real_content
            )
            print(f"Generated content ID {content.id}: {content.title}")
            print(f"Stored in database and in training/adversarial/ directory")
        else:
            # Generate a batch
            print(f"Generating {args.batch} adversarial examples...")
            contents = service.generate_content_batch(
                batch_size=args.batch,
                topics=[args.topic],
                types=[args.type]
            )
            print(f"Generated {len(contents)} content items")
            for content in contents:
                print(f"- ID {content.id}: {content.title}")
            print(f"All items stored in database and in training/adversarial/ directory")

if __name__ == "__main__":
    main()
