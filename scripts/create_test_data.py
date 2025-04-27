#!/usr/bin/env python
"""
Create test data for the CIVILIAN system.
This script creates test narrative instances and narratives for testing.
"""

import os
import sys
import json
import logging
import datetime
from datetime import timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestDataCreator')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import application components
from app import app, db
from models import NarrativeInstance, DetectedNarrative, DataSource

def create_test_narrative():
    """Create a test narrative for testing."""
    with app.app_context():
        # Check if we already have a test narrative
        narrative = DetectedNarrative.query.filter_by(title="Test Narrative for IPFS Evidence").first()
        
        if narrative:
            logger.info(f"Test narrative already exists with ID {narrative.id}")
            return narrative
        
        # Create a new test narrative
        narrative = DetectedNarrative(
            title="Test Narrative for IPFS Evidence",
            description="This is a test narrative created for testing the IPFS evidence storage system.",
            confidence_score=0.85,
            first_detected=datetime.datetime.utcnow() - timedelta(days=1),
            last_updated=datetime.datetime.utcnow(),
            status="active",
            language="en",
            meta_data=json.dumps({
                "test": True,
                "created_by": "create_test_data.py",
                "purpose": "IPFS evidence storage testing",
                "threat_level": 2,
                "propagation_score": 0.5,
                "source_count": 1
            })
        )
        
        db.session.add(narrative)
        db.session.commit()
        
        logger.info(f"Created test narrative with ID {narrative.id}")
        return narrative

def create_test_instance(narrative):
    """Create a test narrative instance for testing."""
    with app.app_context():
        # Create a test data source if it doesn't exist
        source = DataSource.query.filter_by(name="Test Source").first()
        if not source:
            source = DataSource(
                name="Test Source",
                source_type="test",
                config=json.dumps({"test": True}),
                is_active=True
            )
            db.session.add(source)
            db.session.commit()
            logger.info(f"Created test data source with ID {source.id}")
        
        # Create a new test instance
        instance = NarrativeInstance(
            narrative_id=narrative.id,
            source_id=source.id,
            content="This is a test narrative instance created for testing the IPFS evidence storage system. " +
                   "It contains a claim that the Earth is flat, which is false information. " +
                   "This evidence should be stored immutably using the IPFS storage system.",
            url="https://example.com/test-instance",
            detected_at=datetime.datetime.utcnow(),
            meta_data=json.dumps({
                "test": True,
                "created_by": "create_test_data.py",
                "purpose": "IPFS evidence storage testing",
                "claim": "The Earth is flat.",
                "truth": "The Earth is an oblate spheroid.",
                "verified_false": True
            })
        )
        
        db.session.add(instance)
        db.session.commit()
        
        logger.info(f"Created test instance with ID {instance.id}")
        return instance

def main():
    """Main function to create test data."""
    try:
        # Create test narrative
        narrative = create_test_narrative()
        
        # Create test instance
        instance = create_test_instance(narrative)
        
        print(f"\nCreated test data successfully:")
        print(f"  Narrative ID: {narrative.id}")
        print(f"  Instance ID: {instance.id}")
        print(f"\nYou can now use this instance for testing the IPFS evidence storage.")
        print(f"For example, run: python scripts/ipfs_evidence_test.py --instance {instance.id}")
        
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()