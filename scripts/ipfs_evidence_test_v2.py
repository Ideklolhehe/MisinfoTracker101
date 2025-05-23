#!/usr/bin/env python
"""
Test IPFS evidence storage system by storing a test narrative instance.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IPFSEvidenceTest')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import application components
from app import app, db
from models import NarrativeInstance
from storage.ipfs_evidence_store import IPFSEvidenceStore

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test IPFS evidence storage system.')
    parser.add_argument('--instance', type=int, help='ID of the narrative instance to store as evidence')
    parser.add_argument('--list', action='store_true', help='List all stored evidence')
    parser.add_argument('--verify', type=str, help='Verify an evidence hash')
    parser.add_argument('--all-pending', action='store_true', help='Store all pending narrative instances')
    return parser.parse_args()

def store_instance_evidence(instance_id):
    """Store evidence for a specific narrative instance."""
    with app.app_context():
        # Retrieve the instance
        instance = db.session.get(NarrativeInstance, instance_id)
        if not instance:
            logger.error(f"No instance found with ID {instance_id}")
            print(f"Error: No instance found with ID {instance_id}")
            return
        
        # Create IPFS evidence store
        store = IPFSEvidenceStore(ipfs_host='localhost', ipfs_port=5001)
        
        # The IPFSEvidenceStore.store_evidence method expects only the instance ID
        # All content and metadata is fetched from the database
        ipfs_hash = store.store_evidence(instance_id=instance.id)
        
        if ipfs_hash:
            # The method already updates the instance evidence_hash in the database
            
            logger.info(f"Evidence stored successfully for instance {instance_id}")
            logger.info(f"Evidence hash: {ipfs_hash}")
            
            print(f"\nEvidence stored successfully:")
            print(f"  Instance ID: {instance_id}")
            print(f"  Evidence hash: {ipfs_hash}")
            print(f"  IPFS Gateway URL: {store.get_ipfs_gateway_url(ipfs_hash)}")
            
            # Verify the evidence after storing
            is_valid = store.verify_evidence(ipfs_hash)
            print(f"  Verification result: {'Valid' if is_valid else 'Invalid'}")
            
            # Also retrieve the evidence to confirm it's accessible
            evidence_data = store.retrieve_evidence(ipfs_hash)
            if evidence_data:
                print("  Evidence is retrievable")
            else:
                print("  Warning: Evidence could not be retrieved")
        else:
            logger.error(f"Failed to store evidence for instance {instance_id}")
            print(f"\nError: Failed to store evidence for instance {instance_id}")

def list_evidence():
    """List all stored evidence."""
    with app.app_context():
        # Create IPFS evidence store
        store = IPFSEvidenceStore(ipfs_host='localhost', ipfs_port=5001)
        
        # Get all evidence
        evidence_list = store.list_evidence()
        
        if evidence_list:
            print(f"\nStored Evidence ({len(evidence_list)} items):")
            for i, evidence in enumerate(evidence_list, 1):
                print(f"\n{i}. Hash: {evidence.get('hash', 'Unknown')}")
                print(f"   Storage Type: {evidence.get('storage_type', 'Unknown')}")
                print(f"   Instance ID: {evidence.get('instance_id', 'Unknown')}")
                print(f"   Narrative ID: {evidence.get('narrative_id', 'Unknown')}")
                print(f"   Timestamp: {evidence.get('timestamp', 'Unknown')}")
                if evidence.get('url'):
                    print(f"   Gateway URL: {evidence.get('url')}")
        else:
            logger.error("No evidence found or failed to list evidence")
            print("\nNo evidence found or failed to list evidence")

def verify_evidence(evidence_hash):
    """Verify an evidence hash."""
    with app.app_context():
        # Create IPFS evidence store
        store = IPFSEvidenceStore(ipfs_host='localhost', ipfs_port=5001)
        
        # Verify the evidence
        is_valid = store.verify_evidence(evidence_hash)
        
        # Also retrieve the evidence content to show details
        evidence_data = store.retrieve_evidence(evidence_hash)
        
        if is_valid:
            print(f"\nEvidence verified successfully:")
            print(f"  Hash: {evidence_hash}")
            print(f"  Verification Status: Valid")
            
            if evidence_data:
                print(f"  Instance ID: {evidence_data.get('instance_id', 'Unknown')}")
                print(f"  Narrative ID: {evidence_data.get('narrative_id', 'Unknown')}")
                print(f"  Timestamp: {evidence_data.get('timestamp', 'Unknown')}")
                content_preview = evidence_data.get('content', '')[:100]
                if content_preview:
                    print(f"  Content Preview: {content_preview}...")
            
            print(f"  IPFS Gateway URL: {store.get_ipfs_gateway_url(evidence_hash)}")
        else:
            logger.error(f"Failed to verify evidence hash {evidence_hash}")
            print(f"\nError: Failed to verify evidence hash {evidence_hash}")
            print(f"  The evidence may have been tampered with or does not exist.")

def store_all_pending(limit=5):
    """Store all pending narrative instances without evidence hash.
    
    Args:
        limit: Maximum number of instances to process (default=5)
    """
    with app.app_context():
        # Create IPFS evidence store
        store = IPFSEvidenceStore(ipfs_host='localhost', ipfs_port=5001)
        
        # Get instances without evidence hash
        instances = NarrativeInstance.query.filter(
            NarrativeInstance.evidence_hash.is_(None),
            NarrativeInstance.narrative_id.isnot(None)  # Only store confirmed misinformation
        ).limit(limit).all()
        
        print(f"\nFound {len(instances)} instances for evidence storage (limited to {limit})")
        
        # Process each instance manually
        success_count = 0
        failed_count = 0
        for instance in instances:
            print(f"Processing instance {instance.id}...")
            evidence_hash = store.store_evidence(instance_id=instance.id)
            
            if evidence_hash:
                success_count += 1
                print(f"  Success: Evidence stored with hash {evidence_hash[:10]}...")
                
                # Verify after storing
                is_valid = store.verify_evidence(evidence_hash)
                print(f"  Verification: {'Valid' if is_valid else 'Invalid'}")
            else:
                failed_count += 1
                print(f"  Failed: Could not store evidence")
        
        print(f"\nEvidence storing completed:")
        print(f"  Total processed: {len(instances)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {failed_count}")

def main():
    """Main function for IPFS evidence test."""
    try:
        args = parse_args()
        
        if args.instance:
            store_instance_evidence(args.instance)
        elif args.list:
            list_evidence()
        elif args.verify:
            verify_evidence(args.verify)
        elif args.all_pending:
            store_all_pending()
        else:
            print("Please specify an action. Use --help for available options.")
            
    except Exception as e:
        logger.error(f"Error testing IPFS evidence: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()