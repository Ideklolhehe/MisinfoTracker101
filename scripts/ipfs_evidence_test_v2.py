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
        else:
            logger.error(f"Failed to store evidence for instance {instance_id}")
            print(f"\nError: Failed to store evidence for instance {instance_id}")

def list_evidence():
    """List all stored evidence."""
    with app.app_context():
        # Create IPFS evidence store
        store = IPFSEvidenceStore(ipfs_host='localhost', ipfs_port=5001)
        
        # Get all evidence
        result = store.list_evidence()
        
        if result and result.get('success', False):
            evidence_list = result.get('evidence', [])
            
            print(f"\nStored Evidence ({len(evidence_list)} items):")
            for i, evidence in enumerate(evidence_list, 1):
                print(f"\n{i}. Hash: {evidence.get('hash')}")
                print(f"   Date: {evidence.get('date', 'Unknown')}")
                print(f"   Type: {evidence.get('type', 'Unknown')}")
                print(f"   Size: {evidence.get('size', 'Unknown')} bytes")
                print(f"   Gateway URL: {store.get_ipfs_gateway_url(evidence.get('hash'))}")
        else:
            logger.error("Failed to list evidence")
            print("\nError: Failed to list evidence")
            if result:
                print(f"Reason: {result.get('error', 'Unknown error')}")

def verify_evidence(evidence_hash):
    """Verify an evidence hash."""
    with app.app_context():
        # Create IPFS evidence store
        store = IPFSEvidenceStore(ipfs_host='localhost', ipfs_port=5001)
        
        # Verify the evidence
        result = store.verify_evidence(evidence_hash)
        
        if result and result.get('success', False):
            print(f"\nEvidence verified successfully:")
            print(f"  Hash: {evidence_hash}")
            print(f"  Original: {result.get('original_hash')}")
            print(f"  Current: {result.get('current_hash')}")
            print(f"  Verified: {result.get('verified')}")
            print(f"  Content: {result.get('content')[:100]}...") if result.get('content') else None
            print(f"  IPFS Gateway URL: {store.get_ipfs_gateway_url(evidence_hash)}")
        else:
            logger.error(f"Failed to verify evidence hash {evidence_hash}")
            print(f"\nError: Failed to verify evidence hash {evidence_hash}")
            if result:
                print(f"Reason: {result.get('error', 'Unknown error')}")

def store_all_pending():
    """Store all pending narrative instances without evidence hash."""
    with app.app_context():
        # Get all instances without evidence hash
        instances = NarrativeInstance.query.filter(NarrativeInstance.evidence_hash.is_(None)).all()
        
        if not instances:
            logger.info("No pending narrative instances found")
            print("\nNo pending narrative instances found")
            return
        
        # Create IPFS evidence store
        store = IPFSEvidenceStore(ipfs_host='localhost', ipfs_port=5001)
        
        print(f"\nFound {len(instances)} pending narrative instances")
        
        successful = 0
        failed = 0
        
        for instance in instances:
            # Store the evidence using the instance ID
            ipfs_hash = store.store_evidence(instance_id=instance.id)
            
            if ipfs_hash:
                # The method already updates the instance evidence_hash in the database
                
                logger.info(f"Evidence stored successfully for instance {instance.id}")
                print(f"  Stored evidence for instance {instance.id}, hash: {ipfs_hash[:10]}...")
                successful += 1
            else:
                logger.error(f"Failed to store evidence for instance {instance.id}")
                failed += 1
        
        print(f"\nEvidence storing completed:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

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