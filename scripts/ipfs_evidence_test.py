#!/usr/bin/env python
"""
Test script for IPFS evidence storage in the CIVILIAN system.
This script tests the IPFSEvidenceStore class and demonstrates its usage.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IPFSTest')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import application components
from app import app, db
from models import NarrativeInstance, DetectedNarrative
from storage.ipfs_evidence_store import IPFSEvidenceStore

def display_narrative_instances(limit=5):
    """Display recent narrative instances for testing."""
    with app.app_context():
        instances = NarrativeInstance.query.order_by(
            NarrativeInstance.detected_at.desc()
        ).limit(limit).all()
        
        print(f"\n=== Recent Narrative Instances ({len(instances)}) ===")
        for i, instance in enumerate(instances):
            narrative = DetectedNarrative.query.get(instance.narrative_id) if instance.narrative_id else None
            narrative_title = narrative.title if narrative else "No narrative"
            print(f"{i+1}. ID: {instance.id}, Narrative: {narrative_title}")
            print(f"   Content: {instance.content[:100]}...")
            print(f"   Evidence Hash: {instance.evidence_hash or 'None'}")
            print("")
        
        return instances

def test_store_evidence(store, instance_id):
    """Test storing evidence for a specific instance."""
    with app.app_context():
        print(f"\n=== Testing Evidence Storage for Instance {instance_id} ===")
        
        # Get the instance before storage
        instance = NarrativeInstance.query.get(instance_id)
        if not instance:
            print(f"Error: Instance {instance_id} not found")
            return None
        
        print(f"Before: Evidence Hash = {instance.evidence_hash or 'None'}")
        
        # Store evidence
        evidence_hash = store.store_evidence(instance_id)
        
        # Get the updated instance
        instance = NarrativeInstance.query.get(instance_id)
        
        print(f"After: Evidence Hash = {instance.evidence_hash or 'None'}")
        print(f"Result: {'Success' if evidence_hash else 'Failed'}")
        
        if evidence_hash:
            if evidence_hash.startswith('local:'):
                print(f"Stored locally with hash: {evidence_hash}")
            else:
                gateway_url = store.get_ipfs_gateway_url(evidence_hash)
                print(f"Stored on IPFS with CID: {evidence_hash}")
                print(f"Gateway URL: {gateway_url}")
        
        return evidence_hash

def test_retrieve_evidence(store, evidence_hash):
    """Test retrieving evidence by hash."""
    with app.app_context():
        print(f"\n=== Testing Evidence Retrieval for Hash {evidence_hash} ===")
        
        # Retrieve evidence
        evidence_data = store.retrieve_evidence(evidence_hash)
        
        if evidence_data:
            print("Evidence retrieved successfully:")
            print(f"Instance ID: {evidence_data.get('instance_id')}")
            print(f"Narrative ID: {evidence_data.get('narrative_id')}")
            print(f"Detected At: {evidence_data.get('detected_at')}")
            print(f"Content: {evidence_data.get('content', '')[:100]}...")
        else:
            print("Failed to retrieve evidence.")
        
        return evidence_data

def test_verify_evidence(store, evidence_hash):
    """Test verifying evidence integrity."""
    with app.app_context():
        print(f"\n=== Testing Evidence Verification for Hash {evidence_hash} ===")
        
        # Verify evidence
        is_valid = store.verify_evidence(evidence_hash)
        
        print(f"Evidence is {'valid' if is_valid else 'invalid or tampered'}")
        
        return is_valid

def test_list_evidence(store, limit=10):
    """Test listing evidence items."""
    with app.app_context():
        print(f"\n=== Listing Evidence Items (max {limit}) ===")
        
        # List evidence
        evidence_list = store.list_evidence(limit=limit)
        
        for i, evidence in enumerate(evidence_list):
            print(f"{i+1}. Hash: {evidence.get('hash')}")
            print(f"   Storage Type: {evidence.get('storage_type')}")
            print(f"   Instance ID: {evidence.get('instance_id')}")
            print(f"   Timestamp: {evidence.get('timestamp')}")
            if evidence.get('url'):
                print(f"   URL: {evidence.get('url')}")
            print("")
        
        return evidence_list

def test_store_all_pending(store):
    """Test storing all pending evidence."""
    with app.app_context():
        print("\n=== Testing Storage of All Pending Evidence ===")
        
        # Count instances without evidence hash
        count = NarrativeInstance.query.filter(
            NarrativeInstance.evidence_hash.is_(None),
            NarrativeInstance.narrative_id.isnot(None)
        ).count()
        
        print(f"Found {count} instances without evidence hash")
        
        if count > 0:
            # Store all pending
            result = store.store_all_pending()
            
            print("Storage complete:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        return count

def main():
    """Main function to test IPFS evidence storage."""
    parser = argparse.ArgumentParser(description='Test IPFS evidence storage.')
    parser.add_argument('--host', default='localhost', help='IPFS API host')
    parser.add_argument('--port', type=int, default=5001, help='IPFS API port')
    parser.add_argument('--instance', type=int, help='Specific instance ID to test')
    parser.add_argument('--hash', help='Specific evidence hash to test')
    parser.add_argument('--all', action='store_true', help='Store all pending evidence')
    parser.add_argument('--list', action='store_true', help='List evidence items')
    parser.add_argument('--limit', type=int, default=5, help='Limit for listing items')
    args = parser.parse_args()
    
    # Initialize the IPFS evidence store
    store = IPFSEvidenceStore(ipfs_host=args.host, ipfs_port=args.port)
    
    try:
        # Display some recent instances
        instances = display_narrative_instances(limit=args.limit)
        
        # Test storing a specific instance
        if args.instance:
            evidence_hash = test_store_evidence(store, args.instance)
            if evidence_hash:
                test_retrieve_evidence(store, evidence_hash)
                test_verify_evidence(store, evidence_hash)
        
        # Test retrieving and verifying a specific hash
        if args.hash:
            test_retrieve_evidence(store, args.hash)
            test_verify_evidence(store, args.hash)
        
        # Test storing all pending evidence
        if args.all:
            test_store_all_pending(store)
        
        # Test listing evidence
        if args.list or not (args.instance or args.hash or args.all):
            test_list_evidence(store, limit=args.limit)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\nError: {e}")
        print("\nNote: If you're getting connection errors, make sure IPFS daemon is running.")
        print("You can start it with: ipfs daemon")
        print("If IPFS is not installed, you can install it from: https://docs.ipfs.io/install/")
        print("The script will still work in local fallback mode without IPFS.")

if __name__ == "__main__":
    main()