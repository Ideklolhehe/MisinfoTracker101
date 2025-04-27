"""
Evidence routes for the CIVILIAN system.
Provides endpoints for managing and viewing evidence for detected misinformation.
"""

import os
import json
import logging
import datetime
from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from werkzeug.exceptions import NotFound
from flask_login import login_required, current_user

from app import app, db
from models import NarrativeInstance, DetectedNarrative
from storage.evidence_store import EvidenceStore
from storage.ipfs_evidence_store import IPFSEvidenceStore

# Configure logging
logger = logging.getLogger(__name__)

# Initialize blueprint
evidence_bp = Blueprint('evidence', __name__, url_prefix='/evidence')

# Initialize evidence stores
file_evidence_store = EvidenceStore()
ipfs_evidence_store = None

def init_ipfs_store():
    """Initialize the IPFS evidence store if not already initialized."""
    global ipfs_evidence_store
    if ipfs_evidence_store is None:
        try:
            ipfs_host = os.environ.get('IPFS_HOST', 'localhost')
            ipfs_port = int(os.environ.get('IPFS_PORT', '5001'))
            ipfs_evidence_store = IPFSEvidenceStore(
                ipfs_host=ipfs_host,
                ipfs_port=ipfs_port
            )
            logger.info(f"IPFS evidence store initialized in routes (host: {ipfs_host}, port: {ipfs_port})")
        except Exception as e:
            logger.warning(f"IPFS daemon connection failed: {e}")
            ipfs_evidence_store = IPFSEvidenceStore()  # Local fallback mode
            logger.info("IPFS evidence store initialized in routes (local-only mode)")

# Initialize IPFS store on blueprint registration
@evidence_bp.record_once
def on_load(state):
    init_ipfs_store()

@evidence_bp.route('/')
@login_required
def index():
    """Evidence dashboard page."""
    return render_template('evidence/index.html')

@evidence_bp.route('/store/<int:instance_id>', methods=['POST'])
@login_required
def store_evidence(instance_id):
    """Store evidence for a narrative instance."""
    try:
        # Get storage type from request
        storage_type = request.form.get('storage_type', 'ipfs').lower()
        
        # Store evidence in the appropriate store
        if storage_type == 'ipfs':
            evidence_hash = ipfs_evidence_store.store_evidence(instance_id)
            store_name = "IPFS"
        else:
            evidence_hash = file_evidence_store.store_evidence(instance_id)
            store_name = "File"
        
        if evidence_hash:
            return jsonify({
                'success': True,
                'message': f'Evidence stored successfully in {store_name} store',
                'hash': evidence_hash
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to store evidence in {store_name} store'
            }), 500
            
    except Exception as e:
        logger.error(f"Error storing evidence for instance {instance_id}: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@evidence_bp.route('/view/<path:evidence_hash>')
@login_required
def view_evidence(evidence_hash):
    """View evidence by hash."""
    try:
        # Determine storage type based on hash prefix
        if evidence_hash.startswith('local:'):
            storage_type = 'file'
            evidence_data = ipfs_evidence_store.retrieve_evidence(evidence_hash)
            store_name = "IPFS (Local Fallback)"
        elif ':' not in evidence_hash and len(evidence_hash) > 32:
            storage_type = 'ipfs'
            evidence_data = ipfs_evidence_store.retrieve_evidence(evidence_hash)
            store_name = "IPFS"
        else:
            storage_type = 'file'
            evidence_data = file_evidence_store.retrieve_evidence(evidence_hash)
            store_name = "File"
        
        if not evidence_data:
            raise NotFound(f"Evidence not found for hash {evidence_hash}")
        
        # Get related instance and narrative
        instance_id = evidence_data.get('instance_id')
        instance = NarrativeInstance.query.get(instance_id) if instance_id else None
        
        narrative_id = evidence_data.get('narrative_id')
        narrative = DetectedNarrative.query.get(narrative_id) if narrative_id else None
        
        # Get IPFS gateway URL if applicable
        gateway_url = None
        if storage_type == 'ipfs':
            gateway_url = ipfs_evidence_store.get_ipfs_gateway_url(evidence_hash)
        
        return render_template(
            'evidence/view.html',
            hash=evidence_hash,
            data=evidence_data,
            instance=instance,
            narrative=narrative,
            store_type=store_name,
            gateway_url=gateway_url,
            storage_type=storage_type
        )
            
    except NotFound as e:
        logger.warning(f"Evidence not found: {evidence_hash}")
        return render_template('errors/404.html', message=str(e)), 404
    except Exception as e:
        logger.error(f"Error retrieving evidence {evidence_hash}: {e}")
        return render_template('errors/500.html', message=str(e)), 500

@evidence_bp.route('/verify/<path:evidence_hash>')
@login_required
def verify_evidence(evidence_hash):
    """Verify evidence integrity."""
    try:
        # Determine storage type based on hash prefix
        if evidence_hash.startswith('local:') or (not ':' in evidence_hash and len(evidence_hash) > 32):
            is_valid = ipfs_evidence_store.verify_evidence(evidence_hash)
            store_name = "IPFS"
        else:
            is_valid = file_evidence_store.verify_evidence(evidence_hash)
            store_name = "File"
        
        return jsonify({
            'success': True,
            'valid': is_valid,
            'message': f'Evidence {"is valid" if is_valid else "has been tampered with"} ({store_name} store)'
        })
            
    except Exception as e:
        logger.error(f"Error verifying evidence {evidence_hash}: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@evidence_bp.route('/list')
@login_required
def list_evidence():
    """List all evidence items."""
    try:
        # Pagination parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        # Get evidence from both stores
        ipfs_evidence = ipfs_evidence_store.list_evidence(limit=limit, offset=offset)
        file_evidence = file_evidence_store.list_evidence(limit=limit, offset=offset)
        
        # Transform file evidence to match IPFS format
        transformed_file_evidence = []
        for file_hash in file_evidence:
            # Get metadata for the file hash
            file_data = file_evidence_store.retrieve_evidence(file_hash)
            if file_data:
                transformed_file_evidence.append({
                    'hash': file_hash,
                    'storage_type': 'file',
                    'instance_id': file_data.get('instance_id'),
                    'narrative_id': file_data.get('narrative_id'),
                    'timestamp': file_data.get('timestamp')
                })
        
        # Combine and sort evidence by timestamp (newest first)
        all_evidence = ipfs_evidence + transformed_file_evidence
        all_evidence.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return render_template(
            'evidence/list.html',
            evidence_list=all_evidence,
            limit=limit,
            offset=offset,
            has_more=(len(all_evidence) >= limit)
        )
            
    except Exception as e:
        logger.error(f"Error listing evidence: {e}")
        return render_template('errors/500.html', message=str(e)), 500

@evidence_bp.route('/store-all', methods=['POST'])
@login_required
def store_all_pending():
    """Store evidence for all pending instances."""
    try:
        # Get storage type from request
        storage_type = request.form.get('storage_type', 'ipfs').lower()
        
        # Store all pending evidence
        if storage_type == 'ipfs':
            result = ipfs_evidence_store.store_all_pending()
            store_name = "IPFS"
        else:
            result = file_evidence_store.store_all_pending()
            store_name = "File"
        
        return jsonify({
            'success': True,
            'message': f'Stored evidence for {result.get("success", 0)} instances in {store_name} store',
            'result': result
        })
            
    except Exception as e:
        logger.error(f"Error storing all pending evidence: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500