"""
IPFS-based immutable evidence storage for the CIVILIAN system.
This module provides a secure, distributed, and tamper-proof storage system
for evidence related to misinformation narratives using IPFS.
"""

import os
import json
import hashlib
import logging
import tempfile
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import ipfshttpclient
from urllib.parse import urlparse

# Import application components
from models import NarrativeInstance, SystemLog
from app import db

logger = logging.getLogger(__name__)

class IPFSEvidenceStore:
    """Manages immutable storage of evidence for detected misinformation using IPFS."""
    
    def __init__(self, ipfs_host: str = None, ipfs_port: int = None, backup_path: str = None):
        """Initialize the IPFS evidence store.
        
        Args:
            ipfs_host: IPFS API host (defaults to localhost or env variable)
            ipfs_port: IPFS API port (defaults to 5001 or env variable)
            backup_path: Path for storing backup evidence files locally
        """
        # Set defaults from environment or use defaults
        self.ipfs_host = ipfs_host or os.environ.get('IPFS_HOST', 'localhost')
        self.ipfs_port = ipfs_port or int(os.environ.get('IPFS_PORT', '5001'))
        self.backup_path = backup_path or os.environ.get('EVIDENCE_BACKUP_PATH', './evidence/ipfs_backup')
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_path, exist_ok=True)
        
        # Initialize client to None (will be created on first use)
        self._client = None
        
        logger.info(f"IPFSEvidenceStore initialized with host: {self.ipfs_host}, port: {self.ipfs_port}")
    
    @property
    def client(self):
        """Get or create the IPFS client."""
        if self._client is None:
            try:
                # Use format ipfs_host:ipfs_port for IPFS API
                api_url = f'/ip4/{self.ipfs_host}/tcp/{self.ipfs_port}/http'
                self._client = ipfshttpclient.connect(api_url)
                logger.info(f"Connected to IPFS daemon at {api_url}")
            except Exception as e:
                logger.error(f"Failed to connect to IPFS daemon: {e}")
                self._log_error("client_connection", f"Failed to connect to IPFS: {e}")
                raise ConnectionError(f"Failed to connect to IPFS daemon: {e}")
        return self._client
    
    def store_evidence(self, instance_id: int) -> Optional[str]:
        """Store evidence for a narrative instance on IPFS.
        
        Args:
            instance_id: ID of the narrative instance to store
            
        Returns:
            evidence_hash: IPFS CID of the stored evidence, or None if failed
        """
        try:
            # Get the instance from database
            instance = NarrativeInstance.query.get(instance_id)
            if not instance:
                logger.warning(f"Instance {instance_id} not found")
                return None
            
            # Prepare evidence data
            evidence_data = {
                'instance_id': instance.id,
                'narrative_id': instance.narrative_id,
                'source_id': instance.source_id,
                'content': instance.content,
                'metadata': json.loads(instance.meta_data) if instance.meta_data else {},
                'url': instance.url,
                'detected_at': instance.detected_at.isoformat(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Convert to JSON
            evidence_json = json.dumps(evidence_data, sort_keys=True)
            
            # Calculate local hash for verification
            local_hash = hashlib.sha256(evidence_json.encode('utf-8')).hexdigest()
            
            # Store to IPFS
            try:
                # Create a temporary file with the JSON content
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp:
                    temp_path = temp.name
                    temp.write(evidence_json)
                
                # Add to IPFS
                result = self.client.add(temp_path)
                ipfs_hash = result['Hash']
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                # Also store a local backup
                backup_filename = f"{ipfs_hash}.json"
                backup_filepath = os.path.join(self.backup_path, backup_filename)
                with open(backup_filepath, 'w', encoding='utf-8') as f:
                    f.write(evidence_json)
                
                # Update instance with evidence hash
                with db.session.begin():
                    # Store both the IPFS hash and local hash for verification
                    instance.evidence_hash = ipfs_hash
                    if instance.meta_data:
                        meta_data = json.loads(instance.meta_data)
                        meta_data['local_hash'] = local_hash
                        instance.meta_data = json.dumps(meta_data)
                    else:
                        instance.meta_data = json.dumps({'local_hash': local_hash})
                
                logger.info(f"Stored evidence for instance {instance_id} on IPFS with CID {ipfs_hash}")
                return ipfs_hash
            
            except Exception as e:
                logger.error(f"Error storing to IPFS: {e}")
                
                # Fall back to local storage if IPFS fails
                fallback_filename = f"{local_hash}.json"
                fallback_filepath = os.path.join(self.backup_path, fallback_filename)
                
                with open(fallback_filepath, 'w', encoding='utf-8') as f:
                    f.write(evidence_json)
                
                # Update instance with local hash
                with db.session.begin():
                    instance.evidence_hash = f"local:{local_hash}"
                    if instance.meta_data:
                        meta_data = json.loads(instance.meta_data)
                        meta_data['storage_type'] = 'local_fallback'
                        instance.meta_data = json.dumps(meta_data)
                    else:
                        instance.meta_data = json.dumps({'storage_type': 'local_fallback'})
                
                logger.warning(f"Fell back to local storage for instance {instance_id} with hash {local_hash}")
                return f"local:{local_hash}"
                
        except Exception as e:
            logger.error(f"Error storing evidence for instance {instance_id}: {e}")
            self._log_error("store_evidence", f"Instance {instance_id}: {e}")
            return None
    
    def retrieve_evidence(self, evidence_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence by hash.
        
        Args:
            evidence_hash: IPFS CID or local hash of the evidence to retrieve
            
        Returns:
            evidence_data: Dictionary with evidence data, or None if not found
        """
        try:
            # Check if this is a local fallback hash
            if evidence_hash.startswith('local:'):
                local_hash = evidence_hash.split(':', 1)[1]
                return self._retrieve_local(local_hash)
            
            # Try to get from IPFS
            try:
                # Create a temporary file to store the retrieved content
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
                    temp_path = temp.name
                
                # Get the file from IPFS
                self.client.get(evidence_hash, temp_path)
                
                # Read the content
                with open(temp_path, 'r', encoding='utf-8') as f:
                    evidence_data = json.load(f)
                
                # Clean up
                os.unlink(temp_path)
                
                logger.debug(f"Retrieved evidence from IPFS with CID {evidence_hash}")
                return evidence_data
            
            except Exception as e:
                logger.warning(f"Error retrieving from IPFS: {e}, trying local backup")
                
                # Try local backup
                return self._retrieve_local(evidence_hash)
                
        except Exception as e:
            logger.error(f"Error retrieving evidence with hash {evidence_hash}: {e}")
            return None
    
    def _retrieve_local(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence from local backup storage.
        
        Args:
            hash_value: Hash value (without 'local:' prefix)
            
        Returns:
            evidence_data: Dictionary with evidence data, or None if not found
        """
        try:
            # Build filepath
            filename = f"{hash_value}.json"
            filepath = os.path.join(self.backup_path, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Evidence file not found in backup: {filename}")
                return None
            
            # Read evidence data
            with open(filepath, 'r', encoding='utf-8') as f:
                evidence_data = json.load(f)
            
            # Verify integrity
            evidence_json = json.dumps(evidence_data, sort_keys=True)
            calculated_hash = hashlib.sha256(evidence_json.encode('utf-8')).hexdigest()
            
            if not (hash_value == calculated_hash or hash_value.endswith(calculated_hash)):
                logger.error(f"Evidence hash mismatch: {hash_value} vs {calculated_hash}")
                return None
            
            logger.debug(f"Retrieved evidence from local backup with hash {hash_value}")
            return evidence_data
            
        except Exception as e:
            logger.error(f"Error retrieving local evidence with hash {hash_value}: {e}")
            return None
    
    def verify_evidence(self, evidence_hash: str) -> bool:
        """Verify that evidence is intact and matches its hash.
        
        Args:
            evidence_hash: IPFS CID or local hash of the evidence to verify
            
        Returns:
            is_valid: Whether the evidence is valid
        """
        try:
            # Retrieve the evidence
            evidence_data = self.retrieve_evidence(evidence_hash)
            if not evidence_data:
                return False
            
            # Convert to canonical JSON
            evidence_json = json.dumps(evidence_data, sort_keys=True)
            calculated_hash = hashlib.sha256(evidence_json.encode('utf-8')).hexdigest()
            
            # For IPFS hashes, we can't directly compare the calculated_hash with the IPFS CID
            # Instead, we verify by checking if the content can be retrieved from IPFS
            if evidence_hash.startswith('local:'):
                # For local hashes, verify directly
                local_hash = evidence_hash.split(':', 1)[1]
                return local_hash == calculated_hash
            else:
                # For IPFS hashes, we consider it valid if we were able to retrieve it
                return True
            
        except Exception as e:
            logger.error(f"Error verifying evidence with hash {evidence_hash}: {e}")
            return False
    
    def get_ipfs_gateway_url(self, ipfs_hash: str, gateway: str = None) -> str:
        """Get a public gateway URL for an IPFS hash.
        
        Args:
            ipfs_hash: IPFS CID
            gateway: Optional gateway URL (defaults to ipfs.io)
            
        Returns:
            url: Public gateway URL for the IPFS content
        """
        if ipfs_hash.startswith('local:'):
            return f"local://{ipfs_hash.split(':', 1)[1]}"
        
        gateway = gateway or os.environ.get('IPFS_GATEWAY', 'https://ipfs.io')
        
        # Ensure gateway doesn't have trailing slash
        if gateway.endswith('/'):
            gateway = gateway[:-1]
        
        return f"{gateway}/ipfs/{ipfs_hash}"
    
    def store_all_pending(self) -> Dict[str, Any]:
        """Store evidence for all instances that don't have evidence hashes.
        
        Returns:
            result: Dictionary with processing stats
        """
        try:
            # Find instances without evidence hash
            instances = NarrativeInstance.query.filter(
                NarrativeInstance.evidence_hash.is_(None),
                NarrativeInstance.narrative_id.isnot(None)  # Only store confirmed misinformation
            ).all()
            
            logger.info(f"Found {len(instances)} instances needing evidence storage")
            
            # Process each instance
            success_count = 0
            failed_count = 0
            ipfs_count = 0
            local_count = 0
            
            for instance in instances:
                evidence_hash = self.store_evidence(instance.id)
                if evidence_hash:
                    success_count += 1
                    if evidence_hash.startswith('local:'):
                        local_count += 1
                    else:
                        ipfs_count += 1
                else:
                    failed_count += 1
            
            result = {
                'total': len(instances),
                'success': success_count,
                'failed': failed_count,
                'ipfs_stored': ipfs_count,
                'local_stored': local_count
            }
            
            logger.info(f"Stored evidence for {success_count} instances ({failed_count} failed), "
                       f"{ipfs_count} on IPFS, {local_count} locally")
            return result
            
        except Exception as e:
            logger.error(f"Error storing pending evidence: {e}")
            self._log_error("store_all_pending", str(e))
            return {'error': str(e)}
    
    def list_evidence(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List evidence items in the store.
        
        Args:
            limit: Maximum number of items to return
            offset: Offset for pagination
            
        Returns:
            evidence_list: List of evidence items with hash and metadata
        """
        try:
            # Get all JSON files in backup path
            evidence_files = [f for f in os.listdir(self.backup_path) if f.endswith('.json')]
            
            # Sort by modification time (newest first)
            evidence_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.backup_path, f)), reverse=True)
            
            # Apply pagination
            paginated_files = evidence_files[offset:offset+limit]
            
            # Extract data from files
            result = []
            for filename in paginated_files:
                file_path = os.path.join(self.backup_path, filename)
                hash_value = filename[:-5]  # Remove .json extension
                
                try:
                    # Try to read the file to get metadata
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add basic info to result
                    storage_type = "ipfs" if not hash_value.startswith("local:") else "local"
                    result.append({
                        'hash': hash_value,
                        'storage_type': storage_type,
                        'instance_id': data.get('instance_id'),
                        'narrative_id': data.get('narrative_id'),
                        'timestamp': data.get('timestamp'),
                        'url': self.get_ipfs_gateway_url(hash_value) if storage_type == "ipfs" else None
                    })
                except Exception as e:
                    logger.warning(f"Error reading evidence file {filename}: {e}")
                    # Add basic info even if reading fails
                    result.append({
                        'hash': hash_value,
                        'storage_type': "unknown",
                        'error': str(e)
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing evidence: {e}")
            return []
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="ipfs_evidence_store",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")