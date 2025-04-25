import logging
import hashlib
import json
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import application components
from models import NarrativeInstance, SystemLog
from app import db

logger = logging.getLogger(__name__)

class EvidenceStore:
    """Manages immutable storage of evidence for detected misinformation."""
    
    def __init__(self, storage_path: str = None):
        """Initialize the evidence store.
        
        Args:
            storage_path: Path for storing evidence files
        """
        self.storage_path = storage_path or os.environ.get('EVIDENCE_STORAGE_PATH', './evidence')
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"EvidenceStore initialized with path: {self.storage_path}")
    
    def store_evidence(self, instance_id: int) -> Optional[str]:
        """Store evidence for a narrative instance.
        
        Args:
            instance_id: ID of the narrative instance to store
            
        Returns:
            evidence_hash: Hash of the stored evidence, or None if failed
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
            
            # Calculate hash
            evidence_hash = hashlib.sha256(evidence_json.encode('utf-8')).hexdigest()
            
            # Store to file
            filename = f"{evidence_hash}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(evidence_json)
            
            # Update instance with evidence hash
            with db.session.begin():
                instance.evidence_hash = evidence_hash
            
            logger.info(f"Stored evidence for instance {instance_id} with hash {evidence_hash}")
            return evidence_hash
            
        except Exception as e:
            logger.error(f"Error storing evidence for instance {instance_id}: {e}")
            self._log_error("store_evidence", f"Instance {instance_id}: {e}")
            return None
    
    def retrieve_evidence(self, evidence_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence by hash.
        
        Args:
            evidence_hash: Hash of the evidence to retrieve
            
        Returns:
            evidence_data: Dictionary with evidence data, or None if not found
        """
        try:
            # Build filepath
            filename = f"{evidence_hash}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Evidence file not found: {filename}")
                return None
            
            # Read evidence data
            with open(filepath, 'r', encoding='utf-8') as f:
                evidence_data = json.load(f)
            
            # Verify hash
            evidence_json = json.dumps(evidence_data, sort_keys=True)
            calculated_hash = hashlib.sha256(evidence_json.encode('utf-8')).hexdigest()
            
            if calculated_hash != evidence_hash:
                logger.error(f"Evidence hash mismatch: {evidence_hash} vs {calculated_hash}")
                return None
            
            logger.debug(f"Retrieved evidence with hash {evidence_hash}")
            return evidence_data
            
        except Exception as e:
            logger.error(f"Error retrieving evidence with hash {evidence_hash}: {e}")
            return None
    
    def verify_evidence(self, evidence_hash: str) -> bool:
        """Verify that evidence is intact and matches its hash.
        
        Args:
            evidence_hash: Hash of the evidence to verify
            
        Returns:
            is_valid: Whether the evidence is valid
        """
        try:
            # Retrieve evidence
            evidence_data = self.retrieve_evidence(evidence_hash)
            if not evidence_data:
                return False
            
            # Evidence is valid if retrieve_evidence returned data
            # (it already checks the hash)
            return True
            
        except Exception as e:
            logger.error(f"Error verifying evidence with hash {evidence_hash}: {e}")
            return False
    
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
            
            for instance in instances:
                evidence_hash = self.store_evidence(instance.id)
                if evidence_hash:
                    success_count += 1
                else:
                    failed_count += 1
            
            result = {
                'total': len(instances),
                'success': success_count,
                'failed': failed_count
            }
            
            logger.info(f"Stored evidence for {success_count} instances ({failed_count} failed)")
            return result
            
        except Exception as e:
            logger.error(f"Error storing pending evidence: {e}")
            self._log_error("store_all_pending", str(e))
            return {'error': str(e)}
    
    def list_evidence(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List evidence hashes in the store.
        
        Args:
            limit: Maximum number of hashes to return
            offset: Offset for pagination
            
        Returns:
            hashes: List of evidence hashes
        """
        try:
            # Get all JSON files in storage path
            evidence_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
            
            # Sort by modification time (newest first)
            evidence_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.storage_path, f)), reverse=True)
            
            # Apply pagination
            paginated_files = evidence_files[offset:offset+limit]
            
            # Extract hashes from filenames
            hashes = [f[:-5] for f in paginated_files]  # Remove .json extension
            
            return hashes
            
        except Exception as e:
            logger.error(f"Error listing evidence: {e}")
            return []
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="evidence_store",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
