"""
Decentralized Publishing Service for the CIVILIAN system.

This module provides functionality for publishing narratives, counter-narratives,
and analysis reports to decentralized storage networks.
"""

import os
import json
import uuid
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from app import db
from models import (
    DetectedNarrative, 
    CounterMessage, 
    PublishedContent,
    User,
    EvidenceRecord,
    DataSource
)
from services.ipfs_service import IPFSService
from prometheus_client import Counter, Gauge

# Configure logging
logger = logging.getLogger(__name__)

# Prometheus metrics
publish_counter = Counter('content_published_total', 'Total content items published to decentralized networks')
verification_counter = Counter('content_verifications_total', 'Total content verification operations')
ipfs_latency = Gauge('ipfs_publish_latency_seconds', 'IPFS publishing latency in seconds')
ipns_updates = Counter('ipns_updates_total', 'Total IPNS record updates')

class DecentralizedPublishingService:
    """Service for publishing content to decentralized networks."""
    
    def __init__(self, auto_connect: bool = True):
        """
        Initialize the decentralized publishing service.
        
        Args:
            auto_connect: Whether to automatically try connecting to IPFS
        """
        self._ipfs = IPFSService(connect_to_daemon=auto_connect)
        self._publish_lock = threading.Lock()
        self._verified_publishers = set()  # Set of verified publisher IDs
        self._content_index = {}  # Map of content ID to IPFS hash
        self._catalog_hash = None  # Latest IPFS hash of the content catalog
        self._catalog_ipns = None  # IPNS name for the content catalog
        
        # Track successful connection to IPFS
        self.ipfs_available = self._ipfs.connected
        if self.ipfs_available:
            logger.info("Connected to IPFS for decentralized publishing")
            
            # Initialize IPNS catalog key if needed
            try:
                keys = self._ipfs.list_ipns_keys()
                catalog_keys = [k for k in keys if k.get('Name') == 'civilian-catalog']
                if not catalog_keys:
                    logger.info("Creating IPNS key for content catalog")
                    self._catalog_ipns = self._ipfs.create_ipns_key('civilian-catalog')
                else:
                    self._catalog_ipns = catalog_keys[0].get('Id')
                logger.info(f"Using IPNS key for catalog: {self._catalog_ipns}")
                
                # Load existing catalog if available
                if self._catalog_ipns:
                    ipfs_hash = self._ipfs.resolve_ipns(self._catalog_ipns)
                    if ipfs_hash:
                        self._catalog_hash = ipfs_hash
                        catalog_data = self._ipfs.get_content(ipfs_hash)
                        if catalog_data and isinstance(catalog_data, dict):
                            self._content_index = catalog_data.get('content_index', {})
                            logger.info(f"Loaded content catalog with {len(self._content_index)} entries")
            except Exception as e:
                logger.error(f"Error initializing IPNS catalog: {e}")
        else:
            logger.warning("IPFS connection not available, decentralized publishing will be limited")
    
    def publish_narrative_analysis(self, narrative_id: int, include_related: bool = True,
                                   publisher_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Publish a narrative analysis to decentralized storage.
        
        Args:
            narrative_id: ID of the narrative to publish
            include_related: Whether to include related narratives in the publication
            publisher_id: ID of the user publishing the content (for verification)
        
        Returns:
            Publication info dictionary or None if failed
        """
        if not self.ipfs_available:
            logger.error("IPFS not available, cannot publish narrative analysis")
            return None
            
        try:
            # Get the narrative from the database
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.error(f"Narrative with ID {narrative_id} not found")
                return None
                
            # Prepare publication content
            content = {
                "type": "narrative_analysis",
                "id": f"narrative-{narrative_id}",
                "title": narrative.title,
                "description": narrative.description,
                "confidence_score": narrative.confidence_score,
                "first_detected": narrative.first_detected.isoformat() if narrative.first_detected else None,
                "last_updated": narrative.last_updated.isoformat() if narrative.last_updated else None,
                "status": narrative.status,
                "language": narrative.language,
                "metadata": narrative.get_meta_data(),
                "publisher_id": publisher_id,
                "publication_date": datetime.utcnow().isoformat(),
                "version": "1.0",
                "related_narratives": []
            }
            
            # Add instances if available
            content["instances"] = []
            for instance in narrative.instances:
                if instance.evidence_hash:  # Only include instances with evidence
                    instance_data = {
                        "id": instance.id,
                        "content": instance.content,
                        "detected_at": instance.detected_at.isoformat() if instance.detected_at else None,
                        "url": instance.url,
                        "source": instance.source.name if instance.source else None,
                        "evidence_hash": instance.evidence_hash,
                        "metadata": instance.get_meta_data()
                    }
                    content["instances"].append(instance_data)
            
            # Add related narratives if requested
            if include_related:
                # Get related narratives based on metadata (temporal, sequence, or stream clusters)
                meta_data = narrative.get_meta_data()
                cluster_info = {
                    "stream_cluster": meta_data.get("stream_cluster"),
                    "temporal_cluster": meta_data.get("temporal_cluster"),
                    "sequence_cluster": meta_data.get("sequence_cluster")
                }
                
                # Find narratives in the same clusters
                related_narratives = []
                for cluster_type, cluster_id in cluster_info.items():
                    if cluster_id is not None:
                        # Find narratives with matching cluster
                        query = (
                            db.session.query(DetectedNarrative)
                            .filter(DetectedNarrative.id != narrative_id)
                            .limit(5)
                        )
                        
                        for related in query:
                            related_meta = related.get_meta_data()
                            if related_meta.get(cluster_type) == cluster_id:
                                related_data = {
                                    "id": related.id,
                                    "title": related.title,
                                    "description": related.description,
                                    "confidence_score": related.confidence_score,
                                    "first_detected": related.first_detected.isoformat() if related.first_detected else None,
                                    "cluster_type": cluster_type,
                                    "cluster_id": cluster_id
                                }
                                related_narratives.append(related_data)
                
                content["related_narratives"] = related_narratives
            
            # Add counter-narratives if available
            content["counter_narratives"] = []
            for counter in narrative.counter_messages:
                counter_data = {
                    "id": counter.id,
                    "content": counter.content,
                    "dimension": counter.dimension,
                    "strategy": counter.strategy,
                    "status": counter.status,
                    "created_at": counter.created_at.isoformat() if counter.created_at else None,
                    "metadata": counter.get_meta_data()
                }
                content["counter_narratives"].append(counter_data)
            
            # Publish to IPFS
            start_time = time.time()
            metadata = {
                "publisher_id": publisher_id,
                "publish_time": datetime.utcnow().isoformat(),
                "content_type": "application/json",
                "schema_version": "1.0",
                "signature": None  # Would include cryptographic signature for verification
            }
            
            ipfs_hash = self._ipfs.add_content(json.dumps(content), metadata=metadata)
            publish_time = time.time() - start_time
            ipfs_latency.set(publish_time)
            
            if not ipfs_hash:
                logger.error(f"Failed to publish narrative {narrative_id} to IPFS")
                return None
                
            publish_counter.inc()
            logger.info(f"Published narrative {narrative_id} to IPFS with hash: {ipfs_hash}")
            
            # Create a publication record in the database
            try:
                publication = PublishedContent(
                    content_type="narrative_analysis",
                    reference_id=narrative_id,
                    ipfs_hash=ipfs_hash,
                    ipns_name=None,  # Individual publications don't get IPNS names
                    publisher_id=publisher_id,
                    status="published",
                    title=narrative.title,
                    description=f"Analysis of narrative: {narrative.title}",
                    publication_date=datetime.utcnow()
                )
                db.session.add(publication)
                db.session.commit()
                
                # Update the content index and publish catalog
                with self._publish_lock:
                    content_id = f"narrative-{narrative_id}"
                    self._content_index[content_id] = {
                        "ipfs_hash": ipfs_hash,
                        "title": narrative.title,
                        "type": "narrative_analysis",
                        "publication_date": datetime.utcnow().isoformat(),
                        "publisher_id": publisher_id
                    }
                    self._publish_catalog()
                
                return {
                    "success": True,
                    "publication_id": publication.id,
                    "ipfs_hash": ipfs_hash,
                    "title": narrative.title,
                    "publish_time": publish_time,
                    "publication_date": publication.publication_date.isoformat(),
                    "content_url": f"ipfs://{ipfs_hash}"
                }
            except Exception as e:
                logger.error(f"Error creating publication record: {e}")
                return {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "title": narrative.title,
                    "publish_time": publish_time,
                    "content_url": f"ipfs://{ipfs_hash}",
                    "warning": "Publication record not created in database"
                }
                
        except Exception as e:
            logger.error(f"Error publishing narrative analysis: {e}")
            return None
    
    def publish_counter_narrative(self, counter_id: int, publisher_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Publish a counter-narrative to decentralized storage.
        
        Args:
            counter_id: ID of the counter-narrative to publish
            publisher_id: ID of the user publishing the content (for verification)
        
        Returns:
            Publication info dictionary or None if failed
        """
        if not self.ipfs_available:
            logger.error("IPFS not available, cannot publish counter-narrative")
            return None
            
        try:
            # Get the counter-narrative from the database
            counter = CounterMessage.query.get(counter_id)
            if not counter:
                logger.error(f"Counter-narrative with ID {counter_id} not found")
                return None
                
            # Only allow publishing approved counter-narratives
            if counter.status != "approved":
                logger.error(f"Cannot publish counter-narrative {counter_id} with status '{counter.status}'")
                return None
                
            # Get the parent narrative
            narrative = counter.narrative
            if not narrative:
                logger.error(f"Parent narrative not found for counter-narrative {counter_id}")
                return None
                
            # Prepare publication content
            content = {
                "type": "counter_narrative",
                "id": f"counter-{counter_id}",
                "content": counter.content,
                "dimension": counter.dimension,
                "strategy": counter.strategy,
                "parent_narrative": {
                    "id": narrative.id,
                    "title": narrative.title,
                    "description": narrative.description
                },
                "publisher_id": publisher_id,
                "created_by": counter.created_by,
                "approved_by": counter.approved_by,
                "created_at": counter.created_at.isoformat() if counter.created_at else None,
                "last_updated": counter.last_updated.isoformat() if counter.last_updated else None,
                "publication_date": datetime.utcnow().isoformat(),
                "effectiveness": counter.get_meta_data().get("effectiveness", {}),
                "version": "1.0"
            }
            
            # Publish to IPFS
            start_time = time.time()
            metadata = {
                "publisher_id": publisher_id,
                "publish_time": datetime.utcnow().isoformat(),
                "content_type": "application/json",
                "schema_version": "1.0",
                "signature": None  # Would include cryptographic signature for verification
            }
            
            ipfs_hash = self._ipfs.add_content(json.dumps(content), metadata=metadata)
            publish_time = time.time() - start_time
            ipfs_latency.set(publish_time)
            
            if not ipfs_hash:
                logger.error(f"Failed to publish counter-narrative {counter_id} to IPFS")
                return None
                
            publish_counter.inc()
            logger.info(f"Published counter-narrative {counter_id} to IPFS with hash: {ipfs_hash}")
            
            # Create a publication record in the database
            try:
                publication = PublishedContent(
                    content_type="counter_narrative",
                    reference_id=counter_id,
                    ipfs_hash=ipfs_hash,
                    ipns_name=None,  # Individual publications don't get IPNS names
                    publisher_id=publisher_id,
                    status="published",
                    title=f"Counter-narrative for: {narrative.title}",
                    description=counter.content[:100] + "..." if len(counter.content) > 100 else counter.content,
                    publication_date=datetime.utcnow()
                )
                db.session.add(publication)
                db.session.commit()
                
                # Update the content index and publish catalog
                with self._publish_lock:
                    content_id = f"counter-{counter_id}"
                    self._content_index[content_id] = {
                        "ipfs_hash": ipfs_hash,
                        "title": f"Counter-narrative for: {narrative.title}",
                        "type": "counter_narrative",
                        "publication_date": datetime.utcnow().isoformat(),
                        "publisher_id": publisher_id
                    }
                    self._publish_catalog()
                
                return {
                    "success": True,
                    "publication_id": publication.id,
                    "ipfs_hash": ipfs_hash,
                    "title": f"Counter-narrative for: {narrative.title}",
                    "publish_time": publish_time,
                    "publication_date": publication.publication_date.isoformat(),
                    "content_url": f"ipfs://{ipfs_hash}"
                }
            except Exception as e:
                logger.error(f"Error creating publication record: {e}")
                return {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "title": f"Counter-narrative for: {narrative.title}",
                    "publish_time": publish_time,
                    "content_url": f"ipfs://{ipfs_hash}",
                    "warning": "Publication record not created in database"
                }
                
        except Exception as e:
            logger.error(f"Error publishing counter-narrative: {e}")
            return None
    
    def publish_evidence_record(self, evidence_id: str, publisher_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Publish an evidence record to decentralized storage.
        
        Args:
            evidence_id: ID of the evidence record to publish
            publisher_id: ID of the user publishing the content (for verification)
        
        Returns:
            Publication info dictionary or None if failed
        """
        if not self.ipfs_available:
            logger.error("IPFS not available, cannot publish evidence record")
            return None
            
        try:
            # Get the evidence record from the database
            evidence = EvidenceRecord.query.get(evidence_id)
            if not evidence:
                logger.error(f"Evidence record with ID {evidence_id} not found")
                return None
                
            # Prepare publication content
            content = {
                "type": "evidence_record",
                "id": f"evidence-{evidence_id}",
                "hash": evidence.content_hash,
                "source_url": evidence.source_url,
                "capture_date": evidence.capture_date.isoformat() if evidence.capture_date else None,
                "content_type": evidence.content_type,
                "verified": evidence.verified,
                "verification_method": evidence.verification_method,
                "metadata": evidence.get_meta_data(),
                "publisher_id": publisher_id,
                "publication_date": datetime.utcnow().isoformat(),
                "version": "1.0",
            }
            
            # If we have content data, include it
            if evidence.content_data:
                content["content_data"] = evidence.content_data
                
            # Publish to IPFS
            start_time = time.time()
            metadata = {
                "publisher_id": publisher_id,
                "publish_time": datetime.utcnow().isoformat(),
                "content_type": "application/json",
                "schema_version": "1.0",
                "signature": None  # Would include cryptographic signature for verification
            }
            
            ipfs_hash = self._ipfs.add_content(json.dumps(content), metadata=metadata)
            publish_time = time.time() - start_time
            ipfs_latency.set(publish_time)
            
            if not ipfs_hash:
                logger.error(f"Failed to publish evidence record {evidence_id} to IPFS")
                return None
                
            publish_counter.inc()
            logger.info(f"Published evidence record {evidence_id} to IPFS with hash: {ipfs_hash}")
            
            # Create a publication record in the database
            try:
                publication = PublishedContent(
                    content_type="evidence_record",
                    reference_id=evidence_id,
                    ipfs_hash=ipfs_hash,
                    ipns_name=None,  # Individual publications don't get IPNS names
                    publisher_id=publisher_id,
                    status="published",
                    title=f"Evidence: {evidence.source_url}",
                    description=f"Evidence record captured on {evidence.capture_date.isoformat() if evidence.capture_date else 'unknown date'}",
                    publication_date=datetime.utcnow()
                )
                db.session.add(publication)
                db.session.commit()
                
                # Update the content index and publish catalog
                with self._publish_lock:
                    content_id = f"evidence-{evidence_id}"
                    self._content_index[content_id] = {
                        "ipfs_hash": ipfs_hash,
                        "title": f"Evidence: {evidence.source_url}",
                        "type": "evidence_record",
                        "publication_date": datetime.utcnow().isoformat(),
                        "publisher_id": publisher_id
                    }
                    self._publish_catalog()
                
                return {
                    "success": True,
                    "publication_id": publication.id,
                    "ipfs_hash": ipfs_hash,
                    "title": f"Evidence: {evidence.source_url}",
                    "publish_time": publish_time,
                    "publication_date": publication.publication_date.isoformat(),
                    "content_url": f"ipfs://{ipfs_hash}"
                }
            except Exception as e:
                logger.error(f"Error creating publication record: {e}")
                return {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "title": f"Evidence: {evidence.source_url}",
                    "publish_time": publish_time,
                    "content_url": f"ipfs://{ipfs_hash}",
                    "warning": "Publication record not created in database"
                }
                
        except Exception as e:
            logger.error(f"Error publishing evidence record: {e}")
            return None
    
    def publish_source_reliability_analysis(self, source_id: int, publisher_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Publish source reliability analysis to decentralized storage.
        
        Args:
            source_id: ID of the data source to analyze
            publisher_id: ID of the user publishing the content (for verification)
        
        Returns:
            Publication info dictionary or None if failed
        """
        if not self.ipfs_available:
            logger.error("IPFS not available, cannot publish source reliability analysis")
            return None
            
        try:
            # Get the data source from the database
            source = DataSource.query.get(source_id)
            if not source:
                logger.error(f"Data source with ID {source_id} not found")
                return None
                
            # Get misinformation events for this source
            misinfo_events = source.misinfo_events
            
            # Prepare publication content
            content = {
                "type": "source_reliability_analysis",
                "id": f"source-reliability-{source_id}",
                "source_name": source.name,
                "source_type": source.source_type,
                "is_active": source.is_active,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "last_ingestion": source.last_ingestion.isoformat() if source.last_ingestion else None,
                "metadata": source.get_meta_data(),
                "event_count": len(misinfo_events),
                "events": [],
                "publisher_id": publisher_id,
                "publication_date": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
            
            # Add misinformation events
            for event in misinfo_events:
                event_data = {
                    "id": event.id,
                    "narrative_id": event.narrative_id,
                    "narrative_title": event.narrative.title if event.narrative else None,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    "confidence": event.confidence,
                    "correct_detection": event.correct_detection,
                    "evaluation_date": event.evaluation_date.isoformat() if event.evaluation_date else None,
                    "notes": event.notes,
                    "metadata": json.loads(event.meta_data) if event.meta_data else {}
                }
                content["events"].append(event_data)
            
            # Add reliability metrics
            # Calculate overall reliability score based on events
            if misinfo_events:
                reliability_score = sum(1 for e in misinfo_events if e.correct_detection) / len(misinfo_events)
                content["reliability_score"] = reliability_score
            else:
                content["reliability_score"] = None
            
            # Publish to IPFS
            start_time = time.time()
            metadata = {
                "publisher_id": publisher_id,
                "publish_time": datetime.utcnow().isoformat(),
                "content_type": "application/json",
                "schema_version": "1.0",
                "signature": None  # Would include cryptographic signature for verification
            }
            
            ipfs_hash = self._ipfs.add_content(json.dumps(content), metadata=metadata)
            publish_time = time.time() - start_time
            ipfs_latency.set(publish_time)
            
            if not ipfs_hash:
                logger.error(f"Failed to publish source reliability analysis for source {source_id} to IPFS")
                return None
                
            publish_counter.inc()
            logger.info(f"Published source reliability analysis for source {source_id} to IPFS with hash: {ipfs_hash}")
            
            # Create a publication record in the database
            try:
                publication = PublishedContent(
                    content_type="source_reliability_analysis",
                    reference_id=source_id,
                    ipfs_hash=ipfs_hash,
                    ipns_name=None,  # Individual publications don't get IPNS names
                    publisher_id=publisher_id,
                    status="published",
                    title=f"Source Reliability: {source.name}",
                    description=f"Reliability analysis for source: {source.name} ({source.source_type})",
                    publication_date=datetime.utcnow()
                )
                db.session.add(publication)
                db.session.commit()
                
                # Update the content index and publish catalog
                with self._publish_lock:
                    content_id = f"source-reliability-{source_id}"
                    self._content_index[content_id] = {
                        "ipfs_hash": ipfs_hash,
                        "title": f"Source Reliability: {source.name}",
                        "type": "source_reliability_analysis",
                        "publication_date": datetime.utcnow().isoformat(),
                        "publisher_id": publisher_id
                    }
                    self._publish_catalog()
                
                return {
                    "success": True,
                    "publication_id": publication.id,
                    "ipfs_hash": ipfs_hash,
                    "title": f"Source Reliability: {source.name}",
                    "publish_time": publish_time,
                    "publication_date": publication.publication_date.isoformat(),
                    "content_url": f"ipfs://{ipfs_hash}"
                }
            except Exception as e:
                logger.error(f"Error creating publication record: {e}")
                return {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "title": f"Source Reliability: {source.name}",
                    "publish_time": publish_time,
                    "content_url": f"ipfs://{ipfs_hash}",
                    "warning": "Publication record not created in database"
                }
                
        except Exception as e:
            logger.error(f"Error publishing source reliability analysis: {e}")
            return None
    
    def verify_publisher(self, user_id: str) -> bool:
        """
        Verify a user as a trusted publisher.
        
        Args:
            user_id: ID of the user to verify
        
        Returns:
            Whether verification was successful
        """
        try:
            user = User.query.get(user_id)
            if not user:
                logger.error(f"User with ID {user_id} not found")
                return False
                
            # For now, just add to the verified publishers set
            # In a real implementation, would check credentials and potentially use a verification service
            self._verified_publishers.add(user_id)
            verification_counter.inc()
            logger.info(f"Verified publisher: {user.username} ({user_id})")
            return True
        except Exception as e:
            logger.error(f"Error verifying publisher: {e}")
            return False
    
    def is_verified_publisher(self, user_id: str) -> bool:
        """
        Check if a user is a verified publisher.
        
        Args:
            user_id: ID of the user to check
        
        Returns:
            Whether the user is a verified publisher
        """
        return user_id in self._verified_publishers
    
    def get_publication(self, ipfs_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get a publication from IPFS by its hash.
        
        Args:
            ipfs_hash: IPFS hash of the publication
        
        Returns:
            Publication content or None if not found
        """
        if not self.ipfs_available:
            logger.error("IPFS not available, cannot get publication")
            return None
            
        try:
            content = self._ipfs.get_content(ipfs_hash)
            return content
        except Exception as e:
            logger.error(f"Error getting publication: {e}")
            return None
    
    def get_recent_publications(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent publications from the database.
        
        Args:
            limit: Maximum number of publications to return
        
        Returns:
            List of publication info dictionaries
        """
        try:
            publications = (
                PublishedContent.query
                .order_by(PublishedContent.publication_date.desc())
                .limit(limit)
                .all()
            )
            
            result = []
            for pub in publications:
                pub_info = {
                    "id": pub.id,
                    "content_type": pub.content_type,
                    "reference_id": pub.reference_id,
                    "ipfs_hash": pub.ipfs_hash,
                    "ipns_name": pub.ipns_name,
                    "publisher_id": pub.publisher_id,
                    "status": pub.status,
                    "title": pub.title,
                    "description": pub.description,
                    "publication_date": pub.publication_date.isoformat() if pub.publication_date else None,
                    "content_url": f"ipfs://{pub.ipfs_hash}"
                }
                result.append(pub_info)
                
            return result
        except Exception as e:
            logger.error(f"Error getting recent publications: {e}")
            return []
    
    def update_content_catalog(self) -> Optional[str]:
        """
        Update the content catalog in IPFS/IPNS.
        
        Returns:
            IPFS hash of the updated catalog or None if failed
        """
        if not self.ipfs_available or not self._catalog_ipns:
            logger.error("IPFS or IPNS catalog not available, cannot update catalog")
            return None
            
        try:
            with self._publish_lock:
                return self._publish_catalog()
        except Exception as e:
            logger.error(f"Error updating content catalog: {e}")
            return None
    
    def _publish_catalog(self) -> Optional[str]:
        """
        Internal method to publish the content catalog to IPFS and update IPNS.
        
        Returns:
            IPFS hash of the catalog or None if failed
        """
        if not self.ipfs_available or not self._catalog_ipns:
            return None
            
        try:
            # Prepare catalog content
            catalog = {
                "type": "content_catalog",
                "updated_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "content_index": self._content_index
            }
            
            # Publish to IPFS
            ipfs_hash = self._ipfs.add_content(json.dumps(catalog))
            if not ipfs_hash:
                logger.error("Failed to publish content catalog to IPFS")
                return None
                
            # Update IPNS to point to the new catalog
            ipns_name = self._ipfs.publish_to_ipns(ipfs_hash, key_name="civilian-catalog")
            if not ipns_name:
                logger.error("Failed to update IPNS for content catalog")
                return ipfs_hash  # Return hash even if IPNS update failed
                
            self._catalog_hash = ipfs_hash
            ipns_updates.inc()
            logger.info(f"Updated content catalog in IPFS ({ipfs_hash}) and IPNS ({ipns_name})")
            return ipfs_hash
        except Exception as e:
            logger.error(f"Error publishing content catalog: {e}")
            return None