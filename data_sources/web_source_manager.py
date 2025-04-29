"""
Web Source Manager for the CIVILIAN system.
This module handles the integration of web scraping capabilities with the system's
data collection and storage infrastructure.
"""

import logging
import time
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import hashlib
import os
import csv
import concurrent.futures

from flask import current_app
from sqlalchemy import func

from app import db
from models import DataSource, NarrativeInstance, DetectedNarrative
from utils.web_scraper import (get_website_content, crawl_website, search_for_content,
                          process_urls_in_batches)
from utils.app_context import with_app_context
from data_sources.source_base import DataSourceBase

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MAX_CONCURRENT_JOBS = 5  # Maximum number of concurrent scraping jobs
DEFAULT_SEARCH_ENGINE = "bing"
CACHE_DIR = "./storage/web_cache"
DEFAULT_DOMAIN_RATE_LIMITS = {
    "twitter.com": 10,  # Seconds between requests
    "facebook.com": 15,
    "instagram.com": 15,
    "reddit.com": 5,
    "youtube.com": 10,
    "gov": 3,  # For .gov domains
    "edu": 3,  # For .edu domains
}


class WebSourceManager(DataSourceBase):
    """
    Manages web sources for the CIVILIAN system, integrating web scraping
    with the system's data storage and analysis pipeline.
    """
    
    def __init__(self):
        """Initialize the web source manager."""
        super().__init__("web")
        
        self.job_queue = queue.Queue()
        self.active_jobs = set()
        self.completed_jobs = set()
        self.active_jobs_lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Start worker threads
        self.worker_threads = []
        for _ in range(MAX_CONCURRENT_JOBS):
            worker = threading.Thread(target=self._worker_thread, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
            
        logger.info(f"WebSourceManager initialized with {MAX_CONCURRENT_JOBS} worker threads")
    
    def register_source(self, source_data: Dict[str, Any]) -> Optional[int]:
        """
        Register a new web source in the database.
        
        Args:
            source_data: Dictionary containing source information
                Requires:
                - name: Source name
                - url: Base URL
                - source_type: Type of web source (news, social, academic, etc.)
                - config: Configuration for the source (JSON string)
                
        Returns:
            ID of the registered source, or None if registration failed
        """
        try:
            # Validate required fields
            required_fields = ["name", "url", "source_type"]
            for field in required_fields:
                if field not in source_data:
                    logger.error(f"Missing required field '{field}' for web source registration")
                    return None
            
            # Create configuration
            config = source_data.get("config", {})
            if isinstance(config, dict):
                config_str = json.dumps(config)
            else:
                config_str = config
            
            # Create metadata
            meta_data = source_data.get("meta_data", {})
            if isinstance(meta_data, dict):
                meta_data_str = json.dumps(meta_data)
            else:
                meta_data_str = meta_data
            
            # Create new source
            new_source = DataSource(
                name=source_data["name"],
                source_type=f"web_{source_data['source_type']}",
                config=config_str,
                is_active=source_data.get("is_active", True),
                meta_data=meta_data_str
            )
            
            db.session.add(new_source)
            db.session.commit()
            
            logger.info(f"Registered new web source: {source_data['name']} (ID: {new_source.id})")
            return new_source.id
            
        except Exception as e:
            logger.error(f"Error registering web source: {e}")
            db.session.rollback()
            return None
    
    @with_app_context
    def process_sources(self):
        """
        Process all active web sources for content ingestion.
        This method is typically called by a scheduled job.
        """
        try:
            # Get all active web sources
            sources = DataSource.query.filter(
                DataSource.is_active == True,
                DataSource.source_type.like("web_%")
            ).all()
            
            if not sources:
                logger.info("No active web sources to process")
                return
                
            logger.info(f"Processing {len(sources)} active web sources")
            
            # Process each source
            for source in sources:
                self._process_source(source)
                
            logger.info(f"Completed processing {len(sources)} web sources")
            
        except Exception as e:
            logger.error(f"Error processing web sources: {e}")
    
    @with_app_context
    def _process_source(self, source: DataSource):
        """
        Process a single web source and ingest its content.
        
        Args:
            source: DataSource object to process
        """
        try:
            # Parse configuration
            config = json.loads(source.config) if source.config else {}
            
            # Get source URL and type
            source_url = config.get("url", "")
            source_type = source.source_type.split("_")[1] if "_" in source.source_type else "unknown"
            
            if not source_url:
                logger.warning(f"Missing URL for source {source.name} (ID: {source.id})")
                return
                
            logger.info(f"Processing web source: {source.name} (URL: {source_url})")
            
            # Determine processing strategy based on source type
            if source_type == "news":
                # Use crawling for news sites
                max_pages = config.get("max_pages", 10)
                results = crawl_website(source_url, max_pages=max_pages)
                
            elif source_type == "search":
                # Use search for search-based sources
                search_term = config.get("search_term", "")
                search_engine = config.get("search_engine", DEFAULT_SEARCH_ENGINE)
                limit = config.get("limit", 10)
                
                if not search_term:
                    logger.warning(f"Missing search term for search source {source.name}")
                    return
                    
                urls = search_for_content(search_term, search_engine=search_engine, limit=limit)
                results = process_urls_in_batches(urls)
                
            elif source_type == "monitor":
                # Single URL monitoring
                result = get_website_content(source_url)
                results = [result] if result.get("success") else []
                
            else:
                # Default to single URL processing
                result = get_website_content(source_url)
                results = [result] if result.get("success") else []
            
            # Process results
            for result in results:
                if result.get("success"):
                    self._process_result(result, source)
                    
            # Update source last ingestion timestamp
            source.last_ingestion = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Processed {len(results)} pages from source {source.name}")
            
        except Exception as e:
            logger.error(f"Error processing source {source.name}: {e}")
    
    @with_app_context
    def _process_result(self, result: Dict[str, Any], source: DataSource):
        """
        Process a single scraping result and store in the database.
        
        Args:
            result: Dictionary containing scraped content
            source: DataSource that produced the result
        """
        try:
            content = result.get("content", "")
            if not content:
                return
                
            # Check if content already exists (deduplication)
            content_hash = result.get("content_hash") or hashlib.md5(content.encode('utf-8')).hexdigest()
            
            existing = NarrativeInstance.query.filter_by(
                evidence_hash=content_hash
            ).first()
            
            if existing:
                logger.debug(f"Content already exists in database, skipping (hash: {content_hash})")
                return
                
            # Extract metadata
            metadata = result.get("metadata", {})
            title = metadata.get("title", "Untitled Content")
            url = metadata.get("url", result.get("url", ""))
            
            # Determine if this is part of an existing narrative or a new one
            # In a real system, this would use more sophisticated clustering or classification
            # For now, use a simple query to find semantically similar content
            existing_narrative = self._find_related_narrative(content, title)
            
            if existing_narrative:
                # Add as instance to existing narrative
                narrative_id = existing_narrative.id
                logger.debug(f"Adding to existing narrative: {narrative_id}")
            else:
                # Create new narrative
                new_narrative = DetectedNarrative(
                    title=title,
                    description=f"Content from {source.name}: {title}",
                    status="active",
                    detection_method="web_scraping",
                    meta_data=json.dumps({
                        "origin": "web_scraping",
                        "source": source.name,
                        "threat_level": 1,  # Default starting threat level
                        "propagation_rate": 0.1,  # Default starting propagation rate
                        "complexity_score": 0.5,  # Default complexity score
                        "detection_confidence": 1.0  # High confidence since direct source
                    })
                )
                
                db.session.add(new_narrative)
                db.session.commit()
                
                narrative_id = new_narrative.id
                logger.debug(f"Created new narrative: {narrative_id}")
            
            # Create narrative instance
            instance = NarrativeInstance(
                narrative_id=narrative_id,
                source_id=source.id,
                content=content,
                url=url,
                detected_at=datetime.utcnow(),
                evidence_hash=content_hash,
                meta_data=json.dumps(metadata)
            )
            
            db.session.add(instance)
            db.session.commit()
            
            logger.debug(f"Created new narrative instance from {url}")
            
        except Exception as e:
            logger.error(f"Error processing result: {e}")
            db.session.rollback()
    
    @with_app_context
    def _find_related_narrative(self, content: str, title: str) -> Optional[DetectedNarrative]:
        """
        Find a related narrative for the content based on semantic similarity.
        
        Args:
            content: Content text to match
            title: Title of the content
            
        Returns:
            Related DetectedNarrative or None if no match found
        """
        try:
            # For a simplistic approach, search for narratives with similar titles
            # In a real implementation, this would use embeddings or other NLP techniques
            
            # First look for exact title matches (minus common words)
            title_words = set(title.lower().split())
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
            title_keywords = title_words - stop_words
            
            if not title_keywords:
                return None
                
            # Query for narratives with similar titles
            candidates = DetectedNarrative.query.filter(
                DetectedNarrative.status == "active"
            ).order_by(
                DetectedNarrative.created_at.desc()
            ).limit(50).all()
            
            # Score candidates by keyword overlap
            best_match = None
            best_score = 0
            
            for narrative in candidates:
                narrative_words = set(narrative.title.lower().split()) - stop_words
                if not narrative_words:
                    continue
                    
                # Calculate Jaccard similarity
                overlap = len(title_keywords.intersection(narrative_words))
                union = len(title_keywords.union(narrative_words))
                
                if union > 0:
                    score = overlap / union
                    if score > best_score and score > 0.3:  # threshold for similarity
                        best_score = score
                        best_match = narrative
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding related narrative: {e}")
            return None
    
    def add_url_job(self, url: str, job_type: str = "single", params: Dict[str, Any] = None) -> str:
        """
        Add a URL processing job to the queue.
        
        Args:
            url: URL to process
            job_type: Type of job ('single', 'crawl', 'search')
            params: Additional parameters for the job
            
        Returns:
            Job ID for tracking
        """
        if not params:
            params = {}
            
        job_id = hashlib.md5(f"{url}_{job_type}_{time.time()}".encode('utf-8')).hexdigest()
        
        job = {
            "id": job_id,
            "url": url,
            "type": job_type,
            "params": params,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "result": None
        }
        
        self.job_queue.put(job)
        logger.info(f"Added {job_type} job for {url} with ID {job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Job status dictionary
        """
        with self.active_jobs_lock:
            # Check active jobs
            for job in self.active_jobs:
                if job["id"] == job_id:
                    return {
                        "id": job["id"],
                        "status": job["status"],
                        "created_at": job["created_at"],
                        "completed_at": job.get("completed_at")
                    }
            
            # Check completed jobs
            for job in self.completed_jobs:
                if job["id"] == job_id:
                    return {
                        "id": job["id"],
                        "status": job["status"],
                        "created_at": job["created_at"],
                        "completed_at": job.get("completed_at")
                    }
        
        # Job not found
        return {
            "id": job_id,
            "status": "not_found",
            "error": "Job not found"
        }
    
    def _worker_thread(self):
        """Worker thread that processes jobs from the queue."""
        while True:
            try:
                # Get job from queue
                job = self.job_queue.get()
                
                # Process job
                with self.active_jobs_lock:
                    self.active_jobs.add(job)
                    
                try:
                    logger.info(f"Processing job {job['id']}: {job['type']} for {job['url']}")
                    job["status"] = "processing"
                    
                    if job["type"] == "single":
                        result = get_website_content(job["url"])
                    elif job["type"] == "crawl":
                        max_pages = job["params"].get("max_pages", 10)
                        same_domain = job["params"].get("same_domain_only", True)
                        result = crawl_website(job["url"], max_pages=max_pages, same_domain_only=same_domain)
                    elif job["type"] == "search":
                        search_term = job["params"].get("search_term", "")
                        search_engine = job["params"].get("search_engine", DEFAULT_SEARCH_ENGINE)
                        limit = job["params"].get("limit", 10)
                        urls = search_for_content(search_term, search_engine=search_engine, limit=limit)
                        result = process_urls_in_batches(urls)
                    else:
                        logger.error(f"Unknown job type: {job['type']}")
                        job["status"] = "error"
                        job["error"] = f"Unknown job type: {job['type']}"
                        continue
                        
                    # Store result
                    job["result"] = result
                    job["status"] = "completed"
                    job["completed_at"] = datetime.utcnow().isoformat()
                    
                    # Handle results within app context if needed
                    self._handle_job_result(job)
                    
                except Exception as e:
                    logger.error(f"Error processing job {job['id']}: {e}")
                    job["status"] = "error"
                    job["error"] = str(e)
                
                finally:
                    # Remove from active jobs and add to completed
                    with self.active_jobs_lock:
                        if job in self.active_jobs:
                            self.active_jobs.remove(job)
                            self.completed_jobs.add(job)
                            
                            # Limit completed jobs
                            if len(self.completed_jobs) > 100:
                                oldest = sorted(self.completed_jobs, key=lambda j: j["created_at"])[0]
                                self.completed_jobs.remove(oldest)
                    
                    # Mark job as done
                    self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")
                time.sleep(1)  # Prevent spinning
    
    @with_app_context
    def _handle_job_result(self, job: Dict[str, Any]):
        """
        Handle the result of a completed job.
        
        Args:
            job: Completed job dictionary
        """
        try:
            result = job.get("result")
            if not result:
                return
                
            # Handle different job types
            if job["type"] == "single":
                # Single page result
                if result.get("success"):
                    # Find or create source
                    source_name = result.get("metadata", {}).get("title", "Web Source")
                    
                    source = DataSource.query.filter_by(
                        name=source_name,
                        source_type="web_monitor"
                    ).first()
                    
                    if not source:
                        source = DataSource(
                            name=source_name,
                            source_type="web_monitor",
                            config=json.dumps({"url": job["url"]}),
                            is_active=True,
                            meta_data=json.dumps({"origin": "manual_job"})
                        )
                        db.session.add(source)
                        db.session.commit()
                    
                    # Process the result
                    self._process_result(result, source)
                    
            elif job["type"] in ["crawl", "search"]:
                # Multiple page results
                if not result:
                    return
                    
                # Find or create source
                source_name = job["params"].get("source_name", f"Web Source: {job['url']}")
                
                source = DataSource.query.filter_by(
                    name=source_name,
                    source_type=f"web_{job['type']}"
                ).first()
                
                if not source:
                    source = DataSource(
                        name=source_name,
                        source_type=f"web_{job['type']}",
                        config=json.dumps({"url": job["url"], **job["params"]}),
                        is_active=True,
                        meta_data=json.dumps({"origin": "manual_job"})
                    )
                    db.session.add(source)
                    db.session.commit()
                
                # Process each result
                for page_result in result:
                    if page_result.get("success"):
                        self._process_result(page_result, source)
        
        except Exception as e:
            logger.error(f"Error handling job result: {e}")

# Create a singleton instance
web_source_manager = WebSourceManager()