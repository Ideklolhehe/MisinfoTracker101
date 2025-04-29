"""
Web source manager for the CIVILIAN system.
This module coordinates web data sources and job execution.
"""

import os
import json
import logging
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from queue import Queue, Empty
from urllib.parse import urlparse

# Import Flask and SQLAlchemy from app
from app import db
import models

# Import data source classes
from data_sources.source_base import (
    SourceBase, WebPageSource, WebCrawlSource, WebSearchSource
)

# Import web scraping utilities
from utils.web_scraper import WebScraper
from utils.data_scaling import DataScaler

# Configure logger
logger = logging.getLogger(__name__)


class WebSourceJob:
    """A job for processing a web source."""
    
    def __init__(self, job_id: str, source_type: str, config: Dict[str, Any]):
        """
        Initialize a web source job.
        
        Args:
            job_id: Unique identifier for this job
            source_type: Type of web source to use
            config: Configuration for the source
        """
        self.job_id = job_id
        self.source_type = source_type
        self.config = config
        self.status = "pending"
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            'job_id': self.job_id,
            'source_type': self.source_type,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'config': self.config,
            'error': self.error
        }


class WebSourceManager:
    """Manager for web data sources in the CIVILIAN system."""
    
    def __init__(self):
        """Initialize the web source manager."""
        self.active_sources = {}  # source_id -> SourceBase
        self.job_queue = Queue()
        self.jobs = {}  # job_id -> WebSourceJob
        self.scraper = WebScraper()
        self.is_running = False
        self.worker_thread = None
        self.lock = threading.RLock()
    
    def start(self):
        """Start the web source manager."""
        with self.lock:
            if self.is_running:
                logger.warning("Web source manager is already running")
                return
            
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
            self.worker_thread.start()
            
            logger.info("Web source manager started")
    
    def stop(self):
        """Stop the web source manager."""
        with self.lock:
            if not self.is_running:
                logger.warning("Web source manager is not running")
                return
            
            self.is_running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5.0)
                self.worker_thread = None
            
            logger.info("Web source manager stopped")
    
    def _process_jobs(self):
        """Process jobs from the queue."""
        while self.is_running:
            try:
                # Get a job from the queue with timeout
                try:
                    job_id = self.job_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Get the job
                job = self.jobs.get(job_id)
                if not job:
                    logger.warning(f"Job {job_id} not found")
                    continue
                
                # Process the job
                self._process_job(job)
                
                # Mark the job as done
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing job queue: {e}")
                time.sleep(1.0)  # Sleep to avoid tight loop
    
    def _process_job(self, job: WebSourceJob):
        """
        Process a web source job.
        
        Args:
            job: The job to process
        """
        try:
            logger.info(f"Processing job {job.job_id} of type {job.source_type}")
            
            # Update job status
            job.status = "running"
            job.started_at = datetime.now()
            
            # Create source for the job
            source = None
            if job.source_type == 'single':
                source = WebPageSource(name=f"Job {job.job_id}", config=job.config)
            elif job.source_type == 'crawl':
                source = WebCrawlSource(name=f"Job {job.job_id}", config=job.config)
            elif job.source_type == 'search':
                source = WebSearchSource(name=f"Job {job.job_id}", config=job.config)
            else:
                raise ValueError(f"Unknown source type: {job.source_type}")
            
            # Run the source
            success = source.run()
            
            if success:
                # Get content items and stats
                content_items, stats = source.process()
                
                # Process content items
                processed_items = []
                for item in content_items:
                    processed = DataScaler.prepare_web_content(item)
                    processed_items.append(processed)
                
                # Store in job result
                job.result = {
                    'success': True,
                    'results': processed_items,
                    'stats': stats
                }
                job.status = "completed"
            else:
                # Store error in job result
                job.result = {
                    'success': False,
                    'error': source.status_message,
                    'stats': {}
                }
                job.error = source.status_message
                job.status = "error"
            
            # Update job completion time
            job.completed_at = datetime.now()
            
            logger.info(f"Job {job.job_id} completed with status {job.status}")
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {e}")
            job.status = "error"
            job.error = str(e)
            job.completed_at = datetime.now()
    
    def register_source(self, source_data: Dict[str, Any]) -> Optional[int]:
        """
        Register a new web source for monitoring.
        
        Args:
            source_data: Dictionary containing source configuration
            
        Returns:
            Source ID if registration was successful, None otherwise
        """
        try:
            # Validate required fields
            for field in ['name', 'url', 'source_type']:
                if field not in source_data:
                    logger.error(f"Missing required field: {field}")
                    return None
            
            # Parse config
            config = source_data.get('config', {})
            if isinstance(config, str):
                config = json.loads(config)
            
            # Ensure URL is in config
            if 'url' not in config:
                config['url'] = source_data['url']
            
            # Create source model
            with db.session.begin():
                source = models.WebSource(
                    name=source_data['name'],
                    url=source_data['url'],
                    source_type=source_data['source_type'],
                    is_active=source_data.get('is_active', True),
                    config=json.dumps(config),
                    meta_data='{}',
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                db.session.add(source)
                db.session.commit()
                
                logger.info(f"Registered new web source: {source.name} (ID: {source.id})")
                
                return source.id
                
        except Exception as e:
            logger.error(f"Error registering web source: {e}")
            return None
    
    def update_source_status(self, source_id: int, is_active: bool) -> bool:
        """
        Update the active status of a web source.
        
        Args:
            source_id: ID of the source to update
            is_active: New active status
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with db.session.begin():
                source = db.session.query(models.WebSource).get(source_id)
                
                if not source:
                    logger.error(f"Source with ID {source_id} not found")
                    return False
                
                source.is_active = is_active
                source.updated_at = datetime.now()
                
                db.session.commit()
                
                logger.info(f"Updated source {source.name} (ID: {source_id}) status to {is_active}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating source status: {e}")
            return False
    
    def get_source_by_id(self, source_id: int) -> Optional[models.WebSource]:
        """
        Get a web source by ID.
        
        Args:
            source_id: ID of the source to get
            
        Returns:
            WebSource model if found, None otherwise
        """
        try:
            return db.session.query(models.WebSource).get(source_id)
        except Exception as e:
            logger.error(f"Error getting source {source_id}: {e}")
            return None
    
    def get_active_sources(self) -> List[models.WebSource]:
        """
        Get all active web sources.
        
        Returns:
            List of active WebSource models
        """
        try:
            return db.session.query(models.WebSource).filter_by(is_active=True).all()
        except Exception as e:
            logger.error(f"Error getting active sources: {e}")
            return []
    
    def run_source(self, source_id: int) -> bool:
        """
        Run a web source.
        
        Args:
            source_id: ID of the source to run
            
        Returns:
            True if run was successful, False otherwise
        """
        try:
            source_model = self.get_source_by_id(source_id)
            
            if not source_model:
                logger.error(f"Source with ID {source_id} not found")
                return False
            
            if not source_model.is_active:
                logger.warning(f"Source {source_model.name} (ID: {source_id}) is not active")
                return False
            
            # Create source object based on type
            source = None
            config = json.loads(source_model.config) if source_model.config else {}
            
            if source_model.source_type == 'web_page':
                source = WebPageSource(
                    source_id=source_model.id,
                    name=source_model.name,
                    config=config,
                    is_active=source_model.is_active
                )
            elif source_model.source_type == 'web_crawl':
                source = WebCrawlSource(
                    source_id=source_model.id,
                    name=source_model.name,
                    config=config,
                    is_active=source_model.is_active
                )
            elif source_model.source_type == 'web_search':
                source = WebSearchSource(
                    source_id=source_model.id,
                    name=source_model.name,
                    config=config,
                    is_active=source_model.is_active
                )
            else:
                logger.error(f"Unknown source type: {source_model.source_type}")
                return False
            
            # Run the source
            success = source.run()
            
            # Update source metadata in database
            if hasattr(source, 'meta_data') and source.meta_data:
                if isinstance(source.meta_data, dict):
                    meta_data = json.dumps(source.meta_data)
                else:
                    meta_data = source.meta_data
                    
                with db.session.begin():
                    source_model.meta_data = meta_data
                    source_model.last_ingestion = datetime.now()
                    source_model.updated_at = datetime.now()
                    db.session.commit()
            
            return success
            
        except Exception as e:
            logger.error(f"Error running source {source_id}: {e}")
            return False
    
    def run_all_sources(self) -> Dict[int, bool]:
        """
        Run all active web sources.
        
        Returns:
            Dictionary mapping source IDs to success status
        """
        results = {}
        
        try:
            sources = self.get_active_sources()
            
            for source in sources:
                results[source.id] = self.run_source(source.id)
                
            return results
            
        except Exception as e:
            logger.error(f"Error running all sources: {e}")
            return results
    
    def add_url_job(self, url: str, job_type: str = 'single', config: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a job to process a URL.
        
        Args:
            url: URL to process
            job_type: Type of job ('single', 'crawl', or 'search')
            config: Additional configuration for the job
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        # Create config with URL
        job_config = config or {}
        job_config['url'] = url
        
        # Add domain from URL
        parsed_url = urlparse(url)
        job_config['domain'] = parsed_url.netloc
        
        # Create job
        job = WebSourceJob(job_id, job_type, job_config)
        
        # Add job to queue and jobs dictionary
        with self.lock:
            self.jobs[job_id] = job
            self.job_queue.put(job_id)
        
        # Start processing if not already running
        if not self.is_running:
            self.start()
        
        logger.info(f"Added URL job {job_id} of type {job_type} for {url}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a job.
        
        Args:
            job_id: ID of the job to get status for
            
        Returns:
            Dictionary containing job status
        """
        with self.lock:
            job = self.jobs.get(job_id)
            
            if not job:
                return {
                    'job_id': job_id,
                    'status': 'not_found',
                    'error': 'Job not found'
                }
            
            result = job.to_dict()
            
            # Only include result data if job is completed
            if job.status == 'completed' and job.result:
                # Limit result data
                if 'results' in job.result:
                    result_count = len(job.result['results'])
                    result['result_count'] = result_count
                    result['results'] = job.result['results']
                
                # Include stats
                if 'stats' in job.result:
                    result['stats'] = job.result['stats']
            
            return result
    
    def create_source_from_job(self, job_id: str, name: Optional[str] = None) -> Optional[int]:
        """
        Create a persistent web source from a completed job.
        
        Args:
            job_id: ID of the completed job
            name: Name for the new source
            
        Returns:
            Source ID if creation was successful, None otherwise
        """
        with self.lock:
            job = self.jobs.get(job_id)
            
            if not job:
                logger.error(f"Job {job_id} not found")
                return None
            
            if job.status != 'completed':
                logger.error(f"Job {job_id} is not completed")
                return None
            
            # Map job type to source type
            source_type_map = {
                'single': 'web_page',
                'crawl': 'web_crawl',
                'search': 'web_search'
            }
            
            source_type = source_type_map.get(job.source_type, 'web_page')
            
            # Use URL from config for name if not provided
            if not name:
                url = job.config.get('url', '')
                name = f"{source_type.capitalize()}: {url}"
            
            # Create source data
            source_data = {
                'name': name,
                'url': job.config.get('url', ''),
                'source_type': source_type,
                'is_active': True,
                'config': job.config
            }
            
            # Register source
            source_id = self.register_source(source_data)
            
            return source_id
    
    def clean_old_jobs(self, max_age_hours: int = 24):
        """
        Clean up old completed jobs.
        
        Args:
            max_age_hours: Maximum age of jobs to keep in hours
        """
        now = datetime.now()
        jobs_to_remove = []
        
        with self.lock:
            for job_id, job in self.jobs.items():
                if job.completed_at:
                    age = now - job.completed_at
                    if age.total_seconds() > max_age_hours * 3600:
                        jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


# Create singleton instance
web_source_manager = WebSourceManager()