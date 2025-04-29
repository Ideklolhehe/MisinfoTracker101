"""
Web scraping service for the CIVILIAN system.
This service provides functions for web scraping, data collection, and monitoring.
"""

import os
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urlparse

# Import from app
from app import db
import models

# Import utilities
from utils.web_scraper import WebScraper
from utils.data_scaling import DataScaler
from data_sources.web_source_manager import web_source_manager
from data_sources.source_base import WebPageSource, WebCrawlSource, WebSearchSource

# Configure logger
logger = logging.getLogger(__name__)


class WebScrapingService:
    """Service for web scraping and data collection."""
    
    def __init__(self):
        """Initialize the web scraping service."""
        self.scraper = WebScraper()
        self.is_running = False
        self.scheduler_thread = None
        self.lock = threading.RLock()
        
        # Monitoring configuration
        self.focused_domains = []  # List of domains to focus on
        self.search_terms = []  # List of search terms to monitor
        self.monitoring_interval = 3600  # Default to hourly checks
        
        # Load configuration if available
        self._load_configuration()
    
    def _load_configuration(self):
        """Load monitoring configuration from the database."""
        try:
            # Load focused domains
            domains = db.session.query(models.FocusedDomain).filter_by(is_active=True).all()
            self.focused_domains = [
                {
                    'id': domain.id,
                    'domain': domain.domain,
                    'category': domain.category,
                    'priority': domain.priority,
                    'last_check': domain.last_check.isoformat() if domain.last_check else None
                }
                for domain in domains
            ]
            
            # Load search terms
            terms = db.session.query(models.SearchTerm).filter_by(is_active=True).all()
            self.search_terms = [
                {
                    'id': term.id,
                    'term': term.term,
                    'category': term.category,
                    'last_check': term.last_check.isoformat() if term.last_check else None
                }
                for term in terms
            ]
            
            logger.info(f"Loaded {len(self.focused_domains)} domains and {len(self.search_terms)} search terms")
            
        except Exception as e:
            logger.error(f"Error loading monitoring configuration: {e}")
    
    def start_scheduled_scraping(self):
        """Start scheduled web scraping and monitoring."""
        with self.lock:
            if self.is_running:
                logger.warning("Web scraping service is already running")
                return
            
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("Web scraping service started")
    
    def stop_scheduled_scraping(self):
        """Stop scheduled web scraping and monitoring."""
        with self.lock:
            if not self.is_running:
                logger.warning("Web scraping service is not running")
                return
            
            self.is_running = False
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5.0)
                self.scheduler_thread = None
            
            logger.info("Web scraping service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for scheduled scraping."""
        while self.is_running:
            try:
                # Run domain monitoring
                self._monitor_domains()
                
                # Run search term monitoring
                self._monitor_search_terms()
                
                # Run registered web sources
                web_source_manager.run_all_sources()
                
                # Sleep until next cycle
                for _ in range(self.monitoring_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep to avoid tight loop
    
    def _monitor_domains(self):
        """Monitor focused domains."""
        for domain_info in self.focused_domains:
            try:
                if not self.is_running:
                    break
                
                domain = domain_info['domain']
                logger.info(f"Monitoring domain: {domain}")
                
                # Create URL if needed
                url = f"https://{domain}" if not domain.startswith(('http://', 'https://')) else domain
                
                # Create job
                max_pages = 5 if domain_info['priority'] <= 2 else 1
                job_id = web_source_manager.add_url_job(
                    url=url,
                    job_type='crawl',
                    config={
                        'max_pages': max_pages,
                        'same_domain_only': True,
                        'source_name': f"Domain: {domain}",
                        'category': domain_info.get('category', 'general')
                    }
                )
                
                # Update last check time
                try:
                    with db.session.begin():
                        domain_model = db.session.query(models.FocusedDomain).get(domain_info['id'])
                        if domain_model:
                            domain_model.last_check = datetime.now()
                            db.session.commit()
                except Exception as e:
                    logger.error(f"Error updating domain last check time: {e}")
                
                # Add delay between domains
                time.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error monitoring domain {domain_info.get('domain')}: {e}")
    
    def _monitor_search_terms(self):
        """Monitor search terms."""
        for term_info in self.search_terms:
            try:
                if not self.is_running:
                    break
                
                term = term_info['term']
                logger.info(f"Monitoring search term: {term}")
                
                # Add search job
                job_id = web_source_manager.add_url_job(
                    url="https://www.bing.com/search",  # Will be replaced by actual search URL
                    job_type='search',
                    config={
                        'search_term': term,
                        'search_engine': 'bing',
                        'limit': 10,
                        'source_name': f"Search: {term}",
                        'category': term_info.get('category', 'general')
                    }
                )
                
                # Update last check time
                try:
                    with db.session.begin():
                        term_model = db.session.query(models.SearchTerm).get(term_info['id'])
                        if term_model:
                            term_model.last_check = datetime.now()
                            db.session.commit()
                except Exception as e:
                    logger.error(f"Error updating search term last check time: {e}")
                
                # Add delay between search terms
                time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Error monitoring search term {term_info.get('term')}: {e}")
    
    def scan_url(self, url: str, depth: int = 1) -> str:
        """
        Scan a URL for content.
        
        Args:
            url: URL to scan
            depth: How many pages to crawl (1 = single page only)
            
        Returns:
            Job ID for tracking the scan
        """
        job_type = 'single' if depth <= 1 else 'crawl'
        config = {
            'url': url,
            'source_name': f"Manual Scan: {url}",
            'max_pages': depth
        }
        
        return web_source_manager.add_url_job(url, job_type, config)
    
    def search_and_monitor(self, search_term: str, limit: int = 10) -> Optional[str]:
        """
        Search for content and set up monitoring.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            Job ID for tracking the search
        """
        try:
            # Add search job
            job_id = web_source_manager.add_url_job(
                url="https://www.bing.com/search",  # Will be replaced by actual search URL
                job_type='search',
                config={
                    'search_term': search_term,
                    'search_engine': 'bing',
                    'limit': limit,
                    'source_name': f"Search: {search_term}"
                }
            )
            
            # Also add search term to monitoring list
            self.add_search_term(search_term)
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error searching for '{search_term}': {e}")
            return None
    
    def add_focused_domain(self, domain: str, category: str = 'general', priority: int = 2) -> bool:
        """
        Add a domain to the focused domains list.
        
        Args:
            domain: Domain to focus on
            category: Category for this domain
            priority: Priority level (1-3, lower is higher priority)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean domain
            domain = domain.lower()
            if domain.startswith('http://'):
                domain = domain[7:]
            if domain.startswith('https://'):
                domain = domain[8:]
            if domain.startswith('www.'):
                domain = domain[4:]
            domain = domain.split('/')[0]  # Remove path
            
            # Check if domain already exists
            existing = db.session.query(models.FocusedDomain).filter_by(domain=domain).first()
            if existing:
                # Update existing
                with db.session.begin():
                    existing.category = category
                    existing.priority = priority
                    existing.is_active = True
                    existing.updated_at = datetime.now()
                    db.session.commit()
                
                logger.info(f"Updated focused domain: {domain}")
            else:
                # Create new
                with db.session.begin():
                    new_domain = models.FocusedDomain(
                        domain=domain,
                        category=category,
                        priority=priority,
                        is_active=True,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    db.session.add(new_domain)
                    db.session.commit()
                
                logger.info(f"Added new focused domain: {domain}")
            
            # Reload configuration
            self._load_configuration()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding focused domain '{domain}': {e}")
            return False
    
    def add_search_term(self, term: str, category: str = 'general') -> bool:
        """
        Add a search term to the monitoring list.
        
        Args:
            term: Search term to monitor
            category: Category for this term
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean term
            term = term.strip()
            
            # Check if term already exists
            existing = db.session.query(models.SearchTerm).filter_by(term=term).first()
            if existing:
                # Update existing
                with db.session.begin():
                    existing.category = category
                    existing.is_active = True
                    existing.updated_at = datetime.now()
                    db.session.commit()
                
                logger.info(f"Updated search term: {term}")
            else:
                # Create new
                with db.session.begin():
                    new_term = models.SearchTerm(
                        term=term,
                        category=category,
                        is_active=True,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    db.session.add(new_term)
                    db.session.commit()
                
                logger.info(f"Added new search term: {term}")
            
            # Reload configuration
            self._load_configuration()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding search term '{term}': {e}")
            return False
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """
        Get statistics about focused domains.
        
        Returns:
            Dictionary containing domain statistics
        """
        try:
            # Get all domains
            domains = db.session.query(models.FocusedDomain).all()
            
            # Get active domains
            active_domains = [d for d in domains if d.is_active]
            
            # Count domains by category
            domains_by_category = {}
            for domain in active_domains:
                category = domain.category or 'uncategorized'
                domains_by_category[category] = domains_by_category.get(category, 0) + 1
            
            # Get recently processed domains
            recent_domains = db.session.query(models.FocusedDomain) \
                .filter_by(is_active=True) \
                .order_by(models.FocusedDomain.last_check.desc()) \
                .limit(10) \
                .all()
                
            recent = [
                {
                    'name': domain.domain,
                    'category': domain.category,
                    'last_ingestion': domain.last_check.isoformat() if domain.last_check else None
                }
                for domain in recent_domains if domain.last_check
            ]
            
            return {
                'total_domains': len(active_domains),
                'domains_by_category': domains_by_category,
                'recently_processed': recent
            }
            
        except Exception as e:
            logger.error(f"Error getting domain stats: {e}")
            return {
                'total_domains': 0,
                'domains_by_category': {},
                'recently_processed': []
            }
    
    def get_search_term_stats(self) -> Dict[str, Any]:
        """
        Get statistics about search terms.
        
        Returns:
            Dictionary containing search term statistics
        """
        try:
            # Get all terms
            terms = db.session.query(models.SearchTerm).all()
            
            # Get active terms
            active_terms = [t for t in terms if t.is_active]
            
            # Count terms by category
            terms_by_category = {}
            for term in active_terms:
                category = term.category or 'uncategorized'
                terms_by_category[category] = terms_by_category.get(category, 0) + 1
            
            # Get recently processed terms
            recent_terms = db.session.query(models.SearchTerm) \
                .filter_by(is_active=True) \
                .order_by(models.SearchTerm.last_check.desc()) \
                .limit(10) \
                .all()
                
            recent = [
                {
                    'term': term.term,
                    'category': term.category,
                    'last_ingestion': term.last_check.isoformat() if term.last_check else None
                }
                for term in recent_terms if term.last_check
            ]
            
            return {
                'total_terms': len(active_terms),
                'terms_by_category': terms_by_category,
                'recently_processed': recent
            }
            
        except Exception as e:
            logger.error(f"Error getting search term stats: {e}")
            return {
                'total_terms': 0,
                'terms_by_category': {},
                'recently_processed': []
            }
    
    def submit_to_detection_pipeline(self, content_data: Dict[str, Any]) -> bool:
        """
        Submit content to the detection pipeline.
        
        Args:
            content_data: Dictionary containing content data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Format for detection
            detection_data = DataScaler.format_for_detection(content_data)
            
            # Check if we have enough content
            if not detection_data.get('content'):
                logger.warning("Not enough content to submit for detection")
                return False
            
            # Create ContentItem
            with db.session.begin():
                content_item = models.ContentItem(
                    title=detection_data.get('title', ''),
                    content=detection_data.get('content', ''),
                    source=detection_data.get('source', 'web'),
                    url=detection_data.get('url', ''),
                    published_date=detection_data.get('published_date'),
                    meta_data=detection_data.get('meta_data', '{}'),
                    content_type=detection_data.get('content_type', 'web'),
                    is_processed=False,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                db.session.add(content_item)
                db.session.commit()
                
                logger.info(f"Submitted content item ID {content_item.id} to detection pipeline")
                
                return True
                
        except Exception as e:
            logger.error(f"Error submitting content to detection pipeline: {e}")
            return False


# Create singleton instance
web_scraping_service = WebScrapingService()