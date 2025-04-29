"""
Web Scraping Service for the CIVILIAN system.
This module integrates web scraping capabilities and data scaling for real-time
monitoring of internet sources.
"""

import logging
import threading
import time
import json
import os
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib
import queue
import csv

from app import db
from models import DataSource, DetectedNarrative, NarrativeInstance
from utils.web_scraper import WebScraper, get_website_content, crawl_website, search_for_content
from utils.data_scaling import data_scaler
from utils.app_context import with_app_context
from data_sources.web_source_manager import web_source_manager

# Configure logger
logger = logging.getLogger(__name__)

# Constants
SCHEDULED_INTERVAL = 900  # 15 minutes
MAX_SOURCES_PER_INTERVAL = 20
STORAGE_DIR = "./storage/web_data"
FOCUSED_DOMAINS_FILE = f"{STORAGE_DIR}/focused_domains.json"
DEFAULT_DEPTH = 2
DEFAULT_DOMAIN_RATE_LIMIT = 5  # seconds between requests
MIN_CONTENT_LENGTH = 500


class WebScrapingService:
    """Service for managing web scraping operations in the CIVILIAN system."""
    
    def __init__(self):
        """Initialize the web scraping service."""
        self.web_scraper = WebScraper()
        self.is_running = False
        self.stop_event = threading.Event()
        self.scheduled_thread = None
        self.focused_domains = self._load_focused_domains()
        self.search_terms = self._load_search_terms()
        
        # Create storage directory if it doesn't exist
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        # Initialize data cache for web content
        self.cache = data_scaler.get_cache(namespace="web_content", max_size=5000)
        
        logger.info("WebScrapingService initialized")
        
    def _load_focused_domains(self) -> Dict[str, Dict[str, Any]]:
        """Load focused domains configuration."""
        if os.path.exists(FOCUSED_DOMAINS_FILE):
            try:
                with open(FOCUSED_DOMAINS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading focused domains: {e}")
                
        # Default configuration
        default_domains = {
            "news": {
                "cnn.com": {"rate_limit": 10, "priority": 1},
                "bbc.com": {"rate_limit": 10, "priority": 1},
                "reuters.com": {"rate_limit": 10, "priority": 1},
                "apnews.com": {"rate_limit": 10, "priority": 1},
                "nytimes.com": {"rate_limit": 10, "priority": 2},
                "wsj.com": {"rate_limit": 10, "priority": 2}
            },
            "fact_checking": {
                "snopes.com": {"rate_limit": 5, "priority": 1},
                "factcheck.org": {"rate_limit": 5, "priority": 1},
                "politifact.com": {"rate_limit": 5, "priority": 1}
            },
            "think_tanks": {
                "brookings.edu": {"rate_limit": 15, "priority": 3},
                "csis.org": {"rate_limit": 15, "priority": 3},
                "heritage.org": {"rate_limit": 15, "priority": 3}
            },
            "government": {
                "who.int": {"rate_limit": 30, "priority": 3},
                "cdc.gov": {"rate_limit": 30, "priority": 2},
                "nih.gov": {"rate_limit": 30, "priority": 3},
                "un.org": {"rate_limit": 30, "priority": 3}
            }
        }
        
        # Save default configuration
        try:
            with open(FOCUSED_DOMAINS_FILE, 'w') as f:
                json.dump(default_domains, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default focused domains: {e}")
            
        return default_domains
        
    def _load_search_terms(self) -> Dict[str, List[str]]:
        """Load search terms for targeted monitoring."""
        # These would typically come from a configuration file or database
        return {
            "misinformation": [
                "vaccine misinformation",
                "election fraud claims",
                "climate change denial",
                "conspiracy theories"
            ],
            "emerging_narratives": [
                "breaking news",
                "trending story",
                "viral content"
            ],
            "security_threats": [
                "cyber attack",
                "malware campaign",
                "data breach",
                "ransomware"
            ]
        }
    
    def start_scheduled_scraping(self):
        """Start scheduled scraping of web sources."""
        if self.is_running:
            logger.warning("Scheduled scraping is already running")
            return
            
        self.is_running = True
        self.stop_event.clear()
        
        # Start scheduled thread
        self.scheduled_thread = threading.Thread(target=self._scheduled_scraping_thread)
        self.scheduled_thread.daemon = True
        self.scheduled_thread.start()
        
        logger.info("Started scheduled web scraping")
        
    def stop_scheduled_scraping(self):
        """Stop scheduled scraping of web sources."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.scheduled_thread and self.scheduled_thread.is_alive():
            self.scheduled_thread.join(timeout=5)
            
        logger.info("Stopped scheduled web scraping")
    
    def _scheduled_scraping_thread(self):
        """Thread for scheduled scraping of web sources."""
        while self.is_running and not self.stop_event.is_set():
            try:
                self._run_scheduled_scraping()
            except Exception as e:
                logger.error(f"Error in scheduled scraping: {e}")
                
            # Wait for the next interval or until stopped
            for _ in range(SCHEDULED_INTERVAL):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    @with_app_context
    def _run_scheduled_scraping(self):
        """Run a single iteration of scheduled scraping."""
        logger.info("Starting scheduled web scraping iteration")
        
        # Get active web sources from the database
        sources = DataSource.query.filter(
            DataSource.is_active == True,
            DataSource.source_type.like("web_%")
        ).order_by(DataSource.last_ingestion.asc()).limit(MAX_SOURCES_PER_INTERVAL).all()
        
        if not sources:
            logger.info("No active web sources to process")
            return
            
        logger.info(f"Processing {len(sources)} web sources")
        
        # Process sources concurrently
        processed_count = 0
        threads = []
        
        for source in sources[:MAX_SOURCES_PER_INTERVAL]:
            thread = threading.Thread(target=self._process_source, args=(source,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            processed_count += 1
            
        logger.info(f"Completed processing {processed_count} web sources")
        
        # Run focused domain monitoring (subset each time)
        self._run_focused_domain_monitoring()
        
        # Run search term monitoring (subset each time)
        self._run_search_term_monitoring()
    
    @with_app_context
    def _process_source(self, source: DataSource):
        """
        Process a single web source.
        
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
                max_pages = config.get("max_pages", 5)
                results = self.web_scraper.crawl(source_url, max_pages=max_pages)
                
            elif source_type == "search":
                # Use search for search-based sources
                search_term = config.get("search_term", "")
                search_engine = config.get("search_engine", "bing")
                limit = config.get("limit", 10)
                
                if not search_term:
                    logger.warning(f"Missing search term for search source {source.name}")
                    return
                    
                urls = self.web_scraper.search(search_term, search_engine=search_engine, limit=limit)
                results = self.web_scraper.batch_process(urls)
                
            elif source_type == "monitor":
                # Single URL monitoring
                result = self.web_scraper.get_content(source_url)
                results = [result] if result.get("success") else []
                
            else:
                # Default to single URL processing
                result = self.web_scraper.get_content(source_url)
                results = [result] if result.get("success") else []
            
            # Process results
            process_count = 0
            for result in results:
                if result.get("success") and self._is_valid_content(result):
                    web_source_manager._process_result(result, source)
                    process_count += 1
                    
            # Update source last ingestion timestamp
            source.last_ingestion = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Processed {process_count} pages from source {source.name}")
            
        except Exception as e:
            logger.error(f"Error processing source {source.name}: {e}")
            db.session.rollback()
    
    def _is_valid_content(self, result: Dict[str, Any]) -> bool:
        """
        Check if content is valid for processing.
        
        Args:
            result: Scraping result
            
        Returns:
            True if valid, False otherwise
        """
        content = result.get("content", "")
        
        # Check content length
        if not content or len(content) < MIN_CONTENT_LENGTH:
            return False
            
        # Check for duplicate content using hash
        content_hash = result.get("content_hash") or hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Check cache for duplicate
        if self.cache.get(content_hash):
            return False
            
        # Store in cache to prevent duplicates
        self.cache.set(content_hash, True)
        
        return True
    
    def _run_focused_domain_monitoring(self):
        """Run focused domain monitoring for selected domains."""
        # Select a subset of domains to monitor in this iteration
        domains_to_monitor = []
        
        for category, domains in self.focused_domains.items():
            # Select domains based on priority (lower number = higher priority)
            priority_domains = sorted(domains.items(), key=lambda x: x[1].get("priority", 5))
            
            # Take top domains from each category
            domains_to_monitor.extend([domain for domain, _ in priority_domains[:2]])
        
        # Randomize further to avoid patterns
        import random
        random.shuffle(domains_to_monitor)
        
        # Limit total number of domains
        domains_to_monitor = domains_to_monitor[:5]
        
        logger.info(f"Running focused monitoring for domains: {domains_to_monitor}")
        
        # Process each domain
        for domain in domains_to_monitor:
            try:
                self._monitor_focused_domain(domain)
            except Exception as e:
                logger.error(f"Error monitoring domain {domain}: {e}")
    
    @with_app_context
    def _monitor_focused_domain(self, domain: str):
        """
        Monitor a specific focused domain.
        
        Args:
            domain: Domain to monitor
        """
        url = f"https://{domain}"
        
        # Find or create a source for this domain
        source = DataSource.query.filter(
            DataSource.config.like(f'%"{url}"%'),
            DataSource.source_type.like("web_%")
        ).first()
        
        if not source:
            # Create new source
            source = DataSource(
                name=f"Focused Monitor: {domain}",
                source_type="web_monitor",
                config=json.dumps({"url": url}),
                is_active=True,
                meta_data=json.dumps({"origin": "focused_monitoring"})
            )
            db.session.add(source)
            db.session.commit()
        
        # Crawl the domain
        results = self.web_scraper.crawl(url, max_pages=3)
        
        # Process results
        process_count = 0
        for result in results:
            if result.get("success") and self._is_valid_content(result):
                web_source_manager._process_result(result, source)
                process_count += 1
                
        # Update source last ingestion timestamp
        source.last_ingestion = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Focused monitoring of {domain} processed {process_count} pages")
    
    def _run_search_term_monitoring(self):
        """Run search term monitoring for selected terms."""
        # Select a subset of search terms to monitor in this iteration
        terms_to_monitor = []
        
        for category, terms in self.search_terms.items():
            # Take one term from each category
            if terms:
                import random
                terms_to_monitor.append(random.choice(terms))
        
        logger.info(f"Running search term monitoring for: {terms_to_monitor}")
        
        # Process each term
        for term in terms_to_monitor:
            try:
                self._monitor_search_term(term)
            except Exception as e:
                logger.error(f"Error monitoring search term '{term}': {e}")
    
    @with_app_context
    def _monitor_search_term(self, term: str):
        """
        Monitor a specific search term.
        
        Args:
            term: Search term to monitor
        """
        # Find or create a source for this term
        source_name = f"Search Monitor: {term}"
        
        source = DataSource.query.filter_by(
            name=source_name,
            source_type="web_search"
        ).first()
        
        if not source:
            # Create new source
            source = DataSource(
                name=source_name,
                source_type="web_search",
                config=json.dumps({"search_term": term}),
                is_active=True,
                meta_data=json.dumps({"origin": "search_monitoring"})
            )
            db.session.add(source)
            db.session.commit()
        
        # Search for the term
        urls = self.web_scraper.search(term, limit=5)
        
        # Process results
        results = []
        for url in urls:
            result = self.web_scraper.get_content(url)
            if result.get("success"):
                results.append(result)
        
        # Process valid results
        process_count = 0
        for result in results:
            if self._is_valid_content(result):
                web_source_manager._process_result(result, source)
                process_count += 1
                
        # Update source last ingestion timestamp
        source.last_ingestion = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Search term monitoring of '{term}' processed {process_count} pages")
    
    def scan_url(self, url: str, depth: int = DEFAULT_DEPTH) -> Dict[str, Any]:
        """
        Scan a specific URL and its linked pages.
        
        Args:
            url: URL to scan
            depth: Crawl depth
            
        Returns:
            Dictionary with scan results
        """
        logger.info(f"Scanning URL: {url} with depth {depth}")
        
        # Validate URL format
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
            
        # Set up crawl parameters
        domain = urlparse(url).netloc
        
        # Get content with specified depth
        if depth <= 1:
            # Single page
            result = self.web_scraper.get_content(url)
            results = [result] if result.get("success") else []
        else:
            # Crawl
            results = self.web_scraper.crawl(url, max_pages=depth)
            
        # Process and return results
        processed_results = []
        for result in results:
            if result.get("success"):
                # Extract key information
                processed_result = {
                    "url": result.get("url", ""),
                    "title": result.get("metadata", {}).get("title", ""),
                    "description": result.get("metadata", {}).get("description", ""),
                    "content_length": len(result.get("content", "")),
                    "extracted_at": datetime.utcnow().isoformat()
                }
                processed_results.append(processed_result)
                
        return {
            "domain": domain,
            "base_url": url,
            "depth": depth,
            "pages_found": len(results),
            "pages_extracted": len(processed_results),
            "results": processed_results
        }
    
    def search_and_monitor(self, search_term: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for a term and set up monitoring for the results.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results and monitoring info
        """
        logger.info(f"Searching and monitoring term: {search_term}")
        
        # Search for the term
        urls = self.web_scraper.search(search_term, limit=limit)
        
        # Create a source for monitoring
        with db.session.no_autoflush:
            source_name = f"Search Monitor: {search_term}"
            
            # Check if source already exists
            source = DataSource.query.filter_by(
                name=source_name,
                source_type="web_search"
            ).first()
            
            if not source:
                # Create new source
                source = DataSource(
                    name=source_name,
                    source_type="web_search",
                    config=json.dumps({"search_term": search_term}),
                    is_active=True,
                    meta_data=json.dumps({"origin": "search_monitoring"})
                )
                db.session.add(source)
                db.session.commit()
        
        # Process URLs
        results = []
        for url in urls:
            result = self.web_scraper.get_content(url)
            if result.get("success"):
                # Add to results
                results.append({
                    "url": result.get("url", ""),
                    "title": result.get("metadata", {}).get("title", ""),
                    "description": result.get("metadata", {}).get("description", ""),
                    "content_length": len(result.get("content", "")),
                })
                
                # Process for narratives
                if self._is_valid_content(result):
                    web_source_manager._process_result(result, source)
        
        return {
            "search_term": search_term,
            "urls_found": len(urls),
            "results_processed": len(results),
            "source_id": source.id if source else None,
            "results": results
        }
    
    def add_focused_domain(self, domain: str, category: str, priority: int = 2) -> bool:
        """
        Add a domain to focused monitoring.
        
        Args:
            domain: Domain to add
            category: Category for the domain
            priority: Priority level (1=highest)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean domain format
            domain = domain.lower()
            if domain.startswith(("http://", "https://")):
                domain = urlparse(domain).netloc
                
            # Check if category exists, create if not
            if category not in self.focused_domains:
                self.focused_domains[category] = {}
                
            # Add domain with settings
            self.focused_domains[category][domain] = {
                "rate_limit": DEFAULT_DOMAIN_RATE_LIMIT,
                "priority": priority
            }
            
            # Save updated configuration
            with open(FOCUSED_DOMAINS_FILE, 'w') as f:
                json.dump(self.focused_domains, f, indent=2)
                
            logger.info(f"Added domain {domain} to focused monitoring (category: {category}, priority: {priority})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding focused domain: {e}")
            return False
    
    def add_search_term(self, term: str, category: str) -> bool:
        """
        Add a search term for monitoring.
        
        Args:
            term: Search term
            category: Category for the term
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean term
            term = term.strip().lower()
            
            # Check if category exists, create if not
            if category not in self.search_terms:
                self.search_terms[category] = []
                
            # Add term if not already present
            if term not in self.search_terms[category]:
                self.search_terms[category].append(term)
                
            # Save updated configuration
            # In a real implementation, this would be saved to a file or database
            logger.info(f"Added search term '{term}' to monitoring (category: {category})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding search term: {e}")
            return False
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """
        Get statistics about monitored domains.
        
        Returns:
            Dictionary with domain statistics
        """
        stats = {
            "total_domains": 0,
            "domains_by_category": {},
            "recently_processed": []
        }
        
        # Count domains
        for category, domains in self.focused_domains.items():
            stats["domains_by_category"][category] = len(domains)
            stats["total_domains"] += len(domains)
            
        # Get recently processed domains from database
        try:
            recent_sources = DataSource.query.filter(
                DataSource.source_type.like("web_%"),
                DataSource.last_ingestion.isnot(None)
            ).order_by(DataSource.last_ingestion.desc()).limit(10).all()
            
            for source in recent_sources:
                stats["recently_processed"].append({
                    "name": source.name,
                    "source_type": source.source_type,
                    "last_ingestion": source.last_ingestion.isoformat() if source.last_ingestion else None
                })
        except Exception as e:
            logger.error(f"Error getting recently processed domains: {e}")
            
        return stats
    
    def get_search_term_stats(self) -> Dict[str, Any]:
        """
        Get statistics about monitored search terms.
        
        Returns:
            Dictionary with search term statistics
        """
        stats = {
            "total_terms": 0,
            "terms_by_category": {},
            "recently_processed": []
        }
        
        # Count terms
        for category, terms in self.search_terms.items():
            stats["terms_by_category"][category] = len(terms)
            stats["total_terms"] += len(terms)
            
        # Get recently processed search terms from database
        try:
            recent_sources = DataSource.query.filter(
                DataSource.source_type == "web_search",
                DataSource.last_ingestion.isnot(None)
            ).order_by(DataSource.last_ingestion.desc()).limit(10).all()
            
            for source in recent_sources:
                # Extract search term from name or config
                search_term = ""
                if source.name.startswith("Search Monitor:"):
                    search_term = source.name.replace("Search Monitor:", "").strip()
                else:
                    try:
                        config = json.loads(source.config) if source.config else {}
                        search_term = config.get("search_term", "")
                    except:
                        pass
                        
                stats["recently_processed"].append({
                    "term": search_term,
                    "last_ingestion": source.last_ingestion.isoformat() if source.last_ingestion else None
                })
        except Exception as e:
            logger.error(f"Error getting recently processed search terms: {e}")
            
        return stats


# Create a singleton instance
web_scraping_service = WebScrapingService()