"""
Base classes for data sources in the CIVILIAN system.
"""

import os
import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class SourceStatus:
    """Status codes for data sources."""
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class SourceBase(ABC):
    """
    Abstract base class for all data sources.
    All source types should inherit from this class.
    """
    
    def __init__(self, source_id: Optional[int] = None, 
                 name: str = "Unnamed Source",
                 source_type: str = "base", 
                 config: Optional[Dict[str, Any]] = None,
                 is_active: bool = True):
        """
        Initialize the data source.
        
        Args:
            source_id: Unique identifier for this source
            name: Human-readable name for this source
            source_type: Type identifier for this source
            config: Configuration dictionary for the source
            is_active: Whether this source is active
        """
        self.source_id = source_id
        self.name = name
        self.source_type = source_type
        self.config = config or {}
        self.is_active = is_active
        
        # Status tracking
        self.status = SourceStatus.READY
        self.status_message = "Ready"
        self.last_run = None
        self.run_count = 0
        self.error_count = 0
        self.processed_count = 0
        
        # Metadata storage
        self.meta_data = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize source-specific resources
        self.initialize()
    
    def initialize(self):
        """Initialize any source-specific resources. Override in subclasses."""
        pass
    
    @abstractmethod
    def process(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process the data source to extract content.
        
        Returns:
            Tuple containing:
            - List of content items extracted from the source
            - Stats dictionary with metadata about the extraction process
        """
        pass
    
    def run(self) -> bool:
        """
        Run the data source processing.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_active:
            logger.warning(f"Source {self.name} (ID: {self.source_id}) is not active")
            return False
        
        with self.lock:
            try:
                self.status = SourceStatus.RUNNING
                self.status_message = "Processing source data"
                self.last_run = datetime.now()
                self.run_count += 1
                
                logger.info(f"Running source: {self.name} (ID: {self.source_id})")
                
                # Process the source
                content_items, stats = self.process()
                
                # Update metadata
                self.processed_count += len(content_items)
                self.update_metadata(stats)
                
                self.status = SourceStatus.COMPLETED
                self.status_message = "Processing completed successfully"
                
                logger.info(f"Source {self.name} completed successfully, processed {len(content_items)} items")
                
                return True
                
            except Exception as e:
                self.status = SourceStatus.ERROR
                self.status_message = f"Error: {str(e)}"
                self.error_count += 1
                
                logger.error(f"Error processing source {self.name} (ID: {self.source_id}): {e}")
                
                return False
    
    def update_metadata(self, stats: Dict[str, Any]):
        """Update source metadata with processing statistics."""
        try:
            # Merge existing metadata if present
            if hasattr(self, 'meta_data') and self.meta_data:
                if isinstance(self.meta_data, str):
                    # Convert from JSON string if needed
                    current_meta = json.loads(self.meta_data)
                else:
                    current_meta = self.meta_data
            else:
                current_meta = {}
            
            # Add run statistics
            current_meta['last_run'] = datetime.now().isoformat()
            current_meta['run_count'] = self.run_count
            current_meta['error_count'] = self.error_count
            current_meta['processed_count'] = self.processed_count
            
            # Add source-specific stats
            if stats:
                current_meta['stats'] = stats
            
            # Store updated metadata
            self.meta_data = current_meta
            
        except Exception as e:
            logger.error(f"Error updating metadata for source {self.name}: {e}")
    
    def pause(self):
        """Pause the data source."""
        with self.lock:
            self.status = SourceStatus.PAUSED
            self.status_message = "Source paused"
    
    def resume(self):
        """Resume the data source."""
        with self.lock:
            self.status = SourceStatus.READY
            self.status_message = "Source ready"
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the data source."""
        with self.lock:
            return {
                'id': self.source_id,
                'name': self.name,
                'type': self.source_type,
                'status': self.status,
                'status_message': self.status_message,
                'last_run': self.last_run.isoformat() if self.last_run else None,
                'run_count': self.run_count,
                'error_count': self.error_count,
                'processed_count': self.processed_count,
                'is_active': self.is_active
            }


class WebSourceBase(SourceBase):
    """Base class for web-based data sources."""
    
    def __init__(self, source_id: Optional[int] = None, 
                 name: str = "Web Source",
                 source_type: str = "web", 
                 config: Optional[Dict[str, Any]] = None,
                 is_active: bool = True):
        """
        Initialize a web-based data source.
        
        Args:
            source_id: Unique identifier for this source
            name: Human-readable name for this source
            source_type: Type identifier for this source
            config: Configuration dictionary for the source
            is_active: Whether this source is active
        """
        super().__init__(source_id, name, source_type, config, is_active)
        
        # Web-specific configuration
        self.url = self.config.get('url', '')
        self.user_agent = self.config.get('user_agent', 'CIVILIAN Bot')
        self.respect_robots_txt = self.config.get('respect_robots_txt', True)
        self.rate_limit = self.config.get('rate_limit', 2.0)  # seconds between requests
        
        # Initialize web scraper if needed in subclasses
        self.scraper = None
    
    def respect_rate_limit(self):
        """Sleep to respect rate limiting."""
        time.sleep(self.rate_limit)


class WebPageSource(WebSourceBase):
    """Data source for a single web page."""
    
    def __init__(self, source_id: Optional[int] = None, 
                 name: str = "Web Page",
                 config: Optional[Dict[str, Any]] = None,
                 is_active: bool = True):
        """
        Initialize a web page data source.
        
        Args:
            source_id: Unique identifier for this source
            name: Human-readable name for this source
            config: Configuration dictionary for the source
            is_active: Whether this source is active
        """
        super().__init__(source_id, name, 'web_page', config, is_active)
    
    def process(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process the web page to extract content.
        
        Returns:
            Tuple containing:
            - List containing the extracted web page content
            - Stats dictionary with metadata about the extraction process
        """
        from utils.web_scraper import WebScraper
        
        # Create scraper if needed
        if not self.scraper:
            self.scraper = WebScraper()
        
        stats = {
            'url': self.url,
            'started_at': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Fetch the URL content
            html_content = self.scraper.fetch_url(self.url)
            
            if not html_content:
                stats['error'] = 'Failed to fetch URL content'
                return [], stats
            
            # Extract text content
            text_content = self.scraper.extract_text_content(html_content)
            
            if not text_content:
                stats['error'] = 'Failed to extract text content'
                return [], stats
            
            # Extract metadata
            metadata = self.scraper.extract_metadata(html_content, self.url)
            
            # Create content item
            content_item = {
                'url': self.url,
                'title': metadata.get('title', ''),
                'site_name': metadata.get('site_name', ''),
                'author': metadata.get('author', ''),
                'published_date': metadata.get('published_date', ''),
                'content': text_content,
                'domain': metadata.get('domain', ''),
                'scraped_at': datetime.now().isoformat()
            }
            
            # Update stats
            stats['success'] = True
            stats['content_length'] = len(text_content)
            stats['completed_at'] = datetime.now().isoformat()
            
            return [content_item], stats
            
        except Exception as e:
            stats['error'] = str(e)
            stats['completed_at'] = datetime.now().isoformat()
            logger.error(f"Error processing web page {self.url}: {e}")
            return [], stats


class WebCrawlSource(WebSourceBase):
    """Data source for crawling multiple pages from a website."""
    
    def __init__(self, source_id: Optional[int] = None, 
                 name: str = "Web Crawler",
                 config: Optional[Dict[str, Any]] = None,
                 is_active: bool = True):
        """
        Initialize a web crawler data source.
        
        Args:
            source_id: Unique identifier for this source
            name: Human-readable name for this source
            config: Configuration dictionary for the source
            is_active: Whether this source is active
        """
        super().__init__(source_id, name, 'web_crawl', config, is_active)
        
        # Crawler-specific configuration
        self.max_pages = self.config.get('max_pages', 10)
        self.same_domain_only = self.config.get('same_domain_only', True)
    
    def process(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process the website by crawling multiple pages.
        
        Returns:
            Tuple containing:
            - List of content items extracted from the crawled pages
            - Stats dictionary with metadata about the crawling process
        """
        from utils.web_scraper import WebScraper
        
        # Create scraper if needed
        if not self.scraper:
            self.scraper = WebScraper()
        
        stats = {
            'start_url': self.url,
            'max_pages': self.max_pages,
            'same_domain_only': self.same_domain_only,
            'started_at': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Crawl the website
            results = self.scraper.crawl(
                self.url, 
                max_pages=self.max_pages, 
                same_domain_only=self.same_domain_only
            )
            
            if not results:
                stats['error'] = 'No content found while crawling'
                return [], stats
            
            # Update stats
            stats['success'] = True
            stats['pages_processed'] = len(results)
            stats['completed_at'] = datetime.now().isoformat()
            
            return results, stats
            
        except Exception as e:
            stats['error'] = str(e)
            stats['completed_at'] = datetime.now().isoformat()
            logger.error(f"Error crawling website {self.url}: {e}")
            return [], stats


class WebSearchSource(WebSourceBase):
    """Data source for web search results."""
    
    def __init__(self, source_id: Optional[int] = None, 
                 name: str = "Web Search",
                 config: Optional[Dict[str, Any]] = None,
                 is_active: bool = True):
        """
        Initialize a web search data source.
        
        Args:
            source_id: Unique identifier for this source
            name: Human-readable name for this source
            config: Configuration dictionary for the source
            is_active: Whether this source is active
        """
        super().__init__(source_id, name, 'web_search', config, is_active)
        
        # Search-specific configuration
        self.search_term = self.config.get('search_term', '')
        self.search_engine = self.config.get('search_engine', 'bing')
        self.limit = self.config.get('limit', 10)
    
    def process(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process search results to extract content.
        
        Returns:
            Tuple containing:
            - List of content items extracted from search results
            - Stats dictionary with metadata about the search process
        """
        from utils.web_scraper import WebScraper
        
        # Create scraper if needed
        if not self.scraper:
            self.scraper = WebScraper()
        
        stats = {
            'search_term': self.search_term,
            'search_engine': self.search_engine,
            'limit': self.limit,
            'started_at': datetime.now().isoformat(),
            'success': False
        }
        
        if not self.search_term:
            stats['error'] = 'No search term provided'
            return [], stats
        
        try:
            # Search for content
            search_results = self.scraper.search_and_extract(
                self.search_term,
                engine=self.search_engine,
                limit=self.limit
            )
            
            if not search_results:
                stats['error'] = 'No search results found'
                return [], stats
            
            # Update stats
            stats['success'] = True
            stats['results_found'] = len(search_results)
            stats['completed_at'] = datetime.now().isoformat()
            
            return search_results, stats
            
        except Exception as e:
            stats['error'] = str(e)
            stats['completed_at'] = datetime.now().isoformat()
            logger.error(f"Error searching for '{self.search_term}': {e}")
            return [], stats