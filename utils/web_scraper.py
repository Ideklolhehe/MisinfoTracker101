"""
Web scraping utility for the CIVILIAN system.
This utility provides functions for scraping and extracting content from web pages.
"""

import os
import logging
import time
import json
import hashlib
import random
import trafilatura
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

class WebScraper:
    """A utility class for scraping web content with rate limiting and caching."""
    
    def __init__(self, cache_dir='./storage/web_cache', user_agents=None):
        """
        Initialize the web scraper.
        
        Args:
            cache_dir: Directory to store cached web pages
            user_agents: List of user agents to rotate through
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Default user agents to rotate
        self.user_agents = user_agents or [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        # Rate limiting parameters
        self.last_request_time = {}  # domain -> timestamp
        self.min_request_interval = 2  # seconds between requests to same domain
        
        # Request timeout
        self.timeout = 10  # seconds
    
    def _get_cache_path(self, url):
        """Get the cache file path for a URL."""
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.html")
    
    def _is_cached(self, url, max_age=3600):
        """Check if URL content is cached and not expired."""
        cache_path = self._get_cache_path(url)
        
        if not os.path.exists(cache_path):
            return False
        
        # Check if cache is expired
        file_time = os.path.getmtime(cache_path)
        current_time = time.time()
        
        return (current_time - file_time) < max_age
    
    def _cache_content(self, url, content):
        """Cache the content for a URL."""
        if not content:
            return
        
        cache_path = self._get_cache_path(url)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error caching content for {url}: {e}")
    
    def _get_cached_content(self, url):
        """Get cached content for a URL."""
        cache_path = self._get_cache_path(url)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading cached content for {url}: {e}")
            return None
    
    def _respect_rate_limit(self, domain):
        """Ensure rate limits are respected for each domain."""
        current_time = time.time()
        
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                logger.debug(f"Rate limiting: Sleeping {sleep_time:.2f}s for {domain}")
                time.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    def fetch_url(self, url, force_refresh=False, verify_ssl=True):
        """
        Fetch HTML content from a URL with caching and rate limiting.
        
        Args:
            url: The URL to fetch
            force_refresh: Whether to ignore cache and fetch fresh content
            verify_ssl: Whether to verify SSL certificates
            
        Returns:
            HTML content as string or None if failed
        """
        # Parse URL to get domain for rate limiting
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Check cache first unless force refresh
        if not force_refresh and self._is_cached(url):
            logger.debug(f"Using cached content for {url}")
            return self._get_cached_content(url)
        
        # Respect rate limits
        self._respect_rate_limit(domain)
        
        # Rotate user agents
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        try:
            logger.debug(f"Fetching {url}")
            response = requests.get(
                url, 
                headers=headers, 
                timeout=self.timeout,
                verify=verify_ssl
            )
            
            # Check if request was successful
            if response.status_code == 200:
                content = response.text
                self._cache_content(url, content)
                return content
            else:
                logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_text_content(self, html_content):
        """
        Extract main text content from HTML using trafilatura.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Extracted text content
        """
        if not html_content:
            return None
        
        try:
            return trafilatura.extract(html_content)
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return None
    
    def extract_links(self, html_content, base_url):
        """
        Extract links from HTML content.
        
        Args:
            html_content: HTML content as string
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        if not html_content:
            return []
        
        links = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip anchors and javascript links
                if href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # Resolve relative URLs
                abs_url = urljoin(base_url, href)
                
                # Normalize URL
                parsed = urlparse(abs_url)
                abs_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    abs_url += f"?{parsed.query}"
                
                links.append(abs_url)
                
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")
            return []
    
    def extract_metadata(self, html_content, url):
        """
        Extract metadata from HTML content.
        
        Args:
            html_content: HTML content as string
            url: URL of the page
            
        Returns:
            Dictionary containing metadata
        """
        if not html_content:
            return {}
        
        metadata = {
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'title': None,
            'description': None,
            'published_date': None,
            'author': None,
            'site_name': None,
        }
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.text.strip()
            
            # Try to get meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                property = meta.get('property', '').lower()
                content = meta.get('content')
                
                if not content:
                    continue
                
                # Description
                if name == 'description' or property == 'og:description':
                    metadata['description'] = content
                
                # Published date
                elif name == 'date' or property == 'article:published_time':
                    metadata['published_date'] = content
                
                # Author
                elif name == 'author' or property == 'article:author':
                    metadata['author'] = content
                
                # Site name
                elif property == 'og:site_name':
                    metadata['site_name'] = content
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {e}")
            return metadata
    
    def crawl(self, start_url, max_pages=5, same_domain_only=True):
        """
        Crawl a website starting from a URL.
        
        Args:
            start_url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            same_domain_only: Whether to only crawl pages from the same domain
            
        Returns:
            List of dictionaries containing page data
        """
        results = []
        visited = set()
        to_visit = [start_url]
        start_domain = urlparse(start_url).netloc
        
        logger.info(f"Starting crawl from {start_url}, max pages: {max_pages}")
        
        while to_visit and len(results) < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
                
            visited.add(url)
            
            # Skip if different domain and same_domain_only is True
            url_domain = urlparse(url).netloc
            if same_domain_only and url_domain != start_domain:
                continue
            
            # Fetch content
            html_content = self.fetch_url(url)
            if not html_content:
                continue
            
            # Extract data
            text_content = self.extract_text_content(html_content)
            metadata = self.extract_metadata(html_content, url)
            
            if text_content:
                page_data = {
                    **metadata,
                    'content': text_content
                }
                results.append(page_data)
                
                # Extract links for next pages
                links = self.extract_links(html_content, url)
                for link in links:
                    if link not in visited and link not in to_visit:
                        to_visit.append(link)
        
        logger.info(f"Crawl completed, processed {len(results)} pages")
        return results
    
    def search_and_extract(self, query, engine="bing", limit=10):
        """
        Search the web and extract content from results.
        
        Args:
            query: Search query string
            engine: Search engine to use ("bing" or "google")
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        # This is a simplified implementation
        # In a real-world app, you would use search engine APIs or libraries
        
        logger.warning("Search functionality requires API integration with search engines")
        
        # Placeholder for demonstration
        results = []
        
        # Return placeholder
        return results