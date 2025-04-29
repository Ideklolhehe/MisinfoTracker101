"""
Comprehensive web scraping utilities for the CIVILIAN system.
This module provides functions to extract content from various web sources
with robust error handling and rate limiting.
"""

import logging
import time
import random
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import json
import hashlib

import requests
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config
import threading

# Configure logger
logger = logging.getLogger(__name__)

# Configure trafilatura
config = use_config()
config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")
config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "200")

# Constants
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:97.0) Gecko/20100101 Firefox/97.0",
]

# Rate limiting settings
REQUEST_DELAY = 2  # Base delay in seconds between requests
DOMAIN_RATE_LIMITS = {}  # Domain-specific rate limits

# Class wrapper to maintain backwards compatibility
class WebScraper:
    """
    WebScraper class that provides an object-oriented interface to the
    web scraping utilities in this module.
    """
    
    def __init__(self):
        """Initialize the WebScraper with default settings."""
        self.user_agents = USER_AGENTS
        self.request_delay = REQUEST_DELAY
        self.domain_rate_limits = DOMAIN_RATE_LIMITS
        logger.info("WebScraper initialized")
        
    def get_content(self, url: str) -> Dict[str, Any]:
        """Get content from a URL."""
        return get_website_content(url)
        
    def crawl(self, url: str, max_pages: int = 10, same_domain_only: bool = True) -> List[Dict[str, Any]]:
        """Crawl a website starting from a URL."""
        return crawl_website(url, max_pages=max_pages, same_domain_only=same_domain_only)
        
    def search(self, search_term: str, search_engine: str = "bing", limit: int = 10) -> List[str]:
        """Search for URLs related to a specific search term."""
        return search_for_content(search_term, search_engine=search_engine, limit=limit)
        
    def batch_process(self, urls: List[str], batch_size: int = 5) -> List[Dict[str, Any]]:
        """Process a list of URLs in batches."""
        return process_urls_in_batches(urls, batch_size=batch_size)

# Thread-safe domain tracking for rate limiting
domain_locks = {}
domain_last_request = {}
domain_lock = threading.Lock()


class ScraperException(Exception):
    """Exception raised for scraper-specific errors."""
    pass


def get_domain_lock(domain: str) -> threading.Lock:
    """Get or create a lock for a specific domain."""
    with domain_lock:
        if domain not in domain_locks:
            domain_locks[domain] = threading.Lock()
        return domain_locks[domain]


def respect_rate_limits(url: str) -> None:
    """
    Implement rate limiting for web requests to avoid overloading servers.
    Adjusts delay based on domain-specific settings and adds randomization.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Get domain-specific lock
    lock = get_domain_lock(domain)
    
    with lock:
        # Calculate appropriate delay
        domain_delay = DOMAIN_RATE_LIMITS.get(domain, REQUEST_DELAY)
        
        # Add some randomization to avoid pattern detection
        jitter = random.uniform(0, 0.5)
        delay = domain_delay + jitter
        
        # Check if we need to wait
        now = time.time()
        last_request = domain_last_request.get(domain, 0)
        
        if last_request > 0:
            time_since_last = now - last_request
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                time.sleep(sleep_time)
        
        # Update last request time
        domain_last_request[domain] = time.time()


def get_html_content(url: str, max_retries: int = 3, timeout: int = 30) -> Optional[str]:
    """
    Fetch HTML content from a URL with robust error handling and retries.
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        HTML content as string or None if failed
    """
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    # Apply rate limiting
    respect_rate_limits(url)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if not any(ct in content_type.lower() for ct in ["text/html", "application/xhtml", "xml"]):
                logger.warning(f"Unexpected content type: {content_type} for URL: {url}")
                return None
                
            return response.text
            
        except requests.exceptions.SSLError as e:
            logger.warning(f"SSL Error for {url}: {e} (attempt {attempt+1}/{max_retries})")
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for {url}: {e} (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    logger.error(f"Failed to retrieve URL after {max_retries} attempts: {url}")
    return None


def extract_text_with_trafilatura(html: str, url: str) -> Optional[str]:
    """
    Extract main content text from HTML using Trafilatura.
    
    Args:
        html: HTML content string
        url: Original URL for logging
    
    Returns:
        Extracted text or None if extraction failed
    """
    try:
        extracted_text = trafilatura.extract(html, config=config)
        
        if not extracted_text or len(extracted_text) < 200:
            logger.warning(f"Trafilatura extracted too little content from {url}")
            return None
            
        return extracted_text
    
    except Exception as e:
        logger.error(f"Error extracting content with Trafilatura from {url}: {e}")
        return None


def extract_text_with_bs4(html: str, url: str) -> Optional[str]:
    """
    Fallback text extraction using BeautifulSoup.
    
    Args:
        html: HTML content string
        url: Original URL for logging
    
    Returns:
        Extracted text or None if extraction failed
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove common non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
            
        # Extract text and clean up
        text = soup.get_text(separator="\n", strip=True)
        
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        if len(text) < 200:
            logger.warning(f"BeautifulSoup extracted too little content from {url}")
            return None
            
        return text
        
    except Exception as e:
        logger.error(f"Error extracting content with BeautifulSoup from {url}: {e}")
        return None


def extract_metadata(html: str, url: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML content.
    
    Args:
        html: HTML content string
        url: Original URL for reference
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        "url": url,
        "timestamp": datetime.utcnow().isoformat(),
        "title": "",
        "description": "",
        "author": "",
        "published_date": "",
        "keywords": [],
        "language": "",
    }
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract title
        metadata["title"] = soup.title.string if soup.title else ""
        
        # Extract meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            property = meta.get("property", "").lower()
            content = meta.get("content", "")
            
            # Description
            if name == "description" or property == "og:description":
                metadata["description"] = content
                
            # Author
            elif name == "author" or property == "article:author":
                metadata["author"] = content
                
            # Published date
            elif name == "article:published_time" or property == "article:published_time":
                metadata["published_date"] = content
                
            # Keywords
            elif name == "keywords":
                keywords = [k.strip() for k in content.split(",")]
                metadata["keywords"] = keywords
                
            # Language
            elif name == "language" or property == "og:locale":
                metadata["language"] = content
                
        # Extract schema.org metadata
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                ld_json = json.loads(script.string)
                
                # Handle single item or graph
                if isinstance(ld_json, dict):
                    items = [ld_json]
                elif isinstance(ld_json, list):
                    items = ld_json
                elif "@graph" in ld_json:
                    items = ld_json["@graph"]
                else:
                    items = []
                
                for item in items:
                    # Check if it's an article
                    if item.get("@type") in ["Article", "NewsArticle", "BlogPosting"]:
                        if not metadata["title"] and "headline" in item:
                            metadata["title"] = item["headline"]
                            
                        if not metadata["description"] and "description" in item:
                            metadata["description"] = item["description"]
                            
                        if not metadata["author"] and "author" in item:
                            author = item["author"]
                            if isinstance(author, dict) and "name" in author:
                                metadata["author"] = author["name"]
                            elif isinstance(author, list) and len(author) > 0:
                                if isinstance(author[0], dict) and "name" in author[0]:
                                    metadata["author"] = author[0]["name"]
                                    
                        if not metadata["published_date"] and "datePublished" in item:
                            metadata["published_date"] = item["datePublished"]
            
            except json.JSONDecodeError:
                pass
        
    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {e}")
    
    return metadata


def extract_links(html: str, base_url: str, same_domain_only: bool = False) -> List[str]:
    """
    Extract links from HTML content.
    
    Args:
        html: HTML content string
        base_url: Base URL for resolving relative links
        same_domain_only: If True, only return links from the same domain
        
    Returns:
        List of extracted URLs
    """
    links = []
    base_domain = urlparse(base_url).netloc
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # Skip empty links, anchors, javascript
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
                
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            
            # Normalize URL
            parsed = urlparse(absolute_url)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                normalized_url += f"?{parsed.query}"
                
            # Apply domain filter if requested
            if same_domain_only:
                link_domain = urlparse(normalized_url).netloc
                if link_domain != base_domain:
                    continue
                    
            links.append(normalized_url)
            
    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {e}")
        
    # Remove duplicates while preserving order
    unique_links = []
    seen = set()
    for link in links:
        if link not in seen:
            unique_links.append(link)
            seen.add(link)
            
    return unique_links


def get_website_content(url: str) -> Dict[str, Any]:
    """
    Comprehensive function to extract content from a website.
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary containing extracted content and metadata
    """
    result = {
        "url": url,
        "success": False,
        "timestamp": datetime.utcnow().isoformat(),
        "content": "",
        "html": "",
        "metadata": {},
        "links": []
    }
    
    # Fetch HTML
    html = get_html_content(url)
    if not html:
        return result
        
    # Store raw HTML
    result["html"] = html
    
    # Extract content
    content = extract_text_with_trafilatura(html, url)
    if not content:
        # Fallback to BeautifulSoup if Trafilatura fails
        content = extract_text_with_bs4(html, url)
        
    if content:
        result["content"] = content
        result["success"] = True
        
    # Extract metadata
    result["metadata"] = extract_metadata(html, url)
    
    # Extract links
    result["links"] = extract_links(html, url)
    
    # Generate content hash for deduplication
    if content:
        result["content_hash"] = hashlib.md5(content.encode('utf-8')).hexdigest()
    
    return result


def crawl_website(start_url: str, max_pages: int = 10, same_domain_only: bool = True) -> List[Dict[str, Any]]:
    """
    Crawl a website starting from a URL, following links up to a maximum number of pages.
    
    Args:
        start_url: Starting URL for the crawl
        max_pages: Maximum number of pages to crawl
        same_domain_only: If True, only crawl pages on the same domain
        
    Returns:
        List of dictionaries containing extracted content and metadata
    """
    results = []
    visited_urls = set()
    urls_to_visit = [start_url]
    base_domain = urlparse(start_url).netloc
    
    while urls_to_visit and len(results) < max_pages:
        url = urls_to_visit.pop(0)
        
        # Skip if already visited
        if url in visited_urls:
            continue
            
        # Mark as visited
        visited_urls.add(url)
        
        logger.info(f"Crawling: {url}")
        
        # Extract content
        result = get_website_content(url)
        if result["success"]:
            results.append(result)
            
            # Add links to visit queue
            for link in result["links"]:
                if link not in visited_urls and link not in urls_to_visit:
                    # Apply domain filter if requested
                    if same_domain_only:
                        link_domain = urlparse(link).netloc
                        if link_domain != base_domain:
                            continue
                    
                    urls_to_visit.append(link)
    
    return results


def search_for_content(search_term: str, search_engine: str = "bing", limit: int = 10) -> List[str]:
    """
    Search for URLs related to a specific search term.
    
    Args:
        search_term: Term to search for
        search_engine: Search engine to use ('bing', 'duckduckgo')
        limit: Maximum number of results to return
        
    Returns:
        List of URLs from search results
    """
    urls = []
    
    try:
        if search_engine.lower() == "bing":
            urls = _search_bing(search_term, limit)
        elif search_engine.lower() == "duckduckgo":
            urls = _search_duckduckgo(search_term, limit)
        else:
            logger.error(f"Unsupported search engine: {search_engine}")
    except Exception as e:
        logger.error(f"Error searching for '{search_term}' with {search_engine}: {e}")
    
    return urls


def _search_bing(search_term: str, limit: int = 10) -> List[str]:
    """
    Search using Bing.
    
    Args:
        search_term: Term to search for
        limit: Maximum number of results
        
    Returns:
        List of URLs from search results
    """
    urls = []
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    search_url = f"https://www.bing.com/search?q={search_term}"
    
    # Apply rate limiting
    respect_rate_limits("www.bing.com")
    
    try:
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract result links
        for a in soup.select("li.b_algo h2 a"):
            href = a.get("href")
            if href and not href.startswith(("/search", "javascript:")):
                urls.append(href)
                if len(urls) >= limit:
                    break
    
    except Exception as e:
        logger.error(f"Error searching Bing: {e}")
    
    return urls


def _search_duckduckgo(search_term: str, limit: int = 10) -> List[str]:
    """
    Search using DuckDuckGo.
    
    Args:
        search_term: Term to search for
        limit: Maximum number of results
        
    Returns:
        List of URLs from search results
    """
    urls = []
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    search_url = f"https://html.duckduckgo.com/html/?q={search_term}"
    
    # Apply rate limiting
    respect_rate_limits("html.duckduckgo.com")
    
    try:
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract result links
        for a in soup.select(".result__a"):
            href = a.get("href")
            if href:
                # Extract actual URL from DDG redirect
                if "uddg=" in href:
                    href = href.split("uddg=")[1].split("&")[0]
                    
                urls.append(href)
                if len(urls) >= limit:
                    break
    
    except Exception as e:
        logger.error(f"Error searching DuckDuckGo: {e}")
    
    return urls


# Add more scalable content collection methods
def process_urls_in_batches(urls: List[str], batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    Process a list of URLs in batches to prevent overwhelming the system.
    
    Args:
        urls: List of URLs to process
        batch_size: Number of URLs to process in each batch
        
    Returns:
        List of dictionaries containing extracted content and metadata
    """
    results = []
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        batch_results = []
        
        # Process the batch concurrently
        threads = []
        for url in batch:
            thread = threading.Thread(target=lambda u: batch_results.append(get_website_content(u)), args=(url,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        results.extend(batch_results)
        
        # Sleep between batches to prevent overwhelming the system
        time.sleep(2)
    
    return results