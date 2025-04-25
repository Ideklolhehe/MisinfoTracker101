import logging
import trafilatura
from typing import Optional, Dict, Any
import json
from urllib.parse import urlparse
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class WebScraper:
    """Utility for scraping web content from URLs."""
    
    def __init__(self):
        """Initialize the web scraper."""
        logger.info("WebScraper initialized")
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            result: Dictionary with extracted content and metadata
        """
        try:
            logger.debug(f"Scraping URL: {url}")
            
            # Parse the URL to extract domain
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Download the content
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                logger.warning(f"Failed to download content from URL: {url}")
                return {
                    "success": False,
                    "error": "Failed to download content",
                    "url": url,
                    "domain": domain,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Extract the main text content
            content = trafilatura.extract(downloaded)
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            
            # Build result
            result = {
                "success": True,
                "url": url,
                "domain": domain,
                "timestamp": datetime.utcnow().isoformat(),
                "content": content or "",
                "title": metadata.title if metadata else None,
                "author": metadata.author if metadata else None,
                "date": metadata.date if metadata else None,
                "description": metadata.description if metadata else None,
                "sitename": metadata.sitename if metadata else None,
                "categories": metadata.categories if metadata else None,
                "tags": metadata.tags if metadata else None
            }
            
            logger.info(f"Successfully scraped URL: {url}")
            return result
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "domain": urlparse(url).netloc if url else None,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def extract_links(self, url: str) -> Dict[str, Any]:
        """Extract links from a web page.
        
        Args:
            url: URL to extract links from
            
        Returns:
            result: Dictionary with extracted links and metadata
        """
        try:
            logger.debug(f"Extracting links from URL: {url}")
            
            # Download the content
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                logger.warning(f"Failed to download content from URL: {url}")
                return {
                    "success": False,
                    "error": "Failed to download content",
                    "url": url,
                    "links": []
                }
            
            # Extract links using trafilatura's HTML parsing
            links = []
            
            # Use requests to get headers and potential redirects
            response = requests.get(url, timeout=10)
            final_url = response.url
            
            # Use trafilatura to extract links
            links = trafilatura.extract_links(downloaded)
            
            return {
                "success": True,
                "url": url,
                "final_url": final_url,
                "status_code": response.status_code,
                "links": links or [],
                "count": len(links) if links else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting links from URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "links": []
            }
    
    def get_website_text_content(self, url: str) -> str:
        """Get the main text content from a URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            text: Extracted text content
        """
        try:
            # Send a request to the website
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            return text or ""
        except Exception as e:
            logger.error(f"Error getting text content from URL {url}: {e}")
            return ""