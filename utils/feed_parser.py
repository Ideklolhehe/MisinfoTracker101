"""
Enhanced feed parser for the CIVILIAN system.
This module provides robust RSS and Atom feed parsing with error handling.
"""

import logging
import feedparser
import requests
import xml.etree.ElementTree as ET
from requests.exceptions import RequestException
from urllib.parse import urlparse
import time
import random

logger = logging.getLogger(__name__)

class FeedParser:
    """Enhanced feed parser with error handling and rate limiting."""
    
    def __init__(self, timeout=10, retries=2, backoff_factor=0.5):
        """
        Initialize the feed parser.
        
        Args:
            timeout: Request timeout in seconds
            retries: Number of retries for failed requests
            backoff_factor: Exponential backoff factor between retries
        """
        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        
        # Common headers to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/rss+xml, application/atom+xml, application/xml, text/xml, */*'
        }
    
    def parse(self, url):
        """
        Parse feed from URL with robust error handling.
        
        Args:
            url: The URL of the feed to parse
            
        Returns:
            Parsed feed object or None if parsing failed
        """
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid feed URL: {url}")
            return None
            
        # Try direct parsing first
        feed = feedparser.parse(url)
        
        # Check if parsing was successful
        if feed.bozo == 0 and hasattr(feed, 'entries') and len(feed.entries) > 0:
            return feed
            
        # If direct parsing failed, try with requests
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    url, 
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '').lower()
                    
                    # Check for valid content types
                    if any(valid_type in content_type for valid_type in ['xml', 'rss', 'atom']):
                        # Try parsing the content
                        feed = feedparser.parse(response.content)
                        
                        if feed.bozo == 0 and hasattr(feed, 'entries') and len(feed.entries) > 0:
                            return feed
                    
                    # Try to fix common XML issues
                    try:
                        fixed_content = self._fix_xml_content(response.text)
                        feed = feedparser.parse(fixed_content)
                        
                        if feed.bozo == 0 and hasattr(feed, 'entries') and len(feed.entries) > 0:
                            return feed
                    except Exception as e:
                        logger.debug(f"XML fixing failed for {url}: {str(e)}")
                
                # If we reach here, the attempt failed
                logger.warning(f"Feed parsing attempt {attempt+1}/{self.retries+1} failed for {url}")
                
                # Apply backoff before retry
                if attempt < self.retries:
                    sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_time)
                    
            except RequestException as e:
                logger.warning(f"Request error for {url} (attempt {attempt+1}/{self.retries+1}): {str(e)}")
                
                # Apply backoff before retry
                if attempt < self.retries:
                    sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_time)
        
        # All attempts failed
        logger.error(f"Feed parsing failed after {self.retries+1} attempts for {url}")
        return None
    
    def _fix_xml_content(self, content):
        """
        Attempt to fix common XML issues in feed content.
        
        Args:
            content: The XML content to fix
            
        Returns:
            Fixed XML content as string
        """
        # Replace common problematic characters
        fixed = content.replace('&nbsp;', ' ')
        fixed = fixed.replace('&ndash;', '-')
        fixed = fixed.replace('&mdash;', '—')
        
        # Fix unclosed CDATA sections
        if '<![CDATA[' in fixed and ']]>' not in fixed:
            fixed = fixed.replace('<![CDATA[', '') 
            
        # Add XML declaration if missing
        if not fixed.startswith('<?xml'):
            fixed = '<?xml version="1.0" encoding="UTF-8"?>\n' + fixed
            
        # Try parsing with ElementTree to validate
        try:
            ET.fromstring(fixed)
            return fixed
        except Exception:
            # If parsing fails, return the original content
            # feedparser might still be able to handle it
            return content
            
    def get_feed_entries(self, url, max_entries=50):
        """
        Get entries from a feed with a maximum limit.
        
        Args:
            url: The URL of the feed
            max_entries: Maximum number of entries to return
            
        Returns:
            List of feed entries or empty list if parsing failed
        """
        feed = self.parse(url)
        
        if not feed or not hasattr(feed, 'entries'):
            return []
            
        return feed.entries[:max_entries]