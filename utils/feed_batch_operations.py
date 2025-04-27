"""
Batch operations utility for RSS feeds in the CIVILIAN system.
This module provides functions for batch processing of RSS feeds.
"""

import logging
import concurrent.futures
import time
from typing import List, Dict, Tuple, Optional, Any

from app import db
from models import DataSource
from utils.feed_parser import parse_feed_with_retry, get_parser

logger = logging.getLogger(__name__)

def test_all_feeds(max_workers: int = 5, timeout: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Test all RSS feeds in the system and return their status.
    
    Args:
        max_workers: Maximum number of concurrent workers
        timeout: Timeout for each feed test in seconds
        
    Returns:
        Dictionary mapping feed URLs to their status
    """
    feeds = DataSource.query.filter_by(source_type='rss').all()
    feed_urls = [feed.source_url for feed in feeds if feed.source_url]
    
    return test_feed_batch(feed_urls, max_workers, timeout)

def test_feed_batch(feed_urls: List[str], max_workers: int = 5, timeout: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Test a batch of RSS feed URLs and return their status.
    
    Args:
        feed_urls: List of feed URLs to test
        max_workers: Maximum number of concurrent workers
        timeout: Timeout for each feed test in seconds
        
    Returns:
        Dictionary mapping feed URLs to their status
    """
    results = {}
    
    def test_feed(url: str) -> Tuple[str, Dict[str, Any]]:
        """Test a single feed and return its status."""
        start_time = time.time()
        try:
            feed = parse_feed_with_retry(url, max_retries=1)
            
            if feed and hasattr(feed, 'entries'):
                status = {
                    'status': 'success',
                    'entry_count': len(feed.entries),
                    'feed_title': feed.feed.get('title', 'Unknown'),
                    'response_time': time.time() - start_time
                }
            else:
                status = {
                    'status': 'error',
                    'error': 'No entries found',
                    'response_time': time.time() - start_time
                }
        except Exception as e:
            status = {
                'status': 'error',
                'error': str(e),
                'response_time': time.time() - start_time
            }
            
        return url, status
    
    # Use ThreadPoolExecutor to parallelize testing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(test_feed, url): url for url in feed_urls}
        
        for future in concurrent.futures.as_completed(future_to_url, timeout=timeout):
            try:
                url, status = future.result()
                results[url] = status
            except concurrent.futures.TimeoutError:
                url = future_to_url[future]
                results[url] = {
                    'status': 'error',
                    'error': 'Timeout'
                }
            except Exception as e:
                url = future_to_url[future]
                results[url] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    return results

def update_feed_statuses(test_results: Dict[str, Dict[str, Any]]) -> int:
    """
    Update feed status fields in the database based on test results.
    
    Args:
        test_results: Dictionary mapping feed URLs to their status
        
    Returns:
        Number of feeds updated
    """
    updated_count = 0
    
    for url, status in test_results.items():
        try:
            feed = DataSource.query.filter_by(source_url=url).first()
            
            if not feed:
                continue
                
            # Update feed status fields
            if status['status'] == 'success':
                meta_data = feed.meta_data or {}
                meta_data['last_status'] = 'ok'
                meta_data['last_checked'] = time.time()
                meta_data['entry_count'] = status.get('entry_count', 0)
                meta_data['response_time'] = status.get('response_time', 0)
                meta_data['error'] = None
            else:
                meta_data = feed.meta_data or {}
                meta_data['last_status'] = 'error'
                meta_data['last_checked'] = time.time()
                meta_data['error'] = status.get('error', 'Unknown error')
                
            feed.meta_data = meta_data
            db.session.add(feed)
            updated_count += 1
                
        except Exception as e:
            logger.error(f"Error updating feed status for {url}: {e}")
            
    if updated_count > 0:
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error committing feed status updates: {e}")
            db.session.rollback()
            return 0
            
    return updated_count

def find_alternative_feeds(feed_url: str) -> List[str]:
    """
    Try to discover alternative feed URLs for a broken feed.
    
    Args:
        feed_url: Original feed URL that is not working
        
    Returns:
        List of potential alternative feed URLs
    """
    import re
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse, urljoin
    
    alternatives = []
    
    try:
        # Extract the base URL
        parsed_url = urlparse(feed_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Try to fetch the website's homepage
        response = requests.get(base_url, timeout=10)
        if response.status_code != 200:
            return alternatives
            
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for RSS/Atom feed links
        feed_links = soup.find_all('link', rel=['alternate', 'feed'], type=['application/rss+xml', 'application/atom+xml'])
        
        for link in feed_links:
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                if full_url != feed_url and full_url not in alternatives:
                    alternatives.append(full_url)
                    
        # If no feed links found, try common patterns
        common_patterns = [
            '/feed',
            '/rss',
            '/atom',
            '/feed/atom',
            '/feed/rss',
            '/rss.xml',
            '/atom.xml',
            '/feeds/posts/default',
            '/rss/all.xml',
        ]
        
        for pattern in common_patterns:
            alt_url = urljoin(base_url, pattern)
            if alt_url != feed_url and alt_url not in alternatives:
                alternatives.append(alt_url)
                
    except Exception as e:
        logger.error(f"Error finding alternative feeds for {feed_url}: {e}")
        
    return alternatives

def test_alternative_feeds(feed_url: str) -> Optional[str]:
    """
    Test alternative feed URLs and return the first working one.
    
    Args:
        feed_url: Original feed URL that is not working
        
    Returns:
        Working alternative feed URL or None if none found
    """
    alternatives = find_alternative_feeds(feed_url)
    
    if not alternatives:
        return None
        
    # Test each alternative
    test_results = test_feed_batch(alternatives)
    
    # Return the first working alternative
    for url, status in test_results.items():
        if status['status'] == 'success' and status.get('entry_count', 0) > 0:
            return url
            
    return None

def update_feed_urls(broken_feeds: List[str]) -> Dict[str, str]:
    """
    Try to find and update alternative URLs for broken feeds.
    
    Args:
        broken_feeds: List of broken feed URLs
        
    Returns:
        Dictionary mapping original URLs to new URLs
    """
    updates = {}
    
    for feed_url in broken_feeds:
        try:
            alternative = test_alternative_feeds(feed_url)
            
            if alternative:
                # Update the feed URL in the database
                feed = DataSource.query.filter_by(source_url=feed_url).first()
                
                if feed:
                    old_url = feed.source_url
                    feed.source_url = alternative
                    
                    # Update metadata
                    meta_data = feed.meta_data or {}
                    meta_data['previous_url'] = old_url
                    meta_data['url_updated'] = time.time()
                    feed.meta_data = meta_data
                    
                    db.session.add(feed)
                    updates[feed_url] = alternative
        except Exception as e:
            logger.error(f"Error updating feed URL for {feed_url}: {e}")
            
    if updates:
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error committing feed URL updates: {e}")
            db.session.rollback()
            updates = {}
            
    return updates