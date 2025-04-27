"""
RSS Feed Manager for CIVILIAN system.
This module provides tools for managing and validating RSS feed sources.
"""

import logging
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import requests
from sqlalchemy.exc import SQLAlchemyError

from app import db
from models import DataSource, SystemLog
from utils.feed_parser import FeedParser

logger = logging.getLogger(__name__)

class RSSFeedManager:
    """Manages RSS feed sources for the CIVILIAN system."""
    
    def __init__(self):
        """Initialize RSS feed manager."""
        self.feed_parser = FeedParser(timeout=10, retries=1)
        
    def validate_feed(self, feed_url: str) -> Tuple[bool, str]:
        """
        Validate an RSS feed URL to ensure it's accessible and properly formatted.
        
        Args:
            feed_url: The URL of the RSS feed to validate
            
        Returns:
            (is_valid, message): Tuple with validity status and message
        """
        try:
            feed_entries = self.feed_parser.get_feed_entries(feed_url, max_entries=1)
            
            if feed_entries:
                return True, f"Valid feed with {len(feed_entries)} entries available"
            else:
                return False, "Feed parsing successful but no entries found"
        except Exception as e:
            return False, f"Error validating feed: {str(e)}"
            
    def find_alternative_feed_url(self, domain: str) -> Optional[str]:
        """
        Try to find an alternative feed URL for a domain.
        
        Args:
            domain: Domain name to search for feeds
            
        Returns:
            Alternative feed URL if found, None otherwise
        """
        common_paths = [
            '/feed',
            '/rss',
            '/feeds/posts/default',
            '/atom.xml',
            '/rss.xml',
            '/feed/atom',
            '/feed/rss',
            '/index.xml',
            '/blog/feed',
            '/news/feed',
            '/feed.xml',
        ]
        
        # Try common feed paths
        for path in common_paths:
            test_url = f"https://{domain}{path}"
            valid, _ = self.validate_feed(test_url)
            
            if valid:
                return test_url
                
        # Try to scrape the website for feed links
        try:
            response = requests.get(
                f"https://{domain}",
                headers={'User-Agent': 'Mozilla/5.0 (compatible; CIVILIAN/1.0)'},
                timeout=10
            )
            
            if response.status_code == 200:
                # Check for common RSS feed link patterns in the HTML
                html = response.text.lower()
                
                # Look for RSS feed links
                feed_indicators = [
                    'type="application/rss+xml"',
                    'type="application/atom+xml"',
                    'rel="alternate" type="application/rss+xml"',
                    'rel="alternate" type="application/atom+xml"',
                    '<link rel="feed"',
                ]
                
                for indicator in feed_indicators:
                    if indicator in html:
                        # Extract the URL using a simple approach
                        start_idx = html.find('href=', html.find(indicator))
                        if start_idx != -1:
                            start_idx += 6  # Skip past 'href="'
                            end_idx = html.find('"', start_idx)
                            if end_idx != -1:
                                feed_url = html[start_idx:end_idx]
                                
                                # Make relative URLs absolute
                                if feed_url.startswith('/'):
                                    feed_url = f"https://{domain}{feed_url}"
                                
                                # Validate the feed
                                valid, _ = self.validate_feed(feed_url)
                                if valid:
                                    return feed_url
        except Exception as e:
            logger.warning(f"Error searching for feeds on {domain}: {e}")
            
        return None
    
    def check_feeds_in_database(self) -> List[Dict[str, Any]]:
        """
        Check all RSS feeds in the database for validity.
        
        Returns:
            List of feed status information
        """
        feed_status = []
        
        try:
            # Get all RSS sources from the database
            sources = DataSource.query.filter_by(source_type='rss').all()
            
            for source in sources:
                try:
                    config = json.loads(source.config) if source.config else {}
                    feeds = config.get('feeds', [])
                    
                    for feed_url in feeds:
                        valid, message = self.validate_feed(feed_url)
                        alternative = None
                        
                        if not valid:
                            # Try to find an alternative
                            try:
                                domain = feed_url.split('//')[-1].split('/')[0]
                                alternative = self.find_alternative_feed_url(domain)
                            except Exception as e:
                                logger.warning(f"Error finding alternative feed: {e}")
                        
                        feed_status.append({
                            'source_id': source.id,
                            'source_name': source.name,
                            'feed_url': feed_url,
                            'valid': valid,
                            'message': message,
                            'alternative': alternative
                        })
                    
                except Exception as e:
                    logger.error(f"Error checking feeds for source {source.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error querying database for RSS sources: {e}")
            
        return feed_status
        
    def update_feed_url(self, source_id: int, old_url: str, new_url: str) -> bool:
        """
        Update a feed URL in the database.
        
        Args:
            source_id: ID of the data source
            old_url: Original feed URL
            new_url: New feed URL
            
        Returns:
            Success status
        """
        try:
            # Get the data source
            source = DataSource.query.get(source_id)
            if not source:
                logger.error(f"Data source {source_id} not found")
                return False
                
            # Parse the config
            config = json.loads(source.config) if source.config else {}
            feeds = config.get('feeds', [])
            
            # Update the feed URL
            if old_url in feeds:
                feeds[feeds.index(old_url)] = new_url
                config['feeds'] = feeds
                source.config = json.dumps(config)
                
                # Save to database
                db.session.commit()
                
                # Log the change
                log = SystemLog(
                    log_type='info',
                    component='rss_feed_manager',
                    message=f"Updated feed URL from {old_url} to {new_url} for source {source.name}"
                )
                db.session.add(log)
                db.session.commit()
                
                return True
            else:
                logger.warning(f"Feed URL {old_url} not found in source {source.id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating feed URL: {e}")
            db.session.rollback()
            return False
            
    def add_feed_url(self, source_id: int, new_url: str) -> bool:
        """
        Add a new feed URL to a data source.
        
        Args:
            source_id: ID of the data source
            new_url: New feed URL to add
            
        Returns:
            Success status
        """
        try:
            # Get the data source
            source = DataSource.query.get(source_id)
            if not source:
                logger.error(f"Data source {source_id} not found")
                return False
                
            # Parse the config
            config = json.loads(source.config) if source.config else {}
            feeds = config.get('feeds', [])
            
            # Check if URL already exists
            if new_url in feeds:
                logger.warning(f"Feed URL {new_url} already exists in source {source.id}")
                return True
                
            # Validate the feed
            valid, message = self.validate_feed(new_url)
            if not valid:
                logger.warning(f"Invalid feed URL {new_url}: {message}")
                return False
                
            # Add the new URL
            feeds.append(new_url)
            config['feeds'] = feeds
            source.config = json.dumps(config)
            
            # Save to database
            db.session.commit()
            
            # Log the change
            log = SystemLog(
                log_type='info',
                component='rss_feed_manager',
                message=f"Added feed URL {new_url} to source {source.name}"
            )
            db.session.add(log)
            db.session.commit()
            
            return True
                
        except Exception as e:
            logger.error(f"Error adding feed URL: {e}")
            db.session.rollback()
            return False
            
    def remove_feed_url(self, source_id: int, url: str) -> bool:
        """
        Remove a feed URL from a data source.
        
        Args:
            source_id: ID of the data source
            url: Feed URL to remove
            
        Returns:
            Success status
        """
        try:
            # Get the data source
            source = DataSource.query.get(source_id)
            if not source:
                logger.error(f"Data source {source_id} not found")
                return False
                
            # Parse the config
            config = json.loads(source.config) if source.config else {}
            feeds = config.get('feeds', [])
            
            # Check if URL exists
            if url not in feeds:
                logger.warning(f"Feed URL {url} not found in source {source.id}")
                return False
                
            # Remove the URL
            feeds.remove(url)
            config['feeds'] = feeds
            source.config = json.dumps(config)
            
            # Save to database
            db.session.commit()
            
            # Log the change
            log = SystemLog(
                log_type='info',
                component='rss_feed_manager',
                message=f"Removed feed URL {url} from source {source.name}"
            )
            db.session.add(log)
            db.session.commit()
            
            return True
                
        except Exception as e:
            logger.error(f"Error removing feed URL: {e}")
            db.session.rollback()
            return False