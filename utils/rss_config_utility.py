"""
Utility for configuring and managing RSS data sources.
This module provides functions to easily add and manage RSS feeds in the system.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union

from app import db
from models import DataSource
from data_sources.rss_source import RSSSource

logger = logging.getLogger(__name__)

class RSSConfigUtility:
    """Utility class for RSS source configuration."""
    
    def __init__(self):
        """Initialize the RSS configuration utility."""
        self.rss_source = RSSSource()
    
    def add_feed(self, name: str, feed_url: str) -> Optional[int]:
        """Add a single RSS feed to the system.
        
        Args:
            name: Name for the data source
            feed_url: URL of the RSS feed
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Create config with a single feed URL
            config = {'feeds': [feed_url]}
            
            # Create the source
            source_id = self.rss_source.create_source(name, config)
            if source_id:
                logger.info(f"Added RSS feed: {name} ({feed_url})")
                return source_id
            else:
                logger.error(f"Failed to add RSS feed: {name}")
                return None
        except Exception as e:
            logger.error(f"Error adding RSS feed '{name}': {e}")
            return None
    
    def add_multiple_feeds(self, feeds: List[Dict[str, str]]) -> Dict[str, Any]:
        """Add multiple RSS feeds to the system.
        
        Args:
            feeds: List of dictionaries with 'name' and 'url' keys
            
        Returns:
            result: Dictionary with success/failure counts and details
        """
        success = []
        failed = []
        
        for feed in feeds:
            name = feed.get('name')
            url = feed.get('url')
            
            if not name or not url:
                failed.append({'name': name, 'url': url, 'reason': 'Missing name or URL'})
                continue
            
            source_id = self.add_feed(name, url)
            if source_id:
                success.append({'name': name, 'url': url, 'id': source_id})
            else:
                failed.append({'name': name, 'url': url, 'reason': 'Failed to create source'})
        
        return {
            'success_count': len(success),
            'failure_count': len(failed),
            'success': success,
            'failed': failed
        }
    
    def get_all_rss_sources(self) -> List[Dict[str, Any]]:
        """Get all RSS data sources in the system.
        
        Returns:
            sources: List of RSS source dictionaries
        """
        try:
            sources = DataSource.query.filter_by(source_type='rss').all()
            result = []
            
            for source in sources:
                feed_info = {
                    'id': source.id,
                    'name': source.name,
                    'is_active': source.is_active,
                    'created_at': source.created_at.isoformat() if source.created_at else None,
                    'last_ingestion': source.last_ingestion.isoformat() if source.last_ingestion else None,
                    'feeds': []
                }
                
                # Parse config to extract feed URLs
                if source.config:
                    try:
                        config = json.loads(source.config)
                        feed_info['feeds'] = config.get('feeds', [])
                    except json.JSONDecodeError:
                        pass
                
                result.append(feed_info)
            
            return result
        except Exception as e:
            logger.error(f"Error getting RSS sources: {e}")
            return []
    
    def import_from_opml(self, opml_content: str) -> Dict[str, Any]:
        """Import RSS feeds from OPML format.
        
        Args:
            opml_content: OPML file content as string
            
        Returns:
            result: Dictionary with import results
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Parse OPML
            root = ET.fromstring(opml_content)
            feeds = []
            
            # Extract feeds from OPML
            for outline in root.findall('.//outline'):
                if outline.get('type') == 'rss' and outline.get('xmlUrl'):
                    feeds.append({
                        'name': outline.get('title') or outline.get('text', 'Unknown Feed'),
                        'url': outline.get('xmlUrl')
                    })
            
            # Add feeds to the system
            return self.add_multiple_feeds(feeds)
        except Exception as e:
            logger.error(f"Error importing OPML: {e}")
            return {
                'success_count': 0,
                'failure_count': 0,
                'error': str(e)
            }
    
    def test_feed(self, feed_url: str) -> Dict[str, Any]:
        """Test an RSS feed by fetching its content.
        
        Args:
            feed_url: URL of the RSS feed to test
            
        Returns:
            result: Test results including feed title and entry count
        """
        try:
            # Use the RSSSource's fetch_feed method
            entries = self.rss_source.fetch_feed(feed_url, max_entries=5)
            
            if entries:
                # Get feed title if available in first entry
                feed_title = entries[0].get('feed_title', 'Unknown Feed') if entries else 'Unknown Feed'
                
                return {
                    'success': True,
                    'feed_url': feed_url,
                    'feed_title': feed_title,
                    'entry_count': len(entries),
                    'sample_entries': [
                        {
                            'title': entry.get('title', 'No title'),
                            'published': entry.get('published', 'Unknown date')
                        }
                        for entry in entries[:3]  # Include up to 3 sample entries
                    ]
                }
            else:
                return {
                    'success': False,
                    'feed_url': feed_url,
                    'error': 'No entries found or feed not parsable'
                }
        except Exception as e:
            logger.error(f"Error testing feed {feed_url}: {e}")
            return {
                'success': False,
                'feed_url': feed_url,
                'error': str(e)
            }