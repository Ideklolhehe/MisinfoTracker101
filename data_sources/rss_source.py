"""
RSS data source module for the CIVILIAN system.
This module handles ingestion of content from RSS feeds.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import feedparser
from sqlalchemy.exc import SQLAlchemyError

from app import db
from models import DataSource, NarrativeInstance, SystemLog
from utils.app_context import ensure_app_context

logger = logging.getLogger(__name__)

class RSSSource:
    """Data source connector for RSS feeds."""
    
    def __init__(self):
        """Initialize the RSS data source."""
        self._running = False
        self._thread = None
        logger.info("RSSSource initialized")
    
    def start(self):
        """Start monitoring RSS feeds in a background thread."""
        if self._running:
            logger.warning("RSSSource monitoring already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("RSSSource monitoring started")
    
    def stop(self):
        """Stop monitoring RSS feeds."""
        if not self._running:
            logger.warning("RSSSource monitoring not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)  # Wait up to 3 seconds
            logger.info("RSSSource monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        from config import Config
        
        while self._running:
            try:
                logger.debug("Starting RSS monitoring cycle")
                
                # Get active RSS sources from database
                sources = self._get_active_sources()
                
                if not sources:
                    logger.debug("No active RSS sources defined")
                else:
                    # Process each source
                    for source in sources:
                        if not self._running:
                            break
                        
                        try:
                            # Extract feed URLs from config
                            config = json.loads(source.config) if source.config else {}
                            feeds = config.get('feeds', [])
                            
                            for feed_url in feeds:
                                if not self._running:
                                    break
                                self._process_feed(source.id, feed_url)
                            
                        except Exception as e:
                            self._log_error("process_source", f"Error processing source {source.id}: {e}")
                    
                    # Update last ingestion time
                    self._update_ingestion_time(sources)
                
                # Sleep until next cycle
                for _ in range(int(Config.INGESTION_INTERVAL / 2)):
                    if not self._running:
                        break
                    time.sleep(2)  # Check if still running every 2 seconds
            
            except Exception as e:
                self._log_error("monitoring_loop", f"Error in RSS monitoring loop: {e}")
                time.sleep(60)  # Shorter interval after error
    
    @ensure_app_context
    def _get_active_sources(self) -> List[DataSource]:
        """Get active RSS data sources from the database."""
        try:
            return DataSource.query.filter_by(
                source_type='rss',
                is_active=True
            ).all()
        except Exception as e:
            self._log_error("get_sources", f"Error fetching RSS sources: {e}")
            return []
    
    @ensure_app_context
    def _update_ingestion_time(self, sources: List[DataSource]):
        """Update the last ingestion timestamp for sources."""
        current_time = datetime.utcnow()
        try:
            for source in sources:
                source.last_ingestion = current_time
            db.session.commit()
        except Exception as e:
            self._log_error("update_ingestion_time", f"Error updating ingestion time: {e}")
            db.session.rollback()
    
    @ensure_app_context
    def _process_feed(self, source_id: int, feed_url: str, max_entries: int = 50):
        """Process an RSS feed."""
        logger.debug(f"Processing feed: {feed_url}")
        
        try:
            # Fetch and parse the feed
            entries = self.fetch_feed(feed_url, max_entries)
            
            from models import DetectedNarrative
            processed_count = 0
            
            # Process each entry
            for entry in entries:
                content = entry.get('content', '')
                if not content:
                    continue  # Skip empty content
                
                # We need to either find or create a narrative since narrative_id is required
                # For now, we'll create a simple narrative for each feed item
                title = entry.get('title', 'Untitled')
                
                # Check if we have a similar narrative already
                narrative = DetectedNarrative.query.filter_by(
                    title=title
                ).first()
                
                # If no existing narrative, create one
                if not narrative:
                    narrative = DetectedNarrative(
                        title=title,
                        description=content[:500] if len(content) > 500 else content,
                        confidence_score=0.5,  # Default confidence
                        language='en',         # Default language
                        status='unverified'    # Mark as unverified initially
                    )
                    db.session.add(narrative)
                    db.session.flush()  # Get an ID without committing
                
                # Create the narrative instance linked to the narrative
                instance = NarrativeInstance(
                    narrative_id=narrative.id,  # Link to the narrative
                    source_id=source_id,
                    content=content,
                    meta_data=json.dumps(entry),
                    url=entry.get('link', '')
                )
                
                # Add to database
                db.session.add(instance)
                processed_count += 1
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {processed_count} entries from {feed_url}")
        
        except Exception as e:
            self._log_error("process_feed", f"Error processing feed {feed_url}: {e}")
            db.session.rollback()
    
    def fetch_feed(self, feed_url: str, max_entries: int = 20) -> List[Dict[str, Any]]:
        """Fetch an RSS feed and return its entries (for manual API use).
        
        Args:
            feed_url: URL of the RSS feed
            max_entries: Maximum number of entries to retrieve
            
        Returns:
            entries: List of entry dictionaries
        """
        try:
            # Parse the feed
            feed = feedparser.parse(feed_url)
            
            if feed.get('bozo', 0) == 1:
                # Feed parsing had errors
                logger.warning(f"Feed parsing error for {feed_url}: {feed.get('bozo_exception')}")
            
            # Extract feed title
            feed_title = feed.get('feed', {}).get('title', 'Unknown Feed')
            
            # Process entries
            result = []
            for i, entry in enumerate(feed.get('entries', [])):
                if i >= max_entries:
                    break
                
                # Extract content
                content = ""
                if 'content' in entry:
                    content = ' '.join([c.get('value', '') for c in entry.content])
                elif 'summary' in entry:
                    content = entry.summary
                elif 'description' in entry:
                    content = entry.description
                
                # Create entry dict
                entry_data = {
                    'title': entry.get('title', 'No Title'),
                    'content': content,
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'feed_title': feed_title,
                    'author': entry.get('author', '')
                }
                
                result.append(entry_data)
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            return []
    
    @ensure_app_context
    def create_source(self, name: str, config: Dict[str, Any]) -> Optional[int]:
        """Create a new RSS data source in the database.
        
        Args:
            name: Name for the data source
            config: Configuration dictionary with 'feeds' list
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Validate config
            if not isinstance(config, dict) or 'feeds' not in config:
                logger.error("Invalid config: must contain 'feeds' list")
                return None
            
            if not isinstance(config['feeds'], list) or not all(isinstance(f, str) for f in config['feeds']):
                logger.error("Invalid config: 'feeds' must be a list of strings")
                return None
            
            # Create a new data source
            source = DataSource(
                name=name,
                source_type='rss',
                config=json.dumps(config),
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            # Add to database
            db.session.add(source)
            db.session.commit()
            
            logger.info(f"Created RSS source: {name} (ID: {source.id})")
            return source.id
        
        except Exception as e:
            db.session.rollback()
            self._log_error("create_source", f"Error creating RSS source: {e}")
            return None
    
    @ensure_app_context
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log = SystemLog(
                timestamp=datetime.utcnow(),
                log_type='error',
                component='rss_source',
                message=f"{operation}: {message}"
            )
            db.session.add(log)
            db.session.commit()
        except SQLAlchemyError:
            logger.error(f"Failed to log error to database: {message}")
            db.session.rollback()
        except Exception as e:
            logger.error(f"Error logging to database: {e}")