import logging
import time
import threading
import feedparser
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime, timedelta
import hashlib
import trafilatura

# Import application components
from models import DataSource, NarrativeInstance, SystemLog
from app import db

logger = logging.getLogger(__name__)

class RSSSource:
    """Data source connector for RSS feeds."""
    
    def __init__(self):
        """Initialize the RSS data source."""
        self.running = False
        self.thread = None
        self.query_interval = 300  # 5 minutes between queries
        
        logger.info("RSSSource initialized")
    
    def start(self):
        """Start monitoring RSS feeds in a background thread."""
        if self.running:
            logger.warning("RSSSource is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("RSSSource monitoring started")
        
    def stop(self):
        """Stop monitoring RSS feeds."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("RSSSource monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        while self.running:
            try:
                # Log start of monitoring cycle
                logger.debug("Starting RSS monitoring cycle")
                
                # Get active sources from database
                sources = self._get_active_sources()
                
                if not sources:
                    logger.debug("No active RSS sources defined")
                    time.sleep(60)  # Sleep briefly before checking again
                    continue
                
                # Process each source configuration
                for source in sources:
                    try:
                        config = json.loads(source.config) if source.config else {}
                        if 'feeds' in config:
                            for feed_url in config['feeds']:
                                self._process_feed(source.id, feed_url)
                    except Exception as e:
                        logger.error(f"Error processing RSS source {source.id}: {e}")
                
                # Update last ingestion timestamp
                self._update_ingestion_time(sources)
                
                # Wait for next cycle
                logger.debug(f"RSS monitoring cycle complete, sleeping for {self.query_interval} seconds")
                time.sleep(self.query_interval)
                
            except Exception as e:
                logger.error(f"Error in RSS monitoring loop: {e}")
                # Log error to database
                self._log_error("monitoring_loop", str(e))
                time.sleep(60)  # Short sleep on error
    
    def _get_active_sources(self) -> List[DataSource]:
        """Get active RSS data sources from the database."""
        try:
            sources = DataSource.query.filter_by(
                source_type='rss',
                is_active=True
            ).all()
            return sources
        except Exception as e:
            logger.error(f"Error fetching RSS sources: {e}")
            return []
    
    def _update_ingestion_time(self, sources: List[DataSource]):
        """Update the last ingestion timestamp for sources."""
        try:
            with db.session.begin():
                for source in sources:
                    source.last_ingestion = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error updating ingestion time: {e}")
    
    def _process_feed(self, source_id: int, feed_url: str, max_entries: int = 50):
        """Process an RSS feed."""
        try:
            logger.debug(f"Fetching RSS feed: {feed_url}")
            
            # Parse the RSS feed
            feed = feedparser.parse(feed_url)
            
            if not feed or not hasattr(feed, 'entries'):
                logger.warning(f"Failed to parse feed or no entries found: {feed_url}")
                return
            
            # Process entries
            count = 0
            with db.session.begin():
                for entry in feed.entries[:max_entries]:
                    try:
                        # Extract data from entry
                        title = entry.get('title', '')
                        summary = entry.get('summary', '')
                        link = entry.get('link', '')
                        published = entry.get('published', '')
                        author = entry.get('author', '')
                        
                        # Create a unique ID for the entry
                        entry_id = hashlib.md5((link + str(published)).encode()).hexdigest()
                        
                        # Check if this entry already exists in the database
                        existing = NarrativeInstance.query.filter_by(
                            source_id=source_id,
                            url=link
                        ).first()
                        
                        if existing:
                            continue  # Skip this entry as we've already processed it
                        
                        # Get full content if available
                        content = summary
                        try:
                            if link:
                                # Use trafilatura to extract article content
                                downloaded = trafilatura.fetch_url(link)
                                if downloaded:
                                    extracted = trafilatura.extract(downloaded)
                                    if extracted:
                                        content = extracted
                        except Exception as content_err:
                            logger.warning(f"Error extracting content for {link}: {content_err}")
                        
                        # Create metadata
                        metadata = json.dumps({
                            'entry_id': entry_id,
                            'title': title,
                            'author': author,
                            'published': published,
                            'feed_url': feed_url,
                            'feed_title': feed.feed.get('title', ''),
                        })
                        
                        # Create a new narrative instance
                        instance = NarrativeInstance(
                            source_id=source_id,
                            content=f"{title}\n\n{content}",
                            meta_data=metadata,
                            url=link,
                            detected_at=datetime.utcnow()
                        )
                        db.session.add(instance)
                        count += 1
                        
                    except Exception as entry_err:
                        logger.error(f"Error processing RSS entry: {entry_err}")
            
            logger.info(f"Processed {count} entries from feed: {feed_url}")
            
        except Exception as e:
            logger.error(f"Error processing RSS feed '{feed_url}': {e}")
            self._log_error("process_feed", f"Error for feed '{feed_url}': {e}")
    
    def fetch_feed(self, feed_url: str, max_entries: int = 20) -> List[Dict[str, Any]]:
        """Fetch an RSS feed and return its entries (for manual API use).
        
        Args:
            feed_url: URL of the RSS feed
            max_entries: Maximum number of entries to retrieve
            
        Returns:
            entries: List of entry dictionaries
        """
        try:
            logger.debug(f"Fetching RSS feed: {feed_url}")
            
            # Parse the RSS feed
            feed = feedparser.parse(feed_url)
            
            if not feed or not hasattr(feed, 'entries'):
                logger.warning(f"Failed to parse feed or no entries found: {feed_url}")
                return []
            
            # Process entries
            result = []
            for entry in feed.entries[:max_entries]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                link = entry.get('link', '')
                published = entry.get('published', '')
                author = entry.get('author', '')
                
                result.append({
                    'title': title,
                    'summary': summary,
                    'content': summary,  # Will be basic content without extraction
                    'author': author,
                    'published': published,
                    'url': link,
                    'feed_title': feed.feed.get('title', '')
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed '{feed_url}': {e}")
            return []
    
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
            if 'feeds' not in config or not isinstance(config['feeds'], list):
                logger.error("RSS source config must contain 'feeds' list")
                return None
                
            with db.session.begin():
                source = DataSource(
                    name=name,
                    source_type='rss',
                    config=json.dumps(config),
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.session.add(source)
                db.session.flush()
                
                source_id = source.id
                
            logger.info(f"Created RSS data source: {name} (ID: {source_id})")
            return source_id
            
        except Exception as e:
            logger.error(f"Error creating RSS data source '{name}': {e}")
            return None
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="rss_source",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")