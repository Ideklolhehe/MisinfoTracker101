"""
Twitter data source module for the CIVILIAN system.
This module handles ingestion of content from Twitter.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import tweepy
from sqlalchemy.exc import SQLAlchemyError

from app import db
from models import DataSource, NarrativeInstance, SystemLog

logger = logging.getLogger(__name__)

class TwitterSource:
    """Data source connector for Twitter."""
    
    def __init__(self):
        """Initialize the Twitter data source."""
        self._running = False
        self._thread = None
        self._client = None
        
        # Try to initialize the API client
        try:
            self._initialize_client()
        except Exception as e:
            logger.warning(f"Twitter API credentials not provided: {e}")
        
        logger.info("TwitterSource initialized")
    
    def _initialize_client(self):
        """Initialize the Twitter API client with credentials."""
        import os
        
        # Check for required environment variables
        bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        consumer_key = os.environ.get('TWITTER_API_KEY')
        consumer_secret = os.environ.get('TWITTER_API_SECRET')
        access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
        access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')
        
        if bearer_token:
            # Use bearer token authentication (API v2)
            self._client = tweepy.Client(bearer_token=bearer_token)
            logger.info("Twitter API v2 client initialized with bearer token")
        elif consumer_key and consumer_secret and access_token and access_token_secret:
            # Use OAuth 1.0a authentication (API v1.1)
            auth = tweepy.OAuth1UserHandler(
                consumer_key, consumer_secret,
                access_token, access_token_secret
            )
            self._api = tweepy.API(auth)
            logger.info("Twitter API v1.1 client initialized with OAuth")
        else:
            raise ValueError("Twitter API credentials not provided")
    
    def start(self):
        """Start monitoring Twitter in a background thread."""
        if self._running:
            logger.warning("TwitterSource monitoring already running")
            return
        
        if not self._client and not hasattr(self, '_api'):
            logger.error("Twitter API client not initialized, cannot start monitoring")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("TwitterSource monitoring started")
    
    def stop(self):
        """Stop monitoring Twitter."""
        if not self._running:
            logger.warning("TwitterSource monitoring not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)  # Wait up to 3 seconds
            logger.info("TwitterSource monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        from config import Config
        
        while self._running:
            try:
                logger.debug("Starting Twitter monitoring cycle")
                
                # Get active Twitter sources from database
                sources = self._get_active_sources()
                
                if not sources:
                    logger.debug("No active Twitter sources defined")
                else:
                    # Process each source
                    for source in sources:
                        if not self._running:
                            break
                        
                        try:
                            # Extract query terms from config
                            config = json.loads(source.config) if source.config else {}
                            queries = config.get('queries', [])
                            
                            for query in queries:
                                if not self._running:
                                    break
                                self._process_query(source.id, query)
                            
                        except Exception as e:
                            self._log_error("process_source", f"Error processing source {source.id}: {e}")
                    
                    # Update last ingestion time
                    self._update_ingestion_time(sources)
                
                # Sleep until next cycle (respect Twitter API rate limits)
                for _ in range(int(Config.INGESTION_INTERVAL / 2)):
                    if not self._running:
                        break
                    time.sleep(2)  # Check if still running every 2 seconds
            
            except Exception as e:
                self._log_error("monitoring_loop", f"Error in Twitter monitoring loop: {e}")
                time.sleep(60)  # Shorter interval after error
    
    def _get_active_sources(self) -> List[DataSource]:
        """Get active Twitter data sources from the database."""
        try:
            return DataSource.query.filter_by(
                source_type='twitter',
                is_active=True
            ).all()
        except Exception as e:
            self._log_error("get_sources", f"Error fetching Twitter sources: {e}")
            return []
    
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
    
    def _process_query(self, source_id: int, query: str, max_tweets: int = 50):
        """Process a Twitter search query."""
        logger.debug(f"Processing query: {query}")
        
        try:
            # Get tweets matching the query
            tweets = self.search_tweets(query, max_tweets)
            
            # Process each tweet
            for tweet in tweets:
                # Create a narrative instance for detection
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=tweet.get('text', ''),
                    meta_data=json.dumps(tweet),
                    url=tweet.get('url', '')
                )
                
                # Add to database
                db.session.add(instance)
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {len(tweets)} tweets for query: {query}")
        
        except Exception as e:
            self._log_error("process_query", f"Error processing query {query}: {e}")
            db.session.rollback()
    
    def search_tweets(self, query: str, max_tweets: int = 50) -> List[Dict[str, Any]]:
        """Search for tweets matching a query (for manual API use).
        
        Args:
            query: Twitter search query
            max_tweets: Maximum number of tweets to retrieve
            
        Returns:
            tweets: List of tweet dictionaries
        """
        result = []
        
        try:
            if self._client:  # API v2
                # Use Twitter API v2
                tweets = self._client.search_recent_tweets(
                    query=query,
                    max_results=min(max_tweets, 100),  # API limit is 100
                    tweet_fields=['created_at', 'author_id', 'public_metrics']
                )
                
                if not tweets.data:
                    return []
                
                for tweet in tweets.data:
                    tweet_dict = tweet.data
                    tweet_dict['url'] = f"https://twitter.com/user/status/{tweet.id}"
                    result.append(tweet_dict)
            
            elif hasattr(self, '_api'):  # API v1.1
                # Use Twitter API v1.1
                tweets = self._api.search_tweets(
                    q=query,
                    count=min(max_tweets, 100),  # API limit is 100
                    tweet_mode='extended'
                )
                
                for tweet in tweets:
                    result.append({
                        'id': tweet.id_str,
                        'text': tweet.full_text,
                        'created_at': tweet.created_at.isoformat(),
                        'author_id': tweet.user.id_str,
                        'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
                        'user': {
                            'id': tweet.user.id_str,
                            'name': tweet.user.name,
                            'screen_name': tweet.user.screen_name
                        },
                        'retweet_count': tweet.retweet_count,
                        'favorite_count': tweet.favorite_count
                    })
            
            return result
        
        except Exception as e:
            logger.error(f"Error searching tweets for query '{query}': {e}")
            return []
    
    def create_source(self, name: str, config: Dict[str, Any]) -> Optional[int]:
        """Create a new Twitter data source in the database.
        
        Args:
            name: Name for the data source
            config: Configuration dictionary with 'queries' list
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Validate config
            if not isinstance(config, dict) or 'queries' not in config:
                logger.error("Invalid config: must contain 'queries' list")
                return None
            
            if not isinstance(config['queries'], list) or not all(isinstance(q, str) for q in config['queries']):
                logger.error("Invalid config: 'queries' must be a list of strings")
                return None
            
            # Create a new data source
            source = DataSource(
                name=name,
                source_type='twitter',
                config=json.dumps(config),
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            # Add to database
            db.session.add(source)
            db.session.commit()
            
            logger.info(f"Created Twitter source: {name} (ID: {source.id})")
            return source.id
        
        except Exception as e:
            db.session.rollback()
            self._log_error("create_source", f"Error creating Twitter source: {e}")
            return None
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log = SystemLog(
                timestamp=datetime.utcnow(),
                log_type='error',
                component='twitter_source',
                message=f"{operation}: {message}"
            )
            db.session.add(log)
            db.session.commit()
        except SQLAlchemyError:
            logger.error(f"Failed to log error to database: {message}")
            db.session.rollback()
        except Exception as e:
            logger.error(f"Error logging to database: {e}")