import logging
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime, timedelta
import tweepy

# Import application components
from models import DataSource, NarrativeInstance, SystemLog
from app import db

logger = logging.getLogger(__name__)

class TwitterSource:
    """Data source connector for Twitter/X."""
    
    def __init__(self, credentials: Dict[str, str] = None):
        """Initialize the Twitter data source.
        
        Args:
            credentials: Dict containing API keys (optional, can also use env vars)
        """
        self.credentials = credentials or {}
        self.client = None
        self.running = False
        self.thread = None
        self.rate_limit = int(os.environ.get('TWITTER_API_RATE_LIMIT', 60))
        self.query_interval = 300  # 5 minutes between queries
        
        # Initialize the Twitter API client
        self._init_client()
        
        logger.info("TwitterSource initialized")
    
    def _init_client(self):
        """Initialize the Twitter API client."""
        try:
            # Get API credentials from environment or passed dictionary
            api_key = self.credentials.get('api_key') or os.environ.get('TWITTER_API_KEY')
            api_secret = self.credentials.get('api_secret') or os.environ.get('TWITTER_API_SECRET')
            access_token = self.credentials.get('access_token') or os.environ.get('TWITTER_ACCESS_TOKEN')
            access_secret = self.credentials.get('access_secret') or os.environ.get('TWITTER_ACCESS_SECRET')
            bearer_token = self.credentials.get('bearer_token') or os.environ.get('TWITTER_BEARER_TOKEN')
            
            if not bearer_token and not (api_key and api_secret):
                logger.warning("Twitter API credentials not provided")
                return
                
            # Initialize tweepy client
            if bearer_token:
                self.client = tweepy.Client(bearer_token=bearer_token)
            else:
                auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
                self.client = tweepy.API(auth)
                
            logger.info("Twitter API client initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Twitter API client: {e}")
            self.client = None
    
    def start(self):
        """Start monitoring Twitter in a background thread."""
        if self.running:
            logger.warning("TwitterSource is already running")
            return
            
        if not self.client:
            logger.error("Twitter API client not initialized, cannot start monitoring")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("TwitterSource monitoring started")
        
    def stop(self):
        """Stop monitoring Twitter."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("TwitterSource monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        while self.running:
            try:
                # Log start of monitoring cycle
                logger.debug("Starting Twitter monitoring cycle")
                
                # Get active sources from database
                sources = self._get_active_sources()
                
                if not sources:
                    logger.debug("No active Twitter sources defined")
                    time.sleep(60)  # Sleep briefly before checking again
                    continue
                
                # Process each source configuration
                for source in sources:
                    try:
                        config = json.loads(source.config) if source.config else {}
                        if 'query' in config:
                            self._process_search_query(source.id, config['query'])
                        elif 'users' in config:
                            for user in config['users']:
                                self._process_user_timeline(source.id, user)
                    except Exception as e:
                        logger.error(f"Error processing Twitter source {source.id}: {e}")
                
                # Update last ingestion timestamp
                self._update_ingestion_time(sources)
                
                # Wait for next cycle
                logger.debug(f"Twitter monitoring cycle complete, sleeping for {self.query_interval} seconds")
                time.sleep(self.query_interval)
                
            except Exception as e:
                logger.error(f"Error in Twitter monitoring loop: {e}")
                # Log error to database
                self._log_error("monitoring_loop", str(e))
                time.sleep(60)  # Short sleep on error
    
    def _get_active_sources(self) -> List[DataSource]:
        """Get active Twitter data sources from the database."""
        try:
            sources = DataSource.query.filter_by(
                source_type='twitter',
                is_active=True
            ).all()
            return sources
        except Exception as e:
            logger.error(f"Error fetching Twitter sources: {e}")
            return []
    
    def _update_ingestion_time(self, sources: List[DataSource]):
        """Update the last ingestion timestamp for sources."""
        try:
            with db.session.begin():
                for source in sources:
                    source.last_ingestion = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error updating ingestion time: {e}")
    
    def _process_search_query(self, source_id: int, query: str, count: int = 100):
        """Process a Twitter search query."""
        if not self.client:
            logger.error("Twitter API client not initialized")
            return
            
        try:
            logger.debug(f"Executing Twitter search query: {query}")
            
            # Get search results
            # Using tweepy.Client method for Twitter API v2
            if hasattr(self.client, 'search_recent_tweets'):
                # Twitter API v2
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=min(count, 100),
                    tweet_fields=['created_at', 'lang', 'author_id', 'conversation_id']
                )
                
                if hasattr(response, 'data') and response.data:
                    tweets = response.data
                else:
                    tweets = []
            else:
                # Twitter API v1.1 (fallback)
                tweets = self.client.search_tweets(
                    q=query,
                    count=count,
                    tweet_mode='extended',
                    result_type='recent'
                )
            
            # Process tweets
            with db.session.begin():
                for tweet in tweets:
                    # Extract tweet data (different structure in v1.1 vs v2)
                    if hasattr(tweet, 'full_text'):
                        # API v1.1
                        tweet_id = tweet.id_str
                        text = tweet.full_text
                        created_at = tweet.created_at
                        user_id = tweet.user.id_str
                        user_name = tweet.user.screen_name
                        lang = tweet.lang
                    else:
                        # API v2
                        tweet_id = tweet.id
                        text = tweet.text
                        created_at = tweet.created_at
                        user_id = tweet.author_id
                        user_name = None  # Need separate user lookup in v2
                        lang = getattr(tweet, 'lang', 'unknown')
                    
                    # Create metadata
                    metadata = json.dumps({
                        'tweet_id': tweet_id,
                        'user_id': user_id,
                        'user_name': user_name,
                        'created_at': created_at.isoformat() if created_at else None,
                        'lang': lang,
                        'query': query
                    })
                    
                    # Create a new narrative instance
                    instance = NarrativeInstance(
                        source_id=source_id,
                        content=text,
                        metadata=metadata,
                        url=f"https://twitter.com/i/web/status/{tweet_id}",
                        detected_at=datetime.utcnow()
                    )
                    db.session.add(instance)
            
            logger.info(f"Processed {len(tweets)} tweets for query: {query}")
            
        except Exception as e:
            logger.error(f"Error processing Twitter search query '{query}': {e}")
            self._log_error("search_query", f"Error for query '{query}': {e}")
    
    def _process_user_timeline(self, source_id: int, username: str, count: int = 50):
        """Process a Twitter user timeline."""
        if not self.client:
            logger.error("Twitter API client not initialized")
            return
            
        try:
            logger.debug(f"Retrieving Twitter timeline for user: {username}")
            
            # Get user timeline
            # Using tweepy.Client method for Twitter API v2
            if hasattr(self.client, 'get_user') and hasattr(self.client, 'get_users_tweets'):
                # Twitter API v2
                # First get user ID
                user_response = self.client.get_user(username=username)
                if not hasattr(user_response, 'data') or not user_response.data:
                    logger.warning(f"User not found: {username}")
                    return
                    
                user_id = user_response.data.id
                
                # Then get tweets
                response = self.client.get_users_tweets(
                    id=user_id,
                    max_results=min(count, 100),
                    tweet_fields=['created_at', 'lang', 'conversation_id']
                )
                
                if hasattr(response, 'data') and response.data:
                    tweets = response.data
                else:
                    tweets = []
            else:
                # Twitter API v1.1 (fallback)
                tweets = self.client.user_timeline(
                    screen_name=username,
                    count=count,
                    tweet_mode='extended'
                )
            
            # Process tweets
            with db.session.begin():
                for tweet in tweets:
                    # Extract tweet data (different structure in v1.1 vs v2)
                    if hasattr(tweet, 'full_text'):
                        # API v1.1
                        tweet_id = tweet.id_str
                        text = tweet.full_text
                        created_at = tweet.created_at
                        user_id = tweet.user.id_str
                        user_name = tweet.user.screen_name
                        lang = tweet.lang
                    else:
                        # API v2
                        tweet_id = tweet.id
                        text = tweet.text
                        created_at = tweet.created_at
                        user_id = user_id  # From user lookup above
                        user_name = username
                        lang = getattr(tweet, 'lang', 'unknown')
                    
                    # Create metadata
                    metadata = json.dumps({
                        'tweet_id': tweet_id,
                        'user_id': user_id,
                        'user_name': user_name,
                        'created_at': created_at.isoformat() if created_at else None,
                        'lang': lang
                    })
                    
                    # Create a new narrative instance
                    instance = NarrativeInstance(
                        source_id=source_id,
                        content=text,
                        metadata=metadata,
                        url=f"https://twitter.com/i/web/status/{tweet_id}",
                        detected_at=datetime.utcnow()
                    )
                    db.session.add(instance)
            
            logger.info(f"Processed {len(tweets)} tweets for user: {username}")
            
        except Exception as e:
            logger.error(f"Error processing Twitter user timeline '{username}': {e}")
            self._log_error("user_timeline", f"Error for user '{username}': {e}")
    
    def search_tweets(self, query: str, count: int = 100) -> List[Dict[str, Any]]:
        """Search for tweets matching a query (for manual API use).
        
        Args:
            query: Twitter search query
            count: Maximum number of tweets to retrieve
            
        Returns:
            tweets: List of tweet dictionaries
        """
        if not self.client:
            logger.error("Twitter API client not initialized")
            return []
            
        try:
            logger.debug(f"Executing Twitter search query: {query}")
            
            # Get search results
            # Using tweepy.Client method for Twitter API v2
            if hasattr(self.client, 'search_recent_tweets'):
                # Twitter API v2
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=min(count, 100),
                    tweet_fields=['created_at', 'lang', 'author_id', 'conversation_id']
                )
                
                if hasattr(response, 'data') and response.data:
                    tweets = response.data
                else:
                    tweets = []
            else:
                # Twitter API v1.1 (fallback)
                tweets = self.client.search_tweets(
                    q=query,
                    count=count,
                    tweet_mode='extended',
                    result_type='recent'
                )
            
            # Format tweet data
            result = []
            for tweet in tweets:
                # Extract tweet data (different structure in v1.1 vs v2)
                if hasattr(tweet, 'full_text'):
                    # API v1.1
                    tweet_id = tweet.id_str
                    text = tweet.full_text
                    created_at = tweet.created_at
                    user_id = tweet.user.id_str
                    user_name = tweet.user.screen_name
                    lang = tweet.lang
                else:
                    # API v2
                    tweet_id = tweet.id
                    text = tweet.text
                    created_at = tweet.created_at
                    user_id = tweet.author_id
                    user_name = None  # Need separate user lookup in v2
                    lang = getattr(tweet, 'lang', 'unknown')
                
                result.append({
                    'tweet_id': tweet_id,
                    'text': text,
                    'user_id': user_id,
                    'user_name': user_name,
                    'created_at': created_at.isoformat() if created_at else None,
                    'lang': lang,
                    'url': f"https://twitter.com/i/web/status/{tweet_id}"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching tweets for query '{query}': {e}")
            return []
    
    def create_source(self, name: str, config: Dict[str, Any]) -> Optional[int]:
        """Create a new Twitter data source in the database.
        
        Args:
            name: Name for the data source
            config: Configuration dictionary (query or users list)
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Validate config
            if not ('query' in config or 'users' in config):
                logger.error("Twitter source config must contain 'query' or 'users'")
                return None
                
            with db.session.begin():
                source = DataSource(
                    name=name,
                    source_type='twitter',
                    config=json.dumps(config),
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.session.add(source)
                db.session.flush()
                
                source_id = source.id
                
            logger.info(f"Created Twitter data source: {name} (ID: {source_id})")
            return source_id
            
        except Exception as e:
            logger.error(f"Error creating Twitter data source '{name}': {e}")
            return None
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="twitter_source",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
