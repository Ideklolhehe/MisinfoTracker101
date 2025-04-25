"""
YouTube data source module for the CIVILIAN system.
This module handles ingestion of content from YouTube, including videos and comments.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import googleapiclient.discovery
import googleapiclient.errors
from sqlalchemy.exc import SQLAlchemyError

from app import db, app
from models import DataSource, NarrativeInstance, SystemLog

logger = logging.getLogger(__name__)

class YouTubeSource:
    """Data source connector for YouTube."""
    
    def __init__(self):
        """Initialize the YouTube data source."""
        self._running = False
        self._thread = None
        self._api = None
        
        # Try to initialize the API client
        try:
            self._initialize_client()
        except Exception as e:
            logger.warning(f"YouTube API credentials not provided: {e}")
        
        logger.info("YouTubeSource initialized")
    
    def _initialize_client(self):
        """Initialize the YouTube API client with credentials."""
        import os
        
        # Check for required environment variables
        api_key = os.environ.get('YOUTUBE_API_KEY')
        
        if not api_key:
            raise ValueError("YouTube API key not provided")
        
        # Initialize YouTube API client
        self._api = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key, cache_discovery=False
        )
        logger.info("YouTube API client initialized")
    
    def start(self):
        """Start monitoring YouTube in a background thread."""
        if self._running:
            logger.warning("YouTubeSource monitoring already running")
            return
        
        if not self._api:
            logger.error("YouTube API client not initialized, cannot start monitoring")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("YouTubeSource monitoring started")
    
    def stop(self):
        """Stop monitoring YouTube."""
        if not self._running:
            logger.warning("YouTubeSource monitoring not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)  # Wait up to 3 seconds
            logger.info("YouTubeSource monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        from config import Config
        
        while self._running:
            try:
                logger.debug("Starting YouTube monitoring cycle")
                
                # Get active YouTube sources from database
                with app.app_context():
                    sources = self._get_active_sources()
                    
                    if not sources:
                        logger.debug("No active YouTube sources defined")
                    else:
                        # Process each source
                        for source in sources:
                            if not self._running:
                                break
                            
                            try:
                                # Extract monitor targets from config
                                config = json.loads(source.config) if source.config else {}
                                
                                # Process based on monitoring type
                                monitor_type = config.get('monitor_type', 'channel')
                                
                                if monitor_type == 'channel':
                                    # Monitor YouTube channels
                                    channel_ids = config.get('channel_ids', [])
                                    for channel_id in channel_ids:
                                        if not self._running:
                                            break
                                        self._process_channel(source.id, channel_id, config)
                                
                                elif monitor_type == 'search':
                                    # Monitor YouTube search results
                                    search_queries = config.get('search_queries', [])
                                    for query in search_queries:
                                        if not self._running:
                                            break
                                        self._process_search(source.id, query, config)
                                
                                elif monitor_type == 'video':
                                    # Monitor specific YouTube videos
                                    video_ids = config.get('video_ids', [])
                                    for video_id in video_ids:
                                        if not self._running:
                                            break
                                        self._process_video(source.id, video_id, config)
                                
                                elif monitor_type == 'playlist':
                                    # Monitor YouTube playlists
                                    playlist_ids = config.get('playlist_ids', [])
                                    for playlist_id in playlist_ids:
                                        if not self._running:
                                            break
                                        self._process_playlist(source.id, playlist_id, config)
                                
                            except Exception as e:
                                self._log_error("process_source", f"Error processing source {source.id}: {e}")
                        
                        # Update last ingestion time
                        self._update_ingestion_time(sources)
                
                # Sleep until next cycle (respect YouTube API rate limits)
                for _ in range(int(Config.INGESTION_INTERVAL / 2)):
                    if not self._running:
                        break
                    time.sleep(2)  # Check if still running every 2 seconds
            
            except Exception as e:
                with app.app_context():
                    self._log_error("monitoring_loop", f"Error in YouTube monitoring loop: {e}")
                time.sleep(60)  # Shorter interval after error
    
    def _get_active_sources(self) -> List[DataSource]:
        """Get active YouTube data sources from the database."""
        try:
            return DataSource.query.filter_by(
                source_type='youtube',
                is_active=True
            ).all()
        except Exception as e:
            self._log_error("get_sources", f"Error fetching YouTube sources: {e}")
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
    
    def _process_channel(self, source_id: int, channel_id: str, config: Dict[str, Any]):
        """Process videos from a YouTube channel."""
        logger.debug(f"Processing YouTube channel: {channel_id}")
        
        try:
            # Get channel details
            channel_info = self.get_channel_info(channel_id)
            if not channel_info:
                logger.warning(f"Could not find channel info for {channel_id}")
                return
            
            # Get recent videos from the channel
            max_results = config.get('max_videos', 10)
            published_after = datetime.utcnow() - timedelta(days=config.get('days_back', 7))
            
            videos = self.get_channel_videos(
                channel_id,
                max_results=max_results,
                published_after=published_after
            )
            
            # Process each video
            for video in videos:
                # Check if we should process comments
                if config.get('include_comments', False):
                    # Get comments for the video
                    comments = self.get_video_comments(
                        video['id'],
                        max_results=config.get('max_comments', 50)
                    )
                    # Add comments to video data
                    video['comments'] = comments
                
                # Create a narrative instance for detection
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=f"{video.get('title', '')} {video.get('description', '')}",
                    meta_data=json.dumps(video),
                    url=f"https://www.youtube.com/watch?v={video.get('id')}"
                )
                
                # Add to database
                db.session.add(instance)
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {len(videos)} videos from channel {channel_id}")
        
        except Exception as e:
            self._log_error("process_channel", f"Error processing channel {channel_id}: {e}")
            db.session.rollback()
    
    def _process_search(self, source_id: int, query: str, config: Dict[str, Any]):
        """Process videos from a YouTube search."""
        logger.debug(f"Processing YouTube search: {query}")
        
        try:
            # Get search results
            max_results = config.get('max_videos', 10)
            published_after = datetime.utcnow() - timedelta(days=config.get('days_back', 7))
            
            videos = self.search_videos(
                query,
                max_results=max_results,
                published_after=published_after
            )
            
            # Process each video
            for video in videos:
                # Get additional video details
                video_details = self.get_video_details(video['id'])
                if video_details:
                    video.update(video_details)
                
                # Check if we should process comments
                if config.get('include_comments', False):
                    # Get comments for the video
                    comments = self.get_video_comments(
                        video['id'],
                        max_results=config.get('max_comments', 50)
                    )
                    # Add comments to video data
                    video['comments'] = comments
                
                # Create a narrative instance for detection
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=f"{video.get('title', '')} {video.get('description', '')}",
                    meta_data=json.dumps(video),
                    url=f"https://www.youtube.com/watch?v={video.get('id')}"
                )
                
                # Add to database
                db.session.add(instance)
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {len(videos)} videos from search {query}")
        
        except Exception as e:
            self._log_error("process_search", f"Error processing search {query}: {e}")
            db.session.rollback()
    
    def _process_video(self, source_id: int, video_id: str, config: Dict[str, Any]):
        """Process a specific YouTube video."""
        logger.debug(f"Processing YouTube video: {video_id}")
        
        try:
            # Get video details
            video = self.get_video_details(video_id)
            if not video:
                logger.warning(f"Could not find video with ID {video_id}")
                return
            
            # Check if we should process comments
            if config.get('include_comments', False):
                # Get comments for the video
                comments = self.get_video_comments(
                    video_id,
                    max_results=config.get('max_comments', 50)
                )
                # Add comments to video data
                video['comments'] = comments
            
            # Create a narrative instance for detection
            instance = NarrativeInstance(
                source_id=source_id,
                content=f"{video.get('title', '')} {video.get('description', '')}",
                meta_data=json.dumps(video),
                url=f"https://www.youtube.com/watch?v={video_id}"
            )
            
            # Add to database
            db.session.add(instance)
            db.session.commit()
            logger.debug(f"Processed video {video_id}")
        
        except Exception as e:
            self._log_error("process_video", f"Error processing video {video_id}: {e}")
            db.session.rollback()
    
    def _process_playlist(self, source_id: int, playlist_id: str, config: Dict[str, Any]):
        """Process videos from a YouTube playlist."""
        logger.debug(f"Processing YouTube playlist: {playlist_id}")
        
        try:
            # Get videos from the playlist
            max_results = config.get('max_videos', 10)
            
            videos = self.get_playlist_videos(
                playlist_id,
                max_results=max_results
            )
            
            # Process each video
            for video in videos:
                # Get additional video details
                video_details = self.get_video_details(video['id'])
                if video_details:
                    video.update(video_details)
                
                # Check if we should process comments
                if config.get('include_comments', False):
                    # Get comments for the video
                    comments = self.get_video_comments(
                        video['id'],
                        max_results=config.get('max_comments', 50)
                    )
                    # Add comments to video data
                    video['comments'] = comments
                
                # Create a narrative instance for detection
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=f"{video.get('title', '')} {video.get('description', '')}",
                    meta_data=json.dumps(video),
                    url=f"https://www.youtube.com/watch?v={video.get('id')}"
                )
                
                # Add to database
                db.session.add(instance)
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {len(videos)} videos from playlist {playlist_id}")
        
        except Exception as e:
            self._log_error("process_playlist", f"Error processing playlist {playlist_id}: {e}")
            db.session.rollback()
    
    def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            channel_info: Dictionary with channel details
        """
        try:
            # Call the API
            request = self._api.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id
            )
            response = request.execute()
            
            # Extract channel info
            if response.get('items'):
                channel = response['items'][0]
                return {
                    'id': channel['id'],
                    'title': channel['snippet']['title'],
                    'description': channel['snippet'].get('description', ''),
                    'customUrl': channel['snippet'].get('customUrl', ''),
                    'publishedAt': channel['snippet']['publishedAt'],
                    'thumbnails': channel['snippet'].get('thumbnails', {}),
                    'statistics': channel.get('statistics', {}),
                    'uploads_playlist': channel['contentDetails']['relatedPlaylists'].get('uploads', '')
                }
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_id}: {e}")
            return None
    
    def get_channel_videos(self, channel_id: str, max_results: int = 10, 
                          published_after: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get recent videos from a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to retrieve
            published_after: Only get videos published after this date
            
        Returns:
            videos: List of video dictionaries
        """
        try:
            # First, get the channel's uploads playlist
            channel_info = self.get_channel_info(channel_id)
            if not channel_info or not channel_info.get('uploads_playlist'):
                logger.warning(f"Could not find uploads playlist for channel {channel_id}")
                return []
            
            # Get videos from the uploads playlist
            uploads_playlist_id = channel_info['uploads_playlist']
            return self.get_playlist_videos(uploads_playlist_id, max_results, published_after)
        
        except Exception as e:
            logger.error(f"Error getting videos for channel {channel_id}: {e}")
            return []
    
    def get_playlist_videos(self, playlist_id: str, max_results: int = 10,
                           published_after: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get videos from a YouTube playlist.
        
        Args:
            playlist_id: YouTube playlist ID
            max_results: Maximum number of videos to retrieve
            published_after: Only get videos published after this date
            
        Returns:
            videos: List of video dictionaries
        """
        try:
            # Call the API to get playlist items
            request = self._api.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=min(max_results, 50)  # API limit is 50
            )
            response = request.execute()
            
            # Extract video info
            videos = []
            for item in response.get('items', []):
                # Get the video ID
                video_id = item['contentDetails']['videoId']
                
                # Get published date
                published_at = datetime.strptime(
                    item['snippet']['publishedAt'],
                    '%Y-%m-%dT%H:%M:%SZ'
                )
                
                # Skip videos published before the cutoff date
                if published_after and published_at < published_after:
                    continue
                
                # Get detailed video info
                video_details = self.get_video_details(video_id)
                if video_details:
                    videos.append(video_details)
                
                # Check if we've reached the limit
                if len(videos) >= max_results:
                    break
            
            return videos
        
        except Exception as e:
            logger.error(f"Error getting videos for playlist {playlist_id}: {e}")
            return []
    
    def search_videos(self, query: str, max_results: int = 10,
                     published_after: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Search for YouTube videos matching a query.
        
        Args:
            query: Search query
            max_results: Maximum number of videos to retrieve
            published_after: Only get videos published after this date
            
        Returns:
            videos: List of video dictionaries
        """
        try:
            # Format the published_after parameter
            published_after_str = None
            if published_after:
                published_after_str = published_after.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Call the API
            request = self._api.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=min(max_results, 50),  # API limit is 50
                publishedAfter=published_after_str,
                relevanceLanguage="en"  # Prefer English results
            )
            response = request.execute()
            
            # Extract video info
            videos = []
            for item in response.get('items', []):
                # Get the video ID
                video_id = item['id']['videoId']
                
                # Get detailed video info
                video_details = self.get_video_details(video_id)
                if video_details:
                    videos.append(video_details)
                
                # Check if we've reached the limit
                if len(videos) >= max_results:
                    break
            
            return videos
        
        except Exception as e:
            logger.error(f"Error searching videos for query '{query}': {e}")
            return []
    
    def get_video_details(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            video_details: Dictionary with video details
        """
        try:
            # Call the API
            request = self._api.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            response = request.execute()
            
            # Extract video info
            if response.get('items'):
                video = response['items'][0]
                return {
                    'id': video['id'],
                    'title': video['snippet']['title'],
                    'description': video['snippet'].get('description', ''),
                    'publishedAt': video['snippet']['publishedAt'],
                    'channelId': video['snippet']['channelId'],
                    'channelTitle': video['snippet']['channelTitle'],
                    'thumbnails': video['snippet'].get('thumbnails', {}),
                    'tags': video['snippet'].get('tags', []),
                    'categoryId': video['snippet'].get('categoryId', ''),
                    'duration': video['contentDetails']['duration'],
                    'statistics': video.get('statistics', {})
                }
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error getting video details for {video_id}: {e}")
            return None
    
    def get_video_comments(self, video_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get comments for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve
            
        Returns:
            comments: List of comment dictionaries
        """
        try:
            # Call the API
            request = self._api.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=min(max_results, 100)  # API limit is 100
            )
            response = request.execute()
            
            # Extract comment info
            comments = []
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'id': item['id'],
                    'text': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'authorChannelId': comment.get('authorChannelId', {}).get('value'),
                    'publishedAt': comment['publishedAt'],
                    'likeCount': comment['likeCount'],
                    'totalReplyCount': item['snippet']['totalReplyCount']
                })
            
            return comments
        
        except googleapiclient.errors.HttpError as e:
            # Comments might be disabled for the video
            if e.resp.status == 403:
                logger.warning(f"Comments are disabled for video {video_id}")
                return []
            else:
                logger.error(f"Error getting comments for video {video_id}: {e}")
                return []
        except Exception as e:
            logger.error(f"Error getting comments for video {video_id}: {e}")
            return []
    
    def create_source(self, name: str, config: Dict[str, Any]) -> Optional[int]:
        """Create a new YouTube data source in the database.
        
        Args:
            name: Name for the data source
            config: Configuration dictionary with monitoring parameters
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Validate config
            if not isinstance(config, dict):
                logger.error("Invalid config: must be a dictionary")
                return None
            
            # Validate monitor type
            monitor_type = config.get('monitor_type', 'channel')
            if monitor_type not in ['channel', 'search', 'video', 'playlist']:
                logger.error(f"Invalid monitor_type: {monitor_type}")
                return None
            
            # Validate required fields based on monitor_type
            if monitor_type == 'channel' and not config.get('channel_ids'):
                logger.error("Channel IDs are required for channel monitoring")
                return None
            elif monitor_type == 'search' and not config.get('search_queries'):
                logger.error("Search queries are required for search monitoring")
                return None
            elif monitor_type == 'video' and not config.get('video_ids'):
                logger.error("Video IDs are required for video monitoring")
                return None
            elif monitor_type == 'playlist' and not config.get('playlist_ids'):
                logger.error("Playlist IDs are required for playlist monitoring")
                return None
            
            # Create a new data source
            source = DataSource(
                name=name,
                source_type='youtube',
                config=json.dumps(config),
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            # Add to database
            with app.app_context():
                db.session.add(source)
                db.session.commit()
                
                logger.info(f"Created YouTube source: {name} (ID: {source.id})")
                return source.id
        
        except Exception as e:
            with app.app_context():
                db.session.rollback()
                self._log_error("create_source", f"Error creating YouTube source: {e}")
            return None
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log = SystemLog(
                timestamp=datetime.utcnow(),
                log_type='error',
                component='youtube_source',
                message=f"{operation}: {message}"
            )
            db.session.add(log)
            db.session.commit()
        except SQLAlchemyError:
            logger.error(f"Failed to log error to database: {message}")
            db.session.rollback()
        except Exception as e:
            logger.error(f"Error logging to database: {e}")