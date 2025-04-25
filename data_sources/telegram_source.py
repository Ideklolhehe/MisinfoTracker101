import logging
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime, timedelta
import asyncio
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError

# Import application components
from models import DataSource, NarrativeInstance, SystemLog
from app import db

logger = logging.getLogger(__name__)

class TelegramSource:
    """Data source connector for Telegram."""
    
    def __init__(self, credentials: Dict[str, str] = None):
        """Initialize the Telegram data source.
        
        Args:
            credentials: Dict containing API keys (optional, can also use env vars)
        """
        self.credentials = credentials or {}
        self.client = None
        self.running = False
        self.thread = None
        self.rate_limit = int(os.environ.get('TELEGRAM_API_RATE_LIMIT', 30))
        self.channels = []  # List of channels to monitor
        
        # Initialize the Telegram client
        self._init_client()
        
        logger.info("TelegramSource initialized")
    
    def _init_client(self):
        """Initialize the Telegram client."""
        try:
            # Get API credentials from environment or passed dictionary
            api_id = self.credentials.get('api_id') or os.environ.get('TELEGRAM_API_ID')
            api_hash = self.credentials.get('api_hash') or os.environ.get('TELEGRAM_API_HASH')
            phone = self.credentials.get('phone') or os.environ.get('TELEGRAM_PHONE')
            session_name = self.credentials.get('session_name') or 'civilian_telegram_session'
            
            if not api_id or not api_hash:
                logger.warning("Telegram API credentials not provided")
                return
                
            # Will be initialized in the background thread
            self.api_id = api_id
            self.api_hash = api_hash
            self.phone = phone
            self.session_name = session_name
            
            logger.info("Telegram API credentials initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Telegram credentials: {e}")
    
    async def _create_client(self):
        """Create the Telegram client asynchronously."""
        try:
            # Create client
            client = TelegramClient(self.session_name, self.api_id, self.api_hash)
            await client.start(phone=self.phone)
            
            # Check if we're logged in
            if await client.is_user_authorized():
                logger.info("Successfully logged in to Telegram")
                self.client = client
                return client
            else:
                logger.error("Failed to authenticate with Telegram")
                return None
                
        except SessionPasswordNeededError:
            logger.error("Two-factor authentication required for Telegram account")
            return None
            
        except Exception as e:
            logger.error(f"Error creating Telegram client: {e}")
            return None
    
    def start(self):
        """Start monitoring Telegram in a background thread."""
        if self.running:
            logger.warning("TelegramSource is already running")
            return
            
        if not hasattr(self, 'api_id') or not self.api_id:
            logger.error("Telegram API credentials not initialized, cannot start monitoring")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("TelegramSource monitoring started")
        
    def stop(self):
        """Stop monitoring Telegram."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            
        # Close client
        if self.client:
            asyncio.run_coroutine_threadsafe(self.client.disconnect(), asyncio.get_event_loop())
            self.client = None
            
        logger.info("TelegramSource monitoring stopped")
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize client
            client = loop.run_until_complete(self._create_client())
            if not client:
                logger.error("Failed to initialize Telegram client")
                self.running = False
                return
                
            # Register event handlers
            @client.on(events.NewMessage)
            async def handler(event):
                if self.running:
                    chat = await event.get_chat()
                    chat_id = chat.id
                    
                    # Check if this chat is in our monitoring list
                    for source_id, channel in self.channels:
                        if channel == str(chat_id) or (hasattr(chat, 'username') and chat.username and channel.lower() == chat.username.lower()):
                            await self._process_message(source_id, event)
            
            # Main monitoring loop
            while self.running:
                try:
                    # Get active sources from database
                    sources = self._get_active_sources()
                    
                    if not sources:
                        logger.debug("No active Telegram sources defined")
                        time.sleep(60)  # Sleep briefly before checking again
                        continue
                    
                    # Update channels list
                    self.channels = []
                    for source in sources:
                        config = json.loads(source.config) if source.config else {}
                        if 'channels' in config:
                            for channel in config['channels']:
                                self.channels.append((source.id, channel))
                    
                    # Join channels if not already joined
                    for _, channel in self.channels:
                        loop.run_until_complete(self._join_channel(client, channel))
                    
                    # Update last ingestion timestamp
                    self._update_ingestion_time(sources)
                    
                    # Wait for a bit before checking for new sources
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in Telegram monitoring loop: {e}")
                    # Log error to database
                    self._log_error("monitoring_loop", str(e))
                    time.sleep(60)  # Short sleep on error
            
            # Disconnect when stopping
            loop.run_until_complete(client.disconnect())
            
        except Exception as e:
            logger.error(f"Fatal error in Telegram monitoring: {e}")
            self._log_error("telegram_monitor", str(e))
            self.running = False
            
        finally:
            loop.close()
    
    async def _join_channel(self, client, channel):
        """Join a Telegram channel if not already joined."""
        try:
            # Check if channel is numeric (chat ID) or username
            if channel.lstrip('-').isdigit():
                # It's a chat ID
                entity = int(channel)
            else:
                # It's a username
                if not channel.startswith('@'):
                    channel = '@' + channel
                entity = channel
            
            # Try to get entity info (will raise ValueError if not joined)
            try:
                await client.get_entity(entity)
                logger.debug(f"Already joined channel: {channel}")
            except ValueError:
                # Not joined, try to join
                logger.info(f"Joining channel: {channel}")
                await client.join_channel(entity)
                
        except Exception as e:
            logger.error(f"Error joining channel {channel}: {e}")
    
    async def _process_message(self, source_id, event):
        """Process a Telegram message event."""
        try:
            # Extract message data
            message = event.message
            text = message.message
            
            if not text:
                # Skip media-only messages
                return
                
            # Get additional info
            chat = await event.get_chat()
            chat_name = getattr(chat, 'title', None) or getattr(chat, 'username', None) or str(chat.id)
            
            # Create metadata
            metadata = json.dumps({
                'message_id': message.id,
                'chat_id': chat.id,
                'chat_name': chat_name,
                'timestamp': message.date.isoformat(),
                'has_media': message.media is not None,
                'forward': message.forward is not None
            })
            
            # Create chat URL (if possible)
            url = None
            if hasattr(chat, 'username') and chat.username:
                url = f"https://t.me/{chat.username}/{message.id}"
            
            # Create a new narrative instance
            with db.session.begin():
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=text,
                    metadata=metadata,
                    url=url,
                    detected_at=datetime.utcnow()
                )
                db.session.add(instance)
            
            logger.debug(f"Processed message from {chat_name}")
            
        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            self._log_error("process_message", str(e))
    
    def _get_active_sources(self) -> List[DataSource]:
        """Get active Telegram data sources from the database."""
        try:
            sources = DataSource.query.filter_by(
                source_type='telegram',
                is_active=True
            ).all()
            return sources
        except Exception as e:
            logger.error(f"Error fetching Telegram sources: {e}")
            return []
    
    def _update_ingestion_time(self, sources: List[DataSource]):
        """Update the last ingestion timestamp for sources."""
        try:
            with db.session.begin():
                for source in sources:
                    source.last_ingestion = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error updating ingestion time: {e}")
    
    async def get_channel_messages(self, channel: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent messages from a Telegram channel.
        
        Args:
            channel: Channel username or ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            messages: List of message dictionaries
        """
        if not self.client:
            logger.error("Telegram client not initialized")
            return []
            
        try:
            # Get channel entity
            if channel.lstrip('-').isdigit():
                entity = int(channel)
            else:
                if not channel.startswith('@'):
                    channel = '@' + channel
                entity = channel
            
            # Get messages
            messages = await self.client.get_messages(entity, limit=limit)
            
            # Format message data
            result = []
            for message in messages:
                if message.message:  # Only process text messages
                    chat = await message.get_chat()
                    chat_name = getattr(chat, 'title', None) or getattr(chat, 'username', None) or str(chat.id)
                    
                    # Create chat URL (if possible)
                    url = None
                    if hasattr(chat, 'username') and chat.username:
                        url = f"https://t.me/{chat.username}/{message.id}"
                    
                    result.append({
                        'message_id': message.id,
                        'text': message.message,
                        'chat_id': chat.id,
                        'chat_name': chat_name,
                        'timestamp': message.date.isoformat(),
                        'url': url
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting messages from channel '{channel}': {e}")
            return []
    
    def create_source(self, name: str, config: Dict[str, Any]) -> Optional[int]:
        """Create a new Telegram data source in the database.
        
        Args:
            name: Name for the data source
            config: Configuration dictionary (channels list)
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Validate config
            if 'channels' not in config or not config['channels']:
                logger.error("Telegram source config must contain 'channels' list")
                return None
                
            with db.session.begin():
                source = DataSource(
                    name=name,
                    source_type='telegram',
                    config=json.dumps(config),
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.session.add(source)
                db.session.flush()
                
                source_id = source.id
                
            logger.info(f"Created Telegram data source: {name} (ID: {source_id})")
            return source_id
            
        except Exception as e:
            logger.error(f"Error creating Telegram data source '{name}': {e}")
            return None
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log_entry = SystemLog(
                log_type="error",
                component="telegram_source",
                message=f"Error in {operation}: {message}"
            )
            with db.session.begin():
                db.session.add(log_entry)
        except Exception:
            # Just log to console if database logging fails
            logger.error(f"Failed to log error to database: {message}")
