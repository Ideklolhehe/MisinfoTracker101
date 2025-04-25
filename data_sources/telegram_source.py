"""
Telegram data source module for the CIVILIAN system.
This module handles ingestion of content from Telegram channels and groups.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.exc import SQLAlchemyError

from app import db
from models import DataSource, NarrativeInstance, SystemLog

logger = logging.getLogger(__name__)

class TelegramSource:
    """Data source connector for Telegram."""
    
    def __init__(self):
        """Initialize the Telegram data source."""
        self._running = False
        self._thread = None
        self._client = None
        
        # Try to initialize the API client
        try:
            self._initialize_client()
        except Exception as e:
            logger.warning(f"Telegram API credentials not provided: {e}")
        
        logger.info("TelegramSource initialized")
    
    def _initialize_client(self):
        """Initialize the Telegram API client with credentials."""
        import os
        from telethon import TelegramClient
        
        # Check for required environment variables
        api_id = os.environ.get('TELEGRAM_API_ID')
        api_hash = os.environ.get('TELEGRAM_API_HASH')
        
        if not api_id or not api_hash:
            raise ValueError("Telegram API credentials not provided")
        
        # Initialize Telegram client
        self._client = TelegramClient('civilian_telegram_session', int(api_id), api_hash)
        
        # Connect to Telegram
        try:
            self._client.connect()
            logger.info("Telegram client initialized")
        except Exception as e:
            logger.error(f"Failed to connect to Telegram: {e}")
            self._client = None
            raise
    
    def start(self):
        """Start monitoring Telegram in a background thread."""
        if self._running:
            logger.warning("TelegramSource monitoring already running")
            return
        
        if not self._client:
            logger.error("Telegram API credentials not initialized, cannot start monitoring")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("TelegramSource monitoring started")
    
    def stop(self):
        """Stop monitoring Telegram."""
        if not self._running:
            logger.warning("TelegramSource monitoring not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)  # Wait up to 3 seconds
            logger.info("TelegramSource monitoring stopped")
        
        # Disconnect from Telegram
        if self._client:
            self._client.disconnect()
    
    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        from config import Config
        import asyncio
        
        while self._running:
            try:
                logger.debug("Starting Telegram monitoring cycle")
                
                # Get active Telegram sources from database
                sources = self._get_active_sources()
                
                if not sources:
                    logger.debug("No active Telegram sources defined")
                else:
                    # Process each source
                    for source in sources:
                        if not self._running:
                            break
                        
                        try:
                            # Extract channel/group entities from config
                            config = json.loads(source.config) if source.config else {}
                            entities = config.get('entities', [])
                            
                            # Create event loop for async operations
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            for entity in entities:
                                if not self._running:
                                    break
                                # Run async function synchronously
                                loop.run_until_complete(self._process_entity(source.id, entity))
                            
                            loop.close()
                            
                        except Exception as e:
                            self._log_error("process_source", f"Error processing source {source.id}: {e}")
                    
                    # Update last ingestion time
                    self._update_ingestion_time(sources)
                
                # Sleep until next cycle (respect Telegram API rate limits)
                for _ in range(int(Config.INGESTION_INTERVAL / 2)):
                    if not self._running:
                        break
                    time.sleep(2)  # Check if still running every 2 seconds
            
            except Exception as e:
                self._log_error("monitoring_loop", f"Error in Telegram monitoring loop: {e}")
                time.sleep(60)  # Shorter interval after error
    
    def _get_active_sources(self) -> List[DataSource]:
        """Get active Telegram data sources from the database."""
        try:
            return DataSource.query.filter_by(
                source_type='telegram',
                is_active=True
            ).all()
        except Exception as e:
            self._log_error("get_sources", f"Error fetching Telegram sources: {e}")
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
    
    async def _process_entity(self, source_id: int, entity: str, limit: int = 50):
        """Process messages from a Telegram channel or group."""
        logger.debug(f"Processing Telegram entity: {entity}")
        
        try:
            # Get messages from the entity
            messages = await self.get_messages(entity, limit)
            
            # Process each message
            for msg in messages:
                # Create a narrative instance for detection
                instance = NarrativeInstance(
                    source_id=source_id,
                    content=msg.get('text', ''),
                    meta_data=json.dumps(msg),
                    url=f"t.me/{entity.lstrip('@')}/{msg.get('id', '')}"
                )
                
                # Add to database
                db.session.add(instance)
            
            # Commit changes
            db.session.commit()
            logger.debug(f"Processed {len(messages)} messages from {entity}")
        
        except Exception as e:
            self._log_error("process_entity", f"Error processing entity {entity}: {e}")
            db.session.rollback()
    
    async def get_messages(self, entity: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages from a Telegram channel or group (for manual API use).
        
        Args:
            entity: Channel or group username or ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            messages: List of message dictionaries
        """
        result = []
        
        try:
            if not self._client:
                logger.error("Telegram client not initialized")
                return []
            
            # Get messages from the entity
            async for message in self._client.iter_messages(entity, limit=limit):
                if not message.text:
                    continue
                
                # Create message dict
                msg_data = {
                    'id': message.id,
                    'text': message.text,
                    'date': message.date.isoformat(),
                    'entity': entity
                }
                
                # Add user info if available
                if message.sender_id:
                    msg_data['sender_id'] = message.sender_id
                
                # Add forwarded info if available
                if message.forward:
                    msg_data['forwarded_from'] = str(message.forward.sender_id)
                
                result.append(msg_data)
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting messages from {entity}: {e}")
            return []
    
    def create_source(self, name: str, config: Dict[str, Any]) -> Optional[int]:
        """Create a new Telegram data source in the database.
        
        Args:
            name: Name for the data source
            config: Configuration dictionary with 'entities' list
            
        Returns:
            source_id: ID of the created source, or None on error
        """
        try:
            # Validate config
            if not isinstance(config, dict) or 'entities' not in config:
                logger.error("Invalid config: must contain 'entities' list")
                return None
            
            if not isinstance(config['entities'], list) or not all(isinstance(e, str) for e in config['entities']):
                logger.error("Invalid config: 'entities' must be a list of strings")
                return None
            
            # Create a new data source
            source = DataSource(
                name=name,
                source_type='telegram',
                config=json.dumps(config),
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            # Add to database
            db.session.add(source)
            db.session.commit()
            
            logger.info(f"Created Telegram source: {name} (ID: {source.id})")
            return source.id
        
        except Exception as e:
            db.session.rollback()
            self._log_error("create_source", f"Error creating Telegram source: {e}")
            return None
    
    def _log_error(self, operation: str, message: str):
        """Log an error to the database."""
        try:
            log = SystemLog(
                timestamp=datetime.utcnow(),
                log_type='error',
                component='telegram_source',
                message=f"{operation}: {message}"
            )
            db.session.add(log)
            db.session.commit()
        except SQLAlchemyError:
            logger.error(f"Failed to log error to database: {message}")
            db.session.rollback()
        except Exception as e:
            logger.error(f"Error logging to database: {e}")