"""
External API Initializer for CIVILIAN system.
This module handles initialization of external API clients.
"""

import logging
import os
from typing import Dict, Any, Optional

from services.api_credential_manager import APICredentialManager

logger = logging.getLogger(__name__)

class ExternalAPIInitializer:
    """Initializes and manages external API clients."""
    
    @staticmethod
    def init_youtube_client() -> Optional[Any]:
        """
        Initialize YouTube API client.
        
        Returns:
            YouTube API client object or None if initialization fails
        """
        try:
            # Check for required credentials
            if not APICredentialManager.are_credentials_complete('youtube'):
                logger.warning("Missing YouTube API credentials, cannot initialize client")
                APICredentialManager.log_missing_credentials('YouTube')
                return None
                
            # Get credentials
            credentials = APICredentialManager.get_credentials('youtube')
            
            # Import Google API client
            from googleapiclient.discovery import build
            
            # Initialize YouTube client
            youtube = build('youtube', 'v3', developerKey=credentials['api_key'])
            
            logger.info("YouTube API client initialized successfully")
            return youtube
            
        except ImportError:
            logger.error("Google API client library not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API client: {e}")
            return None
    
    @staticmethod
    def init_twitter_client() -> Optional[Any]:
        """
        Initialize Twitter API client.
        
        Returns:
            Twitter API client object or None if initialization fails
        """
        try:
            # Check for required credentials
            if not APICredentialManager.are_credentials_complete('twitter'):
                logger.warning("Missing Twitter API credentials, cannot initialize client")
                APICredentialManager.log_missing_credentials('Twitter')
                return None
                
            # Get credentials
            credentials = APICredentialManager.get_credentials('twitter')
            
            # Import Tweepy
            import tweepy
            
            # Initialize Twitter client
            auth = tweepy.OAuth1UserHandler(
                consumer_key=credentials['api_key'],
                consumer_secret=credentials['api_secret'],
                access_token=credentials['access_token'],
                access_token_secret=credentials['access_secret']
            )
            
            api = tweepy.API(auth)
            
            # Verify credentials
            api.verify_credentials()
            
            logger.info("Twitter API client initialized successfully")
            return api
            
        except ImportError:
            logger.error("Tweepy library not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API client: {e}")
            return None
    
    @staticmethod
    def init_telegram_client() -> Optional[Any]:
        """
        Initialize Telegram API client.
        
        Returns:
            Telegram client object or None if initialization fails
        """
        try:
            # Check for required credentials
            if not APICredentialManager.are_credentials_complete('telegram'):
                logger.warning("Missing Telegram API credentials, cannot initialize client")
                APICredentialManager.log_missing_credentials('Telegram')
                return None
                
            # Get credentials
            credentials = APICredentialManager.get_credentials('telegram')
            
            # Import Telethon
            from telethon import TelegramClient
            
            # Initialize Telegram client
            client = TelegramClient(
                'civilian_session',
                int(credentials['api_id']),
                credentials['api_hash']
            )
            
            logger.info("Telegram API client initialized successfully")
            return client
            
        except ImportError:
            logger.error("Telethon library not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Telegram API client: {e}")
            return None
    
    @staticmethod
    def init_openai_client() -> Optional[Any]:
        """
        Initialize OpenAI API client.
        
        Returns:
            OpenAI client object or None if initialization fails
        """
        try:
            # Check for required credentials
            if not APICredentialManager.are_credentials_complete('openai'):
                logger.warning("Missing OpenAI API credentials, cannot initialize client")
                APICredentialManager.log_missing_credentials('OpenAI')
                return None
                
            # Get credentials
            credentials = APICredentialManager.get_credentials('openai')
            
            # Import OpenAI
            from openai import OpenAI
            
            # Initialize OpenAI client
            client = OpenAI(api_key=credentials['api_key'])
            
            logger.info("OpenAI API client initialized successfully")
            return client
            
        except ImportError:
            logger.error("OpenAI library not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API client: {e}")
            return None
    
    @staticmethod
    def init_tor_client() -> Optional[Any]:
        """
        Initialize Tor client for dark web monitoring.
        
        Returns:
            Tor client object or None if initialization fails
        """
        try:
            # Check for required credentials
            if not APICredentialManager.are_credentials_complete('dark_web'):
                logger.warning("Missing Tor proxy credentials, cannot initialize client")
                APICredentialManager.log_missing_credentials('Dark Web')
                return None
                
            # Get credentials
            credentials = APICredentialManager.get_credentials('dark_web')
            
            # Import required libraries
            import requests
            import stem.process
            from stem.control import Controller
            
            # Set up proxy configuration
            proxy_host = credentials['proxy_host']
            proxy_port = int(credentials['proxy_port'])
            proxy_password = credentials.get('control_password')
            
            # Configure requests session with SOCKS proxy
            session = requests.Session()
            session.proxies = {
                'http': f'socks5h://{proxy_host}:{proxy_port}',
                'https': f'socks5h://{proxy_host}:{proxy_port}'
            }
            
            # Create a controller for Tor if password is provided
            controller = None
            if proxy_password:
                controller = Controller.from_port(address=proxy_host, port=proxy_port)
                controller.authenticate(password=proxy_password)
            
            logger.info("Tor client initialized successfully")
            
            # Return a dict with session and controller
            return {
                'session': session,
                'controller': controller
            }
            
        except ImportError:
            logger.error("Required libraries for Tor not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Tor client: {e}")
            return None
            
    @classmethod
    def initialize_all_clients(cls) -> Dict[str, Any]:
        """
        Initialize all external API clients.
        
        Returns:
            Dictionary of initialized clients
        """
        clients = {}
        
        clients['youtube'] = cls.init_youtube_client()
        clients['twitter'] = cls.init_twitter_client()
        clients['telegram'] = cls.init_telegram_client()
        clients['openai'] = cls.init_openai_client()
        clients['tor'] = cls.init_tor_client()
        
        return clients