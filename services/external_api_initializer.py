import os
import logging
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

# Global API clients
_openai_client = None

def get_openai_client():
    """
    Get or initialize the OpenAI API client.
    
    Returns:
        OpenAI: The initialized OpenAI client
    """
    global _openai_client
    
    if _openai_client is None:
        # Check if API key is available in environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")
        
        try:
            # Initialize the OpenAI client
            _openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI API client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    return _openai_client

class ExternalAPIInitializer:
    """
    Static class for initializing external API clients.
    This class provides methods to initialize various API clients
    used throughout the CIVILIAN system.
    """
    
    @staticmethod
    def init_openai_client():
        """
        Initialize the OpenAI API client.
        
        Returns:
            OpenAI client or None if initialization fails
        """
        try:
            return get_openai_client()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    @staticmethod
    def init_youtube_client():
        """
        Initialize the YouTube API client.
        
        Returns:
            YouTube client or None if initialization fails
        """
        try:
            # Not implemented for complexity analyzer
            return None
        except Exception as e:
            logger.error(f"Failed to initialize YouTube client: {e}")
            return None
    
    @staticmethod
    def init_twitter_client():
        """
        Initialize the Twitter API client.
        
        Returns:
            Twitter client or None if initialization fails
        """
        try:
            # Not implemented for complexity analyzer
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            return None
    
    @staticmethod
    def init_telegram_client():
        """
        Initialize the Telegram API client.
        
        Returns:
            Telegram client or None if initialization fails
        """
        try:
            # Not implemented for complexity analyzer
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            return None
    
    @staticmethod
    def init_tor_client():
        """
        Initialize the Tor client for dark web monitoring.
        
        Returns:
            Tor client or None if initialization fails
        """
        try:
            # Not implemented for complexity analyzer
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Tor client: {e}")
            return None