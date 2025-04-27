"""
API Credential Manager for CIVILIAN system.
This module handles secure storage and validation of API credentials.
"""

import logging
import json
import os
import time
from typing import Dict, Optional, Any, List, Tuple

from app import db
from models import SystemCredential, SystemLog
from utils.encryption import encrypt_dict, decrypt_dict

logger = logging.getLogger(__name__)

class APICredentialManager:
    """Manages API credentials for external services."""
    
    # Define required credentials for each API type
    REQUIRED_CREDENTIALS = {
        'openai': ['api_key'],
        'youtube': ['api_key'],
        'twitter': ['api_key', 'api_secret', 'access_token', 'access_secret'],
        'telegram': ['api_id', 'api_hash'],
        'dark_web': ['proxy_host', 'proxy_port']
    }
    
    @classmethod
    def get_credentials(cls, credential_type: str) -> Dict[str, str]:
        """
        Get credentials for a specific API type.
        
        Args:
            credential_type: Type of credential to retrieve
            
        Returns:
            Dictionary of credential key-value pairs
        """
        try:
            # Check for environment variables first (for OpenAI)
            if credential_type == 'openai' and os.environ.get('OPENAI_API_KEY'):
                return {'api_key': os.environ.get('OPENAI_API_KEY')}
                
            # Get credentials from database
            cred = SystemCredential.query.filter_by(credential_type=credential_type).first()
            
            if cred and cred.credential_data:
                # Decrypt credentials before returning
                encrypted_data = json.loads(cred.credential_data)
                return decrypt_dict(encrypted_data)
            else:
                logger.warning(f"No {credential_type} credentials found in database")
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving {credential_type} credentials: {e}")
            return {}
    
    @classmethod
    def save_credentials(cls, credential_type: str, credentials: Dict[str, str]) -> bool:
        """
        Save or update credentials for a specific API type.
        
        Args:
            credential_type: Type of credential to save
            credentials: Dictionary of credential key-value pairs
            
        Returns:
            Success status
        """
        try:
            # Validate required fields
            required_fields = cls.REQUIRED_CREDENTIALS.get(credential_type, [])
            for field in required_fields:
                if field not in credentials or not credentials[field]:
                    logger.error(f"Missing required field {field} for {credential_type} credentials")
                    return False
            
            # Encrypt the credentials before storing
            encrypted_credentials = encrypt_dict(credentials)
            
            # Check if credentials already exist
            cred = SystemCredential.query.filter_by(credential_type=credential_type).first()
            
            if cred:
                # Update existing credentials
                cred.credential_data = json.dumps(encrypted_credentials)
                cred.updated_at = time.time()
            else:
                # Create new credentials
                cred = SystemCredential(
                    credential_type=credential_type,
                    credential_data=json.dumps(encrypted_credentials),
                    created_at=time.time(),
                    updated_at=time.time()
                )
                db.session.add(cred)
                
            # Save to database
            db.session.commit()
            
            # Log the change
            log = SystemLog(
                log_type='info',
                component='api_credential_manager',
                message=f"Updated {credential_type} API credentials"
            )
            db.session.add(log)
            db.session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving {credential_type} credentials: {e}")
            db.session.rollback()
            return False
    
    @classmethod
    def delete_credentials(cls, credential_type: str) -> bool:
        """
        Delete credentials for a specific API type.
        
        Args:
            credential_type: Type of credential to delete
            
        Returns:
            Success status
        """
        try:
            # Get credentials from database
            cred = SystemCredential.query.filter_by(credential_type=credential_type).first()
            
            if cred:
                # Delete credentials
                db.session.delete(cred)
                
                # Log the change
                log = SystemLog(
                    log_type='warning',
                    component='api_credential_manager',
                    message=f"Deleted {credential_type} API credentials"
                )
                db.session.add(log)
                
                # Save to database
                db.session.commit()
                
                return True
            else:
                logger.warning(f"No {credential_type} credentials found to delete")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting {credential_type} credentials: {e}")
            db.session.rollback()
            return False
    
    @classmethod
    def test_credentials(cls, credential_type: str, credentials: Dict[str, str]) -> Tuple[bool, str]:
        """
        Test if credentials are valid by making a simple API call.
        
        Args:
            credential_type: Type of credential to test
            credentials: Dictionary of credential key-value pairs
            
        Returns:
            (is_valid, message): Tuple with validity status and message
        """
        try:
            # Validate required fields
            required_fields = cls.REQUIRED_CREDENTIALS.get(credential_type, [])
            for field in required_fields:
                if field not in credentials or not credentials[field]:
                    return False, f"Missing required field: {field}"
                    
            # Test credentials based on type
            if credential_type == 'openai':
                return cls._test_openai_credentials(credentials)
            elif credential_type == 'youtube':
                return cls._test_youtube_credentials(credentials)
            elif credential_type == 'twitter':
                return cls._test_twitter_credentials(credentials)
            elif credential_type == 'telegram':
                return cls._test_telegram_credentials(credentials)
            elif credential_type == 'dark_web':
                return cls._test_tor_credentials(credentials)
            else:
                return False, f"Unknown credential type: {credential_type}"
                
        except Exception as e:
            logger.error(f"Error testing {credential_type} credentials: {e}")
            return False, f"Error: {str(e)}"
    
    @staticmethod
    def _test_openai_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Test OpenAI API credentials."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=credentials['api_key'])
            
            # Make a simple API call
            response = client.models.list()
            
            if response:
                return True, f"Valid OpenAI credentials, found {len(response.data)} models"
            else:
                return False, "API call succeeded but returned no data"
                
        except ImportError:
            return False, "OpenAI library not installed"
        except Exception as e:
            return False, f"Invalid OpenAI credentials: {str(e)}"
    
    @staticmethod
    def _test_youtube_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Test YouTube API credentials."""
        try:
            from googleapiclient.discovery import build
            
            youtube = build('youtube', 'v3', developerKey=credentials['api_key'])
            
            # Make a simple API call
            response = youtube.channels().list(
                part='snippet',
                forHandle='@youtube'
            ).execute()
            
            if response and 'items' in response:
                return True, "Valid YouTube API credentials"
            else:
                return False, "API call succeeded but returned no data"
                
        except ImportError:
            return False, "Google API client library not installed"
        except Exception as e:
            return False, f"Invalid YouTube API credentials: {str(e)}"
    
    @staticmethod
    def _test_twitter_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Test Twitter API credentials."""
        try:
            import tweepy
            
            auth = tweepy.OAuth1UserHandler(
                consumer_key=credentials['api_key'],
                consumer_secret=credentials['api_secret'],
                access_token=credentials['access_token'],
                access_token_secret=credentials['access_secret']
            )
            
            api = tweepy.API(auth)
            
            # Make a simple API call
            user = api.verify_credentials()
            
            if user:
                return True, f"Valid Twitter API credentials for user @{user.screen_name}"
            else:
                return False, "API call succeeded but returned no data"
                
        except ImportError:
            return False, "Tweepy library not installed"
        except Exception as e:
            return False, f"Invalid Twitter API credentials: {str(e)}"
    
    @staticmethod
    def _test_telegram_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Test Telegram API credentials."""
        try:
            from telethon import TelegramClient
            from telethon.sessions import MemorySession
            
            client = TelegramClient(
                MemorySession(),
                int(credentials['api_id']),
                credentials['api_hash']
            )
            
            # Just try to connect (we don't need to complete auth)
            client.connect()
            
            if client.is_connected():
                client.disconnect()
                return True, "Valid Telegram API credentials (connection successful)"
            else:
                return False, "Failed to connect to Telegram API"
                
        except ImportError:
            return False, "Telethon library not installed"
        except Exception as e:
            return False, f"Invalid Telegram API credentials: {str(e)}"
    
    @staticmethod
    def _test_tor_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Test Tor proxy credentials."""
        try:
            import requests
            
            # Configure proxy
            proxy_host = credentials['proxy_host']
            proxy_port = int(credentials['proxy_port'])
            
            # Setup proxy configuration
            proxies = {
                'http': f'socks5h://{proxy_host}:{proxy_port}',
                'https': f'socks5h://{proxy_host}:{proxy_port}'
            }
            
            # Try to connect to the Tor check page
            response = requests.get('https://check.torproject.org/', proxies=proxies, timeout=10)
            
            if response.status_code == 200 and 'Congratulations' in response.text:
                return True, "Valid Tor proxy configuration (connected through Tor)"
            else:
                return False, "Connected, but not through Tor network"
                
        except ImportError:
            return False, "Required libraries for Tor not installed"
        except Exception as e:
            return False, f"Invalid Tor proxy configuration: {str(e)}"
    
    @classmethod
    def get_all_credential_types(cls) -> List[str]:
        """
        Get a list of all supported credential types.
        
        Returns:
            List of credential type names
        """
        return list(cls.REQUIRED_CREDENTIALS.keys())
    
    @classmethod
    def get_all_credential_status(cls) -> Dict[str, bool]:
        """
        Check if credentials are available for all API types.
        
        Returns:
            Dictionary mapping credential types to availability status
        """
        result = {}
        
        # Check OpenAI credentials from environment
        if os.environ.get('OPENAI_API_KEY'):
            result['openai'] = True
        else:
            # Check database for all credential types
            for credential_type in cls.get_all_credential_types():
                credentials = cls.get_credentials(credential_type)
                required_fields = cls.REQUIRED_CREDENTIALS.get(credential_type, [])
                
                if all(field in credentials and credentials[field] for field in required_fields):
                    result[credential_type] = True
                else:
                    result[credential_type] = False
        
        return result
    
    @classmethod
    def are_credentials_complete(cls, credential_type: str) -> bool:
        """
        Check if all required credentials are available for a specific API type.
        
        Args:
            credential_type: Type of credential to check
            
        Returns:
            True if all required credentials are available, False otherwise
        """
        # Special case for OpenAI - check environment variable
        if credential_type == 'openai' and os.environ.get('OPENAI_API_KEY'):
            return True
            
        # Check database
        credentials = cls.get_credentials(credential_type)
        required_fields = cls.REQUIRED_CREDENTIALS.get(credential_type, [])
        
        return all(field in credentials and credentials[field] for field in required_fields)
    
    @classmethod
    def log_missing_credentials(cls, service_name: str) -> None:
        """
        Log warning about missing credentials.
        
        Args:
            service_name: Name of the service with missing credentials
        """
        log = SystemLog(
            log_type='warning',
            component='api_credential_manager',
            message=f"Missing {service_name} credentials, some features may not work"
        )
        db.session.add(log)
        db.session.commit()