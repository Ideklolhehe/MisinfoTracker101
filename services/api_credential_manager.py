"""
API Credential Manager for CIVILIAN system.
This module provides a central location for managing external API credentials.
"""

import os
import logging
import json
from typing import Dict, Any, Optional

from models import SystemLog
from app import db

logger = logging.getLogger(__name__)

class APICredentialManager:
    """Manages API credentials for external services."""
    
    # Define credential types and their environment variable names
    CREDENTIAL_TYPES = {
        'youtube': ['YOUTUBE_API_KEY'],
        'twitter': ['TWITTER_API_KEY', 'TWITTER_API_SECRET', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_SECRET'],
        'telegram': ['TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'TELEGRAM_PHONE'],
        'openai': ['OPENAI_API_KEY'],
        'google': ['GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET'],
        'dark_web': ['TOR_PROXY_HOST', 'TOR_PROXY_PORT', 'TOR_CONTROL_PASSWORD']
    }
    
    @classmethod
    def get_credentials(cls, credential_type: str) -> Dict[str, str]:
        """
        Get credentials for a specific API from environment variables.
        
        Args:
            credential_type: Type of credentials to retrieve (youtube, twitter, etc.)
            
        Returns:
            Dictionary of credentials or empty dict if not found
        """
        if credential_type not in cls.CREDENTIAL_TYPES:
            logger.error(f"Unknown credential type: {credential_type}")
            return {}
            
        env_vars = cls.CREDENTIAL_TYPES[credential_type]
        credentials = {}
        
        missing_vars = []
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                # Add to credentials dict with lowercase key (without prefix)
                key = var.lower().split('_', 1)[1] if '_' in var else var.lower()
                credentials[key] = value
            else:
                missing_vars.append(var)
                
        if missing_vars:
            logger.warning(f"Missing {credential_type} credentials: {', '.join(missing_vars)}")
            
        return credentials
    
    @classmethod
    def are_credentials_complete(cls, credential_type: str) -> bool:
        """
        Check if all required credentials for a service are present.
        
        Args:
            credential_type: Type of credentials to check
            
        Returns:
            True if all required credentials are present, False otherwise
        """
        if credential_type not in cls.CREDENTIAL_TYPES:
            return False
            
        env_vars = cls.CREDENTIAL_TYPES[credential_type]
        return all(os.environ.get(var) for var in env_vars)
    
    @classmethod
    def log_missing_credentials(cls, service_name: str) -> None:
        """
        Log missing credentials to the database for monitoring.
        
        Args:
            service_name: Name of the service missing credentials
        """
        try:
            log = SystemLog(
                log_type='warning',
                component='api_credentials',
                message=f"Missing credentials for {service_name} service"
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to log missing credentials: {e}")
            
    @classmethod
    def get_all_credential_status(cls) -> Dict[str, bool]:
        """
        Get the status of all credential types.
        
        Returns:
            Dictionary mapping credential types to their availability status
        """
        return {
            cred_type: cls.are_credentials_complete(cred_type)
            for cred_type in cls.CREDENTIAL_TYPES
        }
        
    @classmethod
    def get_credential_requirements(cls) -> Dict[str, Dict[str, str]]:
        """
        Get the requirements for each credential type.
        
        Returns:
            Dictionary describing all credential requirements
        """
        requirements = {}
        
        for cred_type, env_vars in cls.CREDENTIAL_TYPES.items():
            requirements[cred_type] = {
                'variables': env_vars,
                'status': 'available' if cls.are_credentials_complete(cred_type) else 'missing',
                'description': cls._get_credential_description(cred_type)
            }
            
        return requirements
    
    @staticmethod
    def _get_credential_description(credential_type: str) -> str:
        """Get a human-readable description for a credential type."""
        descriptions = {
            'youtube': "Required for monitoring YouTube channels and videos for misinformation.",
            'twitter': "Required for monitoring Twitter accounts and hashtags for misinformation.",
            'telegram': "Required for monitoring Telegram channels and groups for misinformation.",
            'openai': "Required for AI-powered content verification and analysis.",
            'google': "Used for Google OAuth authentication and Google API access.",
            'dark_web': "Required for monitoring dark web forums and marketplaces."
        }
        
        return descriptions.get(credential_type, "External API credentials")