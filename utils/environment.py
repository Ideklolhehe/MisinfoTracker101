"""
Environment configuration module for the CIVILIAN system.
Provides a unified interface for configuration from environment variables,
with fallback to .env files.
"""

import os
import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Union, List, Set

from decouple import config as decouple_config
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def get_config(
    key: str, 
    default: Any = None, 
    cast: type = str, 
    required: bool = False
) -> Any:
    """
    Get configuration value from environment variables with type casting.
    
    Args:
        key: The environment variable name
        default: Default value if not found
        cast: Type to cast the value to (str, int, float, bool, etc.)
        required: Whether this config is required
        
    Returns:
        The configuration value cast to the specified type
        
    Raises:
        ValueError: If required is True and the value is not found
    """
    try:
        return decouple_config(key, default=default, cast=cast)
    except Exception as e:
        if required:
            logger.error(f"Required configuration {key} not found: {e}")
            raise ValueError(f"Required configuration {key} not found")
        logger.warning(f"Configuration {key} not found, using default: {default}")
        return default


def get_list_config(
    key: str, 
    default: Optional[List] = None, 
    delimiter: str = ",",
    required: bool = False
) -> List:
    """
    Get a list configuration from a comma-separated environment variable.
    
    Args:
        key: The environment variable name
        default: Default list if not found
        delimiter: The delimiter used to split the string
        required: Whether this config is required
        
    Returns:
        A list of string values
    """
    if default is None:
        default = []
        
    value = get_config(key, None, str, required=required)
    if value is None:
        return default
        
    return [item.strip() for item in value.split(delimiter) if item.strip()]


def get_bool_config(key: str, default: bool = False, required: bool = False) -> bool:
    """
    Get a boolean configuration value.
    Treats 'true', 'yes', 'y', '1' as True (case insensitive).
    
    Args:
        key: The environment variable name
        default: Default value if not found
        required: Whether this config is required
        
    Returns:
        Boolean configuration value
    """
    return get_config(
        key, 
        default, 
        cast=lambda v: v.lower() in ('true', 'yes', 'y', '1') 
        if isinstance(v, str) else bool(v),
        required=required
    )


def get_int_config(key: str, default: int = 0, required: bool = False) -> int:
    """
    Get an integer configuration value.
    
    Args:
        key: The environment variable name
        default: Default value if not found
        required: Whether this config is required
        
    Returns:
        Integer configuration value
    """
    return get_config(key, default, int, required=required)


def get_float_config(key: str, default: float = 0.0, required: bool = False) -> float:
    """
    Get a float configuration value.
    
    Args:
        key: The environment variable name
        default: Default value if not found
        required: Whether this config is required
        
    Returns:
        Float configuration value
    """
    return get_config(key, default, float, required=required)


# Core application settings
DEBUG = get_bool_config('DEBUG', default=False)
TESTING = get_bool_config('TESTING', default=False)
SECRET_KEY = get_config('SESSION_SECRET', default='development-key')
DATABASE_URL = get_config('DATABASE_URL', required=True)
LOG_LEVEL = get_config('LOG_LEVEL', default='INFO')

# Feature flags
ENABLE_PROMETHEUS = get_bool_config('ENABLE_PROMETHEUS', default=True)
ENABLE_THREADING = get_bool_config('ENABLE_THREADING', default=True)
ENABLE_FAISS = get_bool_config('ENABLE_FAISS', default=True)

# API settings
CORS_ORIGINS = get_list_config('CORS_ORIGINS', default=['*'])
API_RATE_LIMIT = get_int_config('API_RATE_LIMIT', default=60)  # requests per minute

# Model configuration
NARRATIVE_EMBEDDING_DIM = get_int_config('NARRATIVE_EMBEDDING_DIM', default=768)
CLUSTERING_EPSILON = get_float_config('CLUSTERING_EPSILON', default=0.5)
CLUSTERING_MIN_SAMPLES = get_int_config('CLUSTERING_MIN_SAMPLES', default=5)
SIMILARITY_THRESHOLD = get_float_config('SIMILARITY_THRESHOLD', default=0.85)

# Processing settings
BATCH_SIZE = get_int_config('BATCH_SIZE', default=100)
REFRESH_INTERVAL = get_int_config('REFRESH_INTERVAL', default=300)  # seconds

# Redis configuration (optional)
REDIS_URL = get_config('REDIS_URL', default=None)
REDIS_ENABLED = get_bool_config('REDIS_ENABLED', default=False)