"""
Base class for data sources in the CIVILIAN system.
Provides a common interface for all data sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Generator

# Configure module logger
logger = logging.getLogger(__name__)

class SourceBase(ABC):
    """
    Base class for all data sources in the CIVILIAN system.
    Provides a common interface for configuring, ingesting, and processing data.
    """
    
    # Source type identifier
    SOURCE_TYPE: str = "base"
    
    def __init__(self, **config: Any):
        """
        Initialize the data source.
        
        Args:
            **config: Configuration parameters for the source
        """
        self.config: Dict[str, Any] = {}
        self.is_configured: bool = False
        self.is_active: bool = False
        
        # Configure the source with the provided config
        if config:
            self.configure(**config)
    
    def configure(self, **config: Any) -> None:
        """
        Configure the data source.
        
        Args:
            **config: Configuration parameters for the source
        """
        self.config.update(config)
        self.is_configured = self._validate_config()
        
        if self.is_configured:
            logger.info(f"Source {self.SOURCE_TYPE} configured successfully")
        else:
            logger.error(f"Source {self.SOURCE_TYPE} configuration is invalid")
    
    def _validate_config(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Base implementation always returns True
        # Subclasses should override this to implement specific validation
        return True
    
    @abstractmethod
    def ingest(self, **kwargs: Any) -> Generator[Dict[str, Any], None, None]:
        """
        Ingest data from the source.
        
        Args:
            **kwargs: Additional parameters for the ingestion process
            
        Returns:
            Generator yielding dictionaries of ingested data
        """
        pass
    
    def start(self) -> bool:
        """
        Start the data source.
        
        Returns:
            True if the source was started successfully, False otherwise
        """
        if not self.is_configured:
            logger.error(f"Source {self.SOURCE_TYPE} is not configured")
            return False
        
        self.is_active = True
        logger.info(f"Source {self.SOURCE_TYPE} started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the data source.
        
        Returns:
            True if the source was stopped successfully, False otherwise
        """
        self.is_active = False
        logger.info(f"Source {self.SOURCE_TYPE} stopped")
        return True
    
    def status(self) -> Dict[str, Any]:
        """
        Get the status of the data source.
        
        Returns:
            Dictionary with status information
        """
        return {
            "type": self.SOURCE_TYPE,
            "is_configured": self.is_configured,
            "is_active": self.is_active,
            "config": self.get_safe_config()
        }
    
    def get_safe_config(self) -> Dict[str, Any]:
        """
        Get a safe version of the configuration without sensitive information.
        
        Returns:
            Dictionary with non-sensitive configuration information
        """
        # Implement this in subclasses to remove sensitive information like API keys
        safe_config = self.config.copy()
        
        # Remove any sensitive information
        for key in ['api_key', 'password', 'token', 'secret', 'credentials']:
            if key in safe_config:
                safe_config[key] = "***REDACTED***"
        
        return safe_config