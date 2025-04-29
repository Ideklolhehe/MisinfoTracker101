"""
Base class for all data sources in the CIVILIAN system.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from app import db
from models import DataSource

# Configure logger
logger = logging.getLogger(__name__)

class DataSourceBase:
    """Base class for all data sources."""
    
    def __init__(self, source_type_prefix: str):
        """
        Initialize the data source.
        
        Args:
            source_type_prefix: Prefix for the source type (e.g., 'twitter', 'rss', 'web')
        """
        self.source_type_prefix = source_type_prefix
        self.is_initialized = True
        logger.info(f"{self.__class__.__name__} initialized with source type prefix: {source_type_prefix}")
    
    def register_source(self, source_data: Dict[str, Any]) -> Optional[int]:
        """
        Register a new data source in the database.
        
        Args:
            source_data: Dictionary containing source information
                
        Returns:
            ID of the registered source, or None if registration failed
        """
        raise NotImplementedError("Subclasses must implement register_source")
    
    def process_sources(self):
        """Process all active sources of this type."""
        raise NotImplementedError("Subclasses must implement process_sources")
        
    def get_active_sources(self):
        """Get all active sources of this type."""
        return DataSource.query.filter(
            DataSource.is_active == True,
            DataSource.source_type.like(f"{self.source_type_prefix}_%")
        ).all()
        
    def get_source_by_id(self, source_id: int) -> Optional[DataSource]:
        """Get a source by ID."""
        return DataSource.query.get(source_id)
        
    def get_source_by_name(self, name: str) -> Optional[DataSource]:
        """Get a source by name."""
        return DataSource.query.filter_by(name=name).first()
        
    def update_source_status(self, source_id: int, is_active: bool) -> bool:
        """
        Update the active status of a source.
        
        Args:
            source_id: ID of the source to update
            is_active: New active status
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source = self.get_source_by_id(source_id)
            if not source:
                return False
                
            source.is_active = is_active
            db.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating source status: {e}")
            db.session.rollback()
            return False
            
    def update_source_last_ingestion(self, source_id: int) -> bool:
        """
        Update the last ingestion timestamp of a source.
        
        Args:
            source_id: ID of the source to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source = self.get_source_by_id(source_id)
            if not source:
                return False
                
            source.last_ingestion = datetime.utcnow()
            db.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating source last ingestion: {e}")
            db.session.rollback()
            return False