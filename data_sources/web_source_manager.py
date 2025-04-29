"""
Web source manager for the CIVILIAN system.
Manages and coordinates multiple web sources for data collection.
"""

import logging
import importlib
import pkgutil
import inspect
from typing import Dict, List, Optional, Any, Type, cast

from data_sources.source_base import SourceBase

# Configure module logger
logger = logging.getLogger(__name__)

class WebSourceManager:
    """
    Manages and coordinates multiple web data sources for the CIVILIAN system.
    Automatically discovers and loads source classes from the data_sources package.
    """
    
    def __init__(self):
        """Initialize the web source manager."""
        self.source_classes: Dict[str, Type[SourceBase]] = {}
        self.source_instances: Dict[str, SourceBase] = {}
        
        # Auto-discover source classes
        self._discover_source_classes()
    
    def _discover_source_classes(self) -> None:
        """
        Discover source classes in the data_sources package.
        Any class that inherits from SourceBase will be registered.
        """
        try:
            import data_sources
            package_path = data_sources.__path__
            package_name = data_sources.__name__
            
            for _, module_name, _ in pkgutil.iter_modules(package_path):
                # Skip base modules and utilities
                if module_name in ['__init__', 'source_base', 'web_source_manager']:
                    continue
                
                module_path = f"{package_name}.{module_name}"
                
                try:
                    module = importlib.import_module(module_path)
                    
                    # Find all source classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, SourceBase) and 
                            obj is not SourceBase):
                            
                            source_type = getattr(obj, 'SOURCE_TYPE', None)
                            if source_type:
                                self.source_classes[source_type] = cast(Type[SourceBase], obj)
                                logger.info(f"Discovered source class: {obj.__name__} ({source_type})")
                
                except Exception as e:
                    logger.error(f"Error loading source module {module_path}: {str(e)}")
                    
            logger.info(f"Discovered {len(self.source_classes)} source classes")
            
        except Exception as e:
            logger.error(f"Error discovering source classes: {str(e)}")
    
    def get_source_class(self, source_type: str) -> Optional[Type[SourceBase]]:
        """
        Get a source class by type.
        
        Args:
            source_type: Type of source to get
            
        Returns:
            Source class or None if not found
        """
        return self.source_classes.get(source_type)
    
    def get_source_types(self) -> List[str]:
        """
        Get a list of available source types.
        
        Returns:
            List of source types
        """
        return list(self.source_classes.keys())
    
    def get_source_instance(self, source_type: str, **config: Any) -> Optional[SourceBase]:
        """
        Get a source instance by type.
        If an instance already exists, it will be returned.
        Otherwise, a new instance will be created.
        
        Args:
            source_type: Type of source to get
            **config: Configuration for the source
            
        Returns:
            Source instance or None if the type is not found
        """
        # Check if we already have an instance
        if source_type in self.source_instances:
            return self.source_instances[source_type]
        
        # Get the source class
        source_class = self.get_source_class(source_type)
        if not source_class:
            logger.error(f"Source type not found: {source_type}")
            return None
        
        # Create a new instance
        try:
            instance = source_class(**config)
            self.source_instances[source_type] = instance
            return instance
        except Exception as e:
            logger.error(f"Error creating source instance for {source_type}: {str(e)}")
            return None
    
    def get_all_source_instances(self) -> Dict[str, SourceBase]:
        """
        Get all source instances.
        
        Returns:
            Dictionary of source type to source instance
        """
        return self.source_instances.copy()
    
    def configure_source(self, source_type: str, **config: Any) -> bool:
        """
        Configure a source.
        If the source instance doesn't exist, it will be created.
        
        Args:
            source_type: Type of source to configure
            **config: Configuration for the source
            
        Returns:
            True if the source was configured successfully, False otherwise
        """
        instance = self.get_source_instance(source_type)
        
        if not instance:
            # Try to create a new instance with the provided config
            instance = self.get_source_instance(source_type, **config)
            return instance is not None
        
        # Configure the existing instance
        try:
            instance.configure(**config)
            return True
        except Exception as e:
            logger.error(f"Error configuring source {source_type}: {str(e)}")
            return False