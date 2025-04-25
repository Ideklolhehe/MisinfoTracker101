#!/usr/bin/env python3
"""
Script to fix application context issues in CIVILIAN's background agents.
This script applies a patch to all agents and data sources to ensure they
always operate within a Flask application context.
"""

import sys
import inspect
import logging
from functools import wraps
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_app_context")

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import application components
from app import app
from agents.detector_agent import DetectorAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.counter_agent import CounterAgent
from data_sources.twitter_source import TwitterSource
from data_sources.telegram_source import TelegramSource
from data_sources.rss_source import RSSSource

def with_app_context(func):
    """Decorator to ensure a function runs within app context."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're already in an app context
        if app.app_context().top is not None:
            # We're already in an app context
            return func(*args, **kwargs)
        else:
            # Create an app context for this call
            with app.app_context():
                return func(*args, **kwargs)
    return wrapper

def patch_class_methods(cls, method_prefix=None, excluded_methods=None):
    """Patch all methods in a class to run in app context."""
    excluded_methods = excluded_methods or ['__init__', '__del__', '__str__', '__repr__']
    method_count = 0
    
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        # Skip methods based on exclusion criteria
        if name in excluded_methods:
            continue
        
        # Apply prefix filter if provided
        if method_prefix and not name.startswith(method_prefix):
            continue
        
        # Skip already patched methods
        if hasattr(method, '__app_context_patched__'):
            continue
        
        # Replace the method with a decorated version
        original_method = getattr(cls, name)
        patched_method = with_app_context(original_method)
        patched_method.__app_context_patched__ = True
        setattr(cls, name, patched_method)
        method_count += 1
        
        logger.debug(f"Patched {cls.__name__}.{name} method")
    
    return method_count

def apply_patches():
    """Apply app context patches to all relevant components."""
    patches_applied = 0
    
    # Patch agents (focus on methods that interact with database)
    agent_classes = [DetectorAgent, AnalyzerAgent, CounterAgent]
    for agent_cls in agent_classes:
        # All methods that start with '_' generally interact with the database
        patched = patch_class_methods(agent_cls, method_prefix='_')
        logger.info(f"Patched {patched} methods in {agent_cls.__name__}")
        patches_applied += patched
    
    # Patch data sources (focus on methods that interact with database)
    source_classes = [TwitterSource, TelegramSource, RSSSource]
    for source_cls in source_classes:
        # All methods that start with '_' generally interact with the database
        patched = patch_class_methods(source_cls, method_prefix='_')
        logger.info(f"Patched {patched} methods in {source_cls.__name__}")
        patches_applied += patched
    
    logger.info(f"Total of {patches_applied} methods patched with app context support")
    return patches_applied

def main():
    """Main function to apply the app context patches."""
    logger.info("Starting application context patches")
    
    try:
        patches_applied = apply_patches()
        
        # Print success message
        if patches_applied > 0:
            logger.info("Successfully applied all app context patches")
            logger.info("To apply these changes permanently, the patched code should be written back to the source files")
            logger.info("Please restart the application for changes to take effect")
        else:
            logger.warning("No patches were applied. All methods may already be patched")
    
    except Exception as e:
        logger.error(f"Error applying patches: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()