#!/usr/bin/env python3
"""
Script to configure open news sources in the CIVILIAN system.
This script reads open news sources from a JSON configuration file
and adds them as RSS data sources in the system.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("configure_news_sources")

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import application components
from app import app, db
from models import DataSource
from data_sources.rss_source import RSSSource

def load_news_sources():
    """Load news sources from the configuration file."""
    try:
        config_path = parent_dir / "config" / "open_news_sources.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading news sources: {e}")
        return None

def configure_sources():
    """Configure all news sources in the system."""
    sources = load_news_sources()
    if not sources:
        logger.error("Failed to load news sources.")
        return False
    
    # Initialize RSS source
    rss_source = RSSSource()
    
    # Track configuration status
    success_count = 0
    failure_count = 0
    
    # Process each category
    for category, feeds in sources.items():
        logger.info(f"Configuring {category} feeds...")
        
        for feed in feeds:
            try:
                # Create a name for the source
                name = f"{feed['name']} ({feed['category']})"
                
                # Check if source already exists
                existing = DataSource.query.filter_by(name=name).first()
                if existing:
                    logger.info(f"Source '{name}' already exists, skipping.")
                    continue
                
                # Create the RSS source
                config = {'feeds': [feed['url']]}
                source_id = rss_source.create_source(name, config)
                
                if source_id:
                    logger.info(f"Successfully added source: {name}")
                    success_count += 1
                else:
                    logger.warning(f"Failed to add source: {name}")
                    failure_count += 1
            
            except Exception as e:
                logger.error(f"Error adding source {feed.get('name', 'unknown')}: {e}")
                failure_count += 1
    
    # Summary
    logger.info(f"Configuration complete. Added {success_count} sources. Failed: {failure_count}")
    return success_count > 0

def main():
    """Main function to run the configuration."""
    logger.info("Starting news sources configuration...")
    
    with app.app_context():
        if configure_sources():
            logger.info("News sources configuration completed successfully.")
        else:
            logger.error("News sources configuration failed.")

if __name__ == "__main__":
    main()