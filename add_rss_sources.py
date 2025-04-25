#!/usr/bin/env python3
"""
A simplified script to add RSS sources to the CIVILIAN system.
This only processes the open news sources configuration file.
"""

import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("add_rss_sources")

# Import app components
from app import app, db
from models import DataSource

def add_rss_sources():
    """Add RSS sources defined in the config file."""
    config_path = Path("config/open_news_sources.json")
    success_count = 0
    
    with app.app_context():
        # Load the news sources config
        try:
            with open(config_path, 'r') as f:
                news_sources = json.load(f)
        except Exception as e:
            logger.error(f"Error loading news sources: {e}")
            return
        
        # Process each category
        for category, feeds in news_sources.items():
            logger.info(f"Processing {category} category with {len(feeds)} feeds")
            
            for feed in feeds:
                try:
                    # Create source name
                    name = f"{feed['name']} ({feed['category']})"
                    
                    # Check if source already exists
                    existing = DataSource.query.filter_by(name=name).first()
                    if existing:
                        logger.info(f"Source '{name}' already exists, skipping.")
                        continue
                    
                    # Create config
                    config = {'feeds': [feed['url']]}
                    
                    # Create the source
                    source = DataSource(
                        name=name,
                        source_type='rss',
                        config=json.dumps(config),
                        is_active=True
                    )
                    
                    # Add to database
                    db.session.add(source)
                    db.session.commit()
                    
                    logger.info(f"Added source: {name}")
                    success_count += 1
                
                except Exception as e:
                    logger.error(f"Error adding source {feed['name']}: {e}")
                    db.session.rollback()
        
        logger.info(f"Added {success_count} RSS sources")

if __name__ == "__main__":
    add_rss_sources()