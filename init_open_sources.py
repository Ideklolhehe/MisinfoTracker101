#!/usr/bin/env python
"""
Initialization script to configure open news and data sources for the CIVILIAN system.
This script must be run with the Flask application context to avoid app context errors.
"""

import os
import json
import sys
from app import app, db
from models import DataSource
import logging

logger = logging.getLogger(__name__)

def load_json_config(filepath):
    """Load a JSON configuration file."""
    if not os.path.exists(filepath):
        logger.error(f"Configuration file {filepath} not found")
        return None
    
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration file {filepath}: {e}")
        return None

def configure_rss_sources():
    """Configure RSS data sources from configuration files."""
    # Path to configuration file
    config_file = os.path.join('config', 'open_news_sources.json')
    
    config = load_json_config(config_file)
    if not config:
        return 0
    
    # Process each category of news sources
    total_added = 0
    for category, sources in config.items():
        logger.info(f"Processing category: {category}")
        for source_name, source_config in sources.items():
            # Create a properly formatted name for the source
            full_name = f"RSS: {source_name} ({category})"
            
            # Check if the source already exists
            existing = db.session.query(DataSource).filter_by(name=full_name).first()
            if existing:
                logger.info(f"Source '{full_name}' already exists, skipping.")
                continue
            
            # Create config JSON for the source
            source_data = {
                "url": source_config["url"],
                "category": category,
                "language": source_config.get("language", "en"),
                "update_frequency": source_config.get("update_frequency", 3600),
                "max_entries": source_config.get("max_entries", 50)
            }
            
            # Create and add the source
            new_source = DataSource(
                name=full_name,
                source_type="rss",
                config=json.dumps(source_data),
                is_active=True
            )
            db.session.add(new_source)
            
            logger.info(f"Added source: {full_name}")
            total_added += 1
    
    # Commit changes to the database
    if total_added > 0:
        db.session.commit()
        logger.info(f"Successfully added {total_added} RSS sources")
    
    return total_added

def configure_twitter_sources():
    """Configure Twitter data sources for monitoring."""
    # Path to configuration file
    config_file = os.path.join('config', 'twitter_sources.json')
    
    config = load_json_config(config_file)
    if not config:
        return 0
    
    # Process each monitoring group
    total_added = 0
    for group_name, group_config in config.items():
        # Create a properly formatted name for the source
        full_name = f"Twitter: {group_name}"
        
        # Check if the source already exists
        existing = db.session.query(DataSource).filter_by(name=full_name).first()
        if existing:
            logger.info(f"Twitter source '{full_name}' already exists, skipping.")
            continue
        
        # Create and add the source
        new_source = DataSource(
            name=full_name,
            source_type="twitter",
            config=json.dumps(group_config),
            is_active=True
        )
        db.session.add(new_source)
        
        logger.info(f"Added Twitter source: {full_name}")
        total_added += 1
    
    # Commit changes to the database
    if total_added > 0:
        db.session.commit()
        logger.info(f"Successfully added {total_added} Twitter sources")
    else:
        logger.info("No new Twitter sources added")
    
    return total_added

def configure_telegram_sources():
    """Configure Telegram data sources for monitoring."""
    # Path to configuration file
    config_file = os.path.join('config', 'telegram_sources.json')
    
    config = load_json_config(config_file)
    if not config:
        return 0
    
    # Process each monitoring group
    total_added = 0
    for group_name, group_config in config.items():
        # Create a properly formatted name for the source
        full_name = f"Telegram: {group_name}"
        
        # Check if the source already exists
        existing = db.session.query(DataSource).filter_by(name=full_name).first()
        if existing:
            logger.info(f"Telegram source '{full_name}' already exists, skipping.")
            continue
        
        # Create and add the source
        new_source = DataSource(
            name=full_name,
            source_type="telegram",
            config=json.dumps(group_config),
            is_active=True
        )
        db.session.add(new_source)
        
        logger.info(f"Added Telegram source: {full_name}")
        total_added += 1
    
    # Commit changes to the database
    if total_added > 0:
        db.session.commit()
        logger.info(f"Successfully added {total_added} Telegram sources")
    
    return total_added

def main():
    """Main initialization function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting open sources configuration")
    
    # Configure RSS sources
    rss_count = configure_rss_sources()
    logger.info(f"Total RSS sources added: {rss_count}")
    
    # Configure Twitter sources
    twitter_count = configure_twitter_sources()
    
    # Configure Telegram sources
    telegram_count = configure_telegram_sources()
    
    logger.info("Open sources configuration completed")

if __name__ == "__main__":
    with app.app_context():
        main()
