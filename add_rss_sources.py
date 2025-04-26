#!/usr/bin/env python
"""
A simplified script to add RSS sources to the CIVILIAN system.
This only processes the open news sources configuration file.
"""

import os
import json
import sys
from app import app, db
from models import DataSource

def add_rss_sources():
    """Add RSS sources defined in the config file."""
    # Path to configuration file
    config_file = os.path.join('config', 'open_news_sources.json')
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file {config_file} not found")
        sys.exit(1)
    
    # Load the configuration
    with open(config_file, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    # Process each category of news sources
    total_added = 0
    for category, sources in config.items():
        print(f"Processing category: {category}")
        for source_name, source_config in sources.items():
            # Create a properly formatted name for the source
            full_name = f"RSS: {source_name} ({category})"
            
            # Check if the source already exists
            existing = db.session.query(DataSource).filter_by(name=full_name).first()
            if existing:
                print(f"Source '{full_name}' already exists, skipping.")
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
            
            print(f"Added source: {full_name}")
            total_added += 1
    
    # Commit changes to the database
    if total_added > 0:
        db.session.commit()
        print(f"Successfully added {total_added} RSS sources")
    else:
        print("No new RSS sources added")

if __name__ == "__main__":
    with app.app_context():
        add_rss_sources()
