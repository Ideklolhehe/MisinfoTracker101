#!/usr/bin/env python3
"""
Initialization script to configure open news and data sources for the CIVILIAN system.
This script must be run with the Flask application context to avoid app context errors.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("init_open_sources")

# Load application components
from app import app, db
from models import DataSource
from data_sources.rss_source import RSSSource
from data_sources.twitter_source import TwitterSource
from data_sources.telegram_source import TelegramSource

# Define paths
CONFIG_DIR = Path("config")
OPEN_NEWS_SOURCES_FILE = CONFIG_DIR / "open_news_sources.json"
FACT_CHECK_SOURCES_FILE = CONFIG_DIR / "fact_check_sources.json"
OPEN_DATA_SOURCES_FILE = CONFIG_DIR / "open_data_sources.json"

def load_json_config(filepath):
    """Load a JSON configuration file."""
    try:
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file not found: {filepath}")
            return None
    except Exception as e:
        logger.error(f"Error loading config file {filepath}: {e}")
        return None

def configure_rss_sources():
    """Configure RSS data sources from configuration files."""
    # Initialize RSS source
    rss_source = RSSSource()
    success_count = 0
    failure_count = 0
    
    # Process open news sources
    news_sources = load_json_config(OPEN_NEWS_SOURCES_FILE)
    if news_sources:
        logger.info("Configuring open news sources...")
        for category, feeds in news_sources.items():
            logger.info(f"Processing {category} category with {len(feeds)} feeds")
            
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
                        logger.info(f"Added RSS source: {name}")
                        success_count += 1
                    else:
                        logger.warning(f"Failed to add RSS source: {name}")
                        failure_count += 1
                
                except Exception as e:
                    logger.error(f"Error adding source {feed.get('name', 'unknown')}: {e}")
                    failure_count += 1
    
    # Process fact check sources
    fact_check_sources = load_json_config(FACT_CHECK_SOURCES_FILE)
    if fact_check_sources:
        logger.info("Configuring fact-checking sources...")
        for category, feeds in fact_check_sources.items():
            logger.info(f"Processing {category} category with {len(feeds)} feeds")
            
            for feed in feeds:
                try:
                    # Create a name for the source
                    name = f"FactCheck: {feed['name']} ({feed['category']})"
                    
                    # Check if source already exists
                    existing = DataSource.query.filter_by(name=name).first()
                    if existing:
                        logger.info(f"Source '{name}' already exists, skipping.")
                        continue
                    
                    # Create the RSS source
                    config = {'feeds': [feed['url']]}
                    source_id = rss_source.create_source(name, config)
                    
                    if source_id:
                        logger.info(f"Added fact-check source: {name}")
                        success_count += 1
                    else:
                        logger.warning(f"Failed to add fact-check source: {name}")
                        failure_count += 1
                
                except Exception as e:
                    logger.error(f"Error adding fact-check source {feed.get('name', 'unknown')}: {e}")
                    failure_count += 1
    
    # Process open data sources
    open_data_sources = load_json_config(OPEN_DATA_SOURCES_FILE)
    if open_data_sources:
        logger.info("Configuring open data sources...")
        for category, feeds in open_data_sources.items():
            logger.info(f"Processing {category} category with {len(feeds)} feeds")
            
            for feed in feeds:
                try:
                    # Create a name for the source
                    name = f"OpenData: {feed['name']} ({feed['category']})"
                    
                    # Check if source already exists
                    existing = DataSource.query.filter_by(name=name).first()
                    if existing:
                        logger.info(f"Source '{name}' already exists, skipping.")
                        continue
                    
                    # Create the RSS source
                    config = {'feeds': [feed['url']]}
                    source_id = rss_source.create_source(name, config)
                    
                    if source_id:
                        logger.info(f"Added open data source: {name}")
                        success_count += 1
                    else:
                        logger.warning(f"Failed to add open data source: {name}")
                        failure_count += 1
                
                except Exception as e:
                    logger.error(f"Error adding open data source {feed.get('name', 'unknown')}: {e}")
                    failure_count += 1
    
    logger.info(f"RSS configuration complete. Added {success_count} sources. Failed: {failure_count}")
    return success_count, failure_count

def configure_twitter_sources():
    """Configure Twitter data sources for monitoring."""
    # Initialize Twitter source
    twitter_source = TwitterSource()
    success_count = 0
    failure_count = 0
    
    # Define misinformation monitoring queries
    misinfo_queries = [
        {
            "name": "COVID-19 Misinformation Monitor",
            "queries": [
                "covid hoax",
                "covid conspiracy",
                "vaccine microchip",
                "5G covid",
                "pandemic fake"
            ]
        },
        {
            "name": "Climate Change Misinformation Monitor",
            "queries": [
                "climate hoax",
                "climate change fake",
                "global warming myth",
                "climate scientists lying"
            ]
        },
        {
            "name": "Election Misinformation Monitor",
            "queries": [
                "election rigged",
                "election stolen",
                "voter fraud widespread",
                "voting machines hacked"
            ]
        },
        {
            "name": "Fact-Check Accounts",
            "queries": [
                "from:Snopes",
                "from:PolitiFact",
                "from:FactCheck.org",
                "from:APFactCheck",
                "from:FullFact"
            ]
        }
    ]
    
    # Create Twitter sources
    for source_config in misinfo_queries:
        try:
            name = source_config["name"]
            
            # Check if source already exists
            existing = DataSource.query.filter_by(name=name, source_type='twitter').first()
            if existing:
                logger.info(f"Twitter source '{name}' already exists, skipping.")
                continue
            
            # Create config
            config = {'queries': source_config["queries"]}
            source_id = twitter_source.create_source(name, config)
            
            if source_id:
                logger.info(f"Added Twitter source: {name}")
                success_count += 1
            else:
                logger.warning(f"Failed to add Twitter source: {name}")
                failure_count += 1
        
        except Exception as e:
            logger.error(f"Error adding Twitter source {source_config.get('name', 'unknown')}: {e}")
            failure_count += 1
    
    logger.info(f"Twitter configuration complete. Added {success_count} sources. Failed: {failure_count}")
    return success_count, failure_count

def configure_telegram_sources():
    """Configure Telegram data sources for monitoring."""
    # Initialize Telegram source
    telegram_source = TelegramSource()
    success_count = 0
    failure_count = 0
    
    # Define Telegram channels/groups to monitor
    telegram_sources = [
        {
            "name": "Public Fact-Check Channels",
            "entities": [
                "@FactCheck",
                "@Fullfact",
                "@correctiv_org",
                "@AP_FactCheck"
            ]
        },
        {
            "name": "Science Communication Channels",
            "entities": [
                "@WHO",
                "@CDCgov",
                "@ScienceAlert",
                "@NatureNews"
            ]
        }
    ]
    
    # Create Telegram sources
    for source_config in telegram_sources:
        try:
            name = source_config["name"]
            
            # Check if source already exists
            existing = DataSource.query.filter_by(name=name, source_type='telegram').first()
            if existing:
                logger.info(f"Telegram source '{name}' already exists, skipping.")
                continue
            
            # Create config
            config = {'entities': source_config["entities"]}
            source_id = telegram_source.create_source(name, config)
            
            if source_id:
                logger.info(f"Added Telegram source: {name}")
                success_count += 1
            else:
                logger.warning(f"Failed to add Telegram source: {name}")
                failure_count += 1
        
        except Exception as e:
            logger.error(f"Error adding Telegram source {source_config.get('name', 'unknown')}: {e}")
            failure_count += 1
    
    logger.info(f"Telegram configuration complete. Added {success_count} sources. Failed: {failure_count}")
    return success_count, failure_count

def main():
    """Main initialization function."""
    logger.info("Starting initialization of open sources for CIVILIAN system")
    
    with app.app_context():
        # Configure all data sources
        rss_success, rss_failed = configure_rss_sources()
        twitter_success, twitter_failed = configure_twitter_sources()
        telegram_success, telegram_failed = configure_telegram_sources()
        
        # Summary
        total_success = rss_success + twitter_success + telegram_success
        total_failed = rss_failed + twitter_failed + telegram_failed
        
        logger.info("========== INITIALIZATION SUMMARY ==========")
        logger.info(f"Total sources added: {total_success}")
        logger.info(f"Total sources failed: {total_failed}")
        logger.info(f"RSS sources: {rss_success} added, {rss_failed} failed")
        logger.info(f"Twitter sources: {twitter_success} added, {twitter_failed} failed")
        logger.info(f"Telegram sources: {telegram_success} added, {telegram_failed} failed")
        logger.info("===========================================")

if __name__ == "__main__":
    main()