#!/usr/bin/env python3
"""
Management script for data sources in the CIVILIAN system.
This script provides a command-line interface for:
- Listing all data sources
- Adding new data sources
- Enabling/disabling sources
- Testing sources
- Removing sources
"""

import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("manage_data_sources")

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import application components
from app import app, db
from models import DataSource

def list_sources(args):
    """List all data sources or filtered subset."""
    with app.app_context():
        query = DataSource.query
        
        # Apply filters
        if args.type:
            query = query.filter_by(source_type=args.type)
        
        if args.active:
            query = query.filter_by(is_active=True)
        elif args.inactive:
            query = query.filter_by(is_active=False)
        
        # Execute query
        sources = query.all()
        
        # Further filter by name if provided
        if args.name:
            sources = [s for s in sources if args.name.lower() in s.name.lower()]
        
        # Print results
        print(f"\nFound {len(sources)} sources matching criteria")
        
        for i, source in enumerate(sources, 1):
            status = "ACTIVE" if source.is_active else "INACTIVE"
            last_update = source.last_ingestion.strftime("%Y-%m-%d %H:%M:%S") if source.last_ingestion else "Never"
            
            print(f"\n[{i}] {source.name}")
            print(f"    ID: {source.id}")
            print(f"    Type: {source.source_type}")
            print(f"    Status: {status}")
            print(f"    Last Update: {last_update}")
            
            if args.verbose:
                # Print detailed configuration
                try:
                    if source.config:
                        config = json.loads(source.config)
                        print("    Configuration:")
                        
                        if source.source_type == 'rss' and 'feeds' in config:
                            print(f"        Feeds:")
                            for feed in config['feeds']:
                                print(f"        - {feed}")
                        
                        elif source.source_type == 'twitter' and 'queries' in config:
                            print(f"        Queries:")
                            for query in config['queries']:
                                print(f"        - {query}")
                        
                        elif source.source_type == 'telegram' and 'entities' in config:
                            print(f"        Entities:")
                            for entity in config['entities']:
                                print(f"        - {entity}")
                except Exception as e:
                    print(f"    Error parsing config: {e}")

def add_source(args):
    """Add a new data source."""
    with app.app_context():
        # Check if a source with this name already exists
        existing = DataSource.query.filter_by(name=args.name).first()
        if existing:
            print(f"Error: A source with name '{args.name}' already exists.")
            return
        
        # Create the configuration
        if args.type == 'rss':
            if not args.feed:
                print("Error: RSS sources require at least one feed URL (--feed).")
                return
            
            config = {'feeds': args.feed}
        
        elif args.type == 'twitter':
            if not args.query:
                print("Error: Twitter sources require at least one query (--query).")
                return
            
            config = {'queries': args.query}
        
        elif args.type == 'telegram':
            if not args.entity:
                print("Error: Telegram sources require at least one entity (--entity).")
                return
            
            config = {'entities': args.entity}
        
        else:
            print(f"Error: Unknown source type '{args.type}'.")
            return
        
        # Add optional description
        if args.description:
            config['description'] = args.description
        
        # Create the source
        try:
            source = DataSource(
                name=args.name,
                source_type=args.type,
                config=json.dumps(config),
                is_active=not args.inactive,
                created_at=datetime.utcnow()
            )
            
            # Add to database
            db.session.add(source)
            db.session.commit()
            
            print(f"Successfully added {args.type} source: {args.name} (ID: {source.id})")
        
        except Exception as e:
            db.session.rollback()
            print(f"Error adding source: {e}")

def update_source(args):
    """Update an existing data source."""
    with app.app_context():
        # Get the source
        source = DataSource.query.filter_by(id=args.id).first()
        if not source:
            print(f"Error: Source with ID {args.id} not found.")
            return
        
        changes_made = False
        
        # Update name if provided
        if args.name:
            source.name = args.name
            changes_made = True
            print(f"Updated name to: {args.name}")
        
        # Update active status if provided
        if args.activate:
            source.is_active = True
            changes_made = True
            print(f"Source activated")
        elif args.deactivate:
            source.is_active = False
            changes_made = True
            print(f"Source deactivated")
        
        # Update configuration if needed
        if args.feed or args.query or args.entity or args.description:
            try:
                config = json.loads(source.config) if source.config else {}
                
                # Update feeds
                if args.feed and source.source_type == 'rss':
                    if args.replace:
                        config['feeds'] = args.feed
                        print(f"Replaced feeds with {len(args.feed)} new feeds")
                    else:
                        if 'feeds' not in config:
                            config['feeds'] = []
                        for feed in args.feed:
                            if feed not in config['feeds']:
                                config['feeds'].append(feed)
                                print(f"Added feed: {feed}")
                    changes_made = True
                
                # Update queries
                if args.query and source.source_type == 'twitter':
                    if args.replace:
                        config['queries'] = args.query
                        print(f"Replaced queries with {len(args.query)} new queries")
                    else:
                        if 'queries' not in config:
                            config['queries'] = []
                        for query in args.query:
                            if query not in config['queries']:
                                config['queries'].append(query)
                                print(f"Added query: {query}")
                    changes_made = True
                
                # Update entities
                if args.entity and source.source_type == 'telegram':
                    if args.replace:
                        config['entities'] = args.entity
                        print(f"Replaced entities with {len(args.entity)} new entities")
                    else:
                        if 'entities' not in config:
                            config['entities'] = []
                        for entity in args.entity:
                            if entity not in config['entities']:
                                config['entities'].append(entity)
                                print(f"Added entity: {entity}")
                    changes_made = True
                
                # Update description
                if args.description:
                    config['description'] = args.description
                    print(f"Updated description")
                    changes_made = True
                
                # Save the updated config
                source.config = json.dumps(config)
            
            except Exception as e:
                print(f"Error updating configuration: {e}")
                db.session.rollback()
                return
        
        # Commit changes
        if changes_made:
            try:
                db.session.commit()
                print(f"Successfully updated source: {source.name} (ID: {source.id})")
            except Exception as e:
                db.session.rollback()
                print(f"Error saving changes: {e}")
        else:
            print("No changes were made to the source.")

def delete_source(args):
    """Delete a data source."""
    with app.app_context():
        # Get the source
        source = DataSource.query.filter_by(id=args.id).first()
        if not source:
            print(f"Error: Source with ID {args.id} not found.")
            return
        
        # Confirm deletion if not forced
        if not args.force:
            confirm = input(f"Are you sure you want to delete the source '{source.name}' (ID: {source.id})? (y/N): ")
            if confirm.lower() != 'y':
                print("Deletion cancelled.")
                return
        
        # Delete the source
        try:
            db.session.delete(source)
            db.session.commit()
            print(f"Successfully deleted source: {source.name} (ID: {source.id})")
        except Exception as e:
            db.session.rollback()
            print(f"Error deleting source: {e}")

def test_source(args):
    """Test a data source by trying to fetch content."""
    with app.app_context():
        # Get the source
        source = DataSource.query.filter_by(id=args.id).first()
        if not source:
            print(f"Error: Source with ID {args.id} not found.")
            return
        
        print(f"Testing source: {source.name} (ID: {source.id}, Type: {source.source_type})")
        
        # Import relevant source handler based on type
        try:
            if source.source_type == 'rss':
                from data_sources.rss_source import RSSSource
                handler = RSSSource()
                
                # Get feeds from config
                config = json.loads(source.config) if source.config else {}
                feeds = config.get('feeds', [])
                
                if not feeds:
                    print("Error: Source has no feeds configured.")
                    return
                
                # Test each feed
                for feed_url in feeds:
                    print(f"\nTesting feed: {feed_url}")
                    try:
                        entries = handler.fetch_feed(feed_url, max_entries=5)
                        print(f"Success! Retrieved {len(entries)} entries")
                        
                        # Show sample entries
                        if entries and args.verbose:
                            print("\nSample entries:")
                            for i, entry in enumerate(entries[:3], 1):
                                print(f"[{i}] {entry.get('title', 'No title')}")
                                print(f"    Published: {entry.get('published', 'Unknown')}")
                                print(f"    Link: {entry.get('link', 'No link')}")
                    
                    except Exception as e:
                        print(f"Error fetching feed: {e}")
            
            elif source.source_type == 'twitter':
                from data_sources.twitter_source import TwitterSource
                handler = TwitterSource()
                
                # Check if Twitter API is available
                if not hasattr(handler, '_client') and not hasattr(handler, '_api'):
                    print("Error: Twitter API not configured. Please set the Twitter API credentials.")
                    return
                
                # Get queries from config
                config = json.loads(source.config) if source.config else {}
                queries = config.get('queries', [])
                
                if not queries:
                    print("Error: Source has no queries configured.")
                    return
                
                # Test each query
                for query in queries:
                    print(f"\nTesting query: {query}")
                    try:
                        tweets = handler.search_tweets(query, max_tweets=5)
                        print(f"Success! Retrieved {len(tweets)} tweets")
                        
                        # Show sample tweets
                        if tweets and args.verbose:
                            print("\nSample tweets:")
                            for i, tweet in enumerate(tweets[:3], 1):
                                print(f"[{i}] {tweet.get('text', 'No text')}")
                                print(f"    URL: {tweet.get('url', 'No URL')}")
                    
                    except Exception as e:
                        print(f"Error searching tweets: {e}")
            
            elif source.source_type == 'telegram':
                print("Note: Telegram testing requires authentication")
                from data_sources.telegram_source import TelegramSource
                handler = TelegramSource()
                
                # Check if Telegram client is available
                if not handler._client:
                    print("Error: Telegram client not configured. Please set the Telegram API credentials.")
                    return
                
                # Get entities from config
                config = json.loads(source.config) if source.config else {}
                entities = config.get('entities', [])
                
                if not entities:
                    print("Error: Source has no entities configured.")
                    return
                
                print("Note: Interactive testing of Telegram sources is not available in this script.")
                print("Please use the web interface to test Telegram sources.")
            
            else:
                print(f"Error: Unknown source type '{source.source_type}'.")
                return
        
        except Exception as e:
            print(f"Error testing source: {e}")

def main():
    """Main function to parse arguments and dispatch commands."""
    parser = argparse.ArgumentParser(description='Manage CIVILIAN data sources')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List sources command
    list_parser = subparsers.add_parser('list', help='List data sources')
    list_parser.add_argument('--type', choices=['rss', 'twitter', 'telegram'], help='Filter by source type')
    list_parser.add_argument('--name', help='Filter by source name (substring match)')
    list_parser.add_argument('--active', action='store_true', help='Show only active sources')
    list_parser.add_argument('--inactive', action='store_true', help='Show only inactive sources')
    list_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # Add source command
    add_parser = subparsers.add_parser('add', help='Add a new data source')
    add_parser.add_argument('--name', required=True, help='Name for the source')
    add_parser.add_argument('--type', required=True, choices=['rss', 'twitter', 'telegram'], help='Source type')
    add_parser.add_argument('--feed', action='append', help='RSS feed URL (can be used multiple times)')
    add_parser.add_argument('--query', action='append', help='Twitter search query (can be used multiple times)')
    add_parser.add_argument('--entity', action='append', help='Telegram entity (can be used multiple times)')
    add_parser.add_argument('--description', help='Description for the source')
    add_parser.add_argument('--inactive', action='store_true', help='Create source as inactive')
    
    # Update source command
    update_parser = subparsers.add_parser('update', help='Update an existing data source')
    update_parser.add_argument('--id', required=True, type=int, help='ID of the source to update')
    update_parser.add_argument('--name', help='New name for the source')
    update_parser.add_argument('--feed', action='append', help='RSS feed URL to add (can be used multiple times)')
    update_parser.add_argument('--query', action='append', help='Twitter search query to add (can be used multiple times)')
    update_parser.add_argument('--entity', action='append', help='Telegram entity to add (can be used multiple times)')
    update_parser.add_argument('--description', help='New description for the source')
    update_parser.add_argument('--replace', action='store_true', help='Replace existing feeds/queries/entities instead of adding')
    update_parser.add_argument('--activate', action='store_true', help='Activate the source')
    update_parser.add_argument('--deactivate', action='store_true', help='Deactivate the source')
    
    # Delete source command
    delete_parser = subparsers.add_parser('delete', help='Delete a data source')
    delete_parser.add_argument('--id', required=True, type=int, help='ID of the source to delete')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    
    # Test source command
    test_parser = subparsers.add_parser('test', help='Test a data source')
    test_parser.add_argument('--id', required=True, type=int, help='ID of the source to test')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Dispatch command
    if args.command == 'list':
        list_sources(args)
    elif args.command == 'add':
        add_source(args)
    elif args.command == 'update':
        update_source(args)
    elif args.command == 'delete':
        delete_source(args)
    elif args.command == 'test':
        test_source(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()