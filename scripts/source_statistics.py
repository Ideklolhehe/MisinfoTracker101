#!/usr/bin/env python3
"""
Script to display statistics about configured data sources in the CIVILIAN system.
This script shows detailed information about all the configured data sources,
their types, status, and last update time.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("source_statistics")

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import application components
from app import app, db
from models import DataSource

def display_source_statistics():
    """Display statistics about all configured data sources."""
    with app.app_context():
        # Get all data sources
        all_sources = DataSource.query.all()
        
        # Count by type
        source_types = {}
        for source in all_sources:
            source_type = source.source_type
            if source_type not in source_types:
                source_types[source_type] = {
                    'total': 0,
                    'active': 0,
                    'inactive': 0,
                    'categories': set()
                }
            
            source_types[source_type]['total'] += 1
            
            if source.is_active:
                source_types[source_type]['active'] += 1
            else:
                source_types[source_type]['inactive'] += 1
            
            # Extract category from name
            if '(' in source.name and ')' in source.name:
                category = source.name.split('(')[-1].split(')')[0]
                source_types[source_type]['categories'].add(category)
        
        # Print summary
        print("\n===== CIVILIAN DATA SOURCE STATISTICS =====")
        print(f"Total sources: {len(all_sources)}")
        
        for source_type, stats in source_types.items():
            print(f"\n== {source_type.upper()} SOURCES ==")
            print(f"Total: {stats['total']}")
            print(f"Active: {stats['active']}")
            print(f"Inactive: {stats['inactive']}")
            print(f"Categories: {', '.join(sorted(stats['categories']))}")
        
        # Recent updates
        print("\n== RECENT SOURCE UPDATES ==")
        recent_sources = DataSource.query.filter(DataSource.last_ingestion != None).order_by(
            DataSource.last_ingestion.desc()
        ).limit(10).all()
        
        if recent_sources:
            for source in recent_sources:
                last_update = source.last_ingestion.strftime("%Y-%m-%d %H:%M:%S") if source.last_ingestion else "Never"
                print(f"{source.name} - Last update: {last_update}")
        else:
            print("No recent source updates found.")
        
        # RSS Feed Categories
        print("\n== RSS FEED CATEGORIES ==")
        categories = {}
        for source in all_sources:
            if source.source_type == 'rss':
                # Extract category from name
                if '(' in source.name and ')' in source.name:
                    category = source.name.split('(')[-1].split(')')[0]
                    if category not in categories:
                        categories[category] = 0
                    categories[category] += 1
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"{category}: {count} sources")
        
        print("\n===========================================")

def display_source_details(source_type=None, category=None):
    """Display detailed information about specific sources."""
    with app.app_context():
        query = DataSource.query
        
        # Filter by source type if provided
        if source_type:
            query = query.filter_by(source_type=source_type)
        
        # Get all sources
        sources = query.all()
        
        # Filter by category if provided
        if category:
            filtered_sources = []
            for source in sources:
                if '(' in source.name and ')' in source.name:
                    src_category = source.name.split('(')[-1].split(')')[0]
                    if category.lower() in src_category.lower():
                        filtered_sources.append(source)
            sources = filtered_sources
        
        # Print detailed information
        print(f"\n===== DETAILED SOURCE INFORMATION =====")
        print(f"Filters: Type={source_type or 'All'}, Category={category or 'All'}")
        print(f"Total matching sources: {len(sources)}")
        
        for i, source in enumerate(sources, 1):
            print(f"\n[{i}] {source.name} ({source.source_type})")
            print(f"    Active: {source.is_active}")
            
            last_update = source.last_ingestion.strftime("%Y-%m-%d %H:%M:%S") if source.last_ingestion else "Never"
            print(f"    Last Update: {last_update}")
            
            if source.config:
                try:
                    config = json.loads(source.config)
                    if source.source_type == 'rss' and 'feeds' in config:
                        print(f"    Feeds: {len(config['feeds'])}")
                        for feed in config['feeds']:
                            print(f"        - {feed}")
                    elif source.source_type == 'twitter' and 'queries' in config:
                        print(f"    Queries: {len(config['queries'])}")
                        for query in config['queries']:
                            print(f"        - {query}")
                    elif source.source_type == 'telegram' and 'entities' in config:
                        print(f"    Entities: {len(config['entities'])}")
                        for entity in config['entities']:
                            print(f"        - {entity}")
                except json.JSONDecodeError:
                    print(f"    Config: Invalid JSON")
        
        print("\n=========================================")

def main():
    """Main function to run the statistics display."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Display statistics about CIVILIAN data sources.')
    parser.add_argument('--type', help='Filter sources by type (rss, twitter, telegram)')
    parser.add_argument('--category', help='Filter sources by category substring')
    parser.add_argument('--details', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    if args.details:
        display_source_details(args.type, args.category)
    else:
        display_source_statistics()

if __name__ == "__main__":
    main()