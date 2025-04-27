"""
RSS Feeds route for the CIVILIAN system.
This module provides routes for managing RSS feed sources.
"""

import logging
import json
import time
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from sqlalchemy.exc import SQLAlchemyError

from app import app, db
from models import DataSource, SystemLog
from services.rss_feed_manager import RSSFeedManager
from utils.feed_parser import parse_feed_with_retry
from utils.feed_batch_operations import (
    test_all_feeds, 
    test_feed_batch, 
    update_feed_statuses,
    test_alternative_feeds,
    update_feed_urls
)

logger = logging.getLogger(__name__)

# Create blueprint
rss_feeds_bp = Blueprint('rss_feeds', __name__, url_prefix='/rss-feeds')

# Initialize RSS feed manager
feed_manager = RSSFeedManager()

@rss_feeds_bp.route('/')
@login_required
def index():
    """Display RSS feed management interface."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home.index'))
        
    # Get all RSS sources
    sources = DataSource.query.filter_by(source_type='rss').all()
    
    source_data = []
    for source in sources:
        # Parse the config
        config = json.loads(source.config) if source.config else {}
        feeds = config.get('feeds', [])
        
        feed_count = len(feeds)
        last_ingestion = source.last_ingestion.strftime('%Y-%m-%d %H:%M:%S') if source.last_ingestion else 'Never'
        
        source_data.append({
            'id': source.id,
            'name': source.name,
            'feed_count': feed_count,
            'is_active': source.is_active,
            'last_ingestion': last_ingestion,
            'feeds': feeds
        })
    
    # Get recent feed-related logs
    logs = SystemLog.query.filter_by(
        component='rss_feed_manager'
    ).order_by(
        SystemLog.timestamp.desc()
    ).limit(10).all()
    
    return render_template(
        'rss_feeds/index.html',
        sources=source_data,
        logs=logs,
        title='RSS Feed Management'
    )

@rss_feeds_bp.route('/validate', methods=['POST'])
@login_required
def validate_feed():
    """Validate an RSS feed URL."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get the feed URL from request
    feed_url = request.json.get('feed_url')
    if not feed_url:
        return jsonify({'error': 'No feed URL provided'}), 400
        
    # Validate the feed
    valid, message = feed_manager.validate_feed(feed_url)
    
    # If invalid, try to find an alternative
    alternative = None
    if not valid:
        try:
            domain = feed_url.split('//')[-1].split('/')[0]
            alternative = feed_manager.find_alternative_feed_url(domain)
        except Exception as e:
            logger.warning(f"Error finding alternative feed: {e}")
    
    return jsonify({
        'feed_url': feed_url,
        'valid': valid,
        'message': message,
        'alternative': alternative
    })

@rss_feeds_bp.route('/check-all')
@login_required
def check_all_feeds():
    """Check all RSS feeds in the database for validity."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Check all feeds
    feed_status = feed_manager.check_feeds_in_database()
    
    return jsonify({
        'feed_count': len(feed_status),
        'feeds': feed_status
    })

@rss_feeds_bp.route('/update', methods=['POST'])
@login_required
def update_feed():
    """Update an RSS feed URL."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get request data
    data = request.json
    source_id = data.get('source_id')
    old_url = data.get('old_url')
    new_url = data.get('new_url')
    
    if not all([source_id, old_url, new_url]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    # Update the feed URL
    success = feed_manager.update_feed_url(source_id, old_url, new_url)
    
    if success:
        return jsonify({'success': True, 'message': 'Feed URL updated successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to update feed URL'}), 500

@rss_feeds_bp.route('/add', methods=['POST'])
@login_required
def add_feed():
    """Add a new RSS feed URL to a data source."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get request data
    data = request.json
    source_id = data.get('source_id')
    new_url = data.get('new_url')
    
    if not all([source_id, new_url]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    # Add the feed URL
    success = feed_manager.add_feed_url(source_id, new_url)
    
    if success:
        return jsonify({'success': True, 'message': 'Feed URL added successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to add feed URL'}), 500

@rss_feeds_bp.route('/remove', methods=['POST'])
@login_required
def remove_feed():
    """Remove an RSS feed URL from a data source."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get request data
    data = request.json
    source_id = data.get('source_id')
    url = data.get('url')
    
    if not all([source_id, url]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    # Remove the feed URL
    success = feed_manager.remove_feed_url(source_id, url)
    
    if success:
        return jsonify({'success': True, 'message': 'Feed URL removed successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to remove feed URL'}), 500

# Add batch operations routes
@rss_feeds_bp.route('/batch/test-all', methods=['GET'])
@login_required
def batch_test_all_feeds():
    """Test all RSS feeds in parallel and return their status."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Get the max workers parameter
    max_workers = int(request.args.get('max_workers', 5))
    if max_workers < 1:
        max_workers = 1
    elif max_workers > 20:
        max_workers = 20
    
    # Get the timeout parameter
    timeout = int(request.args.get('timeout', 30))
    if timeout < 5:
        timeout = 5
    elif timeout > 120:
        timeout = 120
        
    # Test all feeds
    try:
        start_time = time.time()
        results = test_all_feeds(max_workers=max_workers, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        # Update feed statuses in the database
        updated_count = update_feed_statuses(results)
        
        # Log the operation
        log = SystemLog(
            log_type='info',
            component='rss_feed_manager',
            message=f"Batch tested {len(results)} feeds in {elapsed_time:.2f} seconds, updated {updated_count} feed statuses"
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'feed_count': len(results),
            'elapsed_time': elapsed_time,
            'updated_count': updated_count,
            'results': results
        })
    except Exception as e:
        logger.error(f"Error testing feeds: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feeds_bp.route('/batch/find-alternatives', methods=['GET'])
@login_required
def batch_find_alternatives():
    """Find alternative URLs for broken feeds."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get all broken feeds
    try:
        broken_feeds = []
        sources = DataSource.query.filter_by(source_type='rss').all()
        
        for source in sources:
            if source.source_url:
                # Check feed status from metadata
                if hasattr(source, 'meta_data') and source.meta_data:
                    meta_data = source.meta_data
                    if isinstance(meta_data, str):
                        try:
                            meta_data = json.loads(meta_data)
                        except:
                            meta_data = {}
                            
                    if meta_data.get('last_status') == 'error':
                        broken_feeds.append(source.source_url)
        
        # Find and update alternatives
        start_time = time.time()
        updates = update_feed_urls(broken_feeds)
        elapsed_time = time.time() - start_time
        
        # Log the operation
        log = SystemLog(
            log_type='info',
            component='rss_feed_manager',
            message=f"Found alternatives for {len(updates)} out of {len(broken_feeds)} broken feeds in {elapsed_time:.2f} seconds"
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'broken_feed_count': len(broken_feeds),
            'updated_count': len(updates),
            'elapsed_time': elapsed_time,
            'updates': updates
        })
    except Exception as e:
        logger.error(f"Error finding alternatives: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feeds_bp.route('/batch/health-check', methods=['GET'])
@login_required
def batch_health_check():
    """Get health status of all RSS feeds."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    try:
        sources = DataSource.query.filter_by(source_type='rss').all()
        
        total_count = len(sources)
        active_count = 0
        error_count = 0
        never_checked = 0
        
        feed_status = {
            'ok': [],
            'error': [],
            'never_checked': []
        }
        
        for source in sources:
            if source.source_url:
                # Check feed status from metadata
                meta_data = source.meta_data
                if isinstance(meta_data, str):
                    try:
                        meta_data = json.loads(meta_data)
                    except:
                        meta_data = {}
                
                if not meta_data or 'last_status' not in meta_data:
                    never_checked += 1
                    feed_status['never_checked'].append({
                        'id': source.id,
                        'name': source.name,
                        'url': source.source_url
                    })
                elif meta_data['last_status'] == 'ok':
                    active_count += 1
                    feed_status['ok'].append({
                        'id': source.id,
                        'name': source.name,
                        'url': source.source_url,
                        'entry_count': meta_data.get('entry_count', 0),
                        'last_checked': meta_data.get('last_checked', 0)
                    })
                else:
                    error_count += 1
                    feed_status['error'].append({
                        'id': source.id,
                        'name': source.name,
                        'url': source.source_url,
                        'error': meta_data.get('error', 'Unknown error'),
                        'last_checked': meta_data.get('last_checked', 0)
                    })
        
        return jsonify({
            'success': True,
            'total_count': total_count,
            'active_count': active_count,
            'error_count': error_count, 
            'never_checked': never_checked,
            'feed_status': feed_status
        })
    except Exception as e:
        logger.error(f"Error checking feed health: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Blueprint will be registered in app.py