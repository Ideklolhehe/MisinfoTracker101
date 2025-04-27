"""
RSS Feeds route for the CIVILIAN system.
This module provides routes for managing RSS feed sources.
"""

import logging
import json
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from sqlalchemy.exc import SQLAlchemyError

from app import app, db
from models import DataSource, SystemLog
from services.rss_feed_manager import RSSFeedManager

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

# Blueprint will be registered in app.py