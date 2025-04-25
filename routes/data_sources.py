"""
Routes for managing data sources in the CIVILIAN system.
This module handles all the web UI routes for viewing and managing data sources.
"""

import json
from datetime import datetime
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from sqlalchemy.exc import SQLAlchemyError

from app import db
from models import DataSource
from data_sources.rss_source import RSSSource
from data_sources.twitter_source import TwitterSource
from data_sources.telegram_source import TelegramSource
from data_sources.youtube_source import YouTubeSource
from data_sources.darkweb_source import DarkWebSource

# Create blueprint
data_sources_bp = Blueprint('data_sources', __name__, url_prefix='/data_sources')

@data_sources_bp.route('/')
def index():
    """Show all data sources."""
    # Get filter parameters
    source_type = request.args.get('type')
    status = request.args.get('status')
    search = request.args.get('search')
    
    # Build query
    query = DataSource.query
    
    if source_type:
        query = query.filter_by(source_type=source_type)
    
    if status == 'active':
        query = query.filter_by(is_active=True)
    elif status == 'inactive':
        query = query.filter_by(is_active=False)
    
    if search:
        query = query.filter(DataSource.name.ilike(f'%{search}%'))
    
    # Get sources
    sources = query.order_by(DataSource.name).all()
    
    # Get counts
    total_count = DataSource.query.count()
    rss_count = DataSource.query.filter_by(source_type='rss').count()
    twitter_count = DataSource.query.filter_by(source_type='twitter').count()
    telegram_count = DataSource.query.filter_by(source_type='telegram').count()
    youtube_count = DataSource.query.filter_by(source_type='youtube').count()
    darkweb_count = DataSource.query.filter_by(source_type='darkweb').count()
    active_count = DataSource.query.filter_by(is_active=True).count()
    
    # Render template
    return render_template(
        'data_sources/index.html',
        sources=sources,
        total_count=total_count,
        rss_count=rss_count,
        twitter_count=twitter_count,
        telegram_count=telegram_count,
        youtube_count=youtube_count,
        darkweb_count=darkweb_count,
        active_count=active_count,
        current_type=source_type,
        current_status=status,
        current_search=search
    )

@data_sources_bp.route('/view/<int:source_id>')
def view(source_id):
    """View a single data source."""
    source = DataSource.query.get_or_404(source_id)
    
    # Parse config
    config = json.loads(source.config) if source.config else {}
    
    # Get source-specific data
    if source.source_type == 'rss':
        feeds = config.get('feeds', [])
        return render_template('data_sources/view_rss.html', source=source, feeds=feeds)
    
    elif source.source_type == 'twitter':
        queries = config.get('queries', [])
        return render_template('data_sources/view_twitter.html', source=source, queries=queries)
    
    elif source.source_type == 'telegram':
        entities = config.get('entities', [])
        return render_template('data_sources/view_telegram.html', source=source, entities=entities)
    
    elif source.source_type == 'youtube':
        return render_template('data_sources/view_youtube.html', source=source)
    
    elif source.source_type == 'darkweb':
        return render_template('data_sources/view_darkweb.html', source=source)
    
    # Default view
    return render_template('data_sources/view.html', source=source, config=config)

@data_sources_bp.route('/add_youtube')
def add_youtube():
    """Add a new YouTube data source."""
    return render_template('data_sources/add_youtube.html')

@data_sources_bp.route('/add_darkweb')
def add_darkweb():
    """Add a new Dark Web data source."""
    return render_template('data_sources/add_darkweb.html')

@data_sources_bp.route('/add', methods=['GET', 'POST'])
def add():
    """Add a new data source."""
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        source_type = request.form.get('type')
        is_active = request.form.get('is_active') == 'on'
        description = request.form.get('description', '')
        
        # Validate inputs
        if not name or not source_type:
            flash('Name and type are required', 'error')
            return redirect(url_for('data_sources.add'))
        
        # Check if source with this name already exists
        existing = DataSource.query.filter_by(name=name).first()
        if existing:
            flash(f'A source with the name "{name}" already exists', 'error')
            return redirect(url_for('data_sources.add'))
        
        # Create config based on source type
        config = {}
        
        if source_type == 'rss':
            feeds = [f.strip() for f in request.form.get('feeds', '').splitlines() if f.strip()]
            if not feeds:
                flash('At least one feed URL is required for RSS sources', 'error')
                return redirect(url_for('data_sources.add'))
            
            config['feeds'] = feeds
        
        elif source_type == 'twitter':
            queries = [q.strip() for q in request.form.get('queries', '').splitlines() if q.strip()]
            if not queries:
                flash('At least one search query is required for Twitter sources', 'error')
                return redirect(url_for('data_sources.add'))
            
            config['queries'] = queries
        
        elif source_type == 'telegram':
            entities = [e.strip() for e in request.form.get('entities', '').splitlines() if e.strip()]
            if not entities:
                flash('At least one entity is required for Telegram sources', 'error')
                return redirect(url_for('data_sources.add'))
            
            config['entities'] = entities
            
        elif source_type == 'youtube':
            monitor_type = request.form.get('monitor_type', 'channel')
            config['monitor_type'] = monitor_type
            
            # Process based on monitor type
            if monitor_type == 'channel':
                channel_ids = [c.strip() for c in request.form.get('channel_ids', '').splitlines() if c.strip()]
                if not channel_ids:
                    flash('At least one channel ID is required for YouTube channel monitoring', 'error')
                    return redirect(url_for('data_sources.add_youtube'))
                config['channel_ids'] = channel_ids
                
            elif monitor_type == 'search':
                search_queries = [q.strip() for q in request.form.get('search_queries', '').splitlines() if q.strip()]
                if not search_queries:
                    flash('At least one search query is required for YouTube search monitoring', 'error')
                    return redirect(url_for('data_sources.add_youtube'))
                config['search_queries'] = search_queries
                
            elif monitor_type == 'video':
                video_ids = [v.strip() for v in request.form.get('video_ids', '').splitlines() if v.strip()]
                if not video_ids:
                    flash('At least one video ID is required for YouTube video monitoring', 'error')
                    return redirect(url_for('data_sources.add_youtube'))
                config['video_ids'] = video_ids
                
            elif monitor_type == 'playlist':
                playlist_ids = [p.strip() for p in request.form.get('playlist_ids', '').splitlines() if p.strip()]
                if not playlist_ids:
                    flash('At least one playlist ID is required for YouTube playlist monitoring', 'error')
                    return redirect(url_for('data_sources.add_youtube'))
                config['playlist_ids'] = playlist_ids
            
            # Common YouTube settings
            config['max_videos'] = int(request.form.get('max_videos', 10))
            config['days_back'] = int(request.form.get('days_back', 7))
            config['include_comments'] = request.form.get('include_comments') == 'on'
            
            if config['include_comments']:
                config['max_comments'] = int(request.form.get('max_comments', 50))
        
        elif source_type == 'darkweb':
            # Process Dark Web site configurations
            site_urls = request.form.getlist('site_urls[]')
            site_types = request.form.getlist('site_types[]')
            site_max_pages = request.form.getlist('site_max_pages[]')
            site_content_selectors = request.form.getlist('site_content_selectors[]')
            site_link_selectors = request.form.getlist('site_link_selectors[]')
            site_exclude_patterns = request.form.getlist('site_exclude_patterns[]')
            
            # Validate
            if not site_urls:
                flash('At least one Dark Web site is required', 'error')
                return redirect(url_for('data_sources.add_darkweb'))
            
            # Create sites array
            sites = []
            for i in range(len(site_urls)):
                if not site_urls[i]:
                    continue
                    
                site = {
                    'url': site_urls[i],
                    'type': site_types[i] if i < len(site_types) else 'generic',
                    'max_pages': int(site_max_pages[i]) if i < len(site_max_pages) and site_max_pages[i] else 5
                }
                
                # Add optional parameters if provided
                if i < len(site_content_selectors) and site_content_selectors[i]:
                    site['content_selector'] = site_content_selectors[i]
                    
                if i < len(site_link_selectors) and site_link_selectors[i]:
                    site['link_selector'] = site_link_selectors[i]
                    
                if i < len(site_exclude_patterns) and site_exclude_patterns[i]:
                    site['exclude_patterns'] = [p.strip() for p in site_exclude_patterns[i].splitlines() if p.strip()]
                
                # Add site-specific selectors
                if site['type'] == 'forum':
                    site_thread_selectors = request.form.getlist('site_thread_selectors[]')
                    site_thread_link_attrs = request.form.getlist('site_thread_link_attrs[]')
                    
                    if i < len(site_thread_selectors) and site_thread_selectors[i]:
                        site['thread_selector'] = site_thread_selectors[i]
                        
                    if i < len(site_thread_link_attrs) and site_thread_link_attrs[i]:
                        site['thread_link_attr'] = site_thread_link_attrs[i]
                        
                elif site['type'] == 'market':
                    site_listing_selectors = request.form.getlist('site_listing_selectors[]')
                    site_price_selectors = request.form.getlist('site_price_selectors[]')
                    
                    if i < len(site_listing_selectors) and site_listing_selectors[i]:
                        site['listing_selector'] = site_listing_selectors[i]
                        
                    if i < len(site_price_selectors) and site_price_selectors[i]:
                        site['price_selector'] = site_price_selectors[i]
                
                sites.append(site)
            
            if not sites:
                flash('At least one valid Dark Web site is required', 'error')
                return redirect(url_for('data_sources.add_darkweb'))
                
            config['sites'] = sites
        
        # Add description if provided
        if description:
            config['description'] = description
        
        # Create the source
        try:
            source = DataSource(
                name=name,
                source_type=source_type,
                config=json.dumps(config),
                is_active=is_active,
                created_at=datetime.utcnow()
            )
            
            db.session.add(source)
            db.session.commit()
            
            flash(f'Source "{name}" added successfully', 'success')
            return redirect(url_for('data_sources.view', source_id=source.id))
        
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding source: {str(e)}', 'error')
            return redirect(url_for('data_sources.add'))
    
    # GET request
    return render_template('data_sources/add.html')

@data_sources_bp.route('/edit/<int:source_id>', methods=['GET', 'POST'])
def edit(source_id):
    """Edit an existing data source."""
    source = DataSource.query.get_or_404(source_id)
    
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        is_active = request.form.get('is_active') == 'on'
        description = request.form.get('description', '')
        
        # Validate inputs
        if not name:
            flash('Name is required', 'error')
            return redirect(url_for('data_sources.edit', source_id=source_id))
        
        # Check if new name conflicts with existing source
        if name != source.name:
            existing = DataSource.query.filter_by(name=name).first()
            if existing and existing.id != source.id:
                flash(f'A source with the name "{name}" already exists', 'error')
                return redirect(url_for('data_sources.edit', source_id=source_id))
        
        # Update the source
        try:
            # Update basic properties
            source.name = name
            source.is_active = is_active
            
            # Parse and update config
            config = json.loads(source.config) if source.config else {}
            
            if source.source_type == 'rss':
                feeds = [f.strip() for f in request.form.get('feeds', '').splitlines() if f.strip()]
                if not feeds:
                    flash('At least one feed URL is required for RSS sources', 'error')
                    return redirect(url_for('data_sources.edit', source_id=source_id))
                
                config['feeds'] = feeds
            
            elif source.source_type == 'twitter':
                queries = [q.strip() for q in request.form.get('queries', '').splitlines() if q.strip()]
                if not queries:
                    flash('At least one search query is required for Twitter sources', 'error')
                    return redirect(url_for('data_sources.edit', source_id=source_id))
                
                config['queries'] = queries
            
            elif source.source_type == 'telegram':
                entities = [e.strip() for e in request.form.get('entities', '').splitlines() if e.strip()]
                if not entities:
                    flash('At least one entity is required for Telegram sources', 'error')
                    return redirect(url_for('data_sources.edit', source_id=source_id))
                
                config['entities'] = entities
                
            elif source.source_type == 'youtube':
                monitor_type = request.form.get('monitor_type', config.get('monitor_type', 'channel'))
                config['monitor_type'] = monitor_type
                
                # Process based on monitor type
                if monitor_type == 'channel':
                    channel_ids = [c.strip() for c in request.form.get('channel_ids', '').splitlines() if c.strip()]
                    if not channel_ids:
                        flash('At least one channel ID is required for YouTube channel monitoring', 'error')
                        return redirect(url_for('data_sources.edit', source_id=source_id))
                    config['channel_ids'] = channel_ids
                    
                elif monitor_type == 'search':
                    search_queries = [q.strip() for q in request.form.get('search_queries', '').splitlines() if q.strip()]
                    if not search_queries:
                        flash('At least one search query is required for YouTube search monitoring', 'error')
                        return redirect(url_for('data_sources.edit', source_id=source_id))
                    config['search_queries'] = search_queries
                    
                elif monitor_type == 'video':
                    video_ids = [v.strip() for v in request.form.get('video_ids', '').splitlines() if v.strip()]
                    if not video_ids:
                        flash('At least one video ID is required for YouTube video monitoring', 'error')
                        return redirect(url_for('data_sources.edit', source_id=source_id))
                    config['video_ids'] = video_ids
                    
                elif monitor_type == 'playlist':
                    playlist_ids = [p.strip() for p in request.form.get('playlist_ids', '').splitlines() if p.strip()]
                    if not playlist_ids:
                        flash('At least one playlist ID is required for YouTube playlist monitoring', 'error')
                        return redirect(url_for('data_sources.edit', source_id=source_id))
                    config['playlist_ids'] = playlist_ids
                
                # Common YouTube settings
                config['max_videos'] = int(request.form.get('max_videos', config.get('max_videos', 10)))
                config['days_back'] = int(request.form.get('days_back', config.get('days_back', 7)))
                config['include_comments'] = request.form.get('include_comments') == 'on'
                
                if config['include_comments']:
                    config['max_comments'] = int(request.form.get('max_comments', config.get('max_comments', 50)))
            
            elif source.source_type == 'darkweb':
                # Process Dark Web site configurations
                site_urls = request.form.getlist('site_urls[]')
                site_types = request.form.getlist('site_types[]')
                site_max_pages = request.form.getlist('site_max_pages[]')
                site_content_selectors = request.form.getlist('site_content_selectors[]')
                site_link_selectors = request.form.getlist('site_link_selectors[]')
                site_exclude_patterns = request.form.getlist('site_exclude_patterns[]')
                
                # Validate
                if not site_urls:
                    flash('At least one Dark Web site is required', 'error')
                    return redirect(url_for('data_sources.edit', source_id=source_id))
                
                # Create sites array
                sites = []
                for i in range(len(site_urls)):
                    if not site_urls[i]:
                        continue
                        
                    site = {
                        'url': site_urls[i],
                        'type': site_types[i] if i < len(site_types) else 'generic',
                        'max_pages': int(site_max_pages[i]) if i < len(site_max_pages) and site_max_pages[i] else 5
                    }
                    
                    # Add optional parameters if provided
                    if i < len(site_content_selectors) and site_content_selectors[i]:
                        site['content_selector'] = site_content_selectors[i]
                        
                    if i < len(site_link_selectors) and site_link_selectors[i]:
                        site['link_selector'] = site_link_selectors[i]
                        
                    if i < len(site_exclude_patterns) and site_exclude_patterns[i]:
                        site['exclude_patterns'] = [p.strip() for p in site_exclude_patterns[i].splitlines() if p.strip()]
                    
                    # Add site-specific selectors
                    if site['type'] == 'forum':
                        site_thread_selectors = request.form.getlist('site_thread_selectors[]')
                        site_thread_link_attrs = request.form.getlist('site_thread_link_attrs[]')
                        
                        if i < len(site_thread_selectors) and site_thread_selectors[i]:
                            site['thread_selector'] = site_thread_selectors[i]
                            
                        if i < len(site_thread_link_attrs) and site_thread_link_attrs[i]:
                            site['thread_link_attr'] = site_thread_link_attrs[i]
                            
                    elif site['type'] == 'market':
                        site_listing_selectors = request.form.getlist('site_listing_selectors[]')
                        site_price_selectors = request.form.getlist('site_price_selectors[]')
                        
                        if i < len(site_listing_selectors) and site_listing_selectors[i]:
                            site['listing_selector'] = site_listing_selectors[i]
                            
                        if i < len(site_price_selectors) and site_price_selectors[i]:
                            site['price_selector'] = site_price_selectors[i]
                    
                    sites.append(site)
                
                if not sites:
                    flash('At least one valid Dark Web site is required', 'error')
                    return redirect(url_for('data_sources.edit', source_id=source_id))
                    
                config['sites'] = sites
            
            # Update description
            if description:
                config['description'] = description
            elif 'description' in config:
                del config['description']
            
            # Save updated config
            source.config = json.dumps(config)
            
            db.session.commit()
            
            flash(f'Source "{name}" updated successfully', 'success')
            return redirect(url_for('data_sources.view', source_id=source.id))
        
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating source: {str(e)}', 'error')
            return redirect(url_for('data_sources.edit', source_id=source_id))
    
    # GET request
    config = json.loads(source.config) if source.config else {}
    
    # Prepare template data based on source type
    template_data = {
        'source': source,
        'description': config.get('description', '')
    }
    
    if source.source_type == 'rss':
        template_data['feeds'] = '\n'.join(config.get('feeds', []))
        return render_template('data_sources/edit_rss.html', **template_data)
    
    elif source.source_type == 'twitter':
        template_data['queries'] = '\n'.join(config.get('queries', []))
        return render_template('data_sources/edit_twitter.html', **template_data)
    
    elif source.source_type == 'telegram':
        template_data['entities'] = '\n'.join(config.get('entities', []))
        return render_template('data_sources/edit_telegram.html', **template_data)
    
    elif source.source_type == 'youtube':
        return render_template('data_sources/edit_youtube.html', **template_data)
    
    elif source.source_type == 'darkweb':
        return render_template('data_sources/edit_darkweb.html', **template_data)
    
    # Default edit view
    return render_template('data_sources/edit.html', **template_data)

@data_sources_bp.route('/delete/<int:source_id>', methods=['POST'])
def delete(source_id):
    """Delete a data source."""
    source = DataSource.query.get_or_404(source_id)
    
    try:
        db.session.delete(source)
        db.session.commit()
        
        flash(f'Source "{source.name}" deleted successfully', 'success')
        return redirect(url_for('data_sources.index'))
    
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting source: {str(e)}', 'error')
        return redirect(url_for('data_sources.view', source_id=source_id))

@data_sources_bp.route('/toggle_active/<int:source_id>', methods=['POST'])
def toggle_active(source_id):
    """Toggle the active status of a data source."""
    source = DataSource.query.get_or_404(source_id)
    
    try:
        source.is_active = not source.is_active
        db.session.commit()
        
        status = 'activated' if source.is_active else 'deactivated'
        flash(f'Source "{source.name}" {status} successfully', 'success')
        
        # Determine redirect target
        if request.args.get('redirect') == 'index':
            return redirect(url_for('data_sources.index'))
        else:
            return redirect(url_for('data_sources.view', source_id=source_id))
    
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating source: {str(e)}', 'error')
        return redirect(url_for('data_sources.view', source_id=source_id))

@data_sources_bp.route('/test/<int:source_id>')
def test(source_id):
    """Test a data source."""
    source = DataSource.query.get_or_404(source_id)
    
    # Parse config
    config = json.loads(source.config) if source.config else {}
    
    # Test based on source type
    results = []
    
    try:
        if source.source_type == 'rss':
            handler = RSSSource()
            feeds = config.get('feeds', [])
            
            for feed_url in feeds:
                try:
                    entries = handler.fetch_feed(feed_url, max_entries=5)
                    results.append({
                        'url': feed_url,
                        'success': True,
                        'count': len(entries),
                        'entries': entries[:3] if entries else []
                    })
                except Exception as e:
                    results.append({
                        'url': feed_url,
                        'success': False,
                        'error': str(e)
                    })
        
        elif source.source_type == 'twitter':
            handler = TwitterSource()
            
            # Check if Twitter API is available
            if not hasattr(handler, '_client') and not hasattr(handler, '_api'):
                flash('Twitter API not configured. Please set Twitter API credentials.', 'error')
                return redirect(url_for('data_sources.view', source_id=source_id))
            
            queries = config.get('queries', [])
            
            for query in queries:
                try:
                    tweets = handler.search_tweets(query, max_tweets=5)
                    results.append({
                        'query': query,
                        'success': True,
                        'count': len(tweets),
                        'tweets': tweets[:3] if tweets else []
                    })
                except Exception as e:
                    results.append({
                        'query': query,
                        'success': False,
                        'error': str(e)
                    })
        
        elif source.source_type == 'telegram':
            handler = TelegramSource()
            
            # Check if Telegram client is available
            if not handler._client:
                flash('Telegram API not configured. Please set Telegram API credentials.', 'error')
                return redirect(url_for('data_sources.view', source_id=source_id))
            
            flash('Telegram sources cannot be tested through the web UI.', 'warning')
            return redirect(url_for('data_sources.view', source_id=source_id))
            
        elif source.source_type == 'youtube':
            handler = YouTubeSource()
            
            # Check if YouTube API is available
            if not handler._service:
                flash('YouTube API not configured. Please set Google API credentials.', 'error')
                return redirect(url_for('data_sources.view', source_id=source_id))
            
            monitor_type = config.get('monitor_type', 'channel')
            
            if monitor_type == 'channel':
                channel_ids = config.get('channel_ids', [])
                
                for channel_id in channel_ids[:3]:  # Test first 3 channels only
                    try:
                        channel_info = handler.get_channel_info(channel_id)
                        videos = handler.get_channel_videos(
                            channel_id, 
                            max_results=3, 
                            include_comments=config.get('include_comments', False),
                            max_comments=config.get('max_comments', 10)
                        )
                        
                        results.append({
                            'channel_id': channel_id,
                            'success': True,
                            'title': channel_info.get('title', 'Unknown'),
                            'video_count': len(videos),
                            'videos': [{'title': v.get('title', 'Unknown'), 'id': v.get('id')} for v in videos[:3]]
                        })
                    except Exception as e:
                        results.append({
                            'channel_id': channel_id,
                            'success': False,
                            'error': str(e)
                        })
            
            elif monitor_type == 'search':
                search_queries = config.get('search_queries', [])
                
                for query in search_queries[:3]:  # Test first 3 queries only
                    try:
                        videos = handler.search_videos(
                            query, 
                            max_results=3, 
                            include_comments=config.get('include_comments', False),
                            max_comments=config.get('max_comments', 10)
                        )
                        
                        results.append({
                            'query': query,
                            'success': True,
                            'video_count': len(videos),
                            'videos': [{'title': v.get('title', 'Unknown'), 'id': v.get('id')} for v in videos[:3]]
                        })
                    except Exception as e:
                        results.append({
                            'query': query,
                            'success': False,
                            'error': str(e)
                        })
            
            elif monitor_type == 'video':
                video_ids = config.get('video_ids', [])
                
                for video_id in video_ids[:3]:  # Test first 3 videos only
                    try:
                        video = handler.get_video_details(
                            video_id, 
                            include_comments=config.get('include_comments', False),
                            max_comments=config.get('max_comments', 10)
                        )
                        
                        results.append({
                            'video_id': video_id,
                            'success': True,
                            'title': video.get('title', 'Unknown'),
                            'channel': video.get('channel_title', 'Unknown'),
                            'comment_count': len(video.get('comments', []))
                        })
                    except Exception as e:
                        results.append({
                            'video_id': video_id,
                            'success': False,
                            'error': str(e)
                        })
            
            elif monitor_type == 'playlist':
                playlist_ids = config.get('playlist_ids', [])
                
                for playlist_id in playlist_ids[:3]:  # Test first 3 playlists only
                    try:
                        videos = handler.get_playlist_videos(
                            playlist_id, 
                            max_results=3, 
                            include_comments=config.get('include_comments', False),
                            max_comments=config.get('max_comments', 10)
                        )
                        
                        results.append({
                            'playlist_id': playlist_id,
                            'success': True,
                            'video_count': len(videos),
                            'videos': [{'title': v.get('title', 'Unknown'), 'id': v.get('id')} for v in videos[:3]]
                        })
                    except Exception as e:
                        results.append({
                            'playlist_id': playlist_id,
                            'success': False,
                            'error': str(e)
                        })
        
        elif source.source_type == 'darkweb':
            handler = DarkWebSource()
            
            # Dark Web sources require Tor, check if it's available
            if not handler._tor_ready:
                flash('Tor is not running or cannot connect to the control port. Dark Web sources require Tor.', 'error')
                return redirect(url_for('data_sources.view', source_id=source_id))
            
            # Test a few sites (limited to avoid causing issues with Tor)
            sites = config.get('sites', [])
            max_test_sites = min(3, len(sites))
            
            for site in sites[:max_test_sites]:
                site_url = site.get('url', '')
                site_type = site.get('type', 'generic')
                
                if not site_url:
                    continue
                
                try:
                    # Just check if we can reach the site, don't scrape content
                    response = handler._fetch_url(site_url, max_retries=1, timeout=15)
                    
                    if response and response.status_code == 200:
                        results.append({
                            'url': site_url,
                            'type': site_type,
                            'success': True,
                            'content_length': len(response.text) if response.text else 0
                        })
                    else:
                        results.append({
                            'url': site_url,
                            'type': site_type,
                            'success': False,
                            'error': f'Failed with status code: {response.status_code if response else "No response"}'
                        })
                except Exception as e:
                    results.append({
                        'url': site_url,
                        'type': site_type,
                        'success': False,
                        'error': str(e)
                    })
            
            # Always warn about limitations
            flash('Dark Web source tests are limited to connectivity checks for security reasons.', 'warning')
    
    except Exception as e:
        flash(f'Error testing source: {str(e)}', 'error')
        return redirect(url_for('data_sources.view', source_id=source_id))
    
    # Render test results
    return render_template(
        'data_sources/test_results.html',
        source=source,
        results=results
    )

@data_sources_bp.route('/api/sources')
def api_list_sources():
    """API endpoint to list data sources in JSON format."""
    # Get filter parameters
    source_type = request.args.get('type')
    status = request.args.get('status')
    
    # Build query
    query = DataSource.query
    
    if source_type:
        query = query.filter_by(source_type=source_type)
    
    if status == 'active':
        query = query.filter_by(is_active=True)
    elif status == 'inactive':
        query = query.filter_by(is_active=False)
    
    # Get sources
    sources = query.all()
    
    # Convert to JSON
    result = []
    for source in sources:
        # Parse config
        config = json.loads(source.config) if source.config else {}
        
        # Base source info
        source_info = {
            'id': source.id,
            'name': source.name,
            'type': source.source_type,
            'is_active': source.is_active,
            'created_at': source.created_at.isoformat() if source.created_at else None,
            'last_ingestion': source.last_ingestion.isoformat() if source.last_ingestion else None,
            'description': config.get('description', '')
        }
        
        # Add source-specific info
        if source.source_type == 'rss':
            source_info['feeds'] = config.get('feeds', [])
        elif source.source_type == 'twitter':
            source_info['queries'] = config.get('queries', [])
        elif source.source_type == 'telegram':
            source_info['entities'] = config.get('entities', [])
        elif source.source_type == 'youtube':
            source_info['monitor_type'] = config.get('monitor_type', 'channel')
            
            if source_info['monitor_type'] == 'channel':
                source_info['channel_ids'] = config.get('channel_ids', [])
            elif source_info['monitor_type'] == 'search':
                source_info['search_queries'] = config.get('search_queries', [])
            elif source_info['monitor_type'] == 'video':
                source_info['video_ids'] = config.get('video_ids', [])
            elif source_info['monitor_type'] == 'playlist':
                source_info['playlist_ids'] = config.get('playlist_ids', [])
                
            source_info['max_videos'] = config.get('max_videos', 10)
            source_info['days_back'] = config.get('days_back', 7)
            source_info['include_comments'] = config.get('include_comments', False)
            
            if source_info['include_comments']:
                source_info['max_comments'] = config.get('max_comments', 50)
        elif source.source_type == 'darkweb':
            source_info['sites'] = config.get('sites', [])
        
        result.append(source_info)
    
    return jsonify(result)