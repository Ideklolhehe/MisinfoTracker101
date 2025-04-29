"""
Web scraping routes for the CIVILIAN system.
These routes handle web scraping, data collection, and monitoring.
"""

import json
import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import login_required, current_user
from urllib.parse import urlparse
from datetime import datetime

from app import db
import models
from services.web_scraping_service import web_scraping_service
from data_sources.web_source_manager import web_source_manager

# Configure logger
logger = logging.getLogger(__name__)

# Create blueprint
web_scraping_bp = Blueprint('web_scraping', __name__)

# Dashboard route
@web_scraping_bp.route('/web-scraping')
@login_required
def dashboard():
    """Web scraping dashboard."""
    # Get statistics
    domain_stats = web_scraping_service.get_domain_stats()
    search_term_stats = web_scraping_service.get_search_term_stats()
    
    # Get recent web sources
    sources = db.session.query(models.WebSource).order_by(
        models.WebSource.updated_at.desc()
    ).limit(10).all()
    
    recent_sources = [
        {
            'id': source.id,
            'name': source.name,
            'url': source.url,
            'source_type': source.source_type,
            'is_active': source.is_active,
            'last_ingestion': source.last_ingestion.isoformat() if source.last_ingestion else None
        }
        for source in sources
    ]
    
    # Get recent content items
    items = db.session.query(models.ContentItem).order_by(
        models.ContentItem.created_at.desc()
    ).limit(10).all()
    
    recent_items = [
        {
            'id': item.id,
            'title': item.title,
            'source': item.source,
            'url': item.url,
            'content_type': item.content_type,
            'created_at': item.created_at.isoformat() if item.created_at else None
        }
        for item in items
    ]
    
    return render_template('web_scraping/dashboard.html',
                          domain_stats=domain_stats,
                          search_term_stats=search_term_stats,
                          recent_sources=recent_sources,
                          recent_items=recent_items)

# URL Scanning route
@web_scraping_bp.route('/web-scraping/scan', methods=['GET', 'POST'])
@login_required
def scan():
    """Scan URLs for content."""
    if request.method == 'POST':
        url = request.form.get('url')
        depth = int(request.form.get('depth', 1))
        
        if not url:
            flash('URL is required', 'error')
            return redirect(url_for('web_scraping.scan'))
        
        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                flash('Invalid URL format', 'error')
                return redirect(url_for('web_scraping.scan'))
        except Exception:
            flash('Invalid URL format', 'error')
            return redirect(url_for('web_scraping.scan'))
        
        # Scan URL
        job_id = web_scraping_service.scan_url(url, depth)
        
        # Store job ID in session for status checking
        session['scan_job_id'] = job_id
        
        # Redirect to results page
        return redirect(url_for('web_scraping.scan_results'))
    
    return render_template('web_scraping/scan.html')

# Scan results route
@web_scraping_bp.route('/web-scraping/scan/results')
@login_required
def scan_results():
    """View scan results."""
    job_id = session.get('scan_job_id')
    
    if not job_id:
        flash('No scan job found', 'error')
        return redirect(url_for('web_scraping.scan'))
    
    # Get job status
    job_status = web_source_manager.get_job_status(job_id)
    
    # Format results for display
    results = []
    if job_status.get('status') == 'completed' and 'results' in job_status:
        for result in job_status.get('results', []):
            item = {
                'title': result.get('title', 'No Title'),
                'url': result.get('url', ''),
                'content': result.get('content', '')[:500] + '...' if result.get('content') and len(result.get('content')) > 500 else result.get('content', ''),
                'source': result.get('source', 'Unknown'),
                'keywords': ', '.join(result.get('keywords', [])),
                'published_date': result.get('published_date')
            }
            results.append(item)
    
    # Check if we need to wait for more results
    done = job_status.get('status') in ['completed', 'error', 'not_found']
    
    return render_template('web_scraping/scan_results.html',
                          job_id=job_id,
                          job_status=job_status,
                          results=results,
                          done=done)

# Submit to detection route
@web_scraping_bp.route('/web-scraping/scan/submit', methods=['POST'])
@login_required
def submit_to_detection():
    """Submit content to the detection pipeline."""
    job_id = request.form.get('job_id')
    result_idx = request.form.get('result_idx')
    
    if not job_id or result_idx is None:
        flash('Missing job ID or result index', 'error')
        return jsonify({'success': False, 'error': 'Missing job ID or result index'})
    
    try:
        result_idx = int(result_idx)
        
        # Get job status
        job_status = web_source_manager.get_job_status(job_id)
        
        # Check if job exists and is completed
        if job_status.get('status') != 'completed' or 'results' not in job_status:
            flash('Job not completed or no results found', 'error')
            return jsonify({'success': False, 'error': 'Job not completed or no results found'})
        
        # Check if result index is valid
        if result_idx < 0 or result_idx >= len(job_status.get('results', [])):
            flash('Invalid result index', 'error')
            return jsonify({'success': False, 'error': 'Invalid result index'})
        
        # Get the content
        content_data = job_status.get('results', [])[result_idx]
        
        # Submit to detection pipeline
        success = web_scraping_service.submit_to_detection_pipeline(content_data)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to submit content to detection pipeline'})
        
    except Exception as e:
        logger.error(f"Error submitting content to detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Create source route
@web_scraping_bp.route('/web-scraping/scan/create-source', methods=['POST'])
@login_required
def create_source():
    """Create a source from scan results."""
    job_id = request.form.get('job_id')
    name = request.form.get('name')
    
    if not job_id:
        flash('Missing job ID', 'error')
        return jsonify({'success': False, 'error': 'Missing job ID'})
    
    # Create source
    source_id = web_source_manager.create_source_from_job(job_id, name)
    
    if source_id:
        return jsonify({'success': True, 'source_id': source_id})
    else:
        return jsonify({'success': False, 'error': 'Failed to create source'})

# Search route
@web_scraping_bp.route('/web-scraping/search', methods=['GET', 'POST'])
@login_required
def search():
    """Search for content."""
    if request.method == 'POST':
        search_term = request.form.get('search_term')
        limit = int(request.form.get('limit', 10))
        add_to_monitoring = request.form.get('add_to_monitoring') == 'on'
        
        if not search_term:
            flash('Search term is required', 'error')
            return redirect(url_for('web_scraping.search'))
        
        # Perform search
        job_id = web_scraping_service.search_and_monitor(search_term, limit) if add_to_monitoring else web_source_manager.add_url_job(
            "https://www.bing.com/search",
            job_type='search',
            config={
                'search_term': search_term,
                'search_engine': 'bing',
                'limit': limit
            }
        )
        
        # Store job ID in session for status checking
        session['search_job_id'] = job_id
        
        # Redirect to results page
        return redirect(url_for('web_scraping.search_results'))
    
    return render_template('web_scraping/search.html')

# Search results route
@web_scraping_bp.route('/web-scraping/search/results')
@login_required
def search_results():
    """View search results."""
    job_id = session.get('search_job_id')
    
    if not job_id:
        flash('No search job found', 'error')
        return redirect(url_for('web_scraping.search'))
    
    # Get job status
    job_status = web_source_manager.get_job_status(job_id)
    
    # Format results for display
    results = []
    if job_status.get('status') == 'completed' and 'results' in job_status:
        for result in job_status.get('results', []):
            item = {
                'title': result.get('title', 'No Title'),
                'url': result.get('url', ''),
                'content': result.get('content', '')[:500] + '...' if result.get('content') and len(result.get('content')) > 500 else result.get('content', ''),
                'source': result.get('source', 'Unknown'),
                'keywords': ', '.join(result.get('keywords', [])),
                'published_date': result.get('published_date')
            }
            results.append(item)
    
    # Check if we need to wait for more results
    done = job_status.get('status') in ['completed', 'error', 'not_found']
    
    return render_template('web_scraping/search_results.html',
                          job_id=job_id,
                          job_status=job_status,
                          results=results,
                          done=done)

# Monitoring route
@web_scraping_bp.route('/web-scraping/monitoring', methods=['GET', 'POST'])
@login_required
def monitoring():
    """Monitor domains and search terms."""
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_domain':
            domain = request.form.get('domain')
            category = request.form.get('category', 'general')
            priority = int(request.form.get('priority', 2))
            
            if not domain:
                flash('Domain is required', 'error')
                return redirect(url_for('web_scraping.monitoring'))
            
            success = web_scraping_service.add_focused_domain(domain, category, priority)
            
            if success:
                flash(f'Domain {domain} added to monitoring', 'success')
            else:
                flash(f'Failed to add domain {domain}', 'error')
                
        elif action == 'add_term':
            term = request.form.get('term')
            category = request.form.get('category', 'general')
            
            if not term:
                flash('Search term is required', 'error')
                return redirect(url_for('web_scraping.monitoring'))
            
            success = web_scraping_service.add_search_term(term, category)
            
            if success:
                flash(f'Search term "{term}" added to monitoring', 'success')
            else:
                flash(f'Failed to add search term "{term}"', 'error')
        
        return redirect(url_for('web_scraping.monitoring'))
    
    # Get domains
    domains = db.session.query(models.FocusedDomain).order_by(
        models.FocusedDomain.priority,
        models.FocusedDomain.domain
    ).all()
    
    # Get search terms
    terms = db.session.query(models.SearchTerm).order_by(
        models.SearchTerm.category,
        models.SearchTerm.term
    ).all()
    
    return render_template('web_scraping/monitoring.html',
                          domains=domains,
                          terms=terms)

# Sources route
@web_scraping_bp.route('/web-scraping/sources', methods=['GET', 'POST'])
@login_required
def sources():
    """Manage web sources."""
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_source':
            name = request.form.get('name')
            url = request.form.get('url')
            source_type = request.form.get('source_type')
            
            if not name or not url or not source_type:
                flash('Name, URL, and source type are required', 'error')
                return redirect(url_for('web_scraping.sources'))
            
            # Create config based on source type
            config = {'url': url}
            
            if source_type == 'web_crawl':
                config['max_pages'] = int(request.form.get('max_pages', 5))
                config['same_domain_only'] = request.form.get('same_domain_only') == 'on'
                
            elif source_type == 'web_search':
                config['search_term'] = request.form.get('search_term', '')
                config['search_engine'] = request.form.get('search_engine', 'bing')
                config['limit'] = int(request.form.get('limit', 10))
            
            # Register source
            source_id = web_source_manager.register_source({
                'name': name,
                'url': url,
                'source_type': source_type,
                'is_active': True,
                'config': config
            })
            
            if source_id:
                flash(f'Source "{name}" added successfully', 'success')
            else:
                flash(f'Failed to add source "{name}"', 'error')
                
        elif action == 'toggle_status':
            source_id = request.form.get('source_id')
            is_active = request.form.get('is_active') == 'true'
            
            if not source_id:
                flash('Source ID is required', 'error')
                return redirect(url_for('web_scraping.sources'))
            
            success = web_source_manager.update_source_status(int(source_id), is_active)
            
            if success:
                status = 'activated' if is_active else 'deactivated'
                flash(f'Source {source_id} {status} successfully', 'success')
            else:
                flash(f'Failed to update source {source_id}', 'error')
                
        elif action == 'run_source':
            source_id = request.form.get('source_id')
            
            if not source_id:
                flash('Source ID is required', 'error')
                return redirect(url_for('web_scraping.sources'))
            
            success = web_source_manager.run_source(int(source_id))
            
            if success:
                flash(f'Source {source_id} executed successfully', 'success')
            else:
                flash(f'Failed to execute source {source_id}', 'error')
        
        return redirect(url_for('web_scraping.sources'))
    
    # Get all sources
    sources = db.session.query(models.WebSource).order_by(
        models.WebSource.source_type,
        models.WebSource.name
    ).all()
    
    return render_template('web_scraping/sources.html', sources=sources)

# Schedule start route
@web_scraping_bp.route('/web-scraping/schedule/start', methods=['POST'])
@login_required
def start_schedule():
    """Start scheduled web scraping."""
    web_scraping_service.start_scheduled_scraping()
    flash('Scheduled web scraping started', 'success')
    return redirect(url_for('web_scraping.dashboard'))

# Schedule stop route
@web_scraping_bp.route('/web-scraping/schedule/stop', methods=['POST'])
@login_required
def stop_schedule():
    """Stop scheduled web scraping."""
    web_scraping_service.stop_scheduled_scraping()
    flash('Scheduled web scraping stopped', 'success')
    return redirect(url_for('web_scraping.dashboard'))