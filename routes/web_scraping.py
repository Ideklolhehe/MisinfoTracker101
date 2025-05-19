import json
import datetime
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from sqlalchemy import desc

from app import db
from models import WebSource, WebSourceJob, WebSourceJobStatus, FocusedDomain, SearchTerm
from utils.web_scraper import WebScraper
from services.web_scraping_service import WebScrapingService
from data_sources.web_source_manager import WebSourceManager

# Blueprint configuration
web_scraping_bp = Blueprint('web_scraping', __name__, url_prefix='/web-scraping')

# Initialize services
web_scraper = WebScraper()
web_scraping_service = WebScrapingService()
web_source_manager = WebSourceManager()

@web_scraping_bp.route('/')
@web_scraping_bp.route('/dashboard')
@login_required
def dashboard():
    """Web scraping dashboard with overview of sources and recent activity."""
    # Get stats for dashboard
    total_sources = WebSource.query.count()
    active_sources = WebSource.query.filter_by(is_active=True).count()
    total_domains = FocusedDomain.query.count()
    total_terms = SearchTerm.query.count()
    
    # Get recent jobs
    recent_jobs = WebSourceJob.query.order_by(desc(WebSourceJob.created_at)).limit(10).all()
    
    # Get recent sources
    recent_sources = WebSource.query.order_by(desc(WebSource.created_at)).limit(5).all()
    
    # Get source types statistics
    source_types = db.session.query(
        WebSource.source_type, 
        db.func.count(WebSource.id)
    ).group_by(WebSource.source_type).all()
    
    source_type_stats = {source_type: count for source_type, count in source_types}
    
    # Get job status statistics
    job_stats = db.session.query(
        WebSourceJob.status, 
        db.func.count(WebSourceJob.id)
    ).group_by(WebSourceJob.status).all()
    
    job_status_stats = {status: count for status, count in job_stats}
    
    return render_template(
        'web_scraping/dashboard.html',
        total_sources=total_sources,
        active_sources=active_sources,
        total_domains=total_domains,
        total_terms=total_terms,
        recent_jobs=recent_jobs,
        recent_sources=recent_sources,
        source_type_stats=source_type_stats,
        job_status_stats=job_status_stats
    )

@web_scraping_bp.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    """Scan a single URL for content."""
    if request.method == 'POST':
        url = request.form.get('url')
        extract_links = 'extract_links' in request.form
        extract_content = 'extract_content' in request.form
        analyze_credibility = 'analyze_credibility' in request.form
        
        if not url:
            flash('URL is required', 'danger')
            return redirect(url_for('web_scraping.scan'))
        
        try:
            # Create a web source job
            job = WebSourceJob()
            job.job_type = 'scan'
            job.status = WebSourceJobStatus.PENDING.value
            job.created_by = current_user.id if current_user.is_authenticated else None
            job.created_at = datetime.datetime.utcnow()
            
            # Set job metadata
            job_meta = {
                'url': url,
                'extract_links': extract_links,
                'extract_content': extract_content,
                'analyze_credibility': analyze_credibility
            }
            job.set_meta_data(job_meta)
            
            db.session.add(job)
            db.session.commit()
            
            # Process the job immediately
            result = web_scraping_service.process_scan_job(job.id)
            
            # Redirect to results page
            return redirect(url_for('web_scraping.scan_results', job_id=job.id))
            
        except Exception as e:
            flash(f'Error processing scan: {str(e)}', 'danger')
            return redirect(url_for('web_scraping.scan'))
    
    return render_template('web_scraping/scan.html')

@web_scraping_bp.route('/scan/results/<int:job_id>')
@login_required
def scan_results(job_id):
    """Show results of a scan job."""
    job = WebSourceJob.query.get_or_404(job_id)
    
    # Check if job is completed
    if job.status != WebSourceJobStatus.COMPLETED.value:
        flash('The scan is still in progress. Please wait.', 'info')
        return redirect(url_for('web_scraping.scan'))
    
    # Get job metadata and results
    meta_data = job.get_meta_data()
    results = job.get_results()
    
    return render_template(
        'web_scraping/scan_results.html',
        job=job,
        meta_data=meta_data,
        results=results
    )

@web_scraping_bp.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    """Search for web content using keywords or phrases."""
    if request.method == 'POST':
        query = request.form.get('query')
        search_engine = request.form.get('search_engine', 'bing')
        limit = int(request.form.get('limit', 10))
        
        if not query:
            flash('Search query is required', 'danger')
            return redirect(url_for('web_scraping.search'))
        
        try:
            # Create a web source job
            job = WebSourceJob()
            job.job_type = 'search'
            job.status = WebSourceJobStatus.PENDING.value
            job.created_by = current_user.id if current_user.is_authenticated else None
            job.created_at = datetime.datetime.utcnow()
            
            # Set job metadata
            job_meta = {
                'query': query,
                'search_engine': search_engine,
                'limit': limit
            }
            job.set_meta_data(job_meta)
            
            db.session.add(job)
            db.session.commit()
            
            # Process the job immediately
            result = web_scraping_service.process_search_job(job.id)
            
            # Redirect to results page
            return redirect(url_for('web_scraping.search_results', job_id=job.id))
            
        except Exception as e:
            flash(f'Error processing search: {str(e)}', 'danger')
            return redirect(url_for('web_scraping.search'))
    
    return render_template('web_scraping/search.html')

@web_scraping_bp.route('/search/results/<int:job_id>')
@login_required
def search_results(job_id):
    """Show results of a search job."""
    job = WebSourceJob.query.get_or_404(job_id)
    
    # Check if job is completed
    if job.status != WebSourceJobStatus.COMPLETED.value:
        flash('The search is still in progress. Please wait.', 'info')
        return redirect(url_for('web_scraping.search'))
    
    # Get job metadata and results
    meta_data = job.get_meta_data()
    results = job.get_results()
    
    return render_template(
        'web_scraping/search_results.html',
        job=job,
        meta_data=meta_data,
        results=results
    )

@web_scraping_bp.route('/monitoring', methods=['GET', 'POST'])
@login_required
def monitoring():
    """Monitor focused domains and search terms."""
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_domain':
            domain = request.form.get('domain')
            category = request.form.get('category')
            priority = int(request.form.get('priority', 2))
            
            if not domain:
                flash('Domain is required', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
            
            # Check if domain already exists
            existing = FocusedDomain.query.filter_by(domain=domain).first()
            if existing:
                flash(f'Domain {domain} is already being monitored', 'warning')
                return redirect(url_for('web_scraping.monitoring'))
            
            try:
                # Create a new focused domain
                focused_domain = FocusedDomain()
                focused_domain.domain = domain
                focused_domain.category = category
                focused_domain.priority = priority
                focused_domain.created_by = current_user.id if current_user.is_authenticated else None
                focused_domain.is_active = True
                
                db.session.add(focused_domain)
                db.session.commit()
                
                flash(f'Domain {domain} added to monitoring', 'success')
                return redirect(url_for('web_scraping.monitoring'))
                
            except Exception as e:
                flash(f'Error adding domain: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
                
        elif action == 'add_term':
            term = request.form.get('term')
            category = request.form.get('category')
            
            if not term:
                flash('Search term is required', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
            
            # Check if term already exists
            existing = SearchTerm.query.filter_by(term=term).first()
            if existing:
                flash(f'Term "{term}" is already being monitored', 'warning')
                return redirect(url_for('web_scraping.monitoring'))
            
            try:
                # Create a new search term
                search_term = SearchTerm()
                search_term.term = term
                search_term.category = category
                search_term.created_by = current_user.id if current_user.is_authenticated else None
                search_term.is_active = True
                
                db.session.add(search_term)
                db.session.commit()
                
                flash(f'Term "{term}" added to monitoring', 'success')
                return redirect(url_for('web_scraping.monitoring'))
                
            except Exception as e:
                flash(f'Error adding search term: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
                
        elif action == 'toggle_domain':
            domain_id = request.form.get('domain_id')
            is_active = request.form.get('is_active', 'false') == 'true'
            
            if not domain_id:
                flash('Domain ID is required', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
            
            try:
                domain = FocusedDomain.query.get(domain_id)
                if not domain:
                    flash('Domain not found', 'danger')
                    return redirect(url_for('web_scraping.monitoring'))
                
                domain.is_active = is_active
                db.session.commit()
                
                status = 'activated' if is_active else 'deactivated'
                flash(f'Domain {domain.domain} {status}', 'success')
                return redirect(url_for('web_scraping.monitoring'))
                
            except Exception as e:
                flash(f'Error updating domain: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
                
        elif action == 'toggle_term':
            term_id = request.form.get('term_id')
            is_active = request.form.get('is_active', 'false') == 'true'
            
            if not term_id:
                flash('Term ID is required', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
            
            try:
                term = SearchTerm.query.get(term_id)
                if not term:
                    flash('Search term not found', 'danger')
                    return redirect(url_for('web_scraping.monitoring'))
                
                term.is_active = is_active
                db.session.commit()
                
                status = 'activated' if is_active else 'deactivated'
                flash(f'Term "{term.term}" {status}', 'success')
                return redirect(url_for('web_scraping.monitoring'))
                
            except Exception as e:
                flash(f'Error updating search term: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
                
        elif action == 'delete_domain':
            domain_id = request.form.get('domain_id')
            
            if not domain_id:
                flash('Domain ID is required', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
            
            try:
                domain = FocusedDomain.query.get(domain_id)
                if not domain:
                    flash('Domain not found', 'danger')
                    return redirect(url_for('web_scraping.monitoring'))
                
                domain_name = domain.domain
                db.session.delete(domain)
                db.session.commit()
                
                flash(f'Domain {domain_name} deleted from monitoring', 'success')
                return redirect(url_for('web_scraping.monitoring'))
                
            except Exception as e:
                flash(f'Error deleting domain: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
                
        elif action == 'delete_term':
            term_id = request.form.get('term_id')
            
            if not term_id:
                flash('Term ID is required', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
            
            try:
                term = SearchTerm.query.get(term_id)
                if not term:
                    flash('Search term not found', 'danger')
                    return redirect(url_for('web_scraping.monitoring'))
                
                term_name = term.term
                db.session.delete(term)
                db.session.commit()
                
                flash(f'Term "{term_name}" deleted from monitoring', 'success')
                return redirect(url_for('web_scraping.monitoring'))
                
            except Exception as e:
                flash(f'Error deleting search term: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.monitoring'))
    
    # Get all focused domains and search terms
    domains = FocusedDomain.query.order_by(FocusedDomain.domain).all()
    terms = SearchTerm.query.order_by(SearchTerm.term).all()
    
    return render_template(
        'web_scraping/monitoring.html',
        domains=domains,
        terms=terms
    )

@web_scraping_bp.route('/sources', methods=['GET', 'POST'])
@login_required
def sources():
    """Manage web scraping sources."""
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_source':
            name = request.form.get('name')
            url = request.form.get('url')
            source_type = request.form.get('source_type')
            
            if not all([name, url, source_type]):
                flash('Name, URL, and source type are required', 'danger')
                return redirect(url_for('web_scraping.sources'))
            
            # Check if source already exists
            existing = WebSource.query.filter_by(name=name).first()
            if existing:
                flash(f'Source with name "{name}" already exists', 'warning')
                return redirect(url_for('web_scraping.sources'))
            
            try:
                # Create source configuration based on source type
                config = {}
                
                if source_type == 'web_page':
                    config['extract_links'] = 'extract_links' in request.form
                    
                elif source_type == 'web_crawl':
                    config['max_pages'] = int(request.form.get('max_pages', 5))
                    config['same_domain_only'] = 'same_domain_only' in request.form
                    
                elif source_type == 'web_search':
                    config['search_term'] = request.form.get('search_term', '')
                    config['search_engine'] = request.form.get('search_engine', 'bing')
                    config['limit'] = int(request.form.get('limit', 10))
                    
                elif source_type == 'rss':
                    # No additional configuration needed for RSS feeds
                    pass
                
                # Create a new web source
                web_source = WebSource()
                web_source.name = name
                web_source.url = url
                web_source.source_type = source_type
                # WebSource doesn't have a created_by field
                web_source.is_active = True
                
                # Set the configuration
                web_source.set_config(config)
                
                db.session.add(web_source)
                db.session.commit()
                
                flash(f'Source "{name}" added successfully', 'success')
                return redirect(url_for('web_scraping.sources'))
                
            except Exception as e:
                flash(f'Error adding source: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.sources'))
                
        elif action == 'delete_source':
            source_id = request.form.get('source_id')
            
            if not source_id:
                flash('Source ID is required', 'danger')
                return redirect(url_for('web_scraping.sources'))
            
            try:
                source = WebSource.query.get(source_id)
                if not source:
                    flash('Source not found', 'danger')
                    return redirect(url_for('web_scraping.sources'))
                
                source_name = source.name
                db.session.delete(source)
                db.session.commit()
                
                flash(f'Source "{source_name}" deleted', 'success')
                return redirect(url_for('web_scraping.sources'))
                
            except Exception as e:
                flash(f'Error deleting source: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.sources'))
                
        elif action == 'toggle_status':
            source_id = request.form.get('source_id')
            is_active = request.form.get('is_active', 'false') == 'true'
            
            if not source_id:
                flash('Source ID is required', 'danger')
                return redirect(url_for('web_scraping.sources'))
            
            try:
                source = WebSource.query.get(source_id)
                if not source:
                    flash('Source not found', 'danger')
                    return redirect(url_for('web_scraping.sources'))
                
                source.is_active = is_active
                db.session.commit()
                
                status = 'activated' if is_active else 'deactivated'
                flash(f'Source "{source.name}" {status}', 'success')
                return redirect(url_for('web_scraping.sources'))
                
            except Exception as e:
                flash(f'Error updating source: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.sources'))
                
        elif action == 'run_source':
            source_id = request.form.get('source_id')
            
            if not source_id:
                flash('Source ID is required', 'danger')
                return redirect(url_for('web_scraping.sources'))
            
            try:
                source = WebSource.query.get(source_id)
                if not source:
                    flash('Source not found', 'danger')
                    return redirect(url_for('web_scraping.sources'))
                
                # Create a job for this source
                job = WebSourceJob()
                job.source_id = source.id
                job.job_type = source.source_type
                job.status = WebSourceJobStatus.PENDING.value
                job.created_by = current_user.id if current_user.is_authenticated else None
                job.created_at = datetime.datetime.utcnow()
                job.job_id = str(uuid.uuid4())
                
                db.session.add(job)
                db.session.commit()
                
                # Process the job
                web_scraping_service.queue_job(job.id)
                
                flash(f'Source "{source.name}" scheduled for processing', 'success')
                return redirect(url_for('web_scraping.sources'))
                
            except Exception as e:
                flash(f'Error running source: {str(e)}', 'danger')
                return redirect(url_for('web_scraping.sources'))
    
    # Get all web sources
    sources = WebSource.query.order_by(WebSource.name).all()
    
    return render_template(
        'web_scraping/sources.html',
        sources=sources
    )

@web_scraping_bp.route('/jobs')
@login_required
def jobs():
    """View and manage web scraping jobs."""
    # Get query parameters
    status = request.args.get('status')
    job_type = request.args.get('job_type')
    
    # Base query
    query = WebSourceJob.query
    
    # Apply filters
    if status:
        query = query.filter(WebSourceJob.status == status)
    
    if job_type:
        query = query.filter(WebSourceJob.job_type == job_type)
    
    # Get jobs with pagination
    page = request.args.get('page', 1, type=int)
    jobs = query.order_by(desc(WebSourceJob.created_at)).paginate(
        page=page, per_page=20, error_out=False
    )
    
    # Get statistics for the sidebar
    job_stats = db.session.query(
        WebSourceJob.status, 
        db.func.count(WebSourceJob.id)
    ).group_by(WebSourceJob.status).all()
    
    job_type_stats = db.session.query(
        WebSourceJob.job_type, 
        db.func.count(WebSourceJob.id)
    ).group_by(WebSourceJob.job_type).all()
    
    return render_template(
        'web_scraping/jobs.html',
        jobs=jobs,
        job_stats=dict(job_stats),
        job_type_stats=dict(job_type_stats),
        current_status=status,
        current_job_type=job_type
    )

@web_scraping_bp.route('/jobs/<int:job_id>')
@login_required
def job_details(job_id):
    """View details of a specific job."""
    job = WebSourceJob.query.get_or_404(job_id)
    
    # Get source if available
    source = None
    if job.source_id:
        source = WebSource.query.get(job.source_id)
    
    # Get metadata and results
    meta_data = job.get_meta_data()
    results = job.get_results()
    
    return render_template(
        'web_scraping/job_details.html',
        job=job,
        source=source,
        meta_data=meta_data,
        results=results
    )

@web_scraping_bp.route('/api/scan', methods=['POST'])
@login_required
def api_scan():
    """API endpoint for scanning a URL."""
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        url = data['url']
        extract_links = data.get('extract_links', False)
        extract_content = data.get('extract_content', True)
        analyze_credibility = data.get('analyze_credibility', False)
        
        # Create a web source job
        job = WebSourceJob(
            job_type='scan',
            status=WebSourceJobStatus.PENDING.value,
            created_by=current_user.id if current_user.is_authenticated else None,
            created_at=datetime.datetime.utcnow(),
            job_id=str(uuid.uuid4())
        )
        
        # Set job metadata
        job_meta = {
            'url': url,
            'extract_links': extract_links,
            'extract_content': extract_content,
            'analyze_credibility': analyze_credibility
        }
        job.set_meta_data(job_meta)
        
        db.session.add(job)
        db.session.commit()
        
        # Process the job (this can be asynchronous)
        web_scraping_service.queue_job(job.id)
        
        return jsonify({
            'job_id': job.id,
            'status': job.status,
            'message': 'Scan job created and queued for processing'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@web_scraping_bp.route('/api/search', methods=['POST'])
@login_required
def api_search():
    """API endpoint for web search."""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
        
    try:
        query = data['query']
        search_engine = data.get('search_engine', 'bing')
        limit = data.get('limit', 10)
        
        # Create a web source job
        job = WebSourceJob()
        job.job_type = 'search'
        job.status = WebSourceJobStatus.PENDING.value
        job.created_by = current_user.id if current_user.is_authenticated else None
        job.created_at = datetime.datetime.utcnow()
        job.job_id = str(uuid.uuid4())
        
        # Set job metadata
        job_meta = {
            'query': query,
            'search_engine': search_engine,
            'limit': limit
        }
        job.set_meta_data(job_meta)
        
        db.session.add(job)
        db.session.commit()
        
        # Queue the job (asynchronous processing)
        web_scraping_service.queue_job(job.id)
        
        return jsonify({
            'job_id': job.id,
            'status': job.status,
            'message': 'Search job created and queued for processing'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
        # Set job metadata
        job_meta = {
            'query': query,
            'search_engine': search_engine,
            'limit': limit
        }
        job.set_meta_data(job_meta)
        
        db.session.add(job)
        db.session.commit()
        
        # Process the job (this can be asynchronous)
        web_scraping_service.queue_job(job.id)
        
        return jsonify({
            'job_id': job.id,
            'status': job.status,
            'message': 'Search job created and queued for processing'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@web_scraping_bp.route('/api/job/<int:job_id>', methods=['GET'])
@login_required
def api_job_status(job_id):
    """API endpoint to check job status."""
    job = WebSourceJob.query.get_or_404(job_id)
    
    # Get metadata and results if available
    meta_data = job.get_meta_data()
    results = job.get_results() if job.status == WebSourceJobStatus.COMPLETED.value else {}
    
    return jsonify({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'meta_data': meta_data,
        'results': results,
        'error_message': job.error_message
    })