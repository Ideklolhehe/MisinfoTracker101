"""
Dashboard routes for the CIVILIAN system.
This module handles the dashboard functionality and analytics.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from replit_auth import require_login
from flask_login import current_user
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, desc
from models import DetectedNarrative, NarrativeInstance, DataSource, MisinformationEvent

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@dashboard_bp.route('/')
@require_login
def index():
    """
    Render the dashboard main page.
    This route is protected by Replit Auth.
    """
    return render_template('dashboard/index.html')

@dashboard_bp.route('/dashboard/source-reliability')
@require_login
def source_reliability():
    """
    Render the source reliability analysis dashboard.
    Shows top misinformation sources and provides tools for detailed analysis.
    """
    # Calculate first day of current month
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Get total event count for current month
    total_events = MisinformationEvent.query.filter(
        MisinformationEvent.timestamp >= month_start
    ).count()
    
    # Get top sources with counts
    top_sources = (
        DataSource.query
        .join(MisinformationEvent, DataSource.id == MisinformationEvent.source_id)
        .filter(MisinformationEvent.timestamp >= month_start)
        .group_by(DataSource.id)
        .order_by(desc(func.count(MisinformationEvent.id)))
        .limit(10)
        .all()
    )
    
    # Get source counts
    source_counts = {}
    for source in top_sources:
        count = MisinformationEvent.query.filter(
            MisinformationEvent.source_id == source.id,
            MisinformationEvent.timestamp >= month_start
        ).count()
        source_counts[source.id] = count
    
    # Get all active sources for form selection
    all_sources = DataSource.query.filter_by(is_active=True).order_by(DataSource.name).all()
    
    return render_template(
        'dashboard/source_reliability.html',
        month_start=month_start.date(),
        top_sources=top_sources,
        source_counts=source_counts,
        total_events=total_events,
        all_sources=all_sources
    )


@dashboard_bp.route('/dashboard/report-misinfo', methods=['GET', 'POST'])
@require_login
def report_misinfo():
    """
    Form for reporting misinformation events.
    POST: Creates a new misinformation event
    GET: Shows the report form
    """
    from prometheus_client import Counter
    misinfo_counter = Counter('misinfo_events_total', 'Total misinformation events reported')
    
    # Get all sources and narratives for dropdown selection
    sources = DataSource.query.filter_by(is_active=True).order_by(DataSource.name).all()
    narratives = DetectedNarrative.query.order_by(DetectedNarrative.last_updated.desc()).limit(100).all()
    
    if request.method == 'POST':
        source_id = request.form.get('source_id')
        narrative_id = request.form.get('narrative_id')
        
        # Validate required fields
        if not source_id or not narrative_id:
            flash('Source and Narrative are required fields.', 'danger')
            return render_template(
                'dashboard/report_misinfo.html',
                sources=sources,
                narratives=narratives
            )
            
        # Safely parse confidence
        try:
            confidence = float(request.form.get('confidence', 1.0))
            if not (0.0 <= confidence <= 1.0):
                raise ValueError("Confidence must be between 0 and 1")
        except ValueError:
            flash('Confidence must be a number between 0 and 1.', 'danger')
            return render_template(
                'dashboard/report_misinfo.html',
                sources=sources,
                narratives=narratives
            )
        
        # Validate and process optional fields
        impact = request.form.get('impact')
        if impact and not impact.isdigit():
            flash('Impact must be a numeric value.', 'danger')
            return render_template(
                'dashboard/report_misinfo.html',
                sources=sources,
                narratives=narratives
            )
            
        reach = request.form.get('reach')
        if reach and not reach.isdigit():
            flash('Reach must be a numeric value.', 'danger')
            return render_template(
                'dashboard/report_misinfo.html',
                sources=sources, 
                narratives=narratives
            )
            
        platform = request.form.get('platform')
        
        # Create misinformation event
        event = MisinformationEvent(
            source_id=source_id,
            narrative_id=narrative_id,
            timestamp=datetime.now(timezone.utc),
            reporter_id=current_user.id,
            confidence=confidence
        )
        
        # Add optional metadata
        metadata = {}
        if impact:
            metadata['impact'] = impact
        if reach:
            metadata['reach'] = reach
        if platform:
            metadata['platform'] = platform
        
        if metadata:
            event.set_meta_data(metadata)
        
        # Save to database
        db = request.app.db.session
        db.add(event)
        db.commit()
        
        # Increment Prometheus counter
        misinfo_counter.inc()
        
        flash('Misinformation event reported successfully.', 'success')
        return redirect(url_for('dashboard.source_reliability'))
    
    return render_template(
        'dashboard/report_misinfo.html',
        sources=sources,
        narratives=narratives
    )


@dashboard_bp.route('/dashboard/source-reliability/<int:source_id>')
@require_login
def source_reliability_detail(source_id):
    """
    Render the detailed view for a specific source's reliability analysis.
    Shows historical data, related narratives, and reliability metrics.
    """
    # Get source or 404
    source = DataSource.query.get_or_404(source_id)
    
    # Calculate dates for analysis
    now = datetime.now(timezone.utc)
    six_months_ago = now - timedelta(days=180)
    one_month_ago = now - timedelta(days=30)
    current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Get monthly data (last 6 months)
    monthly_data = (
        func.date_trunc('month', MisinformationEvent.timestamp).label('month'),
        func.count().label('count')
    )
    
    monthly_counts = (
        MisinformationEvent.query
        .with_entities(*monthly_data)
        .filter(
            MisinformationEvent.source_id == source_id,
            MisinformationEvent.timestamp >= six_months_ago
        )
        .group_by('month')
        .order_by('month')
        .all()
    )
    
    # Get related narratives (most common narratives for this source)
    related_narratives = (
        DetectedNarrative.query
        .join(MisinformationEvent, DetectedNarrative.id == MisinformationEvent.narrative_id)
        .filter(MisinformationEvent.source_id == source_id)
        .with_entities(
            DetectedNarrative.id,
            DetectedNarrative.title, 
            func.count(MisinformationEvent.id).label('event_count')
        )
        .group_by(DetectedNarrative.id, DetectedNarrative.title)
        .order_by(desc('event_count'))
        .limit(5)
        .all()
    )
    
    # Get current month count
    current_month_count = MisinformationEvent.query.filter(
        MisinformationEvent.source_id == source_id,
        MisinformationEvent.timestamp >= current_month_start
    ).count()
    
    # Get previous month count
    previous_month_count = MisinformationEvent.query.filter(
        MisinformationEvent.source_id == source_id,
        MisinformationEvent.timestamp >= one_month_ago,
        MisinformationEvent.timestamp < current_month_start
    ).count()
    
    # Calculate month-over-month change percentage
    if previous_month_count > 0:
        month_change_pct = ((current_month_count - previous_month_count) / previous_month_count) * 100
    else:
        month_change_pct = 100 if current_month_count > 0 else 0
    
    # Get recent events for this source
    recent_events = (
        MisinformationEvent.query
        .filter(MisinformationEvent.source_id == source_id)
        .order_by(MisinformationEvent.timestamp.desc())
        .limit(10)
        .all()
    )
    
    # Prepare datasets for charts (convert to JSON-friendly format)
    monthly_labels = []
    monthly_counts_data = []
    
    for month, count in monthly_counts:
        monthly_labels.append(month.strftime('%b %Y'))
        monthly_counts_data.append(count)
    
    # Get source metadata
    meta_data = source.get_meta_data()
    
    # Calculate reliability score (inverse of misinformation events)
    # Higher event count = lower reliability
    total_events = MisinformationEvent.query.filter(
        MisinformationEvent.source_id == source_id
    ).count()
    
    reliability_score = 100
    if total_events > 0:
        # Simple inverse score - can be refined with more sophisticated calculation
        reliability_score = max(0, 100 - min(100, total_events * 5))
    
    # Update source metadata with new reliability score
    if not meta_data:
        meta_data = {}
    meta_data['reliability_score'] = reliability_score
    source.set_meta_data(meta_data)
    
    return render_template(
        'dashboard/source_reliability_detail.html',
        source=source,
        monthly_labels=monthly_labels,
        monthly_counts=monthly_counts_data,
        related_narratives=related_narratives,
        current_month_count=current_month_count,
        previous_month_count=previous_month_count,
        month_change_pct=month_change_pct,
        reliability_score=reliability_score,
        recent_events=recent_events,
        meta_data=meta_data
    )