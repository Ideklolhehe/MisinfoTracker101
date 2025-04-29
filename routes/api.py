"""
API routes for the CIVILIAN system.
This module handles API endpoints for data access and system interaction.
"""

from flask import Blueprint, jsonify, request, current_app
from replit_auth import require_login
from flask_login import current_user
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, desc
from prometheus_client import Counter
from models import DetectedNarrative, NarrativeInstance, DataSource, MisinformationEvent

# Prometheus metrics
misinfo_counter = Counter('misinfo_events_total', 'Total misinformation events reported')

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/narratives')
@require_login
def get_narratives():
    """Return a list of detected narratives."""
    narratives = DetectedNarrative.query.order_by(DetectedNarrative.last_updated.desc()).limit(10).all()
    result = []
    
    for narrative in narratives:
        result.append({
            'id': narrative.id,
            'title': narrative.title,
            'description': narrative.description,
            'confidence_score': narrative.confidence_score,
            'first_detected': narrative.first_detected.isoformat(),
            'last_updated': narrative.last_updated.isoformat(),
            'status': narrative.status,
            'language': narrative.language
        })
    
    return jsonify(result)

@api_bp.route('/narrative/<int:narrative_id>')
@require_login
def get_narrative(narrative_id):
    """Return details for a specific narrative."""
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    instances = NarrativeInstance.query.filter_by(narrative_id=narrative_id).limit(10).all()
    
    instance_data = []
    for instance in instances:
        source_name = "Unknown"
        if instance.source:
            source_name = instance.source.name
            
        instance_data.append({
            'id': instance.id,
            'content': instance.content,
            'source': source_name,
            'detected_at': instance.detected_at.isoformat(),
            'url': instance.url
        })
    
    result = {
        'id': narrative.id,
        'title': narrative.title,
        'description': narrative.description,
        'confidence_score': narrative.confidence_score,
        'first_detected': narrative.first_detected.isoformat(),
        'last_updated': narrative.last_updated.isoformat(),
        'status': narrative.status,
        'language': narrative.language,
        'instances': instance_data
    }
    
    return jsonify(result)

@api_bp.route('/sources')
@require_login
def get_sources():
    """Return a list of data sources."""
    sources = DataSource.query.filter_by(is_active=True).all()
    result = []
    
    for source in sources:
        result.append({
            'id': source.id,
            'name': source.name,
            'type': source.source_type,
            'is_active': source.is_active,
            'last_ingestion': source.last_ingestion.isoformat() if source.last_ingestion else None
        })
    
    return jsonify(result)

@api_bp.route('/system/status')
@require_login
def system_status():
    """Return system status information."""
    from models import SystemLog
    
    logs = SystemLog.query.order_by(SystemLog.timestamp.desc()).limit(5).all()
    log_data = []
    
    for log in logs:
        log_data.append({
            'timestamp': log.timestamp.isoformat(),
            'type': log.log_type,
            'component': log.component,
            'message': log.message
        })
    
    return jsonify({
        'status': 'operational',
        'recent_logs': log_data
    })


@api_bp.route('/report-misinfo', methods=['POST'])
@require_login
def report_misinfo():
    """
    Report a misinformation event tied to a source and narrative.
    
    Required fields:
    - source_id: ID of the data source
    - narrative_id: ID of the detected narrative
    
    Optional fields:
    - confidence: Confidence score for the misinformation assessment (0.0-1.0)
    - impact: Estimated impact level (in metadata)
    - reach: Estimated reach of the misinformation (in metadata)
    - platform: Platform where misinformation was observed (in metadata)
    """
    data = request.get_json(force=True)
    
    # Validate required fields
    source_id = data.get('source_id')
    narrative_id = data.get('narrative_id')
    
    if not source_id or not narrative_id:
        return jsonify(error='Missing required fields: source_id and narrative_id are required'), 400
    
    # Validate source and narrative exist
    source = DataSource.query.get(source_id)
    narrative = DetectedNarrative.query.get(narrative_id)
    
    if not source:
        return jsonify(error=f'Source with ID {source_id} not found'), 404
    
    if not narrative:
        return jsonify(error=f'Narrative with ID {narrative_id} not found'), 404
    
    # Create misinformation event
    event = MisinformationEvent(
        source_id=source_id,
        narrative_id=narrative_id,
        timestamp=datetime.now(timezone.utc),
        reporter_id=current_user.id,
        confidence=float(data.get('confidence', 1.0))
    )
    
    # Add optional metadata
    metadata = {}
    if 'impact' in data:
        metadata['impact'] = data['impact']
    if 'reach' in data:
        metadata['reach'] = data['reach']
    if 'platform' in data:
        metadata['platform'] = data['platform']
    
    if metadata:
        event.set_meta_data(metadata)
    
    # Save to database
    current_app.db.session.add(event)
    current_app.db.session.commit()
    
    # Increment Prometheus counter
    misinfo_counter.inc()
    
    return jsonify({
        'status': 'success',
        'event_id': event.id,
        'message': 'Misinformation event recorded successfully'
    }), 201


@api_bp.route('/source-reliability', methods=['GET'])
@require_login
def source_reliability():
    """
    Return top misinformation sources for the current month.
    
    Returns:
    - month_start: Start date of the current month
    - top_sources: List of top 10 sources with highest misinformation event counts
    - total_events: Total misinformation events for the current month
    """
    # Calculate first day of current month
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Query for events in the current month
    events_query = MisinformationEvent.query.filter(MisinformationEvent.timestamp >= month_start)
    total_events = events_query.count()
    
    # Get top sources with counts
    top_sources_data = (
        current_app.db.session.query(
            DataSource.id,
            DataSource.name,
            DataSource.source_type,
            func.count(MisinformationEvent.id).label('event_count')
        )
        .join(MisinformationEvent, DataSource.id == MisinformationEvent.source_id)
        .filter(MisinformationEvent.timestamp >= month_start)
        .group_by(DataSource.id, DataSource.name, DataSource.source_type)
        .order_by(desc('event_count'))
        .limit(10)
        .all()
    )
    
    # Format response
    top_sources = []
    for source_id, name, source_type, count in top_sources_data:
        top_sources.append({
            'source_id': source_id,
            'name': name,
            'type': source_type,
            'event_count': count
        })
    
    return jsonify({
        'month_start': month_start.date().isoformat(),
        'top_sources': top_sources,
        'total_events': total_events
    })


@api_bp.route('/source-reliability/<int:source_id>', methods=['GET'])
@require_login
def source_reliability_detail(source_id):
    """
    Return detailed reliability information for a specific source.
    
    Returns:
    - source_info: Basic source information
    - monthly_events: Count of events by month (last 6 months)
    - related_narratives: Top narratives associated with this source
    """
    # Get source or 404
    source = DataSource.query.get_or_404(source_id)
    
    # Calculate date 6 months ago
    now = datetime.now(timezone.utc)
    six_months_ago = now - timedelta(days=180)
    
    # Get monthly data
    monthly_data = (
        current_app.db.session.query(
            func.date_trunc('month', MisinformationEvent.timestamp).label('month'),
            func.count().label('count')
        )
        .filter(
            MisinformationEvent.source_id == source_id,
            MisinformationEvent.timestamp >= six_months_ago
        )
        .group_by('month')
        .order_by('month')
        .all()
    )
    
    # Format monthly data
    monthly_events = []
    for month, count in monthly_data:
        monthly_events.append({
            'month': month.date().isoformat(),
            'count': count
        })
    
    # Get top related narratives
    related_narratives = (
        current_app.db.session.query(
            DetectedNarrative.id,
            DetectedNarrative.title,
            func.count(MisinformationEvent.id).label('event_count')
        )
        .join(MisinformationEvent, DetectedNarrative.id == MisinformationEvent.narrative_id)
        .filter(MisinformationEvent.source_id == source_id)
        .group_by(DetectedNarrative.id, DetectedNarrative.title)
        .order_by(desc('event_count'))
        .limit(5)
        .all()
    )
    
    # Format related narratives
    narratives = []
    for narrative_id, title, count in related_narratives:
        narratives.append({
            'narrative_id': narrative_id,
            'title': title,
            'event_count': count
        })
    
    # Get source metadata
    meta_data = source.get_meta_data()
    
    return jsonify({
        'source_info': {
            'id': source.id,
            'name': source.name,
            'type': source.source_type,
            'is_active': source.is_active,
            'created_at': source.created_at.isoformat() if source.created_at else None,
            'last_ingestion': source.last_ingestion.isoformat() if source.last_ingestion else None,
            'credibility_score': meta_data.get('credibility_score'),
            'reliability_score': meta_data.get('reliability_score')
        },
        'monthly_events': monthly_events,
        'related_narratives': narratives
    })