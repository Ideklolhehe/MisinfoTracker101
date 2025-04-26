"""
API routes for the CIVILIAN system.
This module handles API endpoints for data access and system interaction.
"""

from flask import Blueprint, jsonify, request, current_app
from replit_auth import require_login
from flask_login import current_user
from models import DetectedNarrative, NarrativeInstance, DataSource

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