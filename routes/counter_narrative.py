"""
Routes for counter-narrative generation, dimension-specific strategies, 
and effectiveness tracking.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from flask import Blueprint, request, jsonify, current_app, abort
from flask_login import login_required, current_user
from sqlalchemy import desc

from models import DetectedNarrative, CounterMessage, db
from services.counter_narrative_service import CounterNarrativeService
from utils.metrics import time_operation, record_execution_time
from utils.app_context import with_app_context

# Configure logger
logger = logging.getLogger(__name__)

# Create blueprint
counter_narrative_bp = Blueprint('counter_narrative', __name__, url_prefix='/counter-narrative')

# Initialize service
counter_narrative_service = CounterNarrativeService()


@counter_narrative_bp.route('/')
@login_required
def dashboard():
    """Counter-narrative dashboard."""
    # Get top narratives requiring counter-narratives
    narratives = DetectedNarrative.query.filter(
        DetectedNarrative.status != 'debunked'
    ).order_by(
        DetectedNarrative.threat_score.desc()
    ).limit(10).all()
    
    # Get recent counter-messages
    counter_messages = CounterMessage.query.order_by(
        CounterMessage.created_at.desc()
    ).limit(20).all()
    
    return jsonify({
        'title': 'Counter-Narrative Dashboard',
        'narratives': [n.to_dict() for n in narratives],
        'counter_messages': [m.to_dict() for m in counter_messages]
    })


@counter_narrative_bp.route('/strategies/<string:dimension>')
@login_required
def get_strategies(dimension: str):
    """Get dimension-specific counter-narrative strategies."""
    strategies = counter_narrative_service.get_dimension_strategies(dimension)
    return jsonify({
        'dimension': dimension,
        'strategies': strategies
    })


@counter_narrative_bp.route('/prioritize', methods=['POST'])
@login_required
def prioritize_clusters():
    """Prioritize narrative clusters for counter-narrative deployment."""
    data = request.get_json(force=True)
    
    # Get threshold from request or use default
    threshold = data.get('threshold', 0.75)
    
    # Use service to prioritize
    with time_operation('prioritize_clusters'):
        prioritized = counter_narrative_service.prioritize_clusters(threshold)
    
    return jsonify({
        'prioritized': prioritized,
        'threshold': threshold
    })


@counter_narrative_bp.route('/generate/<int:narrative_id>', methods=['POST'])
@login_required
def generate_counter(narrative_id: int):
    """Generate a counter-narrative for a detected narrative."""
    # Get narrative
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    # Get parameters
    data = request.get_json(force=True) if request.is_json else {}
    dimension = data.get('dimension', 'cognitive')
    strategy = data.get('strategy', 'factual')
    
    # Generate counter-narrative
    with time_operation('generate_counter_narrative'):
        counter_text = counter_narrative_service.generate_counter_narrative(
            narrative.content,
            dimension=dimension,
            strategy=strategy
        )
    
    # Create counter-message
    counter_message = CounterMessage(
        narrative_id=narrative_id,
        message=counter_text,
        dimension=dimension,
        strategy=strategy,
        created_by=current_user.id
    )
    
    # Save to database
    db.session.add(counter_message)
    db.session.commit()
    
    return jsonify({
        'counter_narrative': counter_text,
        'counter_message_id': counter_message.id,
        'dimension': dimension,
        'strategy': strategy
    })


@counter_narrative_bp.route('/track-effectiveness', methods=['POST'])
@login_required
def track_effectiveness():
    """Track the effectiveness of a counter-message."""
    data = request.get_json(force=True)
    
    # Get counter-message ID
    counter_id = data.get('counter_id')
    if not counter_id:
        abort(400, 'Counter message ID is required')
    
    # Get counter-message
    counter_message = CounterMessage.query.get_or_404(counter_id)
    
    # Get metrics
    metrics = data.get('metrics', {})
    if not metrics:
        abort(400, 'Metrics data is required')
    
    # Update counter-message
    counter_message.effectiveness_metrics = metrics
    counter_message.last_updated = db.func.now()
    
    # Save to database
    db.session.commit()
    
    return jsonify({
        'status': 'recorded',
        'counter_id': counter_id
    })


@counter_narrative_bp.route('/optimize-sources', methods=['POST'])
@login_required
def optimize_sources():
    """Optimize sources for counter-narrative targeting."""
    data = request.get_json(force=True)
    
    # Get narrative_id if provided
    narrative_id = data.get('narrative_id')
    
    # Get network edges (optional)
    edges = data.get('edges')
    
    # Use service to optimize sources
    with time_operation('optimize_sources'):
        top_sources = counter_narrative_service.optimize_sources(
            narrative_id=narrative_id,
            edges=edges
        )
    
    return jsonify({
        'top_sources': top_sources
    })


@counter_narrative_bp.route('/view/<int:counter_id>')
@login_required
def view_counter_message(counter_id: int):
    """View a specific counter-message."""
    counter_message = CounterMessage.query.get_or_404(counter_id)
    narrative = DetectedNarrative.query.get(counter_message.narrative_id)
    
    return jsonify({
        'counter_message': counter_message.to_dict(),
        'narrative': narrative.to_dict() if narrative else None
    })


@counter_narrative_bp.route('/list')
@login_required
def list_counter_messages():
    """List all counter-messages."""
    counter_messages = CounterMessage.query.order_by(
        CounterMessage.created_at.desc()
    ).all()
    
    return jsonify({
        'counter_messages': [m.to_dict() for m in counter_messages]
    })