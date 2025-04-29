"""
Routes for comparative analysis of narratives.
"""

import logging
from typing import Dict, List, Any, Optional

from flask import Blueprint, request, jsonify, render_template, abort
from flask_login import login_required, current_user

from models import DetectedNarrative, db
from services.comparative_analysis_service import ComparativeAnalysisService
from utils.metrics import time_operation

# Configure logger
logger = logging.getLogger(__name__)

# Create blueprint
comparative_bp = Blueprint('comparative', __name__, url_prefix='/comparative')

# Initialize service
comparative_service = ComparativeAnalysisService()


@comparative_bp.route('/')
@login_required
def dashboard():
    """Comparative analysis dashboard."""
    # Get top narratives for comparison
    narratives = DetectedNarrative.query.filter(
        DetectedNarrative.status != 'debunked'
    ).order_by(
        DetectedNarrative.last_updated.desc()
    ).limit(20).all()
    
    return render_template(
        'comparative/dashboard.html',
        title='Comparative Analysis Dashboard',
        narratives=narratives
    )


@comparative_bp.route('/side-by-side', methods=['GET', 'POST'])
@login_required
def side_by_side():
    """Side-by-side comparison of narratives across dimensions."""
    if request.method == 'POST':
        data = request.get_json(force=True) if request.is_json else {}
        narrative_ids = data.get('narrative_ids', [])
        dimensions = data.get('dimensions', ['complexity_score', 'propagation_score', 'threat_score'])
        
        if not narrative_ids:
            return jsonify({'error': 'No narratives selected for comparison'}), 400
        
        with time_operation('side_by_side_comparison'):
            result = comparative_service.side_by_side_comparison(narrative_ids, dimensions)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    # GET request - show form
    narratives = DetectedNarrative.query.filter(
        DetectedNarrative.status != 'debunked'
    ).order_by(
        DetectedNarrative.last_updated.desc()
    ).limit(20).all()
    
    return render_template(
        'comparative/side_by_side.html',
        title='Side-by-Side Comparison',
        narratives=narratives
    )


@comparative_bp.route('/growth-rate/<int:narrative_id>')
@login_required
def growth_rate(narrative_id: int):
    """Relative growth rate analysis for a narrative."""
    # Get the narrative
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    # Get number of days (optional parameter)
    days = request.args.get('days', 30, type=int)
    
    with time_operation('growth_rate_analysis'):
        result = comparative_service.relative_growth_rate(narrative_id, days)
    
    if 'error' in result:
        return jsonify(result), 400
    
    if request.headers.get('Accept') == 'application/json':
        return jsonify(result)
    
    return render_template(
        'comparative/growth_rate.html',
        title=f'Growth Rate: {narrative.title}',
        narrative=narrative,
        chart_html=result['chart_html'],
        data=result['data']
    )


@comparative_bp.route('/correlation', methods=['GET', 'POST'])
@login_required
def correlation():
    """Correlation analysis between two narratives."""
    if request.method == 'POST':
        data = request.get_json(force=True) if request.is_json else {}
        narrative_id_1 = data.get('narrative_id_1')
        narrative_id_2 = data.get('narrative_id_2')
        
        if not narrative_id_1 or not narrative_id_2:
            return jsonify({'error': 'Two narratives are required for correlation analysis'}), 400
        
        with time_operation('correlation_analysis'):
            result = comparative_service.identify_correlation(narrative_id_1, narrative_id_2)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    # GET request - show form
    narratives = DetectedNarrative.query.filter(
        DetectedNarrative.status != 'debunked'
    ).order_by(
        DetectedNarrative.last_updated.desc()
    ).limit(20).all()
    
    return render_template(
        'comparative/correlation.html',
        title='Correlation Analysis',
        narratives=narratives
    )


@comparative_bp.route('/themes', methods=['GET', 'POST'])
@login_required
def shared_themes():
    """Shared theme detection across narratives."""
    if request.method == 'POST':
        data = request.get_json(force=True) if request.is_json else {}
        narrative_ids = data.get('narrative_ids', [])
        n_topics = data.get('n_topics', 5)
        
        if not narrative_ids or len(narrative_ids) < 2:
            return jsonify({'error': 'At least two narratives are required for theme detection'}), 400
        
        with time_operation('shared_theme_detection'):
            result = comparative_service.shared_theme_detection(narrative_ids, n_topics)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    # GET request - show form
    narratives = DetectedNarrative.query.filter(
        DetectedNarrative.status != 'debunked'
    ).order_by(
        DetectedNarrative.last_updated.desc()
    ).limit(20).all()
    
    return render_template(
        'comparative/themes.html',
        title='Shared Theme Detection',
        narratives=narratives
    )


@comparative_bp.route('/coordinate-sources', methods=['GET', 'POST'])
@login_required
def coordinate_sources():
    """Coordinated source analysis."""
    if request.method == 'POST':
        data = request.get_json(force=True) if request.is_json else {}
        narrative_ids = data.get('narrative_ids', [])
        edges = data.get('edges')
        
        with time_operation('coordinated_source_analysis'):
            result = comparative_service.coordinated_source_analysis(narrative_ids, edges)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    # GET request - show form
    narratives = DetectedNarrative.query.filter(
        DetectedNarrative.status != 'debunked'
    ).order_by(
        DetectedNarrative.last_updated.desc()
    ).limit(20).all()
    
    return render_template(
        'comparative/coordinate_sources.html',
        title='Coordinated Source Analysis',
        narratives=narratives
    )


@comparative_bp.route('/api/side-by-side', methods=['POST'])
@login_required
def api_side_by_side():
    """API endpoint for side-by-side comparison."""
    data = request.get_json(force=True)
    narrative_ids = data.get('narrative_ids', [])
    dimensions = data.get('dimensions', ['complexity_score', 'propagation_score', 'threat_score'])
    
    if not narrative_ids:
        return jsonify({'error': 'No narratives selected for comparison'}), 400
    
    with time_operation('api_side_by_side_comparison'):
        result = comparative_service.side_by_side_comparison(narrative_ids, dimensions)
    
    return jsonify(result)


@comparative_bp.route('/api/growth-rate', methods=['POST'])
@login_required
def api_growth_rate():
    """API endpoint for growth rate analysis."""
    data = request.get_json(force=True)
    narrative_id = data.get('narrative_id')
    days = data.get('days', 30)
    
    if not narrative_id:
        return jsonify({'error': 'Narrative ID is required'}), 400
    
    with time_operation('api_growth_rate_analysis'):
        result = comparative_service.relative_growth_rate(narrative_id, days)
    
    return jsonify(result)


@comparative_bp.route('/api/correlation', methods=['POST'])
@login_required
def api_correlation():
    """API endpoint for correlation analysis."""
    data = request.get_json(force=True)
    narrative_id_1 = data.get('narrative_id_1')
    narrative_id_2 = data.get('narrative_id_2')
    
    if not narrative_id_1 or not narrative_id_2:
        return jsonify({'error': 'Two narrative IDs are required'}), 400
    
    with time_operation('api_correlation_analysis'):
        result = comparative_service.identify_correlation(narrative_id_1, narrative_id_2)
    
    return jsonify(result)


@comparative_bp.route('/api/shared-themes', methods=['POST'])
@login_required
def api_shared_themes():
    """API endpoint for shared theme detection."""
    data = request.get_json(force=True)
    narrative_ids = data.get('narrative_ids', [])
    n_topics = data.get('n_topics', 5)
    
    if not narrative_ids or len(narrative_ids) < 2:
        return jsonify({'error': 'At least two narratives are required for theme detection'}), 400
    
    with time_operation('api_shared_theme_detection'):
        result = comparative_service.shared_theme_detection(narrative_ids, n_topics)
    
    return jsonify(result)


@comparative_bp.route('/api/coordinate-sources', methods=['POST'])
@login_required
def api_coordinate_sources():
    """API endpoint for coordinated source analysis."""
    data = request.get_json(force=True)
    narrative_ids = data.get('narrative_ids', [])
    edges = data.get('edges')
    
    with time_operation('api_coordinated_source_analysis'):
        result = comparative_service.coordinated_source_analysis(narrative_ids, edges)
    
    return jsonify(result)