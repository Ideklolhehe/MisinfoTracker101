"""
Routes for enhanced clustering features in the CIVILIAN system.

These routes provide access to the enhanced clustering algorithms:
1. EnhancedDenStream - For real-time clustering with improved outlier handling
2. EnhancedCluStream - For temporal analysis with evolution tracking
3. EnhancedSECLEDS - For subtle pattern detection and concept drift
4. Cross-Algorithm Collaboration - For ensemble clustering
"""

import logging
import json
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, abort
from flask_login import login_required, current_user

from services.enhanced_clustering_service import enhanced_clustering_service

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
enhanced_clustering_bp = Blueprint('enhanced_clustering', __name__)

@enhanced_clustering_bp.route('/clusters/enhanced/dashboard', methods=['GET'])
@login_required
def enhanced_clusters_dashboard():
    """
    Dashboard for enhanced clustering of narratives.
    
    Returns:
        HTML page with enhanced clustering dashboard
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access enhanced clustering tools")
        
        # Get recent processing statistics
        stats = enhanced_clustering_service.process_recent_narratives(limit=50)
        
        # Get cluster overview
        overview = enhanced_clustering_service.get_cluster_overview()
        
        # Get temporal alerts
        alerts = enhanced_clustering_service.get_temporal_alerts()
        
        return render_template(
            'clusters/enhanced_dashboard.html',
            stats=stats,
            overview=overview,
            alerts=alerts
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced clusters dashboard: {e}")
        return render_template('error.html', message=str(e)), 500

@enhanced_clustering_bp.route('/clusters/enhanced/algorithms', methods=['GET'])
@login_required
def enhanced_algorithms():
    """
    View descriptions and features of the enhanced clustering algorithms.
    
    Returns:
        HTML page with algorithm descriptions
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access enhanced clustering tools")
        
        return render_template('clusters/enhanced_algorithms.html')
        
    except Exception as e:
        logger.error(f"Error in enhanced algorithms endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@enhanced_clustering_bp.route('/clusters/enhanced/relationship-map', methods=['GET'])
@login_required
def enhanced_relationship_map():
    """
    View relationship map between narratives and clusters.
    
    Returns:
        HTML page with relationship map visualization
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access enhanced clustering tools")
        
        # Get relationship mapping
        mapping = enhanced_clustering_service.get_narrative_relationship_mapping()
        
        return render_template(
            'clusters/enhanced_relationship_map.html',
            mapping=mapping
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced relationship map endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@enhanced_clustering_bp.route('/clusters/enhanced/narrative/<int:narrative_id>', methods=['GET'])
@login_required
def enhanced_narrative_analysis(narrative_id):
    """
    View enhanced analysis for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        HTML page with enhanced narrative analysis
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access enhanced clustering tools")
        
        # Get enhanced analysis
        analysis = enhanced_clustering_service.get_narrative_analysis(narrative_id)
        
        if 'error' in analysis:
            return render_template('error.html', message=analysis['error']), 404
        
        return render_template(
            'clusters/enhanced_narrative_analysis.html',
            analysis=analysis,
            narrative_id=narrative_id
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced narrative analysis endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@enhanced_clustering_bp.route('/clusters/enhanced/process-all', methods=['POST'])
@login_required
def process_all_narratives():
    """
    Process all active narratives with enhanced clustering.
    
    Returns:
        JSON response with processing results
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get days parameter
        days = request.json.get('days', 30)
        
        # Process all narratives
        result = enhanced_clustering_service.process_all_narratives(days=days)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing all narratives: {e}")
        return jsonify({"error": str(e)}), 500

@enhanced_clustering_bp.route('/clusters/enhanced/process-narrative/<int:narrative_id>', methods=['POST'])
@login_required
def process_narrative(narrative_id):
    """
    Process a specific narrative with enhanced clustering.
    
    Args:
        narrative_id: ID of the narrative to process
        
    Returns:
        JSON response with processing results
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Process the narrative
        result = enhanced_clustering_service.process_narrative(narrative_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing narrative {narrative_id}: {e}")
        return jsonify({"error": str(e)}), 500

@enhanced_clustering_bp.route('/clusters/enhanced/feedback', methods=['POST'])
@login_required
def add_feedback():
    """
    Add analyst feedback about a clustering decision.
    
    Returns:
        JSON response with result
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get parameters
        data = request.json
        narrative_id = data.get('narrative_id')
        cluster_id = data.get('cluster_id')
        feedback_score = data.get('feedback_score')
        source = data.get('source', 'analyst')
        
        if not narrative_id or not isinstance(cluster_id, int) or not isinstance(feedback_score, (int, float)):
            return jsonify({"error": "Missing or invalid parameters"}), 400
        
        # Add feedback
        enhanced_clustering_service.add_feedback(
            narrative_id=narrative_id,
            cluster_id=cluster_id,
            feedback_score=feedback_score,
            source=source
        )
        
        return jsonify({"success": True})
        
    except Exception as e:
        logger.error(f"Error adding feedback: {e}")
        return jsonify({"error": str(e)}), 500

# API endpoints for AJAX requests

@enhanced_clustering_bp.route('/api/clusters/enhanced/overview', methods=['GET'])
@login_required
def api_enhanced_cluster_overview():
    """
    API endpoint to get enhanced cluster overview.
    
    Returns:
        JSON with enhanced cluster overview
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get cluster overview
        overview = enhanced_clustering_service.get_cluster_overview()
        
        return jsonify(overview)
        
    except Exception as e:
        logger.error(f"Error in enhanced cluster overview API: {e}")
        return jsonify({"error": str(e)}), 500

@enhanced_clustering_bp.route('/api/clusters/enhanced/alerts', methods=['GET'])
@login_required
def api_enhanced_temporal_alerts():
    """
    API endpoint to get enhanced temporal alerts.
    
    Returns:
        JSON with enhanced temporal alerts
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get temporal alerts
        alerts = enhanced_clustering_service.get_temporal_alerts()
        
        return jsonify(alerts)
        
    except Exception as e:
        logger.error(f"Error in enhanced temporal alerts API: {e}")
        return jsonify({"error": str(e)}), 500

@enhanced_clustering_bp.route('/api/clusters/enhanced/relationship-map', methods=['GET'])
@login_required
def api_enhanced_relationship_map():
    """
    API endpoint to get enhanced relationship map data.
    
    Returns:
        JSON with enhanced relationship map data
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get relationship mapping
        mapping = enhanced_clustering_service.get_narrative_relationship_mapping()
        
        return jsonify(mapping)
        
    except Exception as e:
        logger.error(f"Error in enhanced relationship map API: {e}")
        return jsonify({"error": str(e)}), 500

@enhanced_clustering_bp.route('/api/clusters/enhanced/narrative/<int:narrative_id>', methods=['GET'])
@login_required
def api_enhanced_narrative_analysis(narrative_id):
    """
    API endpoint to get enhanced analysis for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        JSON with enhanced narrative analysis
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get enhanced analysis
        analysis = enhanced_clustering_service.get_narrative_analysis(narrative_id)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error in enhanced narrative analysis API: {e}")
        return jsonify({"error": str(e)}), 500