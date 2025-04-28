"""
Routes for time-series analysis of misinformation narratives.
"""

import logging
import json
from flask import Blueprint, request, jsonify, render_template, abort
from flask_login import login_required, current_user

from app import db
from models import DetectedNarrative
from services.time_series_analyzer import TimeSeriesAnalyzer
from utils.app_context import ensure_app_context

logger = logging.getLogger(__name__)

# Initialize services
time_series_analyzer = TimeSeriesAnalyzer()

# Create Blueprint
time_series_bp = Blueprint('time_series', __name__)

@time_series_bp.route('/time_series/analyze/<int:narrative_id>', methods=['GET'])
@login_required
def analyze_narrative_time_series(narrative_id):
    """
    Analyze time-series data for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        HTML page with time-series analysis
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access time-series analysis")
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            abort(404, f"Narrative with ID {narrative_id} not found")
        
        # Get period parameter (default: 7 days)
        period = request.args.get('period', 7, type=int)
        
        # Perform time-series analysis
        analysis_result = time_series_analyzer.analyze_narrative_time_series(narrative_id, period)
        
        if "error" in analysis_result:
            return render_template(
                'error.html', 
                message=f"Time-series analysis failed: {analysis_result['error']}"
            ), 400
        
        # Generate AI insights
        insights = time_series_analyzer.generate_ai_insights(narrative_id)
        
        return render_template(
            'time_series/view.html',
            narrative=narrative,
            analysis=analysis_result,
            insights=insights,
            period=period
        )
        
    except Exception as e:
        logger.error(f"Error in time-series analysis endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@time_series_bp.route('/time_series/api/analyze/<int:narrative_id>', methods=['GET'])
@login_required
def api_analyze_time_series(narrative_id):
    """
    API endpoint to get time-series analysis for a narrative.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        JSON with time-series analysis
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get period parameter (default: 7 days)
        period = request.args.get('period', 7, type=int)
        
        # Perform time-series analysis
        analysis_result = time_series_analyzer.analyze_narrative_time_series(narrative_id, period)
        
        if "error" in analysis_result:
            return jsonify({"error": analysis_result["error"]}), 400
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error in time-series analysis API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@time_series_bp.route('/time_series/trends', methods=['GET'])
@login_required
def view_time_series_trends():
    """
    View time-series trends across multiple narratives.
    
    Returns:
        HTML page with time-series trends
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access time-series trends")
        
        # Get narratives with complexity data
        narratives_with_data = []
        
        # Get active narratives
        narratives = DetectedNarrative.query.filter_by(status='active').all()
        
        for narrative in narratives:
            # Check if we have time-series data
            try:
                # Calculate trend metrics
                trend_metrics = time_series_analyzer.calculate_trend_metrics(narrative.id)
                
                if "error" not in trend_metrics:
                    # Add to list
                    narratives_with_data.append({
                        'id': narrative.id,
                        'title': narrative.title,
                        'metrics': trend_metrics
                    })
            except Exception as e:
                logger.warning(f"Error calculating trend metrics for narrative {narrative.id}: {e}")
                continue
        
        # Sort by trend significance (highest absolute slope first)
        narratives_with_data.sort(
            key=lambda x: abs(x['metrics']['overall_trend']['slope']), 
            reverse=True
        )
        
        return render_template(
            'time_series/trends.html',
            narratives=narratives_with_data
        )
        
    except Exception as e:
        logger.error(f"Error in time-series trends endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@time_series_bp.route('/time_series/api/trends', methods=['GET'])
@login_required
def api_get_time_series_trends():
    """
    API endpoint to get time-series trends across multiple narratives.
    
    Returns:
        JSON with time-series trends
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get limit parameter (default: 10)
        limit = request.args.get('limit', 10, type=int)
        
        # Get narratives with trend data
        narratives_data = []
        
        # Get active narratives
        narratives = DetectedNarrative.query.filter_by(status='active').limit(100).all()
        
        for narrative in narratives:
            try:
                # Calculate trend metrics
                trend_metrics = time_series_analyzer.calculate_trend_metrics(narrative.id)
                
                if "error" not in trend_metrics:
                    narratives_data.append(trend_metrics)
            except Exception as e:
                logger.warning(f"Error calculating trend metrics for narrative {narrative.id}: {e}")
                continue
        
        # Sort by trend significance (highest absolute slope first)
        narratives_data.sort(
            key=lambda x: abs(x['overall_trend']['slope']), 
            reverse=True
        )
        
        # Limit the results
        narratives_data = narratives_data[:limit]
        
        return jsonify({
            'count': len(narratives_data),
            'narratives': narratives_data
        })
        
    except Exception as e:
        logger.error(f"Error in time-series trends API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@time_series_bp.route('/time_series/insights/<int:narrative_id>', methods=['GET'])
@login_required
def get_time_series_insights(narrative_id):
    """
    Get AI-generated insights for a narrative's time-series data.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        HTML page with time-series insights
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access time-series insights")
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            abort(404, f"Narrative with ID {narrative_id} not found")
        
        # Generate insights
        insights = time_series_analyzer.generate_ai_insights(narrative_id)
        
        if "error" in insights:
            return render_template(
                'error.html', 
                message=f"Failed to generate insights: {insights['error']}"
            ), 400
        
        return render_template(
            'time_series/insights.html',
            narrative=narrative,
            insights=insights
        )
        
    except Exception as e:
        logger.error(f"Error in time-series insights endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@time_series_bp.route('/time_series/api/insights/<int:narrative_id>', methods=['GET'])
@login_required
def api_get_time_series_insights(narrative_id):
    """
    API endpoint to get AI-generated insights for a narrative's time-series data.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        JSON with AI-generated insights
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Generate insights
        insights = time_series_analyzer.generate_ai_insights(narrative_id)
        
        if "error" in insights:
            return jsonify({"error": insights["error"]}), 400
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Error in time-series insights API endpoint: {e}")
        return jsonify({"error": str(e)}), 500