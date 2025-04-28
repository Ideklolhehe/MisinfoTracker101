"""
Routes for predictive modeling of misinformation narratives.
"""

import logging
import json
from flask import Blueprint, request, jsonify, render_template, abort
from flask_login import login_required, current_user

from app import db
from models import DetectedNarrative
from services.predictive_modeling import ComplexityPredictionService
from utils.app_context import ensure_app_context

logger = logging.getLogger(__name__)

# Create Blueprint
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/prediction/complexity/<int:narrative_id>', methods=['GET'])
@login_required
def predict_narrative_complexity(narrative_id):
    """
    Predict future complexity for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to predict
        
    Returns:
        HTML page with complexity prediction
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access prediction tools")
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            abort(404, f"Narrative with ID {narrative_id} not found")
        
        # Get days ahead parameter (default: 7 days)
        days_ahead = request.args.get('days', 7, type=int)
        
        # Get confidence level parameter (default: 0.95)
        confidence_level = request.args.get('confidence', 0.95, type=float)
        
        # Make prediction
        prediction = ComplexityPredictionService.predict_narrative_complexity(
            narrative_id, days_ahead, confidence_level
        )
        
        if "error" in prediction:
            return render_template(
                'error.html', 
                message=f"Prediction failed: {prediction['error']}"
            ), 400
        
        return render_template(
            'prediction/view.html',
            narrative=narrative,
            prediction=prediction,
            days_ahead=days_ahead,
            confidence_level=confidence_level
        )
        
    except Exception as e:
        logger.error(f"Error in complexity prediction endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@prediction_bp.route('/prediction/api/complexity/<int:narrative_id>', methods=['GET'])
@login_required
def api_predict_narrative_complexity(narrative_id):
    """
    API endpoint to predict future complexity for a narrative.
    
    Args:
        narrative_id: ID of the narrative to predict
        
    Returns:
        JSON with prediction results
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get days ahead parameter (default: 7 days)
        days_ahead = request.args.get('days', 7, type=int)
        
        # Get confidence level parameter (default: 0.95)
        confidence_level = request.args.get('confidence', 0.95, type=float)
        
        # Make prediction
        prediction = ComplexityPredictionService.predict_narrative_complexity(
            narrative_id, days_ahead, confidence_level
        )
        
        if "error" in prediction:
            return jsonify({"error": prediction["error"]}), 400
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error in complexity prediction API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@prediction_bp.route('/prediction/multiple', methods=['GET'])
@login_required
def predict_multiple_narratives():
    """
    Predict complexity for multiple narratives.
    
    Returns:
        HTML page with multiple predictions
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access prediction tools")
        
        # Get days ahead parameter (default: 7 days)
        days_ahead = request.args.get('days', 7, type=int)
        
        # Get limit parameter (default: 10)
        limit = request.args.get('limit', 10, type=int)
        
        # Make predictions
        predictions = ComplexityPredictionService.predict_multiple_narratives(
            days_ahead, min_data_points=3, limit=limit
        )
        
        if "error" in predictions:
            return render_template(
                'error.html', 
                message=f"Multiple predictions failed: {predictions['error']}"
            ), 400
        
        return render_template(
            'prediction/multiple.html',
            predictions=predictions,
            days_ahead=days_ahead,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error in multiple predictions endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@prediction_bp.route('/prediction/api/multiple', methods=['GET'])
@login_required
def api_predict_multiple_narratives():
    """
    API endpoint to predict complexity for multiple narratives.
    
    Returns:
        JSON with multiple prediction results
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get days ahead parameter (default: 7 days)
        days_ahead = request.args.get('days', 7, type=int)
        
        # Get limit parameter (default: 10)
        limit = request.args.get('limit', 10, type=int)
        
        # Make predictions
        predictions = ComplexityPredictionService.predict_multiple_narratives(
            days_ahead, min_data_points=3, limit=limit
        )
        
        if "error" in predictions:
            return jsonify({"error": predictions["error"]}), 400
        
        return jsonify(predictions)
        
    except Exception as e:
        logger.error(f"Error in multiple predictions API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@prediction_bp.route('/prediction/what-if/<int:narrative_id>', methods=['GET', 'POST'])
@login_required
def what_if_analysis(narrative_id):
    """
    Perform 'what-if' analysis for different intervention scenarios.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        HTML page with what-if analysis
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst']:
            abort(403, "Insufficient privileges to access what-if analysis")
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            abort(404, f"Narrative with ID {narrative_id} not found")
        
        # Default values
        days_ahead = 14
        scenario = 'counter_narrative'
        analysis_result = None
        
        if request.method == 'POST':
            # Get parameters from form
            days_ahead = request.form.get('days_ahead', 14, type=int)
            scenario = request.form.get('scenario', 'counter_narrative')
            
            # Validate scenario
            valid_scenarios = ['counter_narrative', 'debunking', 'visibility_reduction']
            if scenario not in valid_scenarios:
                return render_template(
                    'error.html', 
                    message=f"Invalid scenario: {scenario}"
                ), 400
            
            # Perform what-if analysis
            analysis_result = ComplexityPredictionService.what_if_analysis(
                narrative_id, scenario, days_ahead
            )
            
            if "error" in analysis_result:
                return render_template(
                    'error.html', 
                    message=f"What-if analysis failed: {analysis_result['error']}"
                ), 400
        
        return render_template(
            'prediction/what_if.html',
            narrative=narrative,
            days_ahead=days_ahead,
            scenario=scenario,
            analysis=analysis_result
        )
        
    except Exception as e:
        logger.error(f"Error in what-if analysis endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@prediction_bp.route('/prediction/api/what-if/<int:narrative_id>', methods=['GET'])
@login_required
def api_what_if_analysis(narrative_id):
    """
    API endpoint to perform 'what-if' analysis.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        JSON with what-if analysis results
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get days ahead parameter (default: 14 days)
        days_ahead = request.args.get('days', 14, type=int)
        
        # Get scenario parameter (default: counter_narrative)
        scenario = request.args.get('scenario', 'counter_narrative')
        
        # Validate scenario
        valid_scenarios = ['counter_narrative', 'debunking', 'visibility_reduction']
        if scenario not in valid_scenarios:
            return jsonify({"error": f"Invalid scenario: {scenario}"}), 400
        
        # Perform what-if analysis
        analysis_result = ComplexityPredictionService.what_if_analysis(
            narrative_id, scenario, days_ahead
        )
        
        if "error" in analysis_result:
            return jsonify({"error": analysis_result["error"]}), 400
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error in what-if analysis API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@prediction_bp.route('/prediction/dashboard', methods=['GET'])
@login_required
def prediction_dashboard():
    """
    View prediction dashboard with summary of predictions across narratives.
    
    Returns:
        HTML page with prediction dashboard
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access prediction dashboard")
        
        # Get predictions for multiple narratives
        predictions = ComplexityPredictionService.predict_multiple_narratives(
            days_ahead=7, min_data_points=3, limit=10
        )
        
        if "error" in predictions:
            return render_template(
                'error.html', 
                message=f"Failed to generate prediction dashboard: {predictions['error']}"
            ), 400
        
        # Get trending narratives
        trending_narratives = []
        
        for pred in predictions.get('predictions', []):
            if pred.get('trend_direction', '') in ['strong_increase', 'moderate_increase']:
                trending_narratives.append(pred)
        
        # Sort by current complexity (highest first)
        trending_narratives.sort(key=lambda x: x.get('current_complexity', 0), reverse=True)
        
        return render_template(
            'prediction/dashboard.html',
            predictions=predictions,
            trending_narratives=trending_narratives
        )
        
    except Exception as e:
        logger.error(f"Error in prediction dashboard endpoint: {e}")
        return render_template('error.html', message=str(e)), 500