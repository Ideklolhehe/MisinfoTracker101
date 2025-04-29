"""
Prediction routes for the CIVILIAN system.

This module handles routes for predictive modeling, including narrative trajectory
forecasting, key factor analysis, anomaly detection, and what-if scenario simulation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from flask import Blueprint, render_template, request, jsonify, url_for, flash, redirect
from flask_login import login_required, current_user

from app import db
from models import DetectedNarrative, NarrativeInstance, CounterMessage
from services.predictive_modeling import PredictiveModeling

# Configure logging
logger = logging.getLogger(__name__)

# Initialize blueprint
prediction_bp = Blueprint('prediction', __name__, url_prefix='/prediction')

# Initialize predictive modeling service
predictive_modeling = PredictiveModeling()

@prediction_bp.route('/')
@login_required
def dashboard():
    """Display the prediction dashboard with available narratives."""
    try:
        # Get narratives with enough data for prediction
        narratives = (DetectedNarrative.query
                    .filter(DetectedNarrative.status == 'active')
                    .order_by(DetectedNarrative.first_detected.desc())
                    .limit(20)
                    .all())
        
        return render_template(
            'prediction/dashboard.html',
            title='Predictive Dashboard',
            narratives=narratives
        )
    except Exception as e:
        logger.exception(f"Error displaying prediction dashboard: {str(e)}")
        flash(f"Error loading dashboard: {str(e)}", 'danger')
        return render_template(
            'prediction/dashboard.html',
            title='Predictive Dashboard',
            narratives=[]
        )

@prediction_bp.route('/forecast/<int:narrative_id>')
@login_required
def forecast(narrative_id):
    """Generate and display a forecast for a narrative."""
    try:
        # Get parameters
        metric = request.args.get('metric', 'complexity')
        model_type = request.args.get('model', 'arima')
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Get narrative
        narrative = DetectedNarrative.query.get_or_404(narrative_id)
        
        # Generate forecast
        forecast_data = predictive_modeling.forecast_narrative(
            narrative_id=narrative_id,
            metric=metric,
            model_type=model_type,
            force_refresh=force_refresh
        )
        
        if not forecast_data.get('success', False):
            flash(forecast_data.get('error', 'Unknown error'), 'danger')
            return render_template(
                'prediction/error.html',
                title='Forecast Error',
                error=forecast_data.get('error', 'Unknown error'),
                narrative=narrative
            )
        
        return render_template(
            'prediction/view.html',
            title=f'Forecast for "{narrative.title}"',
            narrative=narrative,
            forecast=forecast_data,
            metric=metric,
            model_type=model_type
        )
    except Exception as e:
        logger.exception(f"Error forecasting narrative {narrative_id}: {str(e)}")
        narrative = DetectedNarrative.query.get(narrative_id)
        return render_template(
            'prediction/error.html',
            title='Forecast Error',
            error=str(e),
            narrative=narrative
        )

@prediction_bp.route('/api/forecast/<int:narrative_id>')
@login_required
def api_forecast(narrative_id):
    """API endpoint for narrative forecasts."""
    try:
        # Get parameters
        metric = request.args.get('metric', 'complexity')
        model_type = request.args.get('model', 'arima')
        
        # Generate forecast
        forecast_data = predictive_modeling.forecast_narrative(
            narrative_id=narrative_id,
            metric=metric,
            model_type=model_type
        )
        
        return jsonify(forecast_data)
    except Exception as e:
        logger.exception(f"API error forecasting narrative {narrative_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@prediction_bp.route('/key-factors/<int:narrative_id>')
@login_required
def key_factors(narrative_id):
    """Analyze and display key factors influencing a narrative."""
    try:
        # Get parameters
        metric = request.args.get('metric', 'complexity')
        
        # Get narrative
        narrative = DetectedNarrative.query.get_or_404(narrative_id)
        
        # Analyze key factors
        factors_data = predictive_modeling.analyze_key_factors(
            narrative_id=narrative_id,
            metric=metric
        )
        
        return render_template(
            'prediction/key_factors.html',
            title=f'Key Factors for "{narrative.title}"',
            narrative=narrative,
            factors=factors_data,
            metric=metric
        )
    except Exception as e:
        logger.exception(f"Error analyzing key factors for {narrative_id}: {str(e)}")
        narrative = DetectedNarrative.query.get(narrative_id)
        return render_template(
            'prediction/error.html',
            title='Key Factors Analysis Error',
            error=str(e),
            narrative=narrative
        )

@prediction_bp.route('/what-if/<int:narrative_id>')
@login_required
def what_if(narrative_id):
    """Simulate and display a what-if scenario for a narrative."""
    try:
        # Get parameters
        metric = request.args.get('metric', 'complexity')
        model_type = request.args.get('model', 'arima')
        
        # Get narrative
        narrative = DetectedNarrative.query.get_or_404(narrative_id)
        
        # Default interventions (empty/none)
        interventions = {}
        
        # Parse intervention parameters if present
        if request.args.get('counter_strength'):
            interventions['counter_strength'] = float(request.args.get('counter_strength'))
        
        if request.args.get('counter_timing'):
            interventions['counter_timing'] = int(request.args.get('counter_timing'))
        
        # Generate baseline forecast for comparison
        baseline = predictive_modeling.forecast_narrative(
            narrative_id=narrative_id,
            metric=metric,
            model_type=model_type
        )
        
        # Simulate scenario if interventions specified
        if interventions:
            scenario = predictive_modeling.simulate_scenario(
                narrative_id=narrative_id,
                interventions=interventions,
                metric=metric,
                model_type=model_type
            )
        else:
            # If no interventions specified, return template with baseline only
            scenario = None
        
        return render_template(
            'prediction/what_if.html',
            title=f'What-If Analysis for "{narrative.title}"',
            narrative=narrative,
            baseline=baseline,
            scenario=scenario,
            interventions=interventions,
            metric=metric,
            model_type=model_type
        )
    except Exception as e:
        logger.exception(f"Error simulating what-if for {narrative_id}: {str(e)}")
        narrative = DetectedNarrative.query.get(narrative_id)
        return render_template(
            'prediction/error.html',
            title='What-If Analysis Error',
            error=str(e),
            narrative=narrative
        )

@prediction_bp.route('/threshold/<int:narrative_id>')
@login_required
def threshold(narrative_id):
    """Display threshold crossing projections for a narrative."""
    try:
        # Get parameters
        metric = request.args.get('metric', 'complexity')
        model_type = request.args.get('model', 'arima')
        threshold_value = float(request.args.get('threshold', '0.75'))
        direction = request.args.get('direction', 'above')
        
        # Get narrative
        narrative = DetectedNarrative.query.get_or_404(narrative_id)
        
        # Generate forecast
        forecast_data = predictive_modeling.forecast_narrative(
            narrative_id=narrative_id,
            metric=metric,
            model_type=model_type
        )
        
        if not forecast_data.get('success', False):
            flash(forecast_data.get('error', 'Unknown error'), 'danger')
            return render_template(
                'prediction/error.html',
                title='Threshold Analysis Error',
                error=forecast_data.get('error', 'Unknown error'),
                narrative=narrative
            )
        
        # Find threshold crossings
        threshold_data = predictive_modeling.find_threshold_crossing(
            forecast_data=forecast_data,
            threshold_value=threshold_value,
            direction=direction
        )
        
        return render_template(
            'prediction/threshold.html',
            title=f'Threshold Analysis for "{narrative.title}"',
            narrative=narrative,
            forecast=forecast_data,
            threshold=threshold_data,
            threshold_value=threshold_value,
            direction=direction,
            metric=metric
        )
    except Exception as e:
        logger.exception(f"Error analyzing threshold for {narrative_id}: {str(e)}")
        narrative = DetectedNarrative.query.get(narrative_id)
        return render_template(
            'prediction/error.html',
            title='Threshold Analysis Error',
            error=str(e),
            narrative=narrative
        )

@prediction_bp.route('/anomalies/<int:narrative_id>')
@login_required
def anomalies(narrative_id):
    """Detect and display anomalies for a narrative."""
    try:
        # Get parameters
        days = int(request.args.get('days', '30'))
        
        # Get narrative
        narrative = DetectedNarrative.query.get_or_404(narrative_id)
        
        # Detect anomalies
        anomaly_data = predictive_modeling.detect_anomalies(
            narrative_id=narrative_id,
            days=days
        )
        
        return render_template(
            'prediction/anomalies.html',
            title=f'Anomaly Detection for "{narrative.title}"',
            narrative=narrative,
            anomalies=anomaly_data,
            days=days
        )
    except Exception as e:
        logger.exception(f"Error detecting anomalies for {narrative_id}: {str(e)}")
        narrative = DetectedNarrative.query.get(narrative_id)
        return render_template(
            'prediction/error.html',
            title='Anomaly Detection Error',
            error=str(e),
            narrative=narrative
        )

@prediction_bp.route('/multiple')
@login_required
def multiple_forecasts():
    """Generate and display multiple narrative forecasts for comparison."""
    try:
        # Get parameters
        narrative_ids = request.args.getlist('narratives')
        metric = request.args.get('metric', 'complexity')
        days_horizon = int(request.args.get('horizon', '7'))
        
        # If no narratives specified, show selection form
        if not narrative_ids:
            narratives = (DetectedNarrative.query
                        .filter(DetectedNarrative.status == 'active')
                        .order_by(DetectedNarrative.first_detected.desc())
                        .limit(20)
                        .all())
            
            return render_template(
                'prediction/multiple_select.html',
                title='Compare Multiple Narratives',
                narratives=narratives,
                metric=metric,
                days_horizon=days_horizon
            )
        
        # Get narratives and generate forecasts
        selected_narratives = []
        forecasts = {}
        
        for narrative_id in narrative_ids:
            try:
                narrative_id = int(narrative_id)
                narrative = DetectedNarrative.query.get(narrative_id)
                
                if narrative:
                    selected_narratives.append(narrative)
                    
                    # Generate forecast
                    forecast = predictive_modeling.forecast_narrative(
                        narrative_id=narrative_id,
                        metric=metric,
                        days_horizon=days_horizon
                    )
                    
                    if forecast.get('success', False):
                        forecasts[narrative_id] = forecast
            except Exception as e:
                logger.warning(f"Error forecasting narrative {narrative_id}: {str(e)}")
        
        if not forecasts:
            flash("No valid forecasts could be generated for the selected narratives.", "warning")
            return redirect(url_for('prediction.multiple_forecasts'))
        
        return render_template(
            'prediction/multiple_view.html',
            title='Narrative Comparison',
            narratives=selected_narratives,
            forecasts=forecasts,
            metric=metric,
            days_horizon=days_horizon
        )
    except Exception as e:
        logger.exception(f"Error comparing multiple narratives: {str(e)}")
        flash(f"Error generating comparison: {str(e)}", 'danger')
        return redirect(url_for('prediction.dashboard'))

@prediction_bp.route('/trending')
@login_required
def trending():
    """Display trending narratives based on predictive analysis."""
    try:
        # Get parameters
        days = int(request.args.get('days', '7'))
        limit = int(request.args.get('limit', '10'))
        
        # Get trending narratives
        trending_data = predictive_modeling.get_trending_narratives(
            days=days,
            limit=limit
        )
        
        return render_template(
            'prediction/trending.html',
            title='Trending Narratives',
            trending=trending_data,
            days=days
        )
    except Exception as e:
        logger.exception(f"Error getting trending narratives: {str(e)}")
        flash(f"Error loading trending narratives: {str(e)}", 'danger')
        return redirect(url_for('prediction.dashboard'))

@prediction_bp.route('/batch-jobs')
@login_required
def batch_jobs():
    """Manage batch prediction jobs."""
    try:
        # Check if batch job requested
        if request.args.get('run') == 'true':
            # Start batch prediction job
            predictive_modeling.run_batch_predictions()
            flash("Batch prediction job started successfully.", "success")
        
        # Get active models
        models = predictive_modeling.get_all_active_models()
        
        return render_template(
            'prediction/batch_jobs.html',
            title='Batch Prediction Jobs',
            models=models
        )
    except Exception as e:
        logger.exception(f"Error with batch prediction jobs: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.dashboard'))