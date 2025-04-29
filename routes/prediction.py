"""
Routes for predictive modeling, forecasting, and what-if scenarios.
"""

import logging
import json
from typing import Dict, Any, List, Optional

from flask import Blueprint, render_template, request, jsonify, abort, redirect, url_for
from flask_login import login_required, current_user

from services.predictive_modeling import predictive_modeling_service
from models import DetectedNarrative
from app import db
from utils.metrics import time_operation

# Configure module logger
logger = logging.getLogger(__name__)

# Create blueprint
prediction_bp = Blueprint('prediction', __name__, url_prefix='/prediction')


@prediction_bp.route('/')
@login_required
def dashboard():
    """Predictive modeling dashboard."""
    # Get top narratives to display
    narratives = DetectedNarrative.query.filter(
        DetectedNarrative.status != 'debunked'
    ).order_by(
        DetectedNarrative.last_updated.desc()
    ).limit(10).all()
    
    return render_template(
        'prediction/dashboard.html',
        title='Predictive Modeling Dashboard',
        narratives=narratives
    )


@prediction_bp.route('/forecast/<int:narrative_id>')
@login_required
def forecast(narrative_id: int):
    """Display forecast for a narrative."""
    # Get narrative
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    # Get metric (default complexity)
    metric = request.args.get('metric', 'complexity')
    
    # Get model type
    model_type = request.args.get('model', 'prophet')
    
    # Force refresh parameter
    force_refresh = 'refresh' in request.args
    
    # Generate forecast
    with time_operation('generate_forecast'):
        forecast_data = predictive_modeling_service.forecast_narrative(
            narrative_id, metric, model_type, force_refresh
        )
    
    # Check for error
    if 'error' in forecast_data:
        return render_template(
            'prediction/error.html',
            title='Forecast Error',
            narrative=narrative,
            error=forecast_data['error']
        )
    
    return render_template(
        'prediction/view.html',
        title=f'Forecast: {narrative.title}',
        narrative=narrative,
        forecast=forecast_data,
        metric=metric,
        model_type=model_type
    )


@prediction_bp.route('/threshold/<int:narrative_id>')
@login_required
def threshold(narrative_id: int):
    """Display threshold projections for a narrative."""
    # Get narrative
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    # Get metric
    metric = request.args.get('metric', 'complexity')
    
    # Get model type
    model_type = request.args.get('model', 'prophet')
    
    # Get threshold value
    threshold_value = float(request.args.get('value', 0.75))
    
    # Get direction
    direction = request.args.get('direction', 'above')
    
    # Generate forecast first
    forecast_data = predictive_modeling_service.forecast_narrative(
        narrative_id, metric, model_type
    )
    
    # Check for error
    if 'error' in forecast_data:
        return render_template(
            'prediction/error.html',
            title='Threshold Projection Error',
            narrative=narrative,
            error=forecast_data['error']
        )
    
    # Calculate threshold crossing
    threshold_data = predictive_modeling_service.find_threshold_crossing(
        forecast_data, threshold_value, direction
    )
    
    return render_template(
        'prediction/threshold.html',
        title=f'Threshold Projection: {narrative.title}',
        narrative=narrative,
        forecast=forecast_data,
        threshold=threshold_data,
        metric=metric,
        model_type=model_type
    )


@prediction_bp.route('/factors/<int:narrative_id>')
@login_required
def key_factors(narrative_id: int):
    """Display key factors for a narrative."""
    # Get narrative
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    # Get metric
    metric = request.args.get('metric', 'complexity')
    
    # Analyze key factors
    factors_data = predictive_modeling_service.analyze_key_factors(
        narrative_id, metric
    )
    
    # Check for error
    if 'error' in factors_data:
        return render_template(
            'prediction/error.html',
            title='Key Factors Analysis Error',
            narrative=narrative,
            error=factors_data['error']
        )
    
    return render_template(
        'prediction/factors.html',
        title=f'Key Factors: {narrative.title}',
        narrative=narrative,
        factors=factors_data,
        metric=metric
    )


@prediction_bp.route('/multiple')
@login_required
def multiple_forecasts():
    """Display multiple forecasts for comparison."""
    # Get narrative IDs from request
    narrative_ids_str = request.args.get('ids', '')
    
    if not narrative_ids_str:
        # Show form to select narratives
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.status != 'debunked'
        ).order_by(
            DetectedNarrative.last_updated.desc()
        ).limit(20).all()
        
        return render_template(
            'prediction/select_multiple.html',
            title='Compare Multiple Narratives',
            narratives=narratives
        )
    
    # Parse narrative IDs
    try:
        narrative_ids = [int(id_str) for id_str in narrative_ids_str.split(',')]
    except ValueError:
        abort(400, 'Invalid narrative IDs')
    
    # Get metric
    metric = request.args.get('metric', 'complexity')
    
    # Get model type
    model_type = request.args.get('model', 'prophet')
    
    # Get narratives
    narratives = {}
    forecasts = {}
    
    for narrative_id in narrative_ids:
        narrative = DetectedNarrative.query.get(narrative_id)
        if narrative:
            narratives[narrative_id] = narrative
            
            # Generate forecast
            forecast_data = predictive_modeling_service.forecast_narrative(
                narrative_id, metric, model_type
            )
            
            if 'error' not in forecast_data:
                forecasts[narrative_id] = forecast_data
    
    return render_template(
        'prediction/multiple.html',
        title='Multiple Forecasts Comparison',
        narratives=narratives,
        forecasts=forecasts,
        metric=metric,
        model_type=model_type
    )


@prediction_bp.route('/whatif/<int:narrative_id>', methods=['GET', 'POST'])
@login_required
def what_if(narrative_id: int):
    """What-if scenario modeling for a narrative."""
    # Get narrative
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    # Get metric
    metric = request.args.get('metric', 'complexity')
    
    # Get model type
    model_type = request.args.get('model', 'prophet')
    
    if request.method == 'POST':
        # Process scenario inputs
        try:
            scenario_name = request.form.get('scenario_name', 'Custom Scenario')
            
            # Process interventions
            interventions = {}
            
            # Step intervention
            if 'step_date' in request.form and request.form.get('step_value'):
                interventions['step'] = {
                    'type': 'step',
                    'date': request.form.get('step_date'),
                    'value': float(request.form.get('step_value', 0))
                }
            
            # Trend intervention
            if 'trend_factor' in request.form:
                interventions['trend'] = {
                    'type': 'trend',
                    'factor': float(request.form.get('trend_factor', 1.0)),
                    'start_date': request.form.get('trend_start_date')
                }
            
            # Counter-message intervention
            if 'counter_date' in request.form:
                interventions['counter_message'] = {
                    'type': 'counter_message',
                    'date': request.form.get('counter_date'),
                    'impact': float(request.form.get('counter_impact', -0.1)),
                    'decay': float(request.form.get('counter_decay', 0.9))
                }
            
            # Generate scenario
            scenario_data = predictive_modeling_service.simulate_scenario(
                narrative_id, interventions, metric, model_type
            )
            
            # Check for error
            if 'error' in scenario_data:
                return render_template(
                    'prediction/error.html',
                    title='Scenario Error',
                    narrative=narrative,
                    error=scenario_data['error']
                )
            
            # Show results
            return render_template(
                'prediction/scenario_results.html',
                title=f'Scenario Results: {narrative.title}',
                narrative=narrative,
                scenario=scenario_data,
                scenario_name=scenario_name,
                metric=metric,
                model_type=model_type
            )
        except Exception as e:
            logger.error(f"Error processing scenario: {e}")
            abort(400, f"Invalid scenario parameters: {e}")
    
    # Show scenario form
    return render_template(
        'prediction/what_if.html',
        title=f'What-If Scenario: {narrative.title}',
        narrative=narrative,
        metric=metric,
        model_type=model_type
    )


@prediction_bp.route('/api/forecast/<int:narrative_id>')
@login_required
def api_forecast(narrative_id: int):
    """API endpoint for narrative forecasts."""
    # Get parameters
    metric = request.args.get('metric', 'complexity')
    model_type = request.args.get('model', 'prophet')
    force_refresh = 'refresh' in request.args
    
    # Generate forecast
    forecast_data = predictive_modeling_service.forecast_narrative(
        narrative_id, metric, model_type, force_refresh
    )
    
    return jsonify(forecast_data)


@prediction_bp.route('/api/threshold/<int:narrative_id>')
@login_required
def api_threshold(narrative_id: int):
    """API endpoint for threshold projections."""
    # Get parameters
    metric = request.args.get('metric', 'complexity')
    model_type = request.args.get('model', 'prophet')
    threshold_value = float(request.args.get('value', 0.75))
    direction = request.args.get('direction', 'above')
    
    # Generate forecast first
    forecast_data = predictive_modeling_service.forecast_narrative(
        narrative_id, metric, model_type
    )
    
    # Check for error
    if 'error' in forecast_data:
        return jsonify({'error': forecast_data['error']})
    
    # Calculate threshold crossing
    threshold_data = predictive_modeling_service.find_threshold_crossing(
        forecast_data, threshold_value, direction
    )
    
    return jsonify(threshold_data)


@prediction_bp.route('/api/factors/<int:narrative_id>')
@login_required
def api_factors(narrative_id: int):
    """API endpoint for key factors."""
    # Get parameters
    metric = request.args.get('metric', 'complexity')
    
    # Analyze key factors
    factors_data = predictive_modeling_service.analyze_key_factors(
        narrative_id, metric
    )
    
    return jsonify(factors_data)


@prediction_bp.route('/api/scenario/<int:narrative_id>', methods=['POST'])
@login_required
def api_scenario(narrative_id: int):
    """API endpoint for what-if scenarios."""
    try:
        # Get parameters from JSON body
        data = request.get_json(force=True)
        
        metric = data.get('metric', 'complexity')
        model_type = data.get('model_type', 'prophet')
        interventions = data.get('interventions', {})
        
        # Generate scenario
        scenario_data = predictive_modeling_service.simulate_scenario(
            narrative_id, interventions, metric, model_type
        )
        
        return jsonify(scenario_data)
    except Exception as e:
        logger.error(f"Error processing scenario API request: {e}")
        return jsonify({'error': str(e)}), 400