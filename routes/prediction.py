"""
Prediction routes for the CIVILIAN system.

This module handles routes for predictive modeling, including narrative trajectory
forecasting, key factor analysis, anomaly detection, what-if scenario simulation,
counter-narrative effectiveness prediction, source reliability prediction,
category-based models, and narrative pattern detection.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from flask import Blueprint, render_template, request, jsonify, url_for, flash, redirect
from flask_login import login_required, current_user

from app import db
from models import (DetectedNarrative, NarrativeInstance, CounterMessage, 
                   InformationSource, NarrativeCategory, PredictionModel, 
                   PredictionModelRun, NarrativePrediction, NarrativePattern)
from services.predictive_modeling import PredictiveModeling
from utils.time_series import smooth_time_series, detect_seasonality, detect_trends
from utils.concurrency import run_in_thread

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

@prediction_bp.route('/counter-effectiveness/<int:counter_id>')
@login_required
def counter_effectiveness(counter_id):
    """Predict the effectiveness of a counter-narrative."""
    try:
        # Get parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Get counter-narrative
        counter = CounterMessage.query.get_or_404(counter_id)
        
        # Get target narrative
        narrative = DetectedNarrative.query.get_or_404(counter.narrative_id)
        
        # Predict counter-narrative effectiveness
        forecast = predictive_modeling.predict_counter_effectiveness(
            counter_id=counter_id,
            force_refresh=refresh
        )
        
        if not forecast.get('success', False):
            flash(forecast.get('error', 'Unknown error generating effectiveness prediction'), 'danger')
            return render_template(
                'prediction/counter_effectiveness.html',
                title=f'Counter-Narrative Effectiveness for "{counter.title}"',
                counter=counter,
                narrative=narrative,
                success=False,
                error=forecast.get('error', 'Unknown error')
            )
        
        return render_template(
            'prediction/counter_effectiveness.html',
            title=f'Counter-Narrative Effectiveness for "{counter.title}"',
            counter=counter,
            narrative=narrative,
            forecast=forecast,
            success=True
        )
    except Exception as e:
        logger.exception(f"Error predicting counter-narrative effectiveness for {counter_id}: {str(e)}")
        counter = CounterMessage.query.get(counter_id)
        narrative = DetectedNarrative.query.get(counter.narrative_id) if counter else None
        return render_template(
            'prediction/counter_effectiveness.html',
            title='Counter-Narrative Effectiveness Prediction Error',
            counter=counter,
            narrative=narrative,
            success=False,
            error=str(e)
        )

@prediction_bp.route('/source-reliability')
@login_required
def source_reliability():
    """Predict the reliability of information sources."""
    try:
        # Get parameters
        source_id = request.args.get('source_id', None)
        if source_id:
            source_id = int(source_id)
        days = int(request.args.get('days', '90'))
        horizon = int(request.args.get('horizon', '30'))
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Get all sources for selection
        sources = InformationSource.query.filter_by(status='active').order_by(InformationSource.name).all()
        
        # If no source specified, show selection form with top risk sources
        if not source_id:
            # Get top risk sources
            top_risk_sources = predictive_modeling.get_top_risk_sources(limit=10)
            
            return render_template(
                'prediction/source_reliability.html',
                title='Source Reliability Prediction',
                sources=sources,
                top_risk_sources=top_risk_sources,
                source_id=None
            )
        
        # Get source
        source = InformationSource.query.get_or_404(source_id)
        
        # Predict source reliability
        forecast = predictive_modeling.predict_source_reliability(
            source_id=source_id,
            days_history=days,
            days_horizon=horizon,
            force_refresh=refresh
        )
        
        if not forecast.get('success', False):
            flash(forecast.get('error', 'Unknown error generating reliability prediction'), 'danger')
            return redirect(url_for('prediction.source_reliability'))
        
        return render_template(
            'prediction/source_reliability.html',
            title=f'Source Reliability Prediction for "{source.name}"',
            source=source,
            forecast=forecast,
            days=days,
            horizon=horizon,
            source_id=source_id,
            source_name=source.name
        )
    except Exception as e:
        logger.exception(f"Error predicting source reliability: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.source_reliability'))

@prediction_bp.route('/narrative-patterns')
@login_required
def narrative_patterns():
    """Detect patterns in narrative propagation."""
    try:
        # Get parameters
        timeframe = int(request.args.get('timeframe', '90'))
        min_pattern_length = int(request.args.get('min_pattern_length', '5'))
        min_occurrences = int(request.args.get('min_occurrences', '2'))
        category = request.args.get('category', 'all')
        confidence_threshold = float(request.args.get('confidence_threshold', '0.7'))
        similarity_threshold = float(request.args.get('similarity_threshold', '0.7'))
        
        # Get categories for selection
        categories = NarrativeCategory.query.order_by(NarrativeCategory.name).all()
        
        # Detect patterns
        patterns_result = predictive_modeling.detect_narrative_patterns(
            timeframe=timeframe,
            min_pattern_length=min_pattern_length,
            min_occurrences=min_occurrences,
            category=None if category == 'all' else category,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold
        )
        
        # Parse results
        patterns = patterns_result.get('patterns', [])
        insights = patterns_result.get('insights', [])
        
        # Calculate statistics
        high_confidence_count = sum(1 for p in patterns if p.get('confidence', 0) >= 0.8)
        avg_pattern_length = sum(p.get('duration', 0) for p in patterns) / max(1, len(patterns))
        avg_occurrences = sum(p.get('occurrences', 0) for p in patterns) / max(1, len(patterns))
        
        # Get pattern type distribution
        pattern_types = []
        pattern_counts = []
        pattern_colors = []
        
        type_counts = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'Unknown')
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        # Define colors for pattern types
        type_colors = {
            'Cyclic': '#4285F4',
            'Seasonal': '#34A853',
            'Growth': '#FBBC05',
            'Decay': '#EA4335',
            'Spike': '#DB4437',
            'Step': '#673AB7',
            'Unknown': '#757575'
        }
        
        for pattern_type, count in type_counts.items():
            pattern_types.append(pattern_type)
            pattern_counts.append(count)
            pattern_colors.append(type_colors.get(pattern_type, '#757575'))
        
        return render_template(
            'prediction/narrative_patterns.html',
            title='Narrative Pattern Detection',
            timeframe=timeframe,
            min_pattern_length=min_pattern_length,
            min_occurrences=min_occurrences,
            category=category,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
            categories=categories,
            patterns=patterns,
            insights=insights,
            high_confidence_count=high_confidence_count,
            avg_pattern_length=avg_pattern_length,
            avg_occurrences=avg_occurrences,
            pattern_types=pattern_types,
            pattern_counts=pattern_counts,
            pattern_colors=pattern_colors
        )
    except Exception as e:
        logger.exception(f"Error detecting narrative patterns: {str(e)}")
        flash(f"Error detecting patterns: {str(e)}", 'danger')
        
        # Get categories for selection form
        categories = NarrativeCategory.query.order_by(NarrativeCategory.name).all()
        
        return render_template(
            'prediction/narrative_patterns.html',
            title='Narrative Pattern Detection',
            timeframe=90,
            min_pattern_length=5,
            min_occurrences=2,
            category='all',
            confidence_threshold=0.7,
            similarity_threshold=0.7,
            categories=categories,
            patterns=[],
            insights=[],
            high_confidence_count=0,
            avg_pattern_length=0,
            avg_occurrences=0,
            pattern_types=[],
            pattern_counts=[],
            pattern_colors=[]
        )

@prediction_bp.route('/api/pattern/<int:pattern_id>')
@login_required
def api_pattern_detail(pattern_id):
    """API endpoint to get pattern details."""
    try:
        # Get pattern
        pattern = NarrativePattern.query.get_or_404(pattern_id)
        
        # Get pattern details from predictive modeling service
        pattern_data = predictive_modeling.get_pattern_details(pattern_id)
        
        return jsonify({
            "success": True,
            "pattern": pattern_data
        })
    except Exception as e:
        logger.exception(f"Error getting pattern details for {pattern_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@prediction_bp.route('/pattern-forecast/<int:pattern_id>')
@login_required
def pattern_forecast(pattern_id):
    """Forecast the next occurrence of a pattern."""
    try:
        # Get pattern
        pattern = NarrativePattern.query.get_or_404(pattern_id)
        
        # Forecast next occurrence
        forecast_data = predictive_modeling.forecast_pattern_next_occurrence(pattern_id)
        
        if not forecast_data.get('success', False):
            flash(forecast_data.get('error', 'Unknown error forecasting pattern'), 'danger')
            return redirect(url_for('prediction.narrative_patterns'))
        
        return render_template(
            'prediction/pattern_forecast.html',
            title=f'Pattern Forecast for Pattern #{pattern_id}',
            pattern=pattern,
            forecast=forecast_data
        )
    except Exception as e:
        logger.exception(f"Error forecasting pattern {pattern_id}: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.narrative_patterns'))

@prediction_bp.route('/category-models')
@login_required
def category_models():
    """View and manage category-based prediction models."""
    try:
        # Get active and archived models
        active_models = PredictionModel.query.filter_by(status='active').all()
        archived_models = PredictionModel.query.filter_by(status='archived').all()
        
        # Get recent model runs
        recent_runs = PredictionModelRun.query.order_by(PredictionModelRun.run_date.desc()).limit(10).all()
        
        # Calculate model performance metrics
        categories = []
        category_accuracy = []
        
        model_types = []
        type_accuracy = []
        
        # Group by category
        category_models = {}
        for model in active_models:
            if model.category_id not in category_models:
                category_models[model.category_id] = []
            category_models[model.category_id].append(model)
        
        # Calculate average accuracy by category
        for category_id, models in category_models.items():
            category = NarrativeCategory.query.get(category_id)
            if category:
                categories.append(category.name)
                avg_accuracy = sum(m.accuracy for m in models) / len(models)
                category_accuracy.append(avg_accuracy)
        
        # Group by model type
        type_models = {}
        for model in active_models:
            if model.model_type not in type_models:
                type_models[model.model_type] = []
            type_models[model.model_type].append(model)
        
        # Calculate average accuracy by model type
        for model_type, models in type_models.items():
            model_types.append(model_type)
            avg_accuracy = sum(m.accuracy for m in models) / len(models)
            type_accuracy.append(avg_accuracy)
        
        return render_template(
            'prediction/category_models.html',
            title='Category-Based Prediction Models',
            active_models=active_models,
            archived_models=archived_models,
            recent_runs=recent_runs,
            categories=categories,
            category_accuracy=category_accuracy,
            model_types=model_types,
            type_accuracy=type_accuracy
        )
    except Exception as e:
        logger.exception(f"Error viewing category models: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.dashboard'))

@prediction_bp.route('/create-category-model', methods=['GET', 'POST'])
@login_required
def create_category_model():
    """Create a new category-based prediction model."""
    # Require admin or researcher role
    if current_user.role not in ['admin', 'researcher']:
        flash("You don't have permission to create prediction models", "danger")
        return redirect(url_for('prediction.category_models'))
    
    try:
        # Get categories for selection
        categories = NarrativeCategory.query.order_by(NarrativeCategory.name).all()
        
        if request.method == 'POST':
            # Get form data
            name = request.form.get('name')
            description = request.form.get('description')
            category_id = request.form.get('category_id')
            model_type = request.form.get('model_type')
            parameters = request.form.get('parameters')
            
            if not all([name, category_id, model_type]):
                flash("Please fill in all required fields", "danger")
                return render_template(
                    'prediction/create_category_model.html',
                    title='Create Category Model',
                    categories=categories
                )
            
            # Parse parameters as JSON
            try:
                parameters_dict = json.loads(parameters) if parameters else {}
            except json.JSONDecodeError:
                flash("Parameters must be valid JSON", "danger")
                return render_template(
                    'prediction/create_category_model.html',
                    title='Create Category Model',
                    categories=categories
                )
            
            # Create model
            model_data = predictive_modeling.create_category_model(
                name=name,
                description=description,
                category_id=int(category_id),
                model_type=model_type,
                parameters=parameters_dict
            )
            
            if model_data.get('success', False):
                flash(f"Model '{name}' created successfully", "success")
                return redirect(url_for('prediction.category_models'))
            else:
                flash(model_data.get('error', 'Unknown error creating model'), "danger")
        
        return render_template(
            'prediction/create_category_model.html',
            title='Create Category Model',
            categories=categories
        )
    except Exception as e:
        logger.exception(f"Error creating category model: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.category_models'))

@prediction_bp.route('/view-category-model/<int:model_id>')
@login_required
def view_category_model(model_id):
    """View details of a category-based prediction model."""
    try:
        # Get model
        model = PredictionModel.query.get_or_404(model_id)
        
        # Get recent runs for this model
        runs = PredictionModelRun.query.filter_by(model_id=model_id).order_by(PredictionModelRun.run_date.desc()).limit(5).all()
        
        # Get recent predictions from this model
        predictions = NarrativePrediction.query.filter_by(model_id=model_id).order_by(NarrativePrediction.created_at.desc()).limit(10).all()
        
        # Get prediction metrics
        metrics = predictive_modeling.get_category_model_metrics(model_id)
        
        return render_template(
            'prediction/view_category_model.html',
            title=f'Category Model: {model.name}',
            model=model,
            runs=runs,
            predictions=predictions,
            metrics=metrics
        )
    except Exception as e:
        logger.exception(f"Error viewing category model {model_id}: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.category_models'))

@prediction_bp.route('/run-category-model/<int:model_id>')
@login_required
def run_category_model(model_id):
    """Run a category-based prediction model."""
    try:
        # Get model
        model = PredictionModel.query.get_or_404(model_id)
        
        # Check if model is already training
        if model.is_training:
            flash(f"Model '{model.name}' is already running", "warning")
            return redirect(url_for('prediction.view_category_model', model_id=model_id))
        
        # Run model
        @run_in_thread
        def run_model_async():
            try:
                predictive_modeling.run_category_model(model_id)
            except Exception as e:
                logger.exception(f"Error running category model {model_id}: {str(e)}")
        
        run_model_async()
        
        flash(f"Model '{model.name}' run started. Check back soon for results.", "success")
        return redirect(url_for('prediction.view_category_model', model_id=model_id))
    except Exception as e:
        logger.exception(f"Error running category model {model_id}: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.category_models'))

@prediction_bp.route('/archive-category-model', methods=['POST'])
@login_required
def archive_category_model():
    """Archive a category-based prediction model."""
    # Require admin or researcher role
    if current_user.role not in ['admin', 'researcher']:
        flash("You don't have permission to archive prediction models", "danger")
        return redirect(url_for('prediction.category_models'))
    
    try:
        model_id = int(request.form.get('model_id'))
        
        # Archive model
        result = predictive_modeling.update_category_model_status(model_id, 'archived')
        
        if result.get('success', False):
            flash(f"Model archived successfully", "success")
        else:
            flash(result.get('error', 'Unknown error archiving model'), "danger")
        
        return redirect(url_for('prediction.category_models'))
    except Exception as e:
        logger.exception(f"Error archiving category model: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.category_models'))

@prediction_bp.route('/restore-category-model', methods=['POST'])
@login_required
def restore_category_model():
    """Restore an archived category-based prediction model."""
    # Require admin or researcher role
    if current_user.role not in ['admin', 'researcher']:
        flash("You don't have permission to restore prediction models", "danger")
        return redirect(url_for('prediction.category_models'))
    
    try:
        model_id = int(request.form.get('model_id'))
        
        # Restore model
        result = predictive_modeling.update_category_model_status(model_id, 'active')
        
        if result.get('success', False):
            flash(f"Model restored successfully", "success")
        else:
            flash(result.get('error', 'Unknown error restoring model'), "danger")
        
        return redirect(url_for('prediction.category_models'))
    except Exception as e:
        logger.exception(f"Error restoring category model: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.category_models'))

@prediction_bp.route('/delete-category-model', methods=['POST'])
@login_required
def delete_category_model():
    """Delete a category-based prediction model."""
    # Require admin or researcher role
    if current_user.role not in ['admin', 'researcher']:
        flash("You don't have permission to delete prediction models", "danger")
        return redirect(url_for('prediction.category_models'))
    
    try:
        model_id = int(request.form.get('model_id'))
        
        # Delete model
        result = predictive_modeling.delete_category_model(model_id)
        
        if result.get('success', False):
            flash(f"Model deleted successfully", "success")
        else:
            flash(result.get('error', 'Unknown error deleting model'), "danger")
        
        return redirect(url_for('prediction.category_models'))
    except Exception as e:
        logger.exception(f"Error deleting category model: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.category_models'))

@prediction_bp.route('/model-runs')
@login_required
def model_runs():
    """View all model runs."""
    try:
        # Get parameters
        model_id = request.args.get('model_id')
        status = request.args.get('status')
        
        # Build query
        query = PredictionModelRun.query
        
        if model_id:
            query = query.filter_by(model_id=int(model_id))
        
        if status:
            query = query.filter_by(status=status)
        
        # Get runs
        runs = query.order_by(PredictionModelRun.run_date.desc()).all()
        
        # Get models for filter
        models = PredictionModel.query.all()
        
        return render_template(
            'prediction/model_runs.html',
            title='Model Runs',
            runs=runs,
            models=models,
            selected_model_id=model_id,
            selected_status=status
        )
    except Exception as e:
        logger.exception(f"Error viewing model runs: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.dashboard'))

@prediction_bp.route('/view-model-run/<int:run_id>')
@login_required
def view_model_run(run_id):
    """View details of a model run."""
    try:
        # Get run
        run = PredictionModelRun.query.get_or_404(run_id)
        
        # Get model
        model = PredictionModel.query.get(run.model_id)
        
        # Get predictions from this run
        predictions = NarrativePrediction.query.filter_by(run_id=run_id).order_by(NarrativePrediction.created_at.desc()).all()
        
        return render_template(
            'prediction/view_model_run.html',
            title=f'Model Run: {run.id}',
            run=run,
            model=model,
            predictions=predictions
        )
    except Exception as e:
        logger.exception(f"Error viewing model run {run_id}: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('prediction.model_runs'))