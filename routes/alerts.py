"""
Routes for misinformation alert system.
"""

import logging
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, abort, redirect, url_for, flash
from flask_login import login_required, current_user

from app import db
from models import DetectedNarrative, SystemLog
from services.alert_system import AlertSystem, AlertPriority, MisinformationEvent, Alert
from utils.app_context import ensure_app_context
from utils.sms_service import sms_service

logger = logging.getLogger(__name__)

# Initialize services
alert_system = AlertSystem()

# Create Blueprint
alerts_bp = Blueprint('alerts', __name__)

@alerts_bp.route('/alerts/dashboard', methods=['GET'])
@login_required
def alerts_dashboard():
    """
    Display alert system dashboard.
    
    Returns:
        HTML page with alert dashboard
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'moderator']:
            abort(403, "Insufficient privileges to access alert dashboard")
        
        # Get alerts
        recent_alerts = alert_system.get_recent_alerts()
        critical_alerts = alert_system.get_critical_alerts()
        alert_counts = alert_system.get_alert_count_by_priority()
        
        # Get narratives with recent alerts
        narrative_ids = []
        for alert in recent_alerts:
            try:
                narrative_id = int(alert['event_id'])
                if narrative_id not in narrative_ids:
                    narrative_ids.append(narrative_id)
            except (ValueError, TypeError):
                continue
        
        narratives = {}
        if narrative_ids:
            for narrative in DetectedNarrative.query.filter(DetectedNarrative.id.in_(narrative_ids)).all():
                narratives[narrative.id] = narrative
        
        return render_template(
            'alerts/dashboard.html',
            recent_alerts=recent_alerts,
            critical_alerts=critical_alerts,
            alert_counts=alert_counts,
            narratives=narratives
        )
        
    except Exception as e:
        logger.error(f"Error in alerts dashboard endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@alerts_bp.route('/alerts/api/recent', methods=['GET'])
@login_required
def api_get_recent_alerts():
    """
    API endpoint to get recent alerts.
    
    Returns:
        JSON with recent alerts
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'moderator']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get limit parameter (default: 20)
        limit = request.args.get('limit', 20, type=int)
        
        # Get alerts
        recent_alerts = alert_system.get_recent_alerts(limit)
        
        return jsonify({
            'count': len(recent_alerts),
            'alerts': recent_alerts
        })
        
    except Exception as e:
        logger.error(f"Error in recent alerts API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@alerts_bp.route('/alerts/api/priority/<priority>', methods=['GET'])
@login_required
def api_get_alerts_by_priority(priority):
    """
    API endpoint to get alerts by priority.
    
    Args:
        priority: Priority level (low, medium, high, critical)
        
    Returns:
        JSON with alerts of the specified priority
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'moderator']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Validate priority
        try:
            priority_enum = AlertPriority(priority.lower())
        except ValueError:
            return jsonify({"error": f"Invalid priority: {priority}"}), 400
        
        # Get alerts
        alerts = [a for a in alert_system.get_recent_alerts(100) if a['priority'] == priority]
        
        return jsonify({
            'priority': priority,
            'count': len(alerts),
            'alerts': alerts
        })
        
    except Exception as e:
        logger.error(f"Error in alerts by priority API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@alerts_bp.route('/alerts/narrative/<int:narrative_id>', methods=['GET'])
@login_required
def view_narrative_alerts(narrative_id):
    """
    View alerts for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative
        
    Returns:
        HTML page with narrative alerts
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'moderator']:
            abort(403, "Insufficient privileges to access narrative alerts")
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            abort(404, f"Narrative with ID {narrative_id} not found")
        
        # Get alerts for this narrative
        alerts = [a for a in alert_system.get_recent_alerts(100) if a['event_id'] == str(narrative_id)]
        
        # Get alert logs from system logs
        alert_logs = []
        logs = SystemLog.query.filter_by(
            log_type='alert',
            component='alert_system'
        ).order_by(SystemLog.created_at.desc()).limit(50).all()
        
        for log in logs:
            try:
                if log.meta_data:
                    meta_data = json.loads(log.meta_data)
                    if meta_data.get('event_id') == str(narrative_id):
                        alert_logs.append({
                            'id': log.id,
                            'message': log.message,
                            'timestamp': log.created_at.isoformat(),
                            'details': meta_data
                        })
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
        
        return render_template(
            'alerts/narrative.html',
            narrative=narrative,
            alerts=alerts,
            alert_logs=alert_logs
        )
        
    except Exception as e:
        logger.error(f"Error in narrative alerts endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@alerts_bp.route('/alerts/evaluate/<int:narrative_id>', methods=['POST'])
@login_required
def evaluate_narrative_alert(narrative_id):
    """
    Manually evaluate a narrative for alerts.
    
    Args:
        narrative_id: ID of the narrative to evaluate
        
    Returns:
        JSON with evaluation result
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return jsonify({"error": f"Narrative with ID {narrative_id} not found"}), 404
        
        # Evaluate the narrative
        alert = alert_system.evaluate_narrative(narrative_id)
        
        if alert:
            return jsonify({
                'success': True,
                'alert_triggered': True,
                'alert': alert.to_dict()
            })
        else:
            return jsonify({
                'success': True,
                'alert_triggered': False,
                'message': "No alert triggered based on current thresholds"
            })
        
    except Exception as e:
        logger.error(f"Error evaluating narrative for alerts: {e}")
        return jsonify({"error": str(e)}), 500

@alerts_bp.route('/alerts/settings', methods=['GET', 'POST'])
@login_required
def alert_settings():
    """
    View and update alert system settings.
    
    Returns:
        HTML page with alert settings
    """
    try:
        # Check if user has access
        if current_user.role != 'admin':
            abort(403, "Only administrators can access alert settings")
        
        if request.method == 'POST':
            # Update threshold settings
            new_thresholds = {}
            
            for key in alert_system.thresholds.keys():
                try:
                    value = float(request.form.get(key, ''))
                    if 0 <= value <= 1:
                        new_thresholds[key] = value
                except (ValueError, TypeError):
                    continue
            
            if new_thresholds:
                # Update thresholds
                alert_system.thresholds.update(new_thresholds)
                
                return render_template(
                    'alerts/settings.html',
                    thresholds=alert_system.thresholds,
                    success_message="Alert thresholds updated successfully"
                )
        
        # Show current settings
        return render_template(
            'alerts/settings.html',
            thresholds=alert_system.thresholds
        )
        
    except Exception as e:
        logger.error(f"Error in alert settings endpoint: {e}")
        return render_template('error.html', message=str(e)), 500