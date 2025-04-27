"""
Routes for managing the CIVILIAN multi-agent system.
Provides endpoints for monitoring and controlling agents.
"""

import logging
import json
from flask import Blueprint, jsonify, request, render_template
from flask_login import login_required, current_user

from app import db
from models import SystemLog, User

logger = logging.getLogger(__name__)

# Create Blueprint
agents_bp = Blueprint('agents', __name__, url_prefix='/agents')

# Global reference to the coordinator (will be set in main.py)
coordinator = None

def set_coordinator(coordinator_instance):
    """Set the global coordinator reference.
    
    Args:
        coordinator_instance: Instance of MultiAgentCoordinator
    """
    global coordinator
    coordinator = coordinator_instance

@agents_bp.route('/')
def agents_dashboard():
    """Dashboard for monitoring agents."""
    # Temporarily removed login check for development
    return render_template('agents/dashboard.html')

@agents_bp.route('/stats')
def agent_stats():
    """Get statistics for all agents."""
    # Temporarily removed login check for development
    
    if not coordinator:
        return jsonify({'error': 'Agent system not initialized'}), 500
    
    stats = coordinator.get_system_summary()
    
    return jsonify({
        'success': True,
        'system_status': stats
    })

@agents_bp.route('/start', methods=['POST'])
def start_agents():
    """Start all agents."""
    # Temporarily removed login check for development
    
    if not coordinator:
        return jsonify({'error': 'Agent system not initialized'}), 500
    
    try:
        coordinator.start_all_agents()
        
        # Log the action
        log = SystemLog(
            log_type='info',
            component='multi_agent_coordinator',
            message=f"Multi-agent system started by development user"
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Multi-agent system started successfully'
        })
    except Exception as e:
        logger.error(f"Error starting agents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/stop', methods=['POST'])
def stop_agents():
    """Stop all agents."""
    # Temporarily removed login check for development
    
    if not coordinator:
        return jsonify({'error': 'Agent system not initialized'}), 500
    
    try:
        coordinator.stop_all_agents()
        
        # Log the action
        log = SystemLog(
            log_type='info',
            component='multi_agent_coordinator',
            message=f"Multi-agent system stopped by development user"
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Multi-agent system stopped successfully'
        })
    except Exception as e:
        logger.error(f"Error stopping agents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/logs')
def agent_logs():
    """Get recent logs for all agents."""
    # Temporarily removed login check for development
    
    try:
        # Get logs for agent components
        logs = SystemLog.query.filter(
            SystemLog.component.in_(['analyzer_agent', 'detector_agent', 'counter_agent', 'multi_agent_coordinator'])
        ).order_by(SystemLog.timestamp.desc()).limit(100).all()
        
        log_data = [{
            'id': log.id,
            'timestamp': log.timestamp.isoformat(),
            'component': log.component,
            'log_type': log.log_type,
            'message': log.message,
            'meta_data': json.loads(log.meta_data) if log.meta_data else None
        } for log in logs]
        
        return jsonify({
            'success': True,
            'logs': log_data
        })
    except Exception as e:
        logger.error(f"Error getting agent logs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500