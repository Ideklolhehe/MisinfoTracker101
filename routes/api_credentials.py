"""
API Credentials route for the CIVILIAN system.
This module provides routes for managing and monitoring API credentials.
"""

import logging
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from sqlalchemy.exc import SQLAlchemyError

from app import app, db
from models import User, SystemLog
from services.api_credential_manager import APICredentialManager

logger = logging.getLogger(__name__)

# Create blueprint
api_credentials_bp = Blueprint('api_credentials', __name__, url_prefix='/api-credentials')

@api_credentials_bp.route('/')
@login_required
def index():
    """Display API credential status and management interface."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home.index'))
        
    # Get credential status and requirements
    status = APICredentialManager.get_all_credential_status()
    requirements = APICredentialManager.get_credential_requirements()
    
    # Get recent credential-related logs
    logs = SystemLog.query.filter_by(
        component='api_credentials'
    ).order_by(
        SystemLog.timestamp.desc()
    ).limit(10).all()
    
    return render_template(
        'api_credentials/index.html',
        status=status,
        requirements=requirements,
        logs=logs,
        title='API Credentials'
    )

@api_credentials_bp.route('/status')
@login_required
def status():
    """Get the status of all API credentials as JSON."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    status = APICredentialManager.get_all_credential_status()
    return jsonify(status)

@api_credentials_bp.route('/check/<credential_type>')
@login_required
def check_credential(credential_type):
    """Check if specific credentials are available."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    if credential_type not in APICredentialManager.CREDENTIAL_TYPES:
        return jsonify({'error': 'Invalid credential type'}), 400
        
    is_available = APICredentialManager.are_credentials_complete(credential_type)
    status = 'available' if is_available else 'missing'
    
    return jsonify({
        'credential_type': credential_type,
        'status': status,
        'variables': APICredentialManager.CREDENTIAL_TYPES[credential_type]
    })

# Register blueprint with Flask app
app.register_blueprint(api_credentials_bp)