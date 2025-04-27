"""
API Credentials routes for the CIVILIAN system.
This module provides routes for managing API credentials for external services.
"""

import logging
import json
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from sqlalchemy.exc import SQLAlchemyError

from app import app, db
from models import SystemCredential, SystemLog
from services.api_credential_manager import APICredentialManager

logger = logging.getLogger(__name__)

# Create blueprint
api_credentials_bp = Blueprint('api_credentials', __name__, url_prefix='/api-credentials')

@api_credentials_bp.route('/')
@login_required
def index():
    """Display API credential management interface."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home.index'))
        
    # Get credential status
    credential_status = APICredentialManager.get_all_credential_status()
    
    # Get credential requirements
    credential_requirements = {}
    for cred_type, fields in APICredentialManager.REQUIRED_CREDENTIALS.items():
        credential_requirements[cred_type] = fields
    
    # Get recent credential-related logs
    logs = SystemLog.query.filter_by(
        component='api_credential_manager'
    ).order_by(
        SystemLog.timestamp.desc()
    ).limit(10).all()
    
    return render_template(
        'api_credentials/index.html',
        credential_status=credential_status,
        credential_requirements=credential_requirements,
        logs=logs,
        title='API Credential Management'
    )

@api_credentials_bp.route('/test', methods=['POST'])
@login_required
def test_credentials():
    """Test API credentials."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get request data
    data = request.json
    credential_type = data.get('credential_type')
    credentials = data.get('credentials')
    
    if not credential_type or not credentials:
        return jsonify({'error': 'Missing required fields'}), 400
        
    # Test the credentials
    is_valid, message = APICredentialManager.test_credentials(credential_type, credentials)
    
    return jsonify({
        'valid': is_valid,
        'message': message
    })

@api_credentials_bp.route('/save', methods=['POST'])
@login_required
def save_credentials():
    """Save API credentials."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get request data
    data = request.json
    credential_type = data.get('credential_type')
    credentials = data.get('credentials')
    
    if not credential_type or not credentials:
        return jsonify({'error': 'Missing required fields'}), 400
        
    # Save the credentials
    success = APICredentialManager.save_credentials(credential_type, credentials)
    
    if success:
        return jsonify({'success': True, 'message': 'Credentials saved successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to save credentials'}), 500

@api_credentials_bp.route('/delete', methods=['POST'])
@login_required
def delete_credentials():
    """Delete API credentials."""
    # Check if user has admin privileges
    if not current_user.role or current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    # Get request data
    data = request.json
    credential_type = data.get('credential_type')
    
    if not credential_type:
        return jsonify({'error': 'Missing credential type'}), 400
        
    # Delete the credentials
    success = APICredentialManager.delete_credentials(credential_type)
    
    if success:
        return jsonify({'success': True, 'message': 'Credentials deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to delete credentials'}), 500

# Blueprint will be registered in app.py