"""
Dashboard routes for the CIVILIAN system.
This module handles the dashboard functionality and analytics.
"""

from flask import Blueprint, render_template, redirect, url_for, flash
from replit_auth import require_login
from flask_login import current_user

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@dashboard_bp.route('/')
@require_login
def index():
    """
    Render the dashboard main page.
    This route is protected by Replit Auth.
    """
    return render_template('dashboard/index.html')