"""
Profile route for the CIVILIAN system.
This module demonstrates protected routes using Replit Auth.
"""

from flask import Blueprint, render_template, redirect, url_for, flash
from replit_auth import require_login
from flask_login import current_user

# Create blueprint
profile_bp = Blueprint('profile_bp', __name__)

@profile_bp.route('/profile')
@require_login
def profile():
    """
    Render the user profile page.
    This route is protected by Replit Auth.
    """
    return render_template('profile.html')