"""
Development-only authentication routes.
These routes should be disabled in production.
"""

import logging
from flask import Blueprint, redirect, url_for, flash
from flask_login import login_user, current_user

from models import User
from app import db

logger = logging.getLogger(__name__)

dev_auth_bp = Blueprint('dev_auth', __name__, url_prefix='/dev')

@dev_auth_bp.route('/login')
def dev_login():
    """Development-only login route.
    Logs in as 'developer' user without authentication.
    """
    # Check if already logged in
    if current_user.is_authenticated:
        return redirect(url_for('home.index'))
    
    # Get developer user
    dev_user = User.query.filter_by(username='developer').first()
    
    if not dev_user:
        # Create developer user if not exists
        dev_user = User(
            id='dev-user-1',
            username='developer',
            email='dev@example.com',
            role='admin'
        )
        db.session.add(dev_user)
        db.session.commit()
    
    # Log in as developer
    login_user(dev_user)
    logger.warning("Developer login used - NOT FOR PRODUCTION")
    
    return redirect(url_for('agents.agents_dashboard'))