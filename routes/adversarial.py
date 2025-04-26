"""
Adversarial routes for the CIVILIAN system.
This module handles adversarial content generation and training.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from replit_auth import require_login
from flask_login import current_user
from models import AdversarialContent, AdversarialEvaluation
from app import db

# Create blueprint
adversarial_bp = Blueprint('adversarial', __name__)

@adversarial_bp.route('/')
@require_login
def index():
    """
    Render the adversarial content management page.
    This route is protected by Replit Auth.
    """
    content = AdversarialContent.query.all()
    return render_template('adversarial/index.html', content=content)

@adversarial_bp.route('/content/<int:content_id>')
@require_login
def view_content(content_id):
    """
    Render the detailed view for a specific piece of adversarial content.
    This route is protected by Replit Auth.
    """
    content = AdversarialContent.query.get_or_404(content_id)
    evaluations = AdversarialEvaluation.query.filter_by(content_id=content_id).all()
    
    return render_template('adversarial/view.html', content=content, evaluations=evaluations)