"""
Home route for the CIVILIAN system.
This module handles the homepage and redirects to the dashboard.
"""

from flask import Blueprint, render_template, redirect, url_for

# Create blueprint with a unique name
home_bp = Blueprint('home_bp', __name__)

@home_bp.route('/')
def index():
    """Render the CIVILIAN homepage or redirect to dashboard."""
    return render_template('index.html')

@home_bp.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')