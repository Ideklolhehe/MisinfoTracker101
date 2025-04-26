"""
Data Sources routes for the CIVILIAN system.
This module handles data source management and monitoring.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from replit_auth import require_login
from flask_login import current_user
from models import DataSource
from app import db
import json

# Create blueprint
data_sources_bp = Blueprint('data_sources', __name__)

@data_sources_bp.route('/')
@require_login
def index():
    """
    Render the data sources management page.
    This route is protected by Replit Auth.
    """
    sources = DataSource.query.all()
    return render_template('data_sources/index.html', sources=sources)

@data_sources_bp.route('/source/<int:source_id>')
@require_login
def view_source(source_id):
    """
    Render the detailed view for a specific data source.
    This route is protected by Replit Auth.
    """
    source = DataSource.query.get_or_404(source_id)
    config = json.loads(source.config)
    metadata = source.get_meta_data() if source.meta_data else {}
    
    return render_template('data_sources/view.html', source=source, config=config, metadata=metadata)

@data_sources_bp.route('/source/<int:source_id>/toggle', methods=['POST'])
@require_login
def toggle_source(source_id):
    """
    Toggle a data source's active state.
    This route is protected by Replit Auth.
    """
    source = DataSource.query.get_or_404(source_id)
    source.is_active = not source.is_active
    
    try:
        db.session.commit()
        flash(f"Source '{source.name}' {'activated' if source.is_active else 'deactivated'}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating source: {str(e)}", "danger")
    
    return redirect(url_for('data_sources.index'))