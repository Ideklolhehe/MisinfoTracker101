"""
Decentralized Publishing routes for the CIVILIAN system.

These routes handle publishing narratives, counter-narratives, and other content to 
decentralized networks like IPFS.
"""

import logging
from flask import Blueprint, jsonify, request, render_template, redirect, url_for, current_app, flash
from flask_login import current_user, login_required
from sqlalchemy import desc

from app import db
from models import (
    DetectedNarrative, 
    CounterMessage, 
    PublishedContent,
    DataSource,
    EvidenceRecord
)
from services.decentralized_publishing import DecentralizedPublishingService

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
decentralized_bp = Blueprint('decentralized', __name__)

# Initialize publishing service
publishing_service = DecentralizedPublishingService()

@decentralized_bp.route('/decentralized')
@login_required
def index():
    """Main page for decentralized publishing features."""
    # Get recent publications
    publications = (
        PublishedContent.query
        .order_by(desc(PublishedContent.publication_date))
        .limit(10)
        .all()
    )
    
    # Get publishable content
    narratives = (
        DetectedNarrative.query
        .order_by(desc(DetectedNarrative.last_updated))
        .limit(10)
        .all()
    )
    
    counter_messages = (
        CounterMessage.query
        .filter(CounterMessage.status == 'approved')
        .order_by(desc(CounterMessage.last_updated))
        .limit(10)
        .all()
    )
    
    sources = (
        DataSource.query
        .filter(DataSource.is_active == True)
        .order_by(DataSource.name)
        .all()
    )
    
    evidence_records = (
        EvidenceRecord.query
        .filter(EvidenceRecord.verified == True)
        .order_by(desc(EvidenceRecord.capture_date))
        .limit(10)
        .all()
    )
    
    # IPFS node status
    ipfs_status = {
        "available": publishing_service.ipfs_available,
        "message": "IPFS service is available and connected" if publishing_service.ipfs_available else "IPFS service is not available"
    }
    
    return render_template(
        'decentralized/index.html',
        publications=publications,
        narratives=narratives,
        counter_messages=counter_messages,
        sources=sources,
        evidence_records=evidence_records,
        ipfs_status=ipfs_status
    )

@decentralized_bp.route('/decentralized/publications')
@login_required
def publications():
    """View all published content."""
    publications = (
        PublishedContent.query
        .order_by(desc(PublishedContent.publication_date))
        .all()
    )
    
    return render_template(
        'decentralized/publications.html',
        publications=publications
    )

@decentralized_bp.route('/decentralized/publish/narrative/<int:narrative_id>', methods=['GET', 'POST'])
@login_required
def publish_narrative(narrative_id):
    """Publish a narrative to decentralized networks."""
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    if request.method == 'POST':
        include_related = request.form.get('include_related', 'no') == 'yes'
        publisher_id = current_user.id
        
        result = publishing_service.publish_narrative_analysis(
            narrative_id=narrative_id, 
            include_related=include_related,
            publisher_id=publisher_id
        )
        
        if result and result.get('success'):
            flash('Narrative published successfully!', 'success')
            return redirect(url_for('decentralized.publication_detail', publication_id=result.get('publication_id')))
        else:
            flash('Failed to publish narrative. Please check IPFS connection.', 'danger')
            return redirect(url_for('decentralized.publish_narrative', narrative_id=narrative_id))
    
    return render_template(
        'decentralized/publish_narrative.html',
        narrative=narrative
    )

@decentralized_bp.route('/decentralized/publish/counter/<int:counter_id>', methods=['GET', 'POST'])
@login_required
def publish_counter(counter_id):
    """Publish a counter-narrative to decentralized networks."""
    counter = CounterMessage.query.get_or_404(counter_id)
    
    # Only allow publishing approved counter-narratives
    if counter.status != 'approved':
        flash('Only approved counter-narratives can be published.', 'warning')
        return redirect(url_for('decentralized.index'))
    
    if request.method == 'POST':
        publisher_id = current_user.id
        
        result = publishing_service.publish_counter_narrative(
            counter_id=counter_id,
            publisher_id=publisher_id
        )
        
        if result and result.get('success'):
            flash('Counter-narrative published successfully!', 'success')
            return redirect(url_for('decentralized.publication_detail', publication_id=result.get('publication_id')))
        else:
            flash('Failed to publish counter-narrative. Please check IPFS connection.', 'danger')
            return redirect(url_for('decentralized.publish_counter', counter_id=counter_id))
    
    return render_template(
        'decentralized/publish_counter.html',
        counter=counter
    )

@decentralized_bp.route('/decentralized/publish/evidence/<string:evidence_id>', methods=['GET', 'POST'])
@login_required
def publish_evidence(evidence_id):
    """Publish an evidence record to decentralized networks."""
    evidence = EvidenceRecord.query.get_or_404(evidence_id)
    
    if request.method == 'POST':
        publisher_id = current_user.id
        
        result = publishing_service.publish_evidence_record(
            evidence_id=evidence_id,
            publisher_id=publisher_id
        )
        
        if result and result.get('success'):
            flash('Evidence record published successfully!', 'success')
            return redirect(url_for('decentralized.publication_detail', publication_id=result.get('publication_id')))
        else:
            flash('Failed to publish evidence record. Please check IPFS connection.', 'danger')
            return redirect(url_for('decentralized.publish_evidence', evidence_id=evidence_id))
    
    return render_template(
        'decentralized/publish_evidence.html',
        evidence=evidence
    )

@decentralized_bp.route('/decentralized/publish/source/<int:source_id>', methods=['GET', 'POST'])
@login_required
def publish_source_reliability(source_id):
    """Publish source reliability analysis to decentralized networks."""
    source = DataSource.query.get_or_404(source_id)
    
    if request.method == 'POST':
        publisher_id = current_user.id
        
        result = publishing_service.publish_source_reliability_analysis(
            source_id=source_id,
            publisher_id=publisher_id
        )
        
        if result and result.get('success'):
            flash('Source reliability analysis published successfully!', 'success')
            return redirect(url_for('decentralized.publication_detail', publication_id=result.get('publication_id')))
        else:
            flash('Failed to publish source reliability analysis. Please check IPFS connection.', 'danger')
            return redirect(url_for('decentralized.publish_source_reliability', source_id=source_id))
    
    return render_template(
        'decentralized/publish_source_reliability.html',
        source=source
    )

@decentralized_bp.route('/decentralized/publication/<int:publication_id>')
@login_required
def publication_detail(publication_id):
    """View details of a published content."""
    publication = PublishedContent.query.get_or_404(publication_id)
    
    # Get the content from IPFS if available
    ipfs_content = None
    if publishing_service.ipfs_available:
        ipfs_content = publishing_service.get_publication(publication.ipfs_hash)
    
    return render_template(
        'decentralized/publication_detail.html',
        publication=publication,
        ipfs_content=ipfs_content,
        ipfs_available=publishing_service.ipfs_available
    )

# API endpoints
@decentralized_bp.route('/api/decentralized/publications', methods=['GET'])
def api_get_publications():
    """API endpoint to get recent publications."""
    limit = request.args.get('limit', 10, type=int)
    publications = publishing_service.get_recent_publications(limit=limit)
    return jsonify({"publications": publications})

@decentralized_bp.route('/api/decentralized/publish/narrative/<int:narrative_id>', methods=['POST'])
@login_required
def api_publish_narrative(narrative_id):
    """API endpoint to publish a narrative."""
    data = request.get_json() or {}
    include_related = data.get('include_related', False)
    publisher_id = current_user.id
    
    result = publishing_service.publish_narrative_analysis(
        narrative_id=narrative_id,
        include_related=include_related,
        publisher_id=publisher_id
    )
    
    if result and result.get('success'):
        return jsonify({"status": "success", "data": result}), 201
    else:
        return jsonify({"status": "error", "message": "Failed to publish narrative"}), 500

@decentralized_bp.route('/api/decentralized/publish/counter/<int:counter_id>', methods=['POST'])
@login_required
def api_publish_counter(counter_id):
    """API endpoint to publish a counter-narrative."""
    publisher_id = current_user.id
    
    result = publishing_service.publish_counter_narrative(
        counter_id=counter_id,
        publisher_id=publisher_id
    )
    
    if result and result.get('success'):
        return jsonify({"status": "success", "data": result}), 201
    else:
        return jsonify({"status": "error", "message": "Failed to publish counter-narrative"}), 500

@decentralized_bp.route('/api/decentralized/publish/evidence/<string:evidence_id>', methods=['POST'])
@login_required
def api_publish_evidence(evidence_id):
    """API endpoint to publish an evidence record."""
    publisher_id = current_user.id
    
    result = publishing_service.publish_evidence_record(
        evidence_id=evidence_id,
        publisher_id=publisher_id
    )
    
    if result and result.get('success'):
        return jsonify({"status": "success", "data": result}), 201
    else:
        return jsonify({"status": "error", "message": "Failed to publish evidence record"}), 500

@decentralized_bp.route('/api/decentralized/publish/source/<int:source_id>', methods=['POST'])
@login_required
def api_publish_source_reliability(source_id):
    """API endpoint to publish source reliability analysis."""
    publisher_id = current_user.id
    
    result = publishing_service.publish_source_reliability_analysis(
        source_id=source_id,
        publisher_id=publisher_id
    )
    
    if result and result.get('success'):
        return jsonify({"status": "success", "data": result}), 201
    else:
        return jsonify({"status": "error", "message": "Failed to publish source reliability analysis"}), 500

@decentralized_bp.route('/api/decentralized/content/<string:ipfs_hash>', methods=['GET'])
def api_get_content(ipfs_hash):
    """API endpoint to get content from IPFS."""
    content = publishing_service.get_publication(ipfs_hash)
    
    if content:
        return jsonify({"status": "success", "data": content}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to retrieve content"}), 404