"""
Routes for managing adversarial misinformation content for system training.

This module provides endpoints for generating, viewing, and managing
adversarial content used for training the CIVILIAN system.
"""

import logging
import json
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user

from app import db
from models import AdversarialContent, AdversarialEvaluation
from services.adversarial_service import AdversarialService

logger = logging.getLogger(__name__)

# Create blueprint
adversarial_bp = Blueprint('adversarial', __name__, url_prefix='/adversarial')
adversarial_service = AdversarialService()

@adversarial_bp.route('/')
@login_required
def index():
    """Render the adversarial content management page."""
    # Get all active adversarial content
    contents = db.session.query(AdversarialContent)\
        .filter_by(is_active=True, variant_of_id=None)\
        .order_by(AdversarialContent.generated_at.desc())\
        .all()
        
    # Get evaluation statistics
    stats = adversarial_service.get_evaluation_stats()
    
    # Get topic and misinfo type options for the generator form
    topic_options = adversarial_service.generator.topic_areas
    misinfo_options = adversarial_service.generator.misinfo_types
    
    return render_template('adversarial/index.html', 
                         contents=contents, 
                         stats=stats,
                         topic_options=topic_options,
                         misinfo_options=misinfo_options)

@adversarial_bp.route('/content/<int:content_id>')
@login_required
def view_content(content_id):
    """View details of a specific adversarial content piece."""
    content = adversarial_service.get_content_by_id(content_id)
    
    if not content:
        flash("Content not found", "danger")
        return redirect(url_for('adversarial.index'))
    
    # Get variants if any
    variants = content.variants.all() if content.variant_of_id is None else []
    
    # Get evaluations
    evaluations = content.evaluations.all()
    
    # Get parent if this is a variant
    parent = None
    if content.variant_of_id:
        parent = adversarial_service.get_content_by_id(content.variant_of_id)
    
    return render_template('adversarial/view_content.html', 
                         content=content,
                         variants=variants,
                         parent=parent, 
                         evaluations=evaluations)

@adversarial_bp.route('/generate', methods=['POST'])
@login_required
def generate_content():
    """Generate new adversarial content."""
    topic = request.form.get('topic')
    misinfo_type = request.form.get('misinfo_type')
    real_content = request.form.get('real_content', '')
    
    if not topic or not misinfo_type:
        flash("Topic and misinformation type are required", "danger")
        return redirect(url_for('adversarial.index'))
    
    try:
        content = adversarial_service.generate_training_content(
            topic=topic,
            misinfo_type=misinfo_type,
            real_content=real_content if real_content else None
        )
        
        flash(f"Successfully generated adversarial content: {content.title}", "success")
        return redirect(url_for('adversarial.view_content', content_id=content.id))
        
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        flash(f"Error generating content: {str(e)}", "danger")
        return redirect(url_for('adversarial.index'))

@adversarial_bp.route('/batch', methods=['POST'])
@login_required
def generate_batch():
    """Generate a batch of adversarial content."""
    batch_size = int(request.form.get('batch_size', 5))
    topics = request.form.getlist('topics')
    misinfo_types = request.form.getlist('misinfo_types')
    
    if batch_size < 1 or batch_size > 20:
        flash("Batch size must be between 1 and 20", "danger")
        return redirect(url_for('adversarial.index'))
    
    try:
        contents = adversarial_service.generate_content_batch(
            batch_size=batch_size,
            topics=topics if topics else None,
            types=misinfo_types if misinfo_types else None
        )
        
        flash(f"Successfully generated {len(contents)} adversarial content items", "success")
        return redirect(url_for('adversarial.index'))
        
    except Exception as e:
        logger.error(f"Error generating batch content: {e}")
        flash(f"Error generating batch content: {str(e)}", "danger")
        return redirect(url_for('adversarial.index'))

@adversarial_bp.route('/variants/<int:content_id>', methods=['POST'])
@login_required
def generate_variants(content_id):
    """Generate variants of existing content."""
    num_variants = int(request.form.get('num_variants', 3))
    
    if num_variants < 1 or num_variants > 5:
        flash("Number of variants must be between 1 and 5", "danger")
        return redirect(url_for('adversarial.view_content', content_id=content_id))
    
    try:
        variants = adversarial_service.generate_variants(
            content_id=content_id,
            num_variants=num_variants
        )
        
        if variants:
            flash(f"Successfully generated {len(variants)} content variants", "success")
        else:
            flash("No variants could be generated", "warning")
            
        return redirect(url_for('adversarial.view_content', content_id=content_id))
        
    except Exception as e:
        logger.error(f"Error generating variants: {e}")
        flash(f"Error generating variants: {str(e)}", "danger")
        return redirect(url_for('adversarial.view_content', content_id=content_id))

@adversarial_bp.route('/evaluate/<int:content_id>', methods=['POST'])
@login_required
def evaluate_content(content_id):
    """Evaluate adversarial content."""
    detector_version = request.form.get('detector_version', 'current')
    correct_detection = request.form.get('correct_detection') == 'true'
    confidence_score = float(request.form.get('confidence_score', 0.0))
    notes = request.form.get('notes', '')
    
    try:
        evaluation = adversarial_service.evaluate_content(
            content_id=content_id,
            detector_version=detector_version,
            correct_detection=correct_detection,
            confidence_score=confidence_score,
            user_id=current_user.id,
            notes=notes
        )
        
        flash("Evaluation recorded successfully", "success")
        return redirect(url_for('adversarial.view_content', content_id=content_id))
        
    except Exception as e:
        logger.error(f"Error recording evaluation: {e}")
        flash(f"Error recording evaluation: {str(e)}", "danger")
        return redirect(url_for('adversarial.view_content', content_id=content_id))

@adversarial_bp.route('/deactivate/<int:content_id>', methods=['POST'])
@login_required
def deactivate_content(content_id):
    """Deactivate adversarial content (don't use for training)."""
    try:
        if adversarial_service.deactivate_content(content_id):
            flash("Content deactivated successfully", "success")
        else:
            flash("Failed to deactivate content", "danger")
            
        return redirect(url_for('adversarial.index'))
        
    except Exception as e:
        logger.error(f"Error deactivating content: {e}")
        flash(f"Error deactivating content: {str(e)}", "danger")
        return redirect(url_for('adversarial.index'))

@adversarial_bp.route('/api/content/<int:content_id>')
@login_required
def api_get_content(content_id):
    """API endpoint to get content details."""
    content = adversarial_service.get_content_by_id(content_id)
    
    if not content:
        return jsonify({"error": "Content not found"}), 404
    
    # Format data for API response
    result = {
        "id": content.id,
        "title": content.title,
        "content": content.content,
        "topic": content.topic,
        "misinfo_type": content.misinfo_type,
        "generation_method": content.generation_method,
        "generated_at": content.generated_at.isoformat(),
        "is_active": content.is_active,
        "metadata": content.get_meta_data()
    }
    
    # Add variant data if applicable
    if content.variant_of_id:
        result["variant_of_id"] = content.variant_of_id
    
    return jsonify(result)

@adversarial_bp.route('/api/stats')
@login_required
def api_get_stats():
    """API endpoint to get evaluation statistics."""
    stats = adversarial_service.get_evaluation_stats()
    return jsonify(stats)

@adversarial_bp.route('/api/content/topic/<topic>')
@login_required
def api_get_by_topic(topic):
    """API endpoint to get content by topic."""
    limit = request.args.get('limit', 10, type=int)
    contents = adversarial_service.get_content_by_topic(topic, limit)
    
    results = [{
        "id": c.id,
        "title": c.title,
        "topic": c.topic,
        "misinfo_type": c.misinfo_type,
        "generated_at": c.generated_at.isoformat()
    } for c in contents]
    
    return jsonify(results)

@adversarial_bp.route('/api/training-data')
@login_required
def api_get_training_data():
    """API endpoint to get content for training."""
    limit = request.args.get('limit', 100, type=int)
    contents = adversarial_service.get_content_for_training(limit)
    
    results = [{
        "id": c.id,
        "title": c.title,
        "content": c.content,
        "topic": c.topic,
        "misinfo_type": c.misinfo_type,
        "metadata": c.get_meta_data()
    } for c in contents]
    
    return jsonify(results)