"""
Verification routes for the CIVILIAN system.
This module handles content verification and user submission functionality.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from replit_auth import require_login
from flask_login import current_user
from models import UserSubmission, VerificationResult, ContentType, VerificationType, VerificationStatus
from app import db
from services.verification_service import verify_content, analyze_image_content
import os
import uuid
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)

# Create blueprint
verification_bp = Blueprint('verification', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'static/uploads'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@verification_bp.route('/')
def index():
    """Render the verification submission form."""
    return render_template('verification/index.html')

@verification_bp.route('/submit', methods=['POST'])
def submit():
    """
    Handle user submission for verification.
    This accepts both text and media for analysis.
    """
    title = request.form.get('title', '')
    description = request.form.get('description', '')
    text_content = request.form.get('text_content', '')
    source_url = request.form.get('source_url', '')
    content_type = request.form.get('content_type', 'text')
    
    # Validate submission
    if not (text_content or 'media' in request.files):
        flash("Please provide either text content or upload media", "warning")
        return redirect(url_for('verification.index'))
    
    # Handle media upload if present
    media_path = None
    if 'media' in request.files and request.files['media'].filename:
        media_file = request.files['media']
        
        if media_file and allowed_file(media_file.filename):
            # Generate unique filename
            filename = secure_filename(media_file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Save file
            media_file.save(file_path)
            media_path = file_path
            
            # Determine content type based on media
            if text_content:
                content_type = ContentType.TEXT_IMAGE.value
            else:
                content_type = ContentType.IMAGE.value
    
    # Create user submission
    submission = UserSubmission(
        title=title,
        description=description,
        content_type=content_type,
        text_content=text_content,
        media_path=media_path,
        source_url=source_url,
        ip_address=request.remote_addr
    )
    
    # Associate with user if logged in
    if current_user.is_authenticated:
        submission.user_id = current_user.id
    
    # Save submission
    db.session.add(submission)
    db.session.commit()
    
    # Create verification tasks
    if content_type in [ContentType.TEXT.value, ContentType.TEXT_IMAGE.value]:
        # Create misinformation verification task
        misinfo_result = VerificationResult(
            submission_id=submission.id,
            verification_type=VerificationType.MISINFORMATION.value,
            status=VerificationStatus.PENDING.value
        )
        db.session.add(misinfo_result)
        
        # Create AI-generated content verification task
        ai_result = VerificationResult(
            submission_id=submission.id,
            verification_type=VerificationType.AI_GENERATED.value,
            status=VerificationStatus.PENDING.value
        )
        db.session.add(ai_result)
    
    if content_type in [ContentType.IMAGE.value, ContentType.TEXT_IMAGE.value]:
        # Create image authenticity verification task
        authenticity_result = VerificationResult(
            submission_id=submission.id,
            verification_type=VerificationType.AUTHENTICITY.value,
            status=VerificationStatus.PENDING.value
        )
        db.session.add(authenticity_result)
    
    db.session.commit()
    
    # Begin verification process asynchronously
    try:
        # Start verification process for the submission
        process_verification(submission.id)
    except Exception as e:
        logger.error(f"Error initiating verification: {e}")
    
    # Redirect to results page
    return redirect(url_for('verification.results', submission_id=submission.id))

@verification_bp.route('/results/<int:submission_id>')
def results(submission_id):
    """Display verification results for a submission."""
    submission = UserSubmission.query.get_or_404(submission_id)
    results = VerificationResult.query.filter_by(submission_id=submission_id).all()
    
    return render_template('verification/results.html', submission=submission, results=results)

@verification_bp.route('/api/status/<int:submission_id>')
def check_status(submission_id):
    """API endpoint to check verification status."""
    results = VerificationResult.query.filter_by(submission_id=submission_id).all()
    status_data = []
    
    for result in results:
        status_data.append({
            'id': result.id,
            'type': result.verification_type,
            'status': result.status,
            'confidence': result.confidence_score
        })
    
    return jsonify(status_data)

def process_verification(submission_id):
    """Process verification tasks for a submission."""
    submission = UserSubmission.query.get(submission_id)
    if not submission:
        logger.error(f"Submission {submission_id} not found")
        return
    
    results = VerificationResult.query.filter_by(submission_id=submission_id).all()
    
    for result in results:
        # Update status to processing
        result.status = VerificationStatus.PROCESSING.value
        db.session.commit()
        
        try:
            if result.verification_type == VerificationType.MISINFORMATION.value:
                # Verify text content for misinformation
                if submission.text_content:
                    verification_output = verify_content(
                        content=submission.text_content,
                        verification_type="misinformation"
                    )
                    
                    result.confidence_score = verification_output.get('confidence', 0.0)
                    result.result_summary = verification_output.get('summary', '')
                    result.evidence = verification_output.get('evidence', '')
                    result.completed_at = db.func.now()
                    result.status = VerificationStatus.COMPLETED.value
            
            elif result.verification_type == VerificationType.AI_GENERATED.value:
                # Verify text content for AI generation
                if submission.text_content:
                    verification_output = verify_content(
                        content=submission.text_content,
                        verification_type="ai_generated"
                    )
                    
                    result.confidence_score = verification_output.get('confidence', 0.0)
                    result.result_summary = verification_output.get('summary', '')
                    result.evidence = verification_output.get('evidence', '')
                    result.completed_at = db.func.now()
                    result.status = VerificationStatus.COMPLETED.value
            
            elif result.verification_type == VerificationType.AUTHENTICITY.value:
                # Verify image authenticity
                if submission.media_path:
                    verification_output = analyze_image_content(
                        image_path=submission.media_path
                    )
                    
                    result.confidence_score = verification_output.get('confidence', 0.0)
                    result.result_summary = verification_output.get('summary', '')
                    result.evidence = verification_output.get('evidence', '')
                    result.completed_at = db.func.now()
                    result.status = VerificationStatus.COMPLETED.value
            
            db.session.commit()
        
        except Exception as e:
            logger.error(f"Error processing verification {result.id}: {e}")
            result.status = VerificationStatus.FAILED.value
            result.result_summary = f"Verification failed: {str(e)}"
            db.session.commit()