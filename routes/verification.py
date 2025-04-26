"""
Routes for the content verification features of the CIVILIAN system.
Handles user submissions and verification results.
"""

import os
import logging
import uuid
import json
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, session
from werkzeug.utils import secure_filename
from sqlalchemy import desc

from app import db
from models import UserSubmission, VerificationResult, ContentType, VerificationType, VerificationStatus
from services.verification_service import VerificationService

logger = logging.getLogger(__name__)

# Create blueprint
verification_bp = Blueprint('verification', __name__, url_prefix='/verify')

# Initialize verification service
verification_service = VerificationService()

# Configure file upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'webm', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename, content_type):
    """Check if the file has an allowed extension."""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if content_type in ['image', 'text_image']:
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif content_type in ['video', 'text_video']:
        return ext in ALLOWED_VIDEO_EXTENSIONS
    
    return False

@verification_bp.route('/', methods=['GET'])
def index():
    """Main verification page."""
    # Get recent submissions (with pagination)
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    recent_submissions = UserSubmission.query.order_by(
        desc(UserSubmission.submitted_at)
    ).paginate(page=page, per_page=per_page)
    
    # Render the template
    return render_template(
        'verification/index.html',
        recent_submissions=recent_submissions
    )

@verification_bp.route('/submit', methods=['GET', 'POST'])
def submit_content():
    """Submit content for verification."""
    if request.method == 'GET':
        # Display the submission form
        return render_template('verification/submit.html')
    
    elif request.method == 'POST':
        try:
            # Extract form data
            title = request.form.get('title', '')
            description = request.form.get('description', '')
            content_type = request.form.get('content_type')
            text_content = request.form.get('text_content', '')
            source_url = request.form.get('source_url', '')
            
            # Validate content type
            if not content_type or content_type not in [ct.value for ct in ContentType]:
                flash('Invalid content type selected.', 'danger')
                return redirect(url_for('verification.submit_content'))
            
            # Handle file uploads
            media_path = None
            if content_type in ['image', 'text_image', 'video', 'text_video']:
                if 'media_file' not in request.files:
                    flash('No file uploaded.', 'danger')
                    return redirect(request.url)
                
                file = request.files['media_file']
                
                if file.filename == '':
                    flash('No file selected.', 'danger')
                    return redirect(request.url)
                
                if not allowed_file(file.filename, content_type):
                    flash('File type not allowed.', 'danger')
                    return redirect(request.url)
                
                # Generate a unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                
                # Create a subdirectory based on content type
                upload_subdir = os.path.join(UPLOAD_FOLDER, content_type)
                os.makedirs(upload_subdir, exist_ok=True)
                
                # Save the file
                file_path = os.path.join(upload_subdir, unique_filename)
                file.save(file_path)
                
                # Store relative path from the application root
                media_path = os.path.join('static/uploads', content_type, unique_filename)
            
            # Create submission record
            submission = UserSubmission(
                title=title,
                description=description,
                content_type=content_type,
                text_content=text_content,
                media_path=media_path,
                source_url=source_url,
                submitted_at=datetime.utcnow(),
                ip_address=request.remote_addr,
                # user_id would be set for logged-in users if needed
            )
            
            # Add metadata
            metadata = {
                "browser": request.user_agent.browser,
                "version": request.user_agent.version,
                "platform": request.user_agent.platform,
                "request_id": str(uuid.uuid4())
            }
            submission.set_meta_data(metadata)
            
            # Save to database
            db.session.add(submission)
            db.session.commit()
            
            # Start verification process
            # This could be done asynchronously in a real system
            logger.info(f"Starting verification for submission ID {submission.id}")
            verification_result = verification_service.verify_submission(submission.id)
            
            # Redirect to results page
            return redirect(url_for('verification.view_results', submission_id=submission.id))
        
        except Exception as e:
            logger.error(f"Error processing submission: {e}")
            db.session.rollback()
            flash(f"Error processing your submission: {str(e)}", 'danger')
            return redirect(url_for('verification.submit_content'))

@verification_bp.route('/submission/<int:submission_id>', methods=['GET'])
def view_results(submission_id):
    """View verification results for a submission."""
    # Get the submission
    submission = UserSubmission.query.get_or_404(submission_id)
    
    # Get verification results
    verification_results = VerificationResult.query.filter_by(
        submission_id=submission_id
    ).all()
    
    # Render the template
    return render_template(
        'verification/results.html',
        submission=submission,
        verification_results=verification_results
    )

@verification_bp.route('/api/submit', methods=['POST'])
def api_submit_content():
    """API endpoint for submitting content for verification."""
    try:
        # Extract JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['content_type']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract fields
        content_type = data.get('content_type')
        title = data.get('title', '')
        description = data.get('description', '')
        text_content = data.get('text_content', '')
        source_url = data.get('source_url', '')
        
        # Validate content type
        if content_type not in [ct.value for ct in ContentType]:
            return jsonify({
                'success': False,
                'error': 'Invalid content type'
            }), 400
        
        # Handle media for API submissions
        media_path = None
        if 'media_data' in data and data['media_data']:
            # For API-based submissions, media would be handled differently
            # This would need to be implemented for a production system
            pass
        
        # Create submission record
        submission = UserSubmission(
            title=title,
            description=description,
            content_type=content_type,
            text_content=text_content,
            media_path=media_path,
            source_url=source_url,
            submitted_at=datetime.utcnow(),
            ip_address=request.remote_addr,
        )
        
        # Add metadata
        metadata = {
            "api_request": True,
            "request_id": str(uuid.uuid4())
        }
        submission.set_meta_data(metadata)
        
        # Save to database
        db.session.add(submission)
        db.session.commit()
        
        # Start verification process
        # This should be done asynchronously in a real system
        verification_result = verification_service.verify_submission(submission.id)
        
        # Return the submission ID and status
        return jsonify({
            'success': True,
            'submission_id': submission.id,
            'message': 'Content submitted for verification',
            'results_url': url_for('verification.view_results', submission_id=submission.id, _external=True)
        })
    
    except Exception as e:
        logger.error(f"API error processing submission: {e}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@verification_bp.route('/api/results/<int:submission_id>', methods=['GET'])
def api_get_results(submission_id):
    """API endpoint to get verification results for a submission."""
    try:
        # Get the submission
        submission = UserSubmission.query.get(submission_id)
        if not submission:
            return jsonify({
                'success': False,
                'error': f'Submission with ID {submission_id} not found'
            }), 404
        
        # Get verification results
        verification_results = VerificationResult.query.filter_by(
            submission_id=submission_id
        ).all()
        
        # Format the results
        results = []
        for result in verification_results:
            results.append({
                'id': result.id,
                'verification_type': result.verification_type,
                'status': result.status,
                'confidence_score': result.confidence_score,
                'summary': result.result_summary,
                'completed_at': result.completed_at.isoformat() if result.completed_at else None
            })
        
        # Return the results
        return jsonify({
            'success': True,
            'submission_id': submission_id,
            'submission_time': submission.submitted_at.isoformat(),
            'content_type': submission.content_type,
            'title': submission.title,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"API error retrieving results: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500