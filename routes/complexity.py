import logging
import json
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user

from app import app, db
from models import DetectedNarrative, User
from services.complexity_analyzer import ComplexityAnalyzer
from services.complexity_scheduler import ComplexityScheduler
from utils.app_context import ensure_app_context

# Initialize services
complexity_analyzer = ComplexityAnalyzer()
complexity_scheduler = ComplexityScheduler()

# Start the scheduler
complexity_scheduler.start()

logger = logging.getLogger(__name__)

# Create Blueprint
complexity_bp = Blueprint('complexity', __name__)

@complexity_bp.route('/complexity/analyze/<int:narrative_id>', methods=['POST'])
@login_required
def analyze_narrative(narrative_id):
    """
    Analyze complexity for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        Analysis results in JSON format
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Run the analysis
        result = complexity_scheduler.run_single_analysis(narrative_id)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in complexity analysis endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/view/<int:narrative_id>', methods=['GET'])
@login_required
def view_complexity(narrative_id):
    """
    View complexity analysis for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to view
        
    Returns:
        HTML page with complexity analysis
    """
    try:
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return render_template('error.html', message=f"Narrative with ID {narrative_id} not found"), 404
        
        # Extract complexity data from narrative metadata
        complexity_data = {}
        if narrative.meta_data:
            try:
                metadata = json.loads(narrative.meta_data)
                complexity_data = metadata.get('complexity_analysis', {})
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse metadata for narrative {narrative_id}")
        
        # Check if we have complexity data
        has_complexity_data = bool(complexity_data and 'overall_complexity_score' in complexity_data)
        
        return render_template(
            'complexity/view.html', 
            narrative=narrative, 
            complexity_data=complexity_data,
            has_complexity_data=has_complexity_data
        )
        
    except Exception as e:
        logger.error(f"Error in view complexity endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/batch', methods=['POST'])
@login_required
def run_batch_analysis():
    """
    Run batch complexity analysis on recent narratives.
    
    Returns:
        Results summary in JSON format
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get parameters from request
        days = request.json.get('days', 7)
        limit = request.json.get('limit', 50)
        
        # Validate parameters
        if not isinstance(days, int) or days < 1 or days > 30:
            return jsonify({"error": "Invalid 'days' parameter. Must be an integer between 1 and 30"}), 400
            
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return jsonify({"error": "Invalid 'limit' parameter. Must be an integer between 1 and 100"}), 400
        
        # Run batch analysis
        result = complexity_analyzer.batch_analyze_recent_narratives(days=days, limit=limit)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in batch analysis endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/dashboard', methods=['GET'])
@login_required
def complexity_dashboard():
    """
    Display a dashboard of narrative complexity metrics.
    
    Returns:
        HTML page with complexity dashboard
    """
    try:
        # Get recently analyzed narratives
        narratives_with_complexity = []
        
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.status == 'active'
        ).order_by(DetectedNarrative.last_updated.desc()).limit(50).all()
        
        for narrative in narratives:
            complexity_data = {}
            if narrative.meta_data:
                try:
                    metadata = json.loads(narrative.meta_data)
                    complexity_data = metadata.get('complexity_analysis', {})
                except (json.JSONDecodeError, TypeError):
                    pass
            
            if complexity_data and 'overall_complexity_score' in complexity_data:
                narratives_with_complexity.append({
                    'id': narrative.id,
                    'title': narrative.title,
                    'status': narrative.status,
                    'last_updated': narrative.last_updated,
                    'overall_score': complexity_data.get('overall_complexity_score'),
                    'linguistic_score': complexity_data.get('linguistic_complexity', {}).get('score'),
                    'logical_score': complexity_data.get('logical_structure', {}).get('score'),
                    'rhetorical_score': complexity_data.get('rhetorical_techniques', {}).get('score'),
                    'emotional_score': complexity_data.get('emotional_manipulation', {}).get('score'),
                })
        
        return render_template(
            'complexity/dashboard.html',
            narratives=narratives_with_complexity
        )
        
    except Exception as e:
        logger.error(f"Error in complexity dashboard endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/compare', methods=['GET'])
@login_required
def compare_complexity():
    """
    Compare complexity metrics between narratives.
    
    Returns:
        HTML page with narrative complexity comparison
    """
    try:
        # Get narratives with complexity analysis
        narratives_with_complexity = []
        
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.status == 'active'
        ).order_by(DetectedNarrative.id.desc()).all()
        
        for narrative in narratives:
            if narrative.meta_data:
                try:
                    metadata = json.loads(narrative.meta_data)
                    complexity_data = metadata.get('complexity_analysis', {})
                    if complexity_data and 'overall_complexity_score' in complexity_data:
                        narratives_with_complexity.append(narrative)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return render_template(
            'complexity/compare.html',
            narratives=narratives_with_complexity
        )
        
    except Exception as e:
        logger.error(f"Error in complexity comparison endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/api/narrative/<int:narrative_id>', methods=['GET'])
@login_required
def get_narrative_complexity(narrative_id):
    """
    API endpoint to get complexity data for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to retrieve
        
    Returns:
        JSON with narrative complexity data
    """
    try:
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return jsonify({"error": f"Narrative with ID {narrative_id} not found"}), 404
        
        # Extract complexity data from narrative metadata
        complexity_data = {}
        if narrative.meta_data:
            try:
                metadata = json.loads(narrative.meta_data)
                complexity_data = metadata.get('complexity_analysis', {})
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse metadata for narrative {narrative_id}")
        
        # Check if we have complexity data
        if not complexity_data or 'overall_complexity_score' not in complexity_data:
            return jsonify({"error": f"No complexity data available for narrative {narrative_id}"}), 404
        
        # Prepare response data
        response_data = {
            'id': narrative.id,
            'title': narrative.title,
            'status': narrative.status,
            'description': narrative.description,
            'first_detected': narrative.first_detected.isoformat() if narrative.first_detected else None,
            'last_updated': narrative.last_updated.isoformat() if narrative.last_updated else None,
            'complexity_analysis': complexity_data
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in narrative complexity API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/api/trends', methods=['GET'])
@login_required
def get_complexity_trends():
    """
    API endpoint to get complexity trends over time.
    
    Returns:
        JSON with complexity trend data
    """
    try:
        # Get time period from request (default to last 30 days)
        days = request.args.get('days', 30, type=int)
        
        # Calculate time cutoff
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Query narratives created/updated within the time period
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.last_updated >= cutoff_date
        ).order_by(DetectedNarrative.last_updated.asc()).all()
        
        # Process the data
        trend_data = {
            'timestamps': [],
            'overall_complexity': [],
            'linguistic_complexity': [],
            'logical_structure': [],
            'rhetorical_techniques': [],
            'emotional_manipulation': []
        }
        
        for narrative in narratives:
            if not narrative.meta_data:
                continue
                
            try:
                metadata = json.loads(narrative.meta_data)
                complexity_data = metadata.get('complexity_analysis', {})
                
                if not complexity_data or 'overall_complexity_score' not in complexity_data:
                    continue
                
                # Add timestamp
                trend_data['timestamps'].append(narrative.last_updated.isoformat())
                
                # Add complexity scores
                trend_data['overall_complexity'].append(complexity_data.get('overall_complexity_score', 0))
                trend_data['linguistic_complexity'].append(complexity_data.get('linguistic_complexity', {}).get('score', 0))
                trend_data['logical_structure'].append(complexity_data.get('logical_structure', {}).get('score', 0))
                trend_data['rhetorical_techniques'].append(complexity_data.get('rhetorical_techniques', {}).get('score', 0))
                trend_data['emotional_manipulation'].append(complexity_data.get('emotional_manipulation', {}).get('score', 0))
                
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
        
        return jsonify(trend_data)
        
    except Exception as e:
        logger.error(f"Error in complexity trends API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Register blueprint
app.register_blueprint(complexity_bp)