import logging
import json
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify, render_template, send_file, Response, make_response, current_app
from flask_login import login_required, current_user
from io import BytesIO

from app import app, db
from models import DetectedNarrative, User
from services.complexity_analyzer import ComplexityAnalyzer
from services.complexity_scheduler import ComplexityScheduler
from services.export_service import ComplexityReportExporter
from services.complexity_alerts import ComplexityAlertService
from services.complexity_predictor import ComplexityPredictor
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

@complexity_bp.route('/complexity/export/<int:narrative_id>/pdf', methods=['GET'])
@login_required
def export_complexity_to_pdf(narrative_id):
    """
    Export complexity analysis for a specific narrative to PDF.
    
    Args:
        narrative_id: ID of the narrative to export
        
    Returns:
        PDF file download
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get narrative data
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return jsonify({"error": f"Narrative with ID {narrative_id} not found"}), 404
        
        # Generate PDF
        pdf_data = ComplexityReportExporter.export_to_pdf(narrative_id)
        if not pdf_data:
            return jsonify({"error": "Failed to generate PDF report"}), 500
        
        # Prepare filename
        filename = f"narrative_{narrative_id}_complexity_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        # Return the file
        return send_file(
            pdf_data,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error exporting narrative {narrative_id} to PDF: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/export/<int:narrative_id>/csv', methods=['GET'])
@login_required
def export_complexity_to_csv(narrative_id):
    """
    Export complexity analysis for a specific narrative to CSV.
    
    Args:
        narrative_id: ID of the narrative to export
        
    Returns:
        CSV file download
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get narrative data
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return jsonify({"error": f"Narrative with ID {narrative_id} not found"}), 404
        
        # Generate CSV
        csv_data = ComplexityReportExporter.export_to_csv(narrative_id)
        if not csv_data:
            return jsonify({"error": "Failed to generate CSV report"}), 500
        
        # Prepare filename
        filename = f"narrative_{narrative_id}_complexity_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Return the file
        return send_file(
            csv_data,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error exporting narrative {narrative_id} to CSV: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/export/comparison/pdf', methods=['POST'])
@login_required
def export_comparison_to_pdf():
    """
    Export comparison of multiple narratives to PDF.
    
    Returns:
        PDF file download
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get narrative IDs from request
        narrative_ids = request.json.get('narrative_ids', [])
        if not narrative_ids or len(narrative_ids) < 2:
            return jsonify({"error": "At least two narrative IDs are required"}), 400
        
        # Generate PDF
        pdf_data = ComplexityReportExporter.export_comparison_to_pdf(narrative_ids)
        if not pdf_data:
            return jsonify({"error": "Failed to generate comparison PDF report"}), 500
        
        # Prepare filename
        narrative_id_str = '_'.join(str(nid) for nid in narrative_ids[:3])
        if len(narrative_ids) > 3:
            narrative_id_str += f"_plus_{len(narrative_ids) - 3}_more"
        
        filename = f"narrative_comparison_{narrative_id_str}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        # Return the file
        return send_file(
            pdf_data,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error exporting comparison to PDF: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/export/trends/csv', methods=['GET'])
@login_required
def export_trends_to_csv():
    """
    Export complexity trends to CSV.
    
    Returns:
        CSV file download
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get time period from request
        days = request.args.get('days', 30, type=int)
        
        # Generate CSV
        csv_data = ComplexityReportExporter.export_trends_to_csv(days)
        if not csv_data:
            return jsonify({"error": "Failed to generate trends CSV report"}), 500
        
        # Prepare filename
        filename = f"complexity_trends_{days}days_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Return the file
        return send_file(
            csv_data,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error exporting trends to CSV: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/trends', methods=['GET'])
@login_required
def complexity_trends():
    """
    Display time-series trends of narrative complexity.
    
    Returns:
        HTML page with complexity trends visualization
    """
    try:
        # Get recently analyzed narratives for initial stats
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
        
        # Get active alerts
        alerts = ComplexityAlertService.get_all_active_alerts()
        
        return render_template(
            'complexity/trends.html',
            narratives=narratives_with_complexity,
            alerts=alerts
        )
        
    except Exception as e:
        logger.error(f"Error in complexity trends endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/api/alerts', methods=['GET'])
@login_required
def get_complexity_alerts():
    """
    API endpoint to get active complexity alerts.
    
    Returns:
        JSON with alerts data
    """
    try:
        # Get time period from request (default to last 7 days for change alerts)
        days = request.args.get('days', 7, type=int)
        
        # Get alert type filter (optional)
        alert_type = request.args.get('type', None)
        
        if alert_type == 'high_complexity':
            alerts = ComplexityAlertService.check_high_complexity_alerts(days=1)
        elif alert_type == 'rapid_change':
            alerts = ComplexityAlertService.check_rapid_change_alerts(days=days)
        elif alert_type == 'coordinated':
            alerts = ComplexityAlertService.check_coordinated_narratives()
        else:
            # Get all alerts
            alerts = ComplexityAlertService.get_all_active_alerts()
        
        return jsonify(alerts)
        
    except Exception as e:
        logger.error(f"Error in complexity alerts API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/alerts', methods=['GET'])
@login_required
def complexity_alerts():
    """
    Display alerts for narrative complexity.
    
    Returns:
        HTML page with complexity alerts
    """
    try:
        # Get all active alerts
        alerts = ComplexityAlertService.get_all_active_alerts()
        
        # Get narratives for context
        narrative_ids = set()
        
        # Collect all narrative IDs from different alert types
        for alert_type, alert_list in alerts.items():
            if alert_type != 'total_count':
                for alert in alert_list:
                    if 'narrative_id' in alert:
                        narrative_ids.add(alert['narrative_id'])
                    elif 'narrative_ids' in alert:
                        narrative_ids.update(alert['narrative_ids'])
        
        # Get narrative details
        narratives = {}
        for narrative_id in narrative_ids:
            narrative = DetectedNarrative.query.get(narrative_id)
            if narrative:
                narratives[narrative_id] = {
                    'id': narrative.id,
                    'title': narrative.title,
                    'status': narrative.status,
                    'last_updated': narrative.last_updated
                }
        
        return render_template(
            'complexity/alerts.html',
            alerts=alerts,
            narratives=narratives
        )
        
    except Exception as e:
        logger.error(f"Error in complexity alerts endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/clusters', methods=['GET'])
@login_required
def narrative_clusters():
    """
    Display clusters of narratives with similar complexity patterns.
    
    Returns:
        HTML page with narrative clusters
    """
    try:
        # Get narratives with complexity analysis
        narratives_with_complexity = []
        
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.status == 'active'
        ).all()
        
        for narrative in narratives:
            if not narrative.meta_data:
                continue
                
            try:
                metadata = json.loads(narrative.meta_data)
                complexity_data = metadata.get('complexity_analysis', {})
                
                if not complexity_data or 'overall_complexity_score' not in complexity_data:
                    continue
                
                # Extract complexity profile
                narratives_with_complexity.append({
                    'id': narrative.id,
                    'title': narrative.title,
                    'status': narrative.status,
                    'last_updated': narrative.last_updated,
                    'overall_score': complexity_data.get('overall_complexity_score', 0),
                    'linguistic_score': complexity_data.get('linguistic_complexity', {}).get('score', 0),
                    'logical_score': complexity_data.get('logical_structure', {}).get('score', 0),
                    'rhetorical_score': complexity_data.get('rhetorical_techniques', {}).get('score', 0),
                    'emotional_score': complexity_data.get('emotional_manipulation', {}).get('score', 0),
                })
                
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        
        # Get narrative clusters
        clusters = ComplexityAlertService._cluster_similar_narratives(narratives_with_complexity)
        
        return render_template(
            'complexity/clusters.html',
            clusters=clusters,
            narratives_count=len(narratives_with_complexity)
        )
        
    except Exception as e:
        logger.error(f"Error in narrative clusters endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/api/counter-integration/<int:narrative_id>', methods=['GET'])
@login_required
def get_counter_recommendations(narrative_id):
    """
    API endpoint to get counter-narrative recommendations based on complexity analysis.
    
    Args:
        narrative_id: ID of the narrative to get recommendations for
        
    Returns:
        JSON with counter-narrative recommendations
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get narrative data
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
        
        # Generate counter-narrative recommendations based on complexity profile
        # This would normally call a method in a counter-narrative service
        # For now, we'll return a placeholder response
        recommendations = {
            'narrative_id': narrative_id,
            'complexity_profile': {
                'overall': complexity_data.get('overall_complexity_score'),
                'linguistic': complexity_data.get('linguistic_complexity', {}).get('score'),
                'logical': complexity_data.get('logical_structure', {}).get('score'),
                'rhetorical': complexity_data.get('rhetorical_techniques', {}).get('score'),
                'emotional': complexity_data.get('emotional_manipulation', {}).get('score')
            },
            'recommendations': []
        }
        
        # Add recommendations based on high dimension scores
        dimensions = [
            ('linguistic_complexity', 'Linguistic Complexity', 'Simplify language and explain complex terms.'),
            ('logical_structure', 'Logical Structure', 'Focus on clear, linear reasoning and address logical fallacies directly.'),
            ('rhetorical_techniques', 'Rhetorical Techniques', 'Avoid matching rhetorical flourish; instead use evidence-based, direct counters.'),
            ('emotional_manipulation', 'Emotional Manipulation', 'Acknowledge emotions but redirect to factual analysis and reasoned discussion.')
        ]
        
        for dim_key, dim_name, strategy in dimensions:
            dim_score = complexity_data.get(dim_key, {}).get('score', 0)
            if dim_score >= 7.0:
                recommendations['recommendations'].append({
                    'dimension': dim_key,
                    'dimension_name': dim_name,
                    'score': dim_score,
                    'priority': 'high' if dim_score >= 8.0 else 'medium',
                    'strategy': strategy,
                    'techniques': [
                        f"Counter high {dim_name.lower()} with clear, accessible narratives",
                        f"Focus on addressing the specific {dim_name.lower()} techniques identified"
                    ]
                })
        
        return jsonify(recommendations)
        
    except Exception as e:
        logger.error(f"Error in counter recommendations API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/api/predict/<int:narrative_id>', methods=['GET'])
@login_required
def predict_narrative_complexity(narrative_id):
    """
    API endpoint to predict future complexity for a narrative.
    
    Args:
        narrative_id: ID of the narrative to predict
        
    Returns:
        JSON with prediction results
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get prediction days from request
        days_ahead = request.args.get('days', 7, type=int)
        
        # Validate parameters
        if not isinstance(days_ahead, int) or days_ahead < 1 or days_ahead > 30:
            return jsonify({"error": "Invalid 'days' parameter. Must be an integer between 1 and 30"}), 400
        
        # Make prediction
        prediction = ComplexityPredictor.predict_narrative_evolution(narrative_id, days_ahead)
        
        if "error" in prediction:
            return jsonify({"error": prediction["error"]}), 400
            
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error in complexity prediction API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/api/similar-trajectories/<int:narrative_id>', methods=['GET'])
@login_required
def get_similar_trajectories(narrative_id):
    """
    API endpoint to find narratives with similar complexity trajectories.
    
    Args:
        narrative_id: ID of the narrative to compare with others
        
    Returns:
        JSON with similar narratives
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get similar narratives
        similar = ComplexityPredictor.get_similar_trajectories(narrative_id)
        
        if "error" in similar:
            return jsonify({"error": similar["error"]}), 400
            
        return jsonify(similar)
        
    except Exception as e:
        logger.error(f"Error in similar trajectories API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/api/trending', methods=['GET'])
@login_required
def get_trending_narratives():
    """
    API endpoint to get narratives with strong upward complexity trends.
    
    Returns:
        JSON with trending narratives
    """
    try:
        # Check if the user has appropriate role
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Unauthorized: Insufficient privileges"}), 403
        
        # Get parameters from request
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 10, type=int)
        
        # Validate parameters
        if not isinstance(days, int) or days < 1 or days > 90:
            return jsonify({"error": "Invalid 'days' parameter. Must be an integer between 1 and 90"}), 400
            
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            return jsonify({"error": "Invalid 'limit' parameter. Must be an integer between 1 and 50"}), 400
        
        # Get trending narratives
        trending = ComplexityPredictor.get_trending_narratives(days, limit)
        
        if "error" in trending:
            return jsonify({"error": trending["error"]}), 400
            
        return jsonify(trending)
        
    except Exception as e:
        logger.error(f"Error in trending narratives API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@complexity_bp.route('/complexity/predict/<int:narrative_id>', methods=['GET'])
@login_required
def view_narrative_prediction(narrative_id):
    """
    View complexity prediction for a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to predict
        
    Returns:
        HTML page with complexity prediction
    """
    try:
        # Get narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return render_template('error.html', message=f"Narrative with ID {narrative_id} not found"), 404
        
        # Get days from request
        days_ahead = request.args.get('days', 7, type=int)
        
        # Make prediction
        prediction = ComplexityPredictor.predict_narrative_evolution(narrative_id, days_ahead)
        
        # Check if prediction failed
        if "error" in prediction:
            return render_template('error.html', message=prediction["error"]), 400
        
        # Get similar narratives
        similar = ComplexityPredictor.get_similar_trajectories(narrative_id)
        
        return render_template(
            'complexity/predict.html',
            narrative=narrative,
            prediction=prediction,
            similar_narratives=similar.get('similar_narratives', []) if "error" not in similar else []
        )
        
    except Exception as e:
        logger.error(f"Error in view prediction endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/trending', methods=['GET'])
@login_required
def view_trending_narratives():
    """
    View narratives with strong upward complexity trends.
    
    Returns:
        HTML page with trending narratives
    """
    try:
        # Get parameters from request
        days = request.args.get('days', 30, type=int)
        
        # Get trending narratives
        trending = ComplexityPredictor.get_trending_narratives(days, 20)
        
        return render_template(
            'complexity/trending.html',
            trending_narratives=trending.get('trending_narratives', []) if "error" not in trending else [],
            days=days,
            timestamp=trending.get('timestamp') if "error" not in trending else None
        )
        
    except Exception as e:
        logger.error(f"Error in trending narratives endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@complexity_bp.route('/complexity/features', methods=['GET'])
def features():
    """
    Display a page showcasing all features of the Narrative Complexity Analyzer.
    
    Returns:
        HTML page with feature details
    """
    try:
        # Debug messages
        current_app.logger.info("Attempting to render the features page")
        
        # Use a simplified template to debug the issue
        return render_template('complexity/features_simple.html')
        
    except Exception as e:
        # Detailed error logging
        import traceback
        error_traceback = traceback.format_exc()
        current_app.logger.error(f"Error in features page endpoint: {e}")
        current_app.logger.error(f"Traceback: {error_traceback}")
        
        # Return simple error
        return render_template('error.html', message=str(e)), 500

# Register blueprint
app.register_blueprint(complexity_bp)