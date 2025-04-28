"""
Routes for narrative network analysis and visualization.
"""

import logging
import json
from flask import Blueprint, request, jsonify, render_template, abort
from flask_login import login_required, current_user

from app import db
from models import DetectedNarrative
from services.narrative_network import NarrativeNetworkAnalyzer

logger = logging.getLogger(__name__)

# Create Blueprint
network_bp = Blueprint('network', __name__)

@network_bp.route('/network/dashboard', methods=['GET'])
@login_required
def network_dashboard():
    """
    View the narrative network dashboard.
    
    Returns:
        HTML page with network visualization and analysis
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access network analysis tools")
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Build the network
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        analyzer.build_narrative_network(include_archived=include_archived)
        
        # Get network statistics
        statistics = analyzer.get_network_statistics()
        
        # Get central narratives
        top_n = request.args.get('top_n', 5, type=int)
        central_narratives = analyzer.get_central_narratives(top_n=top_n)
        
        # Export network for visualization
        network_data = analyzer.export_network_json()
        
        return render_template(
            'network/dashboard.html',
            network_data=json.dumps(network_data),
            statistics=statistics,
            central_narratives=central_narratives,
            include_archived=include_archived
        )
        
    except Exception as e:
        logger.error(f"Error in network dashboard endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@network_bp.route('/network/campaigns', methods=['GET'])
@login_required
def coordinated_campaigns():
    """
    View potential coordinated misinformation campaigns.
    
    Returns:
        HTML page showing potential coordinated campaigns
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access campaign analysis tools")
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Build the network
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        analyzer.build_narrative_network(include_archived=include_archived)
        
        # Identify campaigns
        min_narratives = request.args.get('min_narratives', 3, type=int)
        min_similarity = request.args.get('min_similarity', 0.5, type=float)
        
        campaigns = analyzer.identify_coordinated_campaigns(
            min_narratives=min_narratives,
            min_similarity=min_similarity
        )
        
        return render_template(
            'network/campaigns.html',
            campaigns=campaigns,
            min_narratives=min_narratives,
            min_similarity=min_similarity,
            include_archived=include_archived
        )
        
    except Exception as e:
        logger.error(f"Error in coordinated campaigns endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@network_bp.route('/network/narrative/<int:narrative_id>', methods=['GET'])
@login_required
def narrative_network(narrative_id):
    """
    View the network surrounding a specific narrative.
    
    Args:
        narrative_id: ID of the narrative to analyze
        
    Returns:
        HTML page with network visualization focused on a specific narrative
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access network analysis tools")
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            abort(404, f"Narrative with ID {narrative_id} not found")
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Build the network
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        analyzer.build_narrative_network(include_archived=include_archived)
        
        # Export full network
        network_data = analyzer.export_network_json()
        
        # Extract related narratives (direct connections)
        related_ids = set()
        for edge in network_data['edges']:
            if edge['source'] == narrative_id:
                related_ids.add(edge['target'])
            elif edge['target'] == narrative_id:
                related_ids.add(edge['source'])
        
        related_narratives = []
        for node in network_data['nodes']:
            if node['id'] in related_ids:
                # Get full narrative object
                related = DetectedNarrative.query.get(node['id'])
                if related:
                    related_narratives.append({
                        'id': related.id,
                        'title': related.title,
                        'status': related.status,
                        'confidence': float(related.confidence_score or 0),
                        'first_detected': related.first_detected.isoformat()
                    })
        
        return render_template(
            'network/narrative.html',
            narrative=narrative,
            network_data=json.dumps(network_data),
            related_narratives=related_narratives,
            include_archived=include_archived
        )
        
    except Exception as e:
        logger.error(f"Error in narrative network endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@network_bp.route('/network/api/data', methods=['GET'])
@login_required
def api_network_data():
    """
    API endpoint to get narrative network data for visualization.
    
    Returns:
        JSON with network data
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Build the network
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        analyzer.build_narrative_network(include_archived=include_archived)
        
        # Export network
        network_data = analyzer.export_network_json()
        
        return jsonify(network_data)
        
    except Exception as e:
        logger.error(f"Error in network data API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@network_bp.route('/network/api/campaigns', methods=['GET'])
@login_required
def api_coordinated_campaigns():
    """
    API endpoint to get potential coordinated campaigns.
    
    Returns:
        JSON with campaign data
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Build the network
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        analyzer.build_narrative_network(include_archived=include_archived)
        
        # Identify campaigns
        min_narratives = request.args.get('min_narratives', 3, type=int)
        min_similarity = request.args.get('min_similarity', 0.5, type=float)
        
        campaigns = analyzer.identify_coordinated_campaigns(
            min_narratives=min_narratives,
            min_similarity=min_similarity
        )
        
        return jsonify({
            "campaigns": campaigns,
            "campaign_count": len(campaigns),
            "generated_at": campaigns[0]["first_narrative_date"] if campaigns else None
        })
        
    except Exception as e:
        logger.error(f"Error in campaigns API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@network_bp.route('/network/communities', methods=['GET'])
@login_required
def louvain_communities():
    """
    View communities detected with Louvain algorithm.
    
    Returns:
        HTML page showing Louvain communities
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access community analysis tools")
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Build the network
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        analyzer.build_narrative_network(include_archived=include_archived)
        
        # Identify communities
        communities = analyzer.identify_communities_with_louvain()
        
        # Export network for visualization
        network_data = analyzer.export_network_json()
        
        return render_template(
            'network/communities.html',
            communities=communities,
            network_data=json.dumps(network_data),
            include_archived=include_archived
        )
        
    except Exception as e:
        logger.error(f"Error in Louvain communities endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@network_bp.route('/network/api/communities', methods=['GET'])
@login_required
def api_louvain_communities():
    """
    API endpoint to get communities detected with Louvain algorithm.
    
    Returns:
        JSON with community data
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Build the network
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        analyzer.build_narrative_network(include_archived=include_archived)
        
        # Identify communities
        communities = analyzer.identify_communities_with_louvain()
        
        return jsonify({
            "communities": communities,
            "community_count": len(communities),
            "generated_at": communities[0]["narratives"][0]["first_detected"] if communities and communities[0]["narratives"] else None
        })
        
    except Exception as e:
        logger.error(f"Error in communities API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@network_bp.route('/network/streaming/denstream', methods=['GET'])
@login_required
def denstream_clusters():
    """
    View clusters from DenStream streaming algorithm.
    
    Returns:
        HTML page showing DenStream clustering results
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access streaming cluster analysis tools")
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Get DenStream clusters
        clusters = analyzer.get_denstream_clusters()
        
        return render_template(
            'network/denstream.html',
            clusters=clusters
        )
        
    except Exception as e:
        logger.error(f"Error in DenStream clusters endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@network_bp.route('/network/api/streaming/denstream', methods=['GET'])
@login_required
def api_denstream_clusters():
    """
    API endpoint to get clusters from DenStream streaming algorithm.
    
    Returns:
        JSON with DenStream clustering results
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Initialize network analyzer
        analyzer = NarrativeNetworkAnalyzer()
        
        # Get DenStream clusters
        clusters = analyzer.get_denstream_clusters()
        
        return jsonify(clusters)
        
    except Exception as e:
        logger.error(f"Error in DenStream API endpoint: {e}")
        return jsonify({"error": str(e)}), 500