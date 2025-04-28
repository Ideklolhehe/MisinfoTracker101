"""
Routes for narrative clustering and relationship analysis.
"""

import logging
import json
from flask import Blueprint, request, jsonify, render_template, abort
from flask_login import login_required, current_user

from app import db
from models import DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge
from services.narrative_analyzer import NarrativeAnalyzer
from utils.app_context import ensure_app_context

logger = logging.getLogger(__name__)

# Initialize services
narrative_analyzer = NarrativeAnalyzer()

# Create Blueprint
clusters_bp = Blueprint('clusters', __name__)

@clusters_bp.route('/clusters/view', methods=['GET'])
@login_required
def view_narrative_clusters():
    """
    View clusters of related narratives.
    
    Returns:
        HTML page with narrative clusters
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access narrative clusters")
        
        # Get days parameter (default: 30 days)
        days = request.args.get('days', 30, type=int)
        
        # Get minimum instances parameter (default: 2)
        min_instances = request.args.get('min_instances', 2, type=int)
        
        # Perform clustering
        clusters_result = narrative_analyzer.cluster_system_narratives(days, min_instances)
        
        if "error" in clusters_result:
            return render_template(
                'error.html', 
                message=f"Clustering failed: {clusters_result['error']}"
            ), 400
        
        return render_template(
            'clusters/view.html',
            clusters=clusters_result,
            days=days,
            min_instances=min_instances
        )
        
    except Exception as e:
        logger.error(f"Error in narrative clusters endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@clusters_bp.route('/clusters/api/data', methods=['GET'])
@login_required
def api_get_narrative_clusters():
    """
    API endpoint to get narrative cluster data.
    
    Returns:
        JSON with narrative clusters
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get days parameter (default: 30 days)
        days = request.args.get('days', 30, type=int)
        
        # Get minimum instances parameter (default: 2)
        min_instances = request.args.get('min_instances', 2, type=int)
        
        # Perform clustering
        clusters_result = narrative_analyzer.cluster_system_narratives(days, min_instances)
        
        if "error" in clusters_result:
            return jsonify({"error": clusters_result["error"]}), 400
        
        return jsonify(clusters_result)
        
    except Exception as e:
        logger.error(f"Error in narrative clusters API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@clusters_bp.route('/clusters/belief-network', methods=['GET'])
@login_required
def view_belief_network():
    """
    View the belief network visualization.
    
    Returns:
        HTML page with belief network visualization
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access belief network")
        
        # Get refresh parameter (if user wants to rebuild the network)
        refresh = request.args.get('refresh', 'false') == 'true'
        
        # Check if we need to build/rebuild the network
        if refresh:
            network_result = narrative_analyzer.create_belief_network()
            
            if "error" in network_result:
                return render_template(
                    'error.html', 
                    message=f"Belief network creation failed: {network_result['error']}"
                ), 400
        
        # Get belief network nodes and edges from database
        nodes = BeliefNode.query.filter_by(node_type='narrative').all()
        edges = BeliefEdge.query.filter_by(relation_type='similar').all()
        
        # Extract narrative information
        narrative_ids = []
        for node in nodes:
            try:
                meta_data = json.loads(node.meta_data) if node.meta_data else {}
                narrative_id = meta_data.get('narrative_id')
                if narrative_id:
                    narrative_ids.append(narrative_id)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        narratives = {}
        if narrative_ids:
            for narrative in DetectedNarrative.query.filter(DetectedNarrative.id.in_(narrative_ids)).all():
                narratives[narrative.id] = narrative
        
        # Prepare network data for visualization
        network_data = {
            'nodes': [],
            'links': []
        }
        
        for node in nodes:
            try:
                meta_data = json.loads(node.meta_data) if node.meta_data else {}
                narrative_id = meta_data.get('narrative_id')
                title = meta_data.get('title', f"Node {node.id}")
                
                if narrative_id in narratives:
                    title = narratives[narrative_id].title
                
                network_data['nodes'].append({
                    'id': node.id,
                    'label': title,
                    'narrative_id': narrative_id
                })
            except (json.JSONDecodeError, KeyError, TypeError):
                network_data['nodes'].append({
                    'id': node.id,
                    'label': f"Node {node.id}"
                })
        
        for edge in edges:
            network_data['links'].append({
                'source': edge.source_id,
                'target': edge.target_id,
                'weight': edge.weight
            })
        
        return render_template(
            'clusters/belief_network.html',
            network=network_data,
            nodes_count=len(nodes),
            edges_count=len(edges)
        )
        
    except Exception as e:
        logger.error(f"Error in belief network endpoint: {e}")
        return render_template('error.html', message=str(e)), 500

@clusters_bp.route('/clusters/api/belief-network', methods=['GET'])
@login_required
def api_get_belief_network():
    """
    API endpoint to get belief network data.
    
    Returns:
        JSON with belief network data
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            return jsonify({"error": "Insufficient privileges"}), 403
        
        # Get refresh parameter (if user wants to rebuild the network)
        refresh = request.args.get('refresh', 'false') == 'true'
        
        # Check if we need to build/rebuild the network
        if refresh:
            network_result = narrative_analyzer.create_belief_network()
            
            if "error" in network_result:
                return jsonify({"error": network_result["error"]}), 400
            
            return jsonify(network_result)
        
        # Get belief network nodes and edges from database
        nodes = BeliefNode.query.filter_by(node_type='narrative').all()
        edges = BeliefEdge.query.filter_by(relation_type='similar').all()
        
        # Prepare network data
        network_data = {
            'nodes': [],
            'links': []
        }
        
        for node in nodes:
            try:
                meta_data = json.loads(node.meta_data) if node.meta_data else {}
                network_data['nodes'].append({
                    'id': node.id,
                    'label': meta_data.get('title', f"Node {node.id}"),
                    'narrative_id': meta_data.get('narrative_id')
                })
            except (json.JSONDecodeError, KeyError, TypeError):
                network_data['nodes'].append({
                    'id': node.id,
                    'label': f"Node {node.id}"
                })
        
        for edge in edges:
            try:
                meta_data = json.loads(edge.meta_data) if edge.meta_data else {}
                network_data['links'].append({
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'weight': edge.weight,
                    'similarity': meta_data.get('similarity_score', edge.weight),
                    'cluster_relationship': meta_data.get('cluster_relationship', False)
                })
            except (json.JSONDecodeError, KeyError, TypeError):
                network_data['links'].append({
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'weight': edge.weight
                })
        
        return jsonify({
            'network': network_data,
            'nodes_count': len(nodes),
            'edges_count': len(edges)
        })
        
    except Exception as e:
        logger.error(f"Error in belief network API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@clusters_bp.route('/clusters/coordinated-campaigns', methods=['GET'])
@login_required
def view_coordinated_campaigns():
    """
    View potential coordinated campaigns based on narrative clusters.
    
    Returns:
        HTML page with coordinated campaigns
    """
    try:
        # Check if user has access
        if current_user.role not in ['admin', 'analyst', 'researcher']:
            abort(403, "Insufficient privileges to access coordinated campaigns")
        
        # Get days parameter (default: 30 days)
        days = request.args.get('days', 30, type=int)
        
        # Get active narratives with instances
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.status == 'active',
            DetectedNarrative.last_updated >= cutoff_date
        ).all()
        
        # Filter narratives with sufficient instances
        valid_narratives = []
        narrative_texts = []
        narrative_metadata = []
        
        for narrative in narratives:
            instance_count = NarrativeInstance.query.filter_by(narrative_id=narrative.id).count()
            
            if instance_count >= 2:
                valid_narratives.append(narrative)
                narrative_texts.append(narrative.description or narrative.title)
                
                # Extract metadata
                meta_data = {}
                if narrative.meta_data:
                    try:
                        meta_data = json.loads(narrative.meta_data)
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                instance_timestamps = [
                    instance.detected_at 
                    for instance in NarrativeInstance.query.filter_by(narrative_id=narrative.id).all()
                ]
                
                narrative_metadata.append({
                    'id': narrative.id,
                    'title': narrative.title,
                    'last_updated': narrative.last_updated.isoformat() if narrative.last_updated else None,
                    'first_detected': narrative.first_detected.isoformat() if narrative.first_detected else None,
                    'language': narrative.language,
                    'complexity': meta_data.get('complexity_analysis', {}).get('overall_complexity_score', 0),
                    'instance_count': instance_count,
                    'instance_timestamps': [ts.isoformat() if ts else None for ts in instance_timestamps]
                })
        
        if not narrative_texts:
            return render_template(
                'error.html', 
                message="No valid narratives found for campaign analysis"
            ), 400
        
        # Perform coordinated campaign detection
        campaigns = narrative_analyzer.detect_coordinated_campaigns(narrative_texts, narrative_metadata)
        
        # Format campaigns for display
        formatted_campaigns = []
        for cluster_id, cluster_metadata in campaigns:
            # Calculate time distribution
            timestamps = []
            for meta in cluster_metadata:
                if 'instance_timestamps' in meta:
                    timestamps.extend([ts for ts in meta['instance_timestamps'] if ts])
            
            timestamps.sort()
            
            # Calculate average complexity
            avg_complexity = sum(meta.get('complexity', 0) for meta in cluster_metadata) / len(cluster_metadata) if cluster_metadata else 0
            
            formatted_campaigns.append({
                'cluster_id': cluster_id,
                'narratives': cluster_metadata,
                'narrative_count': len(cluster_metadata),
                'timestamps': timestamps,
                'avg_complexity': avg_complexity
            })
        
        # Sort by size (largest first)
        formatted_campaigns.sort(key=lambda x: x['narrative_count'], reverse=True)
        
        return render_template(
            'clusters/campaigns.html',
            campaigns=formatted_campaigns,
            days=days,
            total_narratives=len(valid_narratives)
        )
        
    except Exception as e:
        logger.error(f"Error in coordinated campaigns endpoint: {e}")
        return render_template('error.html', message=str(e)), 500