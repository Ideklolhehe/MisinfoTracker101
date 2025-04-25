import json
import logging
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta

from models import (
    DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge, 
    CounterMessage, DataSource, SystemLog
)
from app import db
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from agents.detector_agent import DetectorAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.counter_agent import CounterAgent
from storage.evidence_store import EvidenceStore
from storage.graph_store import GraphStore

# Initialize API blueprint
api_bp = Blueprint('api', __name__)

# Initialize utilities
text_processor = TextProcessor(languages=['en', 'es'])
vector_store = VectorStore(dimension=768)
evidence_store = EvidenceStore()
graph_store = GraphStore()

# Initialize agents
detector = DetectorAgent(text_processor, vector_store)
analyzer = AnalyzerAgent(text_processor)
counter_agent = CounterAgent(text_processor)

logger = logging.getLogger(__name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '0.1.0'
    })

@api_bp.route('/narratives', methods=['GET'])
def get_narratives():
    """Get detected narratives with optional filtering."""
    try:
        # Parse query parameters
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        language = request.args.get('language')
        days = request.args.get('days')
        
        # Build query
        query = DetectedNarrative.query
        
        # Apply filters
        if language:
            query = query.filter_by(language=language)
            
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=int(days))
            query = query.filter(DetectedNarrative.last_updated >= cutoff_date)
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination and sort
        narratives = query.order_by(
            DetectedNarrative.last_updated.desc()
        ).limit(limit).offset(offset).all()
        
        # Format response
        result = []
        for narrative in narratives:
            # Parse metadata if available
            metadata = {}
            if narrative.meta_data:
                try:
                    metadata = json.loads(narrative.meta_data)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Count instances
            instance_count = NarrativeInstance.query.filter_by(
                narrative_id=narrative.id
            ).count()
            
            result.append({
                'id': narrative.id,
                'title': narrative.title,
                'description': narrative.description,
                'confidence_score': narrative.confidence_score,
                'first_detected': narrative.first_detected.isoformat(),
                'last_updated': narrative.last_updated.isoformat(),
                'status': narrative.status,
                'language': narrative.language,
                'instance_count': instance_count,
                'propagation_score': metadata.get('propagation_score', 0),
                'viral_threat': metadata.get('viral_threat', 0)
            })
        
        return jsonify({
            'narratives': result,
            'total': total_count,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error in get_narratives: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/narratives/<int:narrative_id>', methods=['GET'])
def get_narrative(narrative_id):
    """Get details for a specific narrative."""
    try:
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return jsonify({'error': 'Narrative not found'}), 404
        
        # Get instances
        instances = NarrativeInstance.query.filter_by(
            narrative_id=narrative_id
        ).order_by(NarrativeInstance.detected_at.desc()).all()
        
        # Parse metadata
        metadata = {}
        if narrative.meta_data:
            try:
                metadata = json.loads(narrative.meta_data)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Format instances
        formatted_instances = []
        for instance in instances:
            instance_metadata = {}
            if instance.meta_data:
                try:
                    instance_metadata = json.loads(instance.meta_data)
                except (json.JSONDecodeError, TypeError):
                    pass
                    
            formatted_instances.append({
                'id': instance.id,
                'content': instance.content,
                'url': instance.url,
                'detected_at': instance.detected_at.isoformat(),
                'source_id': instance.source_id,
                'metadata': instance_metadata,
                'evidence_hash': instance.evidence_hash
            })
        
        # Get counter messages
        counter_messages = CounterMessage.query.filter_by(
            narrative_id=narrative_id
        ).order_by(CounterMessage.created_at.desc()).all()
        
        formatted_counter_messages = []
        for message in counter_messages:
            formatted_counter_messages.append({
                'id': message.id,
                'content': message.content,
                'strategy': message.strategy,
                'status': message.status,
                'created_at': message.created_at.isoformat()
            })
        
        result = {
            'id': narrative.id,
            'title': narrative.title,
            'description': narrative.description,
            'confidence_score': narrative.confidence_score,
            'first_detected': narrative.first_detected.isoformat(),
            'last_updated': narrative.last_updated.isoformat(),
            'status': narrative.status,
            'language': narrative.language,
            'instances': formatted_instances,
            'counter_messages': formatted_counter_messages,
            'metadata': metadata
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_narrative: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/narratives/<int:narrative_id>/analyze', methods=['POST'])
def analyze_narrative(narrative_id):
    """Trigger analysis for a specific narrative."""
    try:
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return jsonify({'error': 'Narrative not found'}), 404
        
        # Run analysis
        analysis_result = analyzer.analyze_narrative(narrative_id)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error in analyze_narrative: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/counter-messages', methods=['GET'])
def get_counter_messages():
    """Get counter messages with optional filtering."""
    try:
        # Parse query parameters
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        status = request.args.get('status')
        
        # Build query
        query = CounterMessage.query
        
        # Apply filters
        if status:
            query = query.filter_by(status=status)
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination and sort
        messages = query.order_by(
            CounterMessage.created_at.desc()
        ).limit(limit).offset(offset).all()
        
        # Format response
        result = []
        for message in messages:
            # Get narrative title
            narrative_title = "Unknown"
            if message.narrative_id:
                narrative = DetectedNarrative.query.get(message.narrative_id)
                if narrative:
                    narrative_title = narrative.title
            
            result.append({
                'id': message.id,
                'narrative_id': message.narrative_id,
                'narrative_title': narrative_title,
                'content': message.content,
                'strategy': message.strategy,
                'status': message.status,
                'created_at': message.created_at.isoformat()
            })
        
        return jsonify({
            'counter_messages': result,
            'total': total_count,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error in get_counter_messages: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/counter-messages/<int:message_id>/approve', methods=['POST'])
def approve_counter_message(message_id):
    """Approve a counter message."""
    try:
        # In a real system, we would verify user authentication here
        user_id = 1  # Default admin user for now
        
        result = counter_agent.approve_counter_message(message_id, user_id)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in approve_counter_message: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/counter-messages/generate', methods=['POST'])
def generate_counter_message():
    """Generate a counter message for a narrative."""
    try:
        data = request.get_json()
        if not data or 'narrative_id' not in data:
            return jsonify({'error': 'narrative_id is required'}), 400
            
        narrative_id = data['narrative_id']
        
        # Check if narrative exists
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return jsonify({'error': 'Narrative not found'}), 404
        
        # Generate counter message
        result = counter_agent.generate_counter_message(narrative_id)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in generate_counter_message: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/evidence/<evidence_hash>', methods=['GET'])
def get_evidence(evidence_hash):
    """Retrieve stored evidence by hash."""
    try:
        evidence = evidence_store.retrieve_evidence(evidence_hash)
        if not evidence:
            return jsonify({'error': 'Evidence not found'}), 404
            
        return jsonify(evidence)
        
    except Exception as e:
        logger.error(f"Error in get_evidence: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/evidence/verify/<evidence_hash>', methods=['GET'])
def verify_evidence(evidence_hash):
    """Verify evidence integrity."""
    try:
        is_valid = evidence_store.verify_evidence(evidence_hash)
        return jsonify({'evidence_hash': evidence_hash, 'is_valid': is_valid})
        
    except Exception as e:
        logger.error(f"Error in verify_evidence: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/graph/node/<int:node_id>', methods=['GET'])
def get_node_connections(node_id):
    """Get a node and its connections in the belief graph."""
    try:
        depth = int(request.args.get('depth', 1))
        result = graph_store.get_node_connections(node_id, depth)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_node_connections: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/graph/path', methods=['GET'])
def get_path_between_nodes():
    """Find path between two nodes in the belief graph."""
    try:
        source_id = int(request.args.get('source_id'))
        target_id = int(request.args.get('target_id'))
        
        result = graph_store.get_path_between(source_id, target_id)
        return jsonify(result)
        
    except ValueError:
        return jsonify({'error': 'source_id and target_id must be integers'}), 400
    except Exception as e:
        logger.error(f"Error in get_path_between_nodes: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/process', methods=['POST'])
def process_content():
    """Process content to detect misinformation."""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'content is required'}), 400
            
        content = data['content']
        source = data.get('source')
        metadata = data.get('metadata')
        
        # Process content using detector agent
        result = detector.process_content(content, None, source, metadata)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in process_content: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        # Get narrative stats
        total_narratives = DetectedNarrative.query.count()
        active_narratives = DetectedNarrative.query.filter_by(status='active').count()
        
        # Get instance stats
        total_instances = NarrativeInstance.query.count()
        
        # Get counter message stats
        total_counter_messages = CounterMessage.query.count()
        draft_messages = CounterMessage.query.filter_by(status='draft').count()
        approved_messages = CounterMessage.query.filter_by(status='approved').count()
        
        # Get source stats
        active_sources = DataSource.query.filter_by(is_active=True).count()
        
        # Get recent logs
        recent_logs = SystemLog.query.order_by(
            SystemLog.timestamp.desc()
        ).limit(10).all()
        
        formatted_logs = []
        for log in recent_logs:
            formatted_logs.append({
                'timestamp': log.timestamp.isoformat(),
                'log_type': log.log_type,
                'component': log.component,
                'message': log.message
            })
        
        # Get vector store stats
        vector_stats = vector_store.get_stats()
        
        return jsonify({
            'narratives': {
                'total': total_narratives,
                'active': active_narratives
            },
            'instances': {
                'total': total_instances
            },
            'counter_messages': {
                'total': total_counter_messages,
                'draft': draft_messages,
                'approved': approved_messages
            },
            'sources': {
                'active': active_sources
            },
            'recent_logs': formatted_logs,
            'vector_store': vector_stats
        })
        
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/logs', methods=['GET'])
def get_logs():
    """Get system logs with optional filtering."""
    try:
        # Parse query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        log_type = request.args.get('type')
        component = request.args.get('component')
        
        # Build query
        query = SystemLog.query
        
        # Apply filters
        if log_type:
            query = query.filter_by(log_type=log_type)
            
        if component:
            query = query.filter_by(component=component)
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination and sort
        logs = query.order_by(
            SystemLog.timestamp.desc()
        ).limit(limit).offset(offset).all()
        
        # Format response
        result = []
        for log in logs:
            metadata = {}
            if log.meta_data:
                try:
                    metadata = json.loads(log.meta_data)
                except (json.JSONDecodeError, TypeError):
                    pass
                    
            result.append({
                'id': log.id,
                'timestamp': log.timestamp.isoformat(),
                'log_type': log.log_type,
                'component': log.component,
                'message': log.message,
                'metadata': metadata
            })
        
        return jsonify({
            'logs': result,
            'total': total_count,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error in get_logs: {e}")
        return jsonify({'error': str(e)}), 500
