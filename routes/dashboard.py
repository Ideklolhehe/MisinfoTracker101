from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import json
import logging
from datetime import datetime, timedelta

from models import (
    DetectedNarrative, NarrativeInstance, BeliefNode, BeliefEdge, 
    CounterMessage, DataSource, SystemLog
)
from app import db

# Initialize dashboard blueprint
dashboard_bp = Blueprint('dashboard', __name__)

logger = logging.getLogger(__name__)

@dashboard_bp.route('/')
def index():
    """Render the dashboard homepage."""
    try:
        # Get recent narratives (last 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        recent_narratives = DetectedNarrative.query.filter(
            DetectedNarrative.last_updated >= cutoff_date
        ).order_by(DetectedNarrative.last_updated.desc()).limit(5).all()
        
        # Get pending counter messages
        pending_messages = CounterMessage.query.filter_by(
            status='draft'
        ).order_by(CounterMessage.created_at.desc()).limit(5).all()
        
        # Get active data sources
        active_sources = DataSource.query.filter_by(
            is_active=True
        ).all()
        
        # Get recent logs
        recent_logs = SystemLog.query.order_by(
            SystemLog.timestamp.desc()
        ).limit(10).all()
        
        # Get overall statistics
        total_narratives = DetectedNarrative.query.count()
        active_narratives = DetectedNarrative.query.filter_by(status='active').count()
        total_instances = NarrativeInstance.query.count()
        
        # Get high-threat narratives
        high_threat_narratives = []
        for narrative in recent_narratives:
            threat_level = 0
            if narrative.meta_data:
                try:
                    metadata = json.loads(narrative.meta_data)
                    threat_level = metadata.get('viral_threat', 0)
                    if threat_level >= 3:  # Consider high threat if >= 3 out of 5
                        high_threat_narratives.append({
                            'id': narrative.id,
                            'title': narrative.title,
                            'threat_level': threat_level,
                            'last_updated': narrative.last_updated
                        })
                except (json.JSONDecodeError, TypeError):
                    pass
                    
        return render_template(
            'dashboard/index.html',
            recent_narratives=recent_narratives,
            pending_messages=pending_messages,
            active_sources=active_sources,
            recent_logs=recent_logs,
            total_narratives=total_narratives,
            active_narratives=active_narratives,
            total_instances=total_instances,
            high_threat_narratives=sorted(high_threat_narratives, key=lambda x: x['threat_level'], reverse=True)
        )
    except Exception as e:
        logger.error(f"Error in dashboard index: {e}")
        return render_template('dashboard/index.html', error=str(e))

@dashboard_bp.route('/narratives')
def narratives():
    """Render the narratives page."""
    try:
        # Get query parameters
        status = request.args.get('status', 'all')
        language = request.args.get('language', 'all')
        days = request.args.get('days', '30')
        search = request.args.get('search', '')
        
        # Build query
        query = DetectedNarrative.query
        
        # Apply filters
        if status != 'all':
            query = query.filter_by(status=status)
            
        if language != 'all':
            query = query.filter_by(language=language)
            
        if days != 'all':
            cutoff_date = datetime.utcnow() - timedelta(days=int(days))
            query = query.filter(DetectedNarrative.last_updated >= cutoff_date)
            
        if search:
            query = query.filter(
                (DetectedNarrative.title.ilike(f'%{search}%')) | 
                (DetectedNarrative.description.ilike(f'%{search}%'))
            )
        
        # Get total count for pagination
        total_narratives = query.count()
        
        # Get paginated results
        page = request.args.get('page', 1, type=int)
        per_page = 20
        narratives = query.order_by(
            DetectedNarrative.last_updated.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        # Get available languages for filter
        languages = db.session.query(DetectedNarrative.language.distinct()).all()
        available_languages = [lang[0] for lang in languages]
        
        return render_template(
            'dashboard/narratives.html',
            narratives=narratives,
            total_narratives=total_narratives,
            status=status,
            language=language,
            days=days,
            search=search,
            available_languages=available_languages
        )
    except Exception as e:
        logger.error(f"Error in narratives page: {e}")
        return render_template('dashboard/narratives.html', error=str(e))

@dashboard_bp.route('/narratives/<int:narrative_id>')
def narrative_detail(narrative_id):
    """Render the narrative detail page."""
    try:
        narrative = DetectedNarrative.query.get_or_404(narrative_id)
        
        # Get instances
        instances = NarrativeInstance.query.filter_by(
            narrative_id=narrative_id
        ).order_by(NarrativeInstance.detected_at.desc()).all()
        
        # Get counter messages
        counter_messages = CounterMessage.query.filter_by(
            narrative_id=narrative_id
        ).order_by(CounterMessage.created_at.desc()).all()
        
        # Parse metadata
        metadata = {}
        if narrative.meta_data:
            try:
                metadata = json.loads(narrative.meta_data)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # If there are no counter messages and there's at least one instance,
        # we can show a button to generate a counter message
        show_generate_button = len(counter_messages) == 0 and len(instances) > 0
        
        return render_template(
            'dashboard/narratives.html',
            narrative=narrative,
            instances=instances,
            counter_messages=counter_messages,
            metadata=metadata,
            show_generate_button=show_generate_button,
            detail_view=True
        )
    except Exception as e:
        logger.error(f"Error in narrative detail: {e}")
        return render_template('dashboard/narratives.html', error=str(e))

@dashboard_bp.route('/counter-messaging')
def counter_messaging():
    """Render the counter messaging page."""
    try:
        # Get query parameters
        status = request.args.get('status', 'all')
        search = request.args.get('search', '')
        
        # Build query
        query = CounterMessage.query
        
        # Apply filters
        if status != 'all':
            query = query.filter_by(status=status)
            
        if search:
            query = query.filter(CounterMessage.content.ilike(f'%{search}%'))
        
        # Get total count for pagination
        total_messages = query.count()
        
        # Get paginated results
        page = request.args.get('page', 1, type=int)
        per_page = 20
        messages = query.order_by(
            CounterMessage.created_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        # Get narratives for each counter message
        narratives = {}
        for message in messages.items:
            if message.narrative_id:
                narrative = DetectedNarrative.query.get(message.narrative_id)
                if narrative:
                    narratives[message.id] = narrative
        
        return render_template(
            'dashboard/counter_messaging.html',
            messages=messages,
            total_messages=total_messages,
            status=status,
            search=search,
            narratives=narratives
        )
    except Exception as e:
        logger.error(f"Error in counter messaging page: {e}")
        return render_template('dashboard/counter_messaging.html', error=str(e))

@dashboard_bp.route('/settings')
def settings():
    """Render the settings page."""
    try:
        # Get active sources
        sources = DataSource.query.order_by(DataSource.name).all()
        
        # Group sources by type
        twitter_sources = [s for s in sources if s.source_type == 'twitter']
        telegram_sources = [s for s in sources if s.source_type == 'telegram']
        rss_sources = [s for s in sources if s.source_type == 'rss']
        other_sources = [s for s in sources if s.source_type not in ('twitter', 'telegram', 'rss')]
        
        # Get logs for system status
        recent_logs = SystemLog.query.order_by(
            SystemLog.timestamp.desc()
        ).limit(20).all()
        
        # Group logs by type
        error_logs = [log for log in recent_logs if log.log_type == 'error']
        info_logs = [log for log in recent_logs if log.log_type == 'info']
        
        return render_template(
            'dashboard/settings.html',
            twitter_sources=twitter_sources,
            telegram_sources=telegram_sources,
            rss_sources=rss_sources,
            other_sources=other_sources,
            error_logs=error_logs,
            info_logs=info_logs
        )
    except Exception as e:
        logger.error(f"Error in settings page: {e}")
        return render_template('dashboard/settings.html', error=str(e))

@dashboard_bp.route('/create-twitter-source', methods=['POST'])
def create_twitter_source():
    """Create a new Twitter data source."""
    try:
        name = request.form.get('name')
        source_type = request.form.get('type', 'query')
        query = request.form.get('query', '')
        users = request.form.get('users', '')
        
        if not name:
            flash('Source name is required', 'danger')
            return redirect(url_for('dashboard.settings'))
            
        # Build config based on type
        if source_type == 'query' and query:
            config = {'query': query}
        elif source_type == 'users' and users:
            # Split users by comma and strip whitespace
            user_list = [u.strip() for u in users.split(',') if u.strip()]
            config = {'users': user_list}
        else:
            flash('Invalid source configuration', 'danger')
            return redirect(url_for('dashboard.settings'))
        
        # Create source
        source = DataSource(
            name=name,
            source_type='twitter',
            config=json.dumps(config),
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.session.add(source)
        db.session.commit()
        
        flash(f'Twitter source "{name}" created successfully', 'success')
        return redirect(url_for('dashboard.settings'))
        
    except Exception as e:
        logger.error(f"Error creating Twitter source: {e}")
        flash(f'Error creating source: {str(e)}', 'danger')
        return redirect(url_for('dashboard.settings'))

@dashboard_bp.route('/create-telegram-source', methods=['POST'])
def create_telegram_source():
    """Create a new Telegram data source."""
    try:
        name = request.form.get('name')
        channels = request.form.get('channels', '')
        
        if not name or not channels:
            flash('Source name and channels are required', 'danger')
            return redirect(url_for('dashboard.settings'))
            
        # Split channels by comma and strip whitespace
        channel_list = [c.strip() for c in channels.split(',') if c.strip()]
        
        if not channel_list:
            flash('At least one valid channel is required', 'danger')
            return redirect(url_for('dashboard.settings'))
        
        # Create config
        config = {'channels': channel_list}
        
        # Create source
        source = DataSource(
            name=name,
            source_type='telegram',
            config=json.dumps(config),
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.session.add(source)
        db.session.commit()
        
        flash(f'Telegram source "{name}" created successfully', 'success')
        return redirect(url_for('dashboard.settings'))
        
    except Exception as e:
        logger.error(f"Error creating Telegram source: {e}")
        flash(f'Error creating source: {str(e)}', 'danger')
        return redirect(url_for('dashboard.settings'))

@dashboard_bp.route('/toggle-source/<int:source_id>', methods=['POST'])
def toggle_source(source_id):
    """Toggle a data source active status."""
    try:
        source = DataSource.query.get_or_404(source_id)
        
        # Toggle status
        source.is_active = not source.is_active
        
        db.session.commit()
        
        status = "activated" if source.is_active else "deactivated"
        flash(f'Source "{source.name}" {status} successfully', 'success')
        return redirect(url_for('dashboard.settings'))
        
    except Exception as e:
        logger.error(f"Error toggling source {source_id}: {e}")
        flash(f'Error updating source: {str(e)}', 'danger')
        return redirect(url_for('dashboard.settings'))

@dashboard_bp.route('/create-rss-source', methods=['POST'])
def create_rss_source():
    """Create a new RSS data source."""
    try:
        name = request.form.get('name')
        feeds = request.form.get('feeds', '')
        
        if not name or not feeds:
            flash('Source name and feeds are required', 'danger')
            return redirect(url_for('dashboard.settings'))
            
        # Split feeds by comma and strip whitespace
        feed_list = [f.strip() for f in feeds.split(',') if f.strip()]
        
        if not feed_list:
            flash('At least one valid feed URL is required', 'danger')
            return redirect(url_for('dashboard.settings'))
        
        # Create config
        config = {'feeds': feed_list}
        
        # Import and use RSSSource
        from data_sources.rss_source import RSSSource
        rss_source = RSSSource()
        source_id = rss_source.create_source(name, config)
        
        if source_id:
            flash(f'RSS source "{name}" created successfully', 'success')
        else:
            flash('Error creating RSS source', 'danger')
            
        return redirect(url_for('dashboard.settings'))
        
    except Exception as e:
        logger.error(f"Error creating RSS source: {e}")
        flash(f'Error creating source: {str(e)}', 'danger')
        return redirect(url_for('dashboard.settings'))

@dashboard_bp.route('/test-detector', methods=['GET', 'POST'])
def test_detector():
    """Test the detector with custom content."""
    if request.method == 'POST':
        try:
            content = request.form.get('content', '')
            if not content:
                flash('Content is required', 'danger')
                return redirect(url_for('dashboard.settings'))
                
            # Import detector
            from agents.detector_agent import DetectorAgent
            from utils.text_processor import TextProcessor
            from utils.vector_store import VectorStore
            
            # Initialize components
            text_processor = TextProcessor()
            vector_store = VectorStore()
            detector = DetectorAgent(text_processor, vector_store)
            
            # Process content
            result = detector.process_content(content)
            
            # Format result for display
            detection_result = {
                'is_misinformation': result['is_misinformation'],
                'confidence': result['confidence'],
                'language': result['language'],
                'claims': result['claims']
            }
            
            flash('Detection completed successfully', 'success')
            return render_template(
                'dashboard/settings.html',
                detection_result=detection_result,
                test_content=content
            )
            
        except Exception as e:
            logger.error(f"Error testing detector: {e}")
            flash(f'Error running detection: {str(e)}', 'danger')
            return redirect(url_for('dashboard.settings'))
    
    return redirect(url_for('dashboard.settings'))
