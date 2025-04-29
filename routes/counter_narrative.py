"""
Routes for counter-narrative generation and management.
"""

import logging
from typing import Dict, List, Any, Optional
import json

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from sqlalchemy import func

from app import db
from models import DetectedNarrative, NarrativeInstance, CounterMessage, DataSource
from utils.metrics import record_execution_time

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize blueprint
counter_narrative_bp = Blueprint("counter_narrative", __name__, url_prefix="/counter-narrative")

class MockCounterNarrativeService:
    """
    A simple mock version of the CounterNarrativeService that doesn't depend on transformers.
    This is a temporary solution until we can install the required dependencies.
    """
    
    def __init__(self):
        """Initialize the counter narrative service."""
        logger.info("Mock Counter Narrative Service initialized")
        
    def generate_counter_narrative(self, narrative_id: int, strategy: str = None) -> Dict[str, Any]:
        """Generate a counter-narrative for a detected narrative."""
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return {"error": f"Narrative with ID {narrative_id} not found"}
            
        # Get existing counter messages
        existing_counter_messages = CounterMessage.query.filter_by(narrative_id=narrative_id).all()
        
        # Create a mock counter narrative
        counter_message = {
            "id": len(existing_counter_messages) + 1,
            "content": f"This is a mock counter-narrative for '{narrative.title}'. In a real implementation, this would be generated using AI.",
            "strategy": strategy or "reframing",
            "effectiveness_score": 0.75,
            "channels": ["twitter", "facebook", "news_outlets"],
            "targeting": {
                "age_groups": ["18-24", "25-34"],
                "regions": ["North America", "Europe"],
                "demographics": ["general_public", "opinion_leaders"]
            }
        }
        
        return {
            "narrative": {
                "id": narrative.id,
                "title": narrative.title,
                "description": narrative.description
            },
            "counter_message": counter_message
        }
        
    def save_counter_message(self, narrative_id: int, content: str, strategy: str, targeting: Dict[str, Any] = None) -> CounterMessage:
        """Save a counter message to the database."""
        counter_message = CounterMessage(
            narrative_id=narrative_id,
            content=content,
            strategy=strategy,
            effectiveness_score=0.0,  # Initial score
            meta_data=json.dumps(targeting) if targeting else "{}"
        )
        
        db.session.add(counter_message)
        db.session.commit()
        
        return counter_message
        
    def get_counter_messages(self, narrative_id: int = None) -> List[Dict[str, Any]]:
        """Get counter messages for a narrative or all counter messages."""
        if narrative_id:
            query = CounterMessage.query.filter_by(narrative_id=narrative_id)
        else:
            query = CounterMessage.query
            
        counter_messages = query.order_by(CounterMessage.created_at.desc()).all()
        
        result = []
        for cm in counter_messages:
            narrative = DetectedNarrative.query.get(cm.narrative_id)
            
            result.append({
                "id": cm.id,
                "narrative_id": cm.narrative_id,
                "narrative_title": narrative.title if narrative else "Unknown",
                "content": cm.content,
                "strategy": cm.strategy,
                "effectiveness_score": cm.effectiveness_score,
                "created_at": cm.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "targeting": json.loads(cm.meta_data) if cm.meta_data and isinstance(cm.meta_data, str) else {}
            })
            
        return result
        
    def get_strategies(self) -> List[Dict[str, Any]]:
        """Get available counter-narrative strategies."""
        return [
            {
                "id": "fact_checking",
                "name": "Fact Checking",
                "description": "Directly debunk false claims with factual information from trusted sources."
            },
            {
                "id": "reframing",
                "name": "Reframing",
                "description": "Shift the perspective on the issue to highlight different aspects and provide context."
            },
            {
                "id": "alternative_narrative",
                "name": "Alternative Narrative",
                "description": "Present an entirely different narrative that addresses the same concerns but with accurate information."
            },
            {
                "id": "inoculation",
                "name": "Inoculation",
                "description": "Preemptively expose people to weakened forms of misinformation along with counter-arguments."
            },
            {
                "id": "emotional_appeal",
                "name": "Emotional Appeal",
                "description": "Address the emotional core of the narrative rather than just the factual content."
            }
        ]
        
    def track_effectiveness(self, counter_message_id: int, engagement_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track the effectiveness of a counter message based on engagement metrics."""
        counter_message = CounterMessage.query.get(counter_message_id)
        if not counter_message:
            return {"error": f"Counter message with ID {counter_message_id} not found"}
            
        # Calculate a mock effectiveness score based on engagement metrics
        reach = engagement_metrics.get("reach", 0)
        shares = engagement_metrics.get("shares", 0)
        positive_reactions = engagement_metrics.get("positive_reactions", 0)
        
        # Simple formula for demonstration purposes
        effectiveness = min(1.0, (0.4 * (reach / 1000) + 0.3 * (shares / 100) + 0.3 * (positive_reactions / 100)))
        
        # Update the counter message
        counter_message.effectiveness_score = effectiveness
        
        # Store the engagement metrics in meta_data
        meta_data = json.loads(counter_message.meta_data) if counter_message.meta_data and isinstance(counter_message.meta_data, str) else {}
        meta_data["engagement_metrics"] = engagement_metrics
        counter_message.meta_data = json.dumps(meta_data)
        
        db.session.commit()
        
        return {
            "counter_message_id": counter_message.id,
            "effectiveness_score": effectiveness,
            "engagement_metrics": engagement_metrics
        }

# Initialize service
counter_narrative_service = MockCounterNarrativeService()

@counter_narrative_bp.route("/dashboard")
@login_required
@record_execution_time
def dashboard():
    """Counter-narrative dashboard."""
    # Get counts
    narrative_count = DetectedNarrative.query.count()
    counter_message_count = CounterMessage.query.count()
    
    # Get narratives with counter messages
    narratives_with_counters = db.session.query(
        DetectedNarrative, func.count(CounterMessage.id).label('counter_count')
    ).outerjoin(
        CounterMessage, DetectedNarrative.id == CounterMessage.narrative_id
    ).group_by(
        DetectedNarrative.id
    ).having(
        func.count(CounterMessage.id) > 0
    ).order_by(
        func.count(CounterMessage.id).desc()
    ).all()
    
    # Get recent counter messages
    recent_counter_messages = counter_narrative_service.get_counter_messages()[:10]
    
    # Get strategies
    strategies = counter_narrative_service.get_strategies()
    
    return render_template(
        "counter_narrative/dashboard.html",
        narrative_count=narrative_count,
        counter_message_count=counter_message_count,
        narratives_with_counters=narratives_with_counters,
        recent_counter_messages=recent_counter_messages,
        strategies=strategies
    )

@counter_narrative_bp.route("/generate/<int:narrative_id>", methods=["GET", "POST"])
@login_required
@record_execution_time
def generate_counter(narrative_id):
    """Generate a counter-narrative for a detected narrative."""
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    if request.method == "POST":
        strategy = request.form.get("strategy")
        
        # Generate counter narrative
        result = counter_narrative_service.generate_counter_narrative(narrative_id, strategy)
        
        if "error" in result:
            flash(result["error"], "danger")
            return redirect(url_for("counter_narrative.generate_counter", narrative_id=narrative_id))
            
        # Extract counter message
        counter_message = result["counter_message"]
        
        # Prepare targeting data
        targeting = {
            "channels": request.form.getlist("channels"),
            "age_groups": request.form.getlist("age_groups"),
            "regions": request.form.getlist("regions"),
            "demographics": request.form.getlist("demographics")
        }
        
        # Save counter message
        saved_message = counter_narrative_service.save_counter_message(
            narrative_id=narrative_id,
            content=counter_message["content"],
            strategy=strategy,
            targeting=targeting
        )
        
        flash("Counter-narrative generated and saved successfully!", "success")
        return redirect(url_for("counter_narrative.view_counter", counter_id=saved_message.id))
    
    # Get available strategies for the form
    strategies = counter_narrative_service.get_strategies()
    
    return render_template(
        "counter_narrative/generate.html",
        narrative=narrative,
        strategies=strategies
    )

@counter_narrative_bp.route("/view/<int:counter_id>")
@login_required
@record_execution_time
def view_counter(counter_id):
    """View a counter-narrative."""
    counter_message = CounterMessage.query.get_or_404(counter_id)
    narrative = DetectedNarrative.query.get(counter_message.narrative_id)
    
    # Parse meta_data JSON
    targeting = {}
    if counter_message.meta_data and isinstance(counter_message.meta_data, str):
        try:
            targeting = json.loads(counter_message.meta_data)
        except:
            targeting = {}
    
    return render_template(
        "counter_narrative/view.html",
        counter_message=counter_message,
        narrative=narrative,
        targeting=targeting
    )

@counter_narrative_bp.route("/list")
@login_required
@record_execution_time
def list_counters():
    """List all counter-narratives."""
    counter_messages = counter_narrative_service.get_counter_messages()
    
    return render_template(
        "counter_narrative/list.html",
        counter_messages=counter_messages
    )

@counter_narrative_bp.route("/track-effectiveness/<int:counter_id>", methods=["GET", "POST"])
@login_required
@record_execution_time
def track_effectiveness(counter_id):
    """Track the effectiveness of a counter-narrative."""
    counter_message = CounterMessage.query.get_or_404(counter_id)
    
    if request.method == "POST":
        # Extract metrics from form
        engagement_metrics = {
            "reach": int(request.form.get("reach", 0)),
            "shares": int(request.form.get("shares", 0)),
            "positive_reactions": int(request.form.get("positive_reactions", 0)),
            "negative_reactions": int(request.form.get("negative_reactions", 0)),
            "comments": int(request.form.get("comments", 0))
        }
        
        # Track effectiveness
        result = counter_narrative_service.track_effectiveness(counter_id, engagement_metrics)
        
        if "error" in result:
            flash(result["error"], "danger")
        else:
            flash(f"Effectiveness tracked successfully! Score: {result['effectiveness_score']:.2f}", "success")
            
        return redirect(url_for("counter_narrative.view_counter", counter_id=counter_id))
    
    return render_template(
        "counter_narrative/track_effectiveness.html",
        counter_message=counter_message
    )

# API Routes

@counter_narrative_bp.route("/api/generate", methods=["POST"])
@login_required
@record_execution_time
def api_generate():
    """API endpoint for generating a counter-narrative."""
    data = request.json
    narrative_id = data.get("narrative_id")
    strategy = data.get("strategy")
    
    if not narrative_id:
        return jsonify({"error": "Narrative ID is required"}), 400
        
    result = counter_narrative_service.generate_counter_narrative(narrative_id, strategy)
    return jsonify(result)

@counter_narrative_bp.route("/api/save", methods=["POST"])
@login_required
@record_execution_time
def api_save():
    """API endpoint for saving a counter-narrative."""
    data = request.json
    narrative_id = data.get("narrative_id")
    content = data.get("content")
    strategy = data.get("strategy")
    targeting = data.get("targeting")
    
    if not narrative_id or not content or not strategy:
        return jsonify({"error": "Narrative ID, content, and strategy are required"}), 400
        
    counter_message = counter_narrative_service.save_counter_message(
        narrative_id=narrative_id,
        content=content,
        strategy=strategy,
        targeting=targeting
    )
    
    return jsonify({
        "id": counter_message.id,
        "narrative_id": counter_message.narrative_id,
        "content": counter_message.content,
        "strategy": counter_message.strategy,
        "created_at": counter_message.created_at.strftime("%Y-%m-%d %H:%M:%S")
    })

@counter_narrative_bp.route("/api/list", methods=["GET"])
@login_required
@record_execution_time
def api_list():
    """API endpoint for listing counter-narratives."""
    narrative_id = request.args.get("narrative_id", None)
    if narrative_id:
        try:
            narrative_id = int(narrative_id)
        except:
            narrative_id = None
            
    counter_messages = counter_narrative_service.get_counter_messages(narrative_id)
    return jsonify(counter_messages)

@counter_narrative_bp.route("/api/strategies", methods=["GET"])
@login_required
@record_execution_time
def api_strategies():
    """API endpoint for getting available strategies."""
    strategies = counter_narrative_service.get_strategies()
    return jsonify(strategies)

@counter_narrative_bp.route("/api/track-effectiveness", methods=["POST"])
@login_required
@record_execution_time
def api_track_effectiveness():
    """API endpoint for tracking counter-narrative effectiveness."""
    data = request.json
    counter_id = data.get("counter_id")
    engagement_metrics = data.get("engagement_metrics")
    
    if not counter_id or not engagement_metrics:
        return jsonify({"error": "Counter ID and engagement metrics are required"}), 400
        
    result = counter_narrative_service.track_effectiveness(counter_id, engagement_metrics)
    return jsonify(result)