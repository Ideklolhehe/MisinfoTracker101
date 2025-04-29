"""
Routes for comparative analysis.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from sqlalchemy import func

from app import db
from models import DetectedNarrative, NarrativeInstance, CounterMessage, DataSource
from utils.metrics import record_execution_time

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize blueprint
comparative_bp = Blueprint("comparative", __name__, url_prefix="/comparative")

class MockComparativeAnalysisService:
    """
    Mock service for comparative analysis that doesn't depend on external libraries.
    This is a temporary solution until required dependencies can be installed.
    """
    
    def __init__(self):
        """Initialize the comparative analysis service."""
        logger.info("Mock Comparative Analysis Service initialized")
        
    def compare_dimensions(self, narrative_ids: List[int], dimensions: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple narratives across different dimensions.
        
        Args:
            narrative_ids: List of narrative IDs to compare
            dimensions: List of dimensions to compare (threat_level, complexity_score, etc.)
            
        Returns:
            Dictionary with comparison data
        """
        if not dimensions:
            dimensions = ["threat_level", "propagation_rate", "complexity_score"]
            
        narratives = []
        for narrative_id in narrative_ids:
            narrative = DetectedNarrative.query.get(narrative_id)
            if narrative:
                narrative_data = {
                    "id": narrative.id,
                    "title": narrative.title,
                    "description": narrative.description,
                    "first_detected": narrative.created_at.strftime("%Y-%m-%d"),
                    "status": narrative.status,
                    "dimensions": {}
                }
                
                # Get meta_data for the narrative
                meta_data = json.loads(narrative.meta_data) if narrative.meta_data and isinstance(narrative.meta_data, str) else {}
                
                # Mock dimension values
                for dimension in dimensions:
                    if dimension == "threat_level":
                        narrative_data["dimensions"][dimension] = meta_data.get("threat_level", 3)
                    elif dimension == "propagation_rate":
                        narrative_data["dimensions"][dimension] = meta_data.get("propagation_rate", 0.75)
                    elif dimension == "complexity_score":
                        narrative_data["dimensions"][dimension] = meta_data.get("complexity_score", 0.5)
                    elif dimension == "instance_count":
                        # Count actual instances
                        count = NarrativeInstance.query.filter_by(narrative_id=narrative.id).count()
                        narrative_data["dimensions"][dimension] = count
                    elif dimension == "source_diversity":
                        # Count unique sources
                        unique_sources = db.session.query(NarrativeInstance.source_id).filter(
                            NarrativeInstance.narrative_id == narrative.id,
                            NarrativeInstance.source_id.isnot(None)
                        ).distinct().count()
                        narrative_data["dimensions"][dimension] = unique_sources
                        
                narratives.append(narrative_data)
        
        # Generate a simple HTML chart
        chart_html = f"""
        <div>
            <h4>Dimension Comparison Chart</h4>
            <p>This is a placeholder for the actual chart. In a real implementation, this would be a radar chart or bar chart comparing the narratives across the selected dimensions.</p>
        </div>
        """
        
        return {
            "narratives": narratives,
            "dimensions": dimensions,
            "chart_html": chart_html
        }
        
    def analyze_growth_rate(self, narrative_id: int, time_period: str = "30d") -> Dict[str, Any]:
        """
        Analyze the growth rate of a narrative over time.
        
        Args:
            narrative_id: ID of the narrative to analyze
            time_period: Time period for analysis (7d, 30d, 90d, all)
            
        Returns:
            Dictionary with growth rate data
        """
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return {"error": f"Narrative with ID {narrative_id} not found"}
            
        # Set time periods
        today = datetime.utcnow()
        if time_period == "7d":
            start_date = today - timedelta(days=7)
        elif time_period == "30d":
            start_date = today - timedelta(days=30)
        elif time_period == "90d":
            start_date = today - timedelta(days=90)
        else:  # "all"
            start_date = narrative.created_at
            
        # Get instances within the time period
        instances = NarrativeInstance.query.filter(
            NarrativeInstance.narrative_id == narrative_id,
            NarrativeInstance.detected_at >= start_date
        ).order_by(NarrativeInstance.detected_at).all()
        
        # Generate mock time series data
        time_points = []
        current_date = start_date
        while current_date <= today:
            time_points.append(current_date)
            current_date += timedelta(days=1)
            
        # Create cumulative counts
        cumulative_counts = [0] * len(time_points)
        daily_counts = [0] * len(time_points)
        
        for instance in instances:
            for i, date in enumerate(time_points):
                if instance.detected_at.date() <= date.date():
                    cumulative_counts[i] += 1
                if instance.detected_at.date() == date.date():
                    daily_counts[i] += 1
        
        # Calculate mock growth rates
        growth_rates = [0.0]
        for i in range(1, len(cumulative_counts)):
            if cumulative_counts[i-1] > 0:
                rate = (cumulative_counts[i] - cumulative_counts[i-1]) / cumulative_counts[i-1]
                growth_rates.append(rate)
            else:
                growth_rates.append(0.0)
                
        # Generate mock growth rate data
        chart_html = f"""
        <div>
            <h4>Growth Rate Chart</h4>
            <p>This is a placeholder for the actual chart. In a real implementation, this would be a line chart showing the growth rate over time.</p>
        </div>
        """
        
        # Calculate average growth rate
        avg_growth_rate = sum(growth_rates) / len(growth_rates) if growth_rates else 0
        
        # Detect acceleration/deceleration
        recent_growth = growth_rates[-7:] if len(growth_rates) >= 7 else growth_rates
        early_growth = growth_rates[:7] if len(growth_rates) >= 14 else growth_rates
        
        avg_recent = sum(recent_growth) / len(recent_growth) if recent_growth else 0
        avg_early = sum(early_growth) / len(early_growth) if early_growth else 0
        
        status = "accelerating" if avg_recent > avg_early else "decelerating" if avg_recent < avg_early else "stable"
        
        return {
            "narrative": {
                "id": narrative.id,
                "title": narrative.title,
                "description": narrative.description,
                "first_detected": narrative.created_at.strftime("%Y-%m-%d")
            },
            "time_points": [date.strftime("%Y-%m-%d") for date in time_points],
            "cumulative_counts": cumulative_counts,
            "daily_counts": daily_counts,
            "growth_rates": growth_rates,
            "avg_growth_rate": avg_growth_rate,
            "status": status,
            "chart_html": chart_html,
            "time_period": time_period
        }
        
    def analyze_correlation(self, narrative_id_1: int, narrative_id_2: int) -> Dict[str, Any]:
        """
        Analyze correlation between two narratives.
        
        Args:
            narrative_id_1: ID of the first narrative
            narrative_id_2: ID of the second narrative
            
        Returns:
            Dictionary with correlation data
        """
        narrative_1 = DetectedNarrative.query.get(narrative_id_1)
        narrative_2 = DetectedNarrative.query.get(narrative_id_2)
        
        if not narrative_1 or not narrative_2:
            return {"error": "One or both narratives not found"}
            
        # Generate mock correlation data
        correlation = 0.75  # Mock correlation coefficient
        
        # Determine correlation strength
        if correlation >= 0.8:
            correlation_strength = "Very Strong"
        elif correlation >= 0.6:
            correlation_strength = "Strong"
        elif correlation >= 0.4:
            correlation_strength = "Moderate"
        elif correlation >= 0.2:
            correlation_strength = "Weak"
        else:
            correlation_strength = "Very Weak"
            
        # Mock lag analysis
        lag_analysis = {
            "lag": 2,  # Positive means narrative_1 leads narrative_2
            "significance": "high"
        }
        
        # Identify shared sources
        sources_1 = set(db.session.query(NarrativeInstance.source_id).filter(
            NarrativeInstance.narrative_id == narrative_id_1,
            NarrativeInstance.source_id.isnot(None)
        ).distinct())
        
        sources_2 = set(db.session.query(NarrativeInstance.source_id).filter(
            NarrativeInstance.narrative_id == narrative_id_2,
            NarrativeInstance.source_id.isnot(None)
        ).distinct())
        
        shared_source_ids = sources_1.intersection(sources_2)
        shared_sources = []
        
        for source_id in shared_source_ids:
            source = DataSource.query.get(source_id[0])
            if source:
                shared_sources.append({
                    "id": source.id,
                    "name": source.name,
                    "type": source.source_type
                })
        
        # Generate a simple HTML chart
        chart_html = f"""
        <div>
            <h4>Correlation Analysis Chart</h4>
            <p>This is a placeholder for the actual chart. In a real implementation, this would be a scatter plot or line chart showing the correlation between the two narratives.</p>
        </div>
        """
        
        return {
            "correlation": correlation,
            "correlation_strength": correlation_strength,
            "lag_analysis": lag_analysis,
            "shared_sources": {
                "count": len(shared_sources),
                "percentage1": len(shared_sources) / len(sources_1) if sources_1 else 0,
                "percentage2": len(shared_sources) / len(sources_2) if sources_2 else 0,
                "sources": shared_sources
            },
            "chart_html": chart_html,
        }
        
    def detect_shared_themes(self, narrative_ids: List[int], n_topics: int = 5) -> Dict[str, Any]:
        """
        Detect shared themes across multiple narratives.
        
        Args:
            narrative_ids: List of narrative IDs to analyze
            n_topics: Number of themes to extract
            
        Returns:
            Dictionary with shared theme data
        """
        narratives = []
        for narrative_id in narrative_ids:
            narrative = DetectedNarrative.query.get(narrative_id)
            if narrative:
                narratives.append(narrative)
                
        if not narratives:
            return {"error": "No valid narratives found"}
            
        # Mock theme data
        themes = []
        for i in range(n_topics):
            theme = {
                "id": i,
                "name": f"Theme {i+1}",
                "top_words": [f"keyword{j}" for j in range(1, 6)],
                "narratives": {}
            }
            
            # Assign random weights to narratives
            total_weight = 1.0
            for narrative in narratives:
                weight = min(0.8, max(0.1, 0.4 + (hash(f"{narrative.id}{i}") % 50) / 100))
                theme["narratives"][narrative.id] = {
                    "title": narrative.title,
                    "weight": weight
                }
                
            themes.append(theme)
            
        # Generate a simple HTML chart
        chart_html = f"""
        <div>
            <h4>Shared Themes Chart</h4>
            <p>This is a placeholder for the actual chart. In a real implementation, this would be a heatmap or network diagram showing the themes shared across narratives.</p>
        </div>
        """
        
        return {
            "themes": themes,
            "chart_html": chart_html
        }
        
    def analyze_coordinated_sources(self, narrative_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze coordinated sources across narratives.
        
        Args:
            narrative_ids: Optional list of narrative IDs to analyze. If None, analyze all narratives.
            
        Returns:
            Dictionary with coordinated source data
        """
        # Create a mock network of sources
        nodes = []
        edges = []
        communities = []
        
        # Get source data from the database
        if narrative_ids:
            # Get sources for specified narratives
            sources_query = db.session.query(DataSource).join(
                NarrativeInstance, DataSource.id == NarrativeInstance.source_id
            ).filter(
                NarrativeInstance.narrative_id.in_(narrative_ids)
            ).distinct()
        else:
            # Get all active sources
            sources_query = DataSource.query.filter_by(is_active=True)
            
        sources = sources_query.all()
        
        # Create nodes
        for i, source in enumerate(sources):
            # Count narratives per source
            if narrative_ids:
                narrative_count = db.session.query(NarrativeInstance.narrative_id).filter(
                    NarrativeInstance.source_id == source.id,
                    NarrativeInstance.narrative_id.in_(narrative_ids)
                ).distinct().count()
            else:
                narrative_count = db.session.query(NarrativeInstance.narrative_id).filter(
                    NarrativeInstance.source_id == source.id
                ).distinct().count()
                
            nodes.append({
                "id": source.id,
                "name": source.name,
                "type": source.source_type,
                "narrative_count": narrative_count,
                "community": i % 5  # Assign to mock communities
            })
            
        # Create edges between sources that share narratives
        if len(nodes) > 1:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    # Create a mock edge with random weight
                    source_id_1 = nodes[i]["id"]
                    source_id_2 = nodes[j]["id"]
                    
                    # In a real implementation, get shared narratives
                    shared_narratives = []
                    
                    # Only create edges with some probability to prevent a fully connected graph
                    if (source_id_1 + source_id_2) % 3 == 0:
                        edges.append({
                            "source": source_id_1,
                            "target": source_id_2,
                            "weight": ((source_id_1 + source_id_2) % 5) + 1,
                            "shared_narratives": shared_narratives
                        })
                        
        # Create mock communities
        community_count = min(5, len(nodes) // 2 + 1)
        for i in range(community_count):
            community_nodes = [node["id"] for node in nodes if node["community"] == i]
            if community_nodes:
                communities.append({
                    "id": i,
                    "name": f"Community {i+1}",
                    "nodes": community_nodes,
                    "source_type": "mixed" if i % 2 == 0 else "homogeneous",
                    "modularity": 0.65 + (i % 6) / 20
                })
                
        # Generate a simple HTML chart
        chart_html = f"""
        <div>
            <h4>Coordinated Sources Network</h4>
            <p>This is a placeholder for the actual chart. In a real implementation, this would be a network diagram showing the relationships between sources spreading multiple narratives.</p>
        </div>
        """
        
        return {
            "nodes": nodes,
            "edges": edges,
            "communities": communities,
            "chart_html": chart_html
        }

# Initialize service
comparative_analysis_service = MockComparativeAnalysisService()

@comparative_bp.route("/dashboard")
@login_required
@record_execution_time
def dashboard():
    """Comparative analysis dashboard."""
    # Get counts
    narrative_count = DetectedNarrative.query.count()
    
    # Get narratives for selection
    narratives = DetectedNarrative.query.order_by(DetectedNarrative.title).all()
    
    return render_template(
        "comparative/dashboard.html",
        narrative_count=narrative_count,
        narratives=narratives
    )

@comparative_bp.route("/side-by-side", methods=["GET", "POST"])
@login_required
@record_execution_time
def side_by_side():
    """Side-by-side comparison of narratives."""
    narratives = DetectedNarrative.query.order_by(DetectedNarrative.title).all()
    
    if request.method == "POST":
        # Get selected narrative IDs
        narrative_ids = request.form.getlist("narrative_ids")
        dimensions = request.form.getlist("dimensions")
        
        if not narrative_ids or len(narrative_ids) < 2:
            flash("Please select at least two narratives for comparison.", "danger")
            return render_template(
                "comparative/side_by_side.html",
                narratives=narratives,
                error="Please select at least two narratives for comparison."
            )
            
        if not dimensions:
            flash("Please select at least one dimension for comparison.", "danger")
            return render_template(
                "comparative/side_by_side.html",
                narratives=narratives,
                error="Please select at least one dimension for comparison."
            )
            
        # Convert to integers
        narrative_ids = [int(nid) for nid in narrative_ids]
        
        # Get comparison data
        comparison_data = comparative_analysis_service.compare_dimensions(narrative_ids, dimensions)
        
        # Get selected narratives for template context
        selected_narratives = DetectedNarrative.query.filter(DetectedNarrative.id.in_(narrative_ids)).all()
        
        return render_template(
            "comparative/side_by_side.html",
            narratives=narratives,
            selected_narratives=selected_narratives,
            selected_dimensions=dimensions,
            comparison_data=comparison_data
        )
        
    return render_template(
        "comparative/side_by_side.html",
        narratives=narratives
    )

@comparative_bp.route("/growth-rate/<int:narrative_id>", methods=["GET", "POST"])
@login_required
@record_execution_time
def growth_rate(narrative_id):
    """Analyze growth rate for a narrative."""
    narrative = DetectedNarrative.query.get_or_404(narrative_id)
    
    time_period = request.args.get("time_period", "30d")
    if request.method == "POST":
        time_period = request.form.get("time_period", "30d")
        
    growth_data = comparative_analysis_service.analyze_growth_rate(narrative_id, time_period)
    
    if "error" in growth_data:
        flash(growth_data["error"], "danger")
        return redirect(url_for("comparative.dashboard"))
        
    return render_template(
        "comparative/growth_rate.html",
        narrative=narrative,
        time_period=time_period,
        growth_data=growth_data
    )

@comparative_bp.route("/correlation", methods=["GET", "POST"])
@login_required
@record_execution_time
def correlation():
    """Analyze correlation between two narratives."""
    narratives = DetectedNarrative.query.order_by(DetectedNarrative.title).all()
    
    if request.method == "POST":
        narrative_id_1 = request.form.get("narrative_id_1")
        narrative_id_2 = request.form.get("narrative_id_2")
        
        if not narrative_id_1 or not narrative_id_2:
            flash("Please select two narratives for correlation analysis.", "danger")
            return render_template(
                "comparative/correlation.html",
                narratives=narratives,
                error="Please select two narratives for correlation analysis."
            )
            
        if narrative_id_1 == narrative_id_2:
            flash("Please select two different narratives for correlation analysis.", "danger")
            return render_template(
                "comparative/correlation.html",
                narratives=narratives,
                error="Please select two different narratives for correlation analysis."
            )
            
        # Convert to integers
        narrative_id_1 = int(narrative_id_1)
        narrative_id_2 = int(narrative_id_2)
        
        # Get correlation data
        correlation_data = comparative_analysis_service.analyze_correlation(narrative_id_1, narrative_id_2)
        
        if "error" in correlation_data:
            flash(correlation_data["error"], "danger")
            return render_template(
                "comparative/correlation.html",
                narratives=narratives,
                error=correlation_data["error"]
            )
            
        # Get narratives for template context
        narrative_1 = DetectedNarrative.query.get(narrative_id_1)
        narrative_2 = DetectedNarrative.query.get(narrative_id_2)
        
        return render_template(
            "comparative/correlation.html",
            narratives=narratives,
            narrative_1=narrative_1,
            narrative_2=narrative_2,
            correlation_data=correlation_data
        )
        
    return render_template(
        "comparative/correlation.html",
        narratives=narratives
    )

@comparative_bp.route("/shared-themes", methods=["GET", "POST"])
@login_required
@record_execution_time
def shared_themes():
    """Detect shared themes across narratives."""
    narratives = DetectedNarrative.query.order_by(DetectedNarrative.title).all()
    
    if request.method == "POST":
        narrative_ids = request.form.getlist("narrative_ids")
        n_topics = int(request.form.get("n_topics", 5))
        
        if not narrative_ids or len(narrative_ids) < 2:
            flash("Please select at least two narratives for theme detection.", "danger")
            return render_template(
                "comparative/shared_themes.html",
                narratives=narratives,
                error="Please select at least two narratives for theme detection."
            )
            
        # Convert to integers
        narrative_ids = [int(nid) for nid in narrative_ids]
        
        # Get themes data
        themes_data = comparative_analysis_service.detect_shared_themes(narrative_ids, n_topics)
        
        if "error" in themes_data:
            flash(themes_data["error"], "danger")
            return render_template(
                "comparative/shared_themes.html",
                narratives=narratives,
                error=themes_data["error"]
            )
            
        # Get selected narratives for template context
        selected_narratives = DetectedNarrative.query.filter(DetectedNarrative.id.in_(narrative_ids)).all()
        
        return render_template(
            "comparative/shared_themes.html",
            narratives=narratives,
            selected_narratives=selected_narratives,
            n_topics=n_topics,
            themes_data=themes_data
        )
        
    return render_template(
        "comparative/shared_themes.html",
        narratives=narratives
    )

@comparative_bp.route("/coordinate-sources", methods=["GET", "POST"])
@login_required
@record_execution_time
def coordinate_sources():
    """Analyze coordinated sources."""
    narratives = DetectedNarrative.query.order_by(DetectedNarrative.title).all()
    
    if request.method == "POST":
        analysis_type = request.form.get("analysis_type", "global")
        narrative_ids = None
        
        if analysis_type == "specific":
            narrative_ids = request.form.getlist("narrative_ids")
            if not narrative_ids:
                flash("Please select at least one narrative for specific analysis.", "danger")
                return render_template(
                    "comparative/coordinate_sources.html",
                    narratives=narratives,
                    error="Please select at least one narrative for specific analysis."
                )
                
            # Convert to integers
            narrative_ids = [int(nid) for nid in narrative_ids]
            
        # Get source data
        source_data = comparative_analysis_service.analyze_coordinated_sources(narrative_ids)
        
        # Get selected narratives for template context
        selected_narratives = DetectedNarrative.query.filter(DetectedNarrative.id.in_(narrative_ids)).all() if narrative_ids else None
        
        return render_template(
            "comparative/coordinate_sources.html",
            narratives=narratives,
            selected_narratives=selected_narratives,
            source_data=source_data
        )
        
    return render_template(
        "comparative/coordinate_sources.html",
        narratives=narratives
    )

# API Routes

@comparative_bp.route("/api/side-by-side", methods=["POST"])
@login_required
@record_execution_time
def api_side_by_side():
    """API endpoint for side-by-side comparison."""
    data = request.json
    narrative_ids = data.get("narrative_ids", [])
    dimensions = data.get("dimensions", [])
    
    if not narrative_ids or len(narrative_ids) < 2:
        return jsonify({"error": "Please select at least two narratives for comparison."}), 400
        
    if not dimensions:
        return jsonify({"error": "Please select at least one dimension for comparison."}), 400
        
    # Get comparison data
    comparison_data = comparative_analysis_service.compare_dimensions(narrative_ids, dimensions)
    return jsonify(comparison_data)

@comparative_bp.route("/api/growth-rate/<int:narrative_id>", methods=["GET"])
@login_required
@record_execution_time
def api_growth_rate(narrative_id):
    """API endpoint for growth rate analysis."""
    time_period = request.args.get("time_period", "30d")
    growth_data = comparative_analysis_service.analyze_growth_rate(narrative_id, time_period)
    return jsonify(growth_data)

@comparative_bp.route("/api/correlation", methods=["POST"])
@login_required
@record_execution_time
def api_correlation():
    """API endpoint for correlation analysis."""
    data = request.json
    narrative_id_1 = data.get("narrative_id_1")
    narrative_id_2 = data.get("narrative_id_2")
    
    if not narrative_id_1 or not narrative_id_2:
        return jsonify({"error": "Please provide two narrative IDs for correlation analysis."}), 400
        
    if narrative_id_1 == narrative_id_2:
        return jsonify({"error": "Please provide two different narrative IDs for correlation analysis."}), 400
        
    # Get correlation data
    correlation_data = comparative_analysis_service.analyze_correlation(narrative_id_1, narrative_id_2)
    return jsonify(correlation_data)

@comparative_bp.route("/api/shared-themes", methods=["POST"])
@login_required
@record_execution_time
def api_shared_themes():
    """API endpoint for shared theme detection."""
    data = request.json
    narrative_ids = data.get("narrative_ids", [])
    n_topics = data.get("n_topics", 5)
    
    if not narrative_ids or len(narrative_ids) < 2:
        return jsonify({"error": "Please select at least two narratives for theme detection."}), 400
        
    # Get themes data
    themes_data = comparative_analysis_service.detect_shared_themes(narrative_ids, n_topics)
    return jsonify(themes_data)

@comparative_bp.route("/api/coordinate-sources", methods=["POST"])
@login_required
@record_execution_time
def api_coordinate_sources():
    """API endpoint for coordinated source analysis."""
    data = request.json
    narrative_ids = data.get("narrative_ids")
    
    # Get source data
    source_data = comparative_analysis_service.analyze_coordinated_sources(narrative_ids)
    return jsonify(source_data)