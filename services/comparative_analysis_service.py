"""
Service for comparative analysis of narratives and counter-narratives.
"""

import logging
import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Import database models
from app import db
from models import DetectedNarrative, NarrativeInstance, DataSource, CounterMessage
from utils.app_context import with_app_context

# Configure logger
logger = logging.getLogger(__name__)

class ComparativeAnalysisService:
    """Service for comparative analysis of narratives."""

    def __init__(self):
        """Initialize the comparative analysis service."""
        logger.info("Comparative Analysis Service initialized")
        self.lock = threading.RLock()
        
    def side_by_side_comparison(self, narrative_ids: List[int], dimensions: List[str] = None) -> Dict[str, Any]:
        """Perform side-by-side comparison of narratives across dimensions."""
        if dimensions is None:
            dimensions = ["threat_level", "propagation_rate", "complexity_score"]
            
        narratives = DetectedNarrative.query.filter(DetectedNarrative.id.in_(narrative_ids)).all()
        if not narratives:
            return {"error": "No narratives found with the provided IDs"}
            
        comparison_data = {
            "dimensions": dimensions,
            "narratives": [],
            "chart_data": {}
        }
        
        # Create default chart data structure for each dimension
        for dimension in dimensions:
            comparison_data["chart_data"][dimension] = {
                "labels": [],
                "values": []
            }
        
        # Collect data for each narrative
        for narrative in narratives:
            narrative_data = {
                "id": narrative.id,
                "title": narrative.title,
                "description": narrative.description,
                "first_detected": narrative.created_at.strftime("%Y-%m-%d"),
                "status": narrative.status,
                "dimensions": {}
            }
            
            # Get data for each dimension
            for dimension in dimensions:
                if dimension == "threat_level":
                    value = narrative.threat_level or 0
                elif dimension == "propagation_rate":
                    value = narrative.propagation_rate or 0.0
                elif dimension == "complexity_score":
                    value = narrative.complexity_score or 0.0
                elif dimension == "instance_count":
                    value = len(narrative.instances)
                elif dimension == "source_diversity":
                    # Count unique sources
                    sources = set()
                    for instance in narrative.instances:
                        if instance.source_id:
                            sources.add(instance.source_id)
                    value = len(sources)
                else:
                    # Try to get from meta_data
                    meta_data = narrative.meta_data or {}
                    if isinstance(meta_data, str):
                        try:
                            meta_data = json.loads(meta_data)
                        except:
                            meta_data = {}
                    value = meta_data.get(dimension, 0)
                
                narrative_data["dimensions"][dimension] = value
                
                # Add to chart data
                comparison_data["chart_data"][dimension]["labels"].append(narrative.title)
                comparison_data["chart_data"][dimension]["values"].append(value)
            
            comparison_data["narratives"].append(narrative_data)
            
        # Generate mock chart HTML for visualization
        chart_html = self._generate_mock_chart(comparison_data)
        comparison_data["chart_html"] = chart_html
            
        return comparison_data
    
    def relative_growth_rate(self, narrative_id: int, days: int = 30) -> Dict[str, Any]:
        """Calculate and visualize the relative growth rate for a narrative."""
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return {"error": f"Narrative with ID {narrative_id} not found"}
            
        # Set time range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get instances within time range
        instances = db.session.query(NarrativeInstance).filter(
            NarrativeInstance.narrative_id == narrative_id,
            NarrativeInstance.created_at >= start_date,
            NarrativeInstance.created_at <= end_date
        ).order_by(NarrativeInstance.created_at).all()
        
        # Group instances by day
        daily_counts = defaultdict(int)
        for instance in instances:
            day_key = instance.created_at.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1
            
        # Fill in missing days
        current_date = start_date
        all_days = []
        while current_date <= end_date:
            day_key = current_date.strftime("%Y-%m-%d")
            if day_key not in daily_counts:
                daily_counts[day_key] = 0
            all_days.append(day_key)
            current_date += timedelta(days=1)
            
        # Sort days
        all_days.sort()
        
        # Calculate cumulative counts and growth rates
        counts = [daily_counts[day] for day in all_days]
        cumulative_counts = []
        running_total = 0
        for count in counts:
            running_total += count
            cumulative_counts.append(running_total)
            
        # Calculate daily growth rates
        growth_rates = [0]  # First day has no growth rate
        for i in range(1, len(cumulative_counts)):
            prev = cumulative_counts[i-1]
            curr = cumulative_counts[i]
            if prev > 0:
                growth_rate = (curr - prev) / prev
            else:
                growth_rate = 0 if curr == 0 else 1  # 100% growth if starting from 0
            growth_rates.append(growth_rate)
        
        # Prepare data for response
        data = {
            "dates": all_days,
            "daily_counts": counts,
            "cumulative_counts": cumulative_counts,
            "growth_rates": growth_rates
        }
        
        # Generate mock chart HTML
        chart_html = self._generate_mock_growth_chart(data)
        
        return {
            "data": data,
            "chart_html": chart_html
        }
    
    def identify_correlation(self, narrative_id_1: int, narrative_id_2: int) -> Dict[str, Any]:
        """Identify correlation between two narratives over time."""
        # Get both narratives
        narrative1 = DetectedNarrative.query.get(narrative_id_1)
        narrative2 = DetectedNarrative.query.get(narrative_id_2)
        
        if not narrative1 or not narrative2:
            return {"error": "One or both narratives not found"}
            
        # Set time range - use last 90 days by default
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # Get instances for both narratives within time range
        instances1 = db.session.query(NarrativeInstance).filter(
            NarrativeInstance.narrative_id == narrative_id_1,
            NarrativeInstance.created_at >= start_date,
            NarrativeInstance.created_at <= end_date
        ).order_by(NarrativeInstance.created_at).all()
        
        instances2 = db.session.query(NarrativeInstance).filter(
            NarrativeInstance.narrative_id == narrative_id_2,
            NarrativeInstance.created_at >= start_date,
            NarrativeInstance.created_at <= end_date
        ).order_by(NarrativeInstance.created_at).all()
        
        # Group instances by day
        daily_counts1 = defaultdict(int)
        for instance in instances1:
            day_key = instance.created_at.strftime("%Y-%m-%d")
            daily_counts1[day_key] += 1
            
        daily_counts2 = defaultdict(int)
        for instance in instances2:
            day_key = instance.created_at.strftime("%Y-%m-%d")
            daily_counts2[day_key] += 1
            
        # Get all unique days
        all_days = sorted(set(list(daily_counts1.keys()) + list(daily_counts2.keys())))
        
        # Fill in missing days for both narratives
        for day in all_days:
            if day not in daily_counts1:
                daily_counts1[day] = 0
            if day not in daily_counts2:
                daily_counts2[day] = 0
                
        # Sort all days
        all_days.sort()
        
        # Extract counts in order of days
        counts1 = [daily_counts1[day] for day in all_days]
        counts2 = [daily_counts2[day] for day in all_days]
        
        # Simple correlation calculation
        correlation = 0.0
        if len(counts1) > 1:  # Need at least 2 points for correlation
            # Calculate means
            mean1 = sum(counts1) / len(counts1)
            mean2 = sum(counts2) / len(counts2)
            
            # Calculate numerator (covariance)
            numerator = sum((c1 - mean1) * (c2 - mean2) for c1, c2 in zip(counts1, counts2))
            
            # Calculate denominators (standard deviations)
            sum_sq1 = sum((c1 - mean1) ** 2 for c1 in counts1)
            sum_sq2 = sum((c2 - mean2) ** 2 for c2 in counts2)
            
            # Calculate correlation
            if sum_sq1 > 0 and sum_sq2 > 0:
                correlation = numerator / ((sum_sq1 ** 0.5) * (sum_sq2 ** 0.5))
        
        # Determine lag or lead relationship
        lag_analysis = self._analyze_lag(counts1, counts2)
        
        # Calculate shared sources
        shared_sources = self._calculate_shared_sources(narrative_id_1, narrative_id_2)
        
        # Prepare data for response
        data = {
            "dates": all_days,
            "counts_narrative1": counts1,
            "counts_narrative2": counts2,
            "correlation": correlation,
            "correlation_strength": self._interpret_correlation(correlation),
            "lag_analysis": lag_analysis,
            "shared_sources": shared_sources
        }
        
        # Generate mock chart HTML
        chart_html = self._generate_mock_correlation_chart(data)
        
        return {
            "data": data,
            "chart_html": chart_html
        }
    
    def shared_theme_detection(self, narrative_ids: List[int], n_topics: int = 5) -> Dict[str, Any]:
        """Detect shared themes across narratives using simple text analysis."""
        narratives = DetectedNarrative.query.filter(DetectedNarrative.id.in_(narrative_ids)).all()
        if not narratives:
            return {"error": "No narratives found with the provided IDs"}
            
        # Collect all text from each narrative and its instances
        narrative_texts = {}
        all_text = ""
        
        for narrative in narratives:
            # Concatenate narrative title and description
            narrative_text = f"{narrative.title} {narrative.description}"
            
            # Add text from instances
            for instance in narrative.instances:
                narrative_text += f" {instance.content}"
                
            narrative_texts[narrative.id] = narrative_text
            all_text += f" {narrative_text}"
            
        # Simple word frequency analysis for themes
        # In a real implementation, this would use proper NLP with LDA topic modeling
        word_freq = self._simple_word_frequency(all_text)
        
        # Create mock themes
        themes = []
        for i in range(1, n_topics + 1):
            theme = {
                "id": i,
                "name": f"Theme {i}",
                "top_words": [word for word, _ in word_freq[:5]],
                "weight": 1.0 / n_topics,
                "narratives": {}
            }
            
            # Assign random theme distribution to each narrative
            for narrative in narratives:
                theme["narratives"][narrative.id] = {
                    "title": narrative.title,
                    "weight": 0.5 + (i * 0.1) % 0.5  # Mock weight between 0.5 and 1.0
                }
                
            themes.append(theme)
            
        # Generate mock chart HTML
        chart_html = self._generate_mock_theme_chart(themes)
        
        return {
            "themes": themes,
            "chart_html": chart_html
        }
    
    def coordinated_source_analysis(
        self, 
        narrative_ids: Optional[List[int]] = None, 
        edges: Optional[List[List[int]]] = None
    ) -> Dict[str, Any]:
        """Analyze source coordination using network analysis."""
        # Determine if we're analyzing specific narratives or all
        if narrative_ids:
            # Get instances for specified narratives
            instances = db.session.query(NarrativeInstance).filter(
                NarrativeInstance.narrative_id.in_(narrative_ids)
            ).all()
        else:
            # Get all instances
            instances = db.session.query(NarrativeInstance).all()
            
        # Build a dictionary of sources and their connections
        source_connections = defaultdict(set)
        for instance in instances:
            if instance.source_id and instance.narrative_id:
                source_connections[instance.source_id].add(instance.narrative_id)
                
        # Build graph nodes (sources)
        nodes = []
        for source_id, narratives in source_connections.items():
            source = DataSource.query.get(source_id)
            if source:
                nodes.append({
                    "id": source.id,
                    "name": source.name,
                    "type": source.type,
                    "narrative_count": len(narratives),
                    "narratives": list(narratives)
                })
        
        # Build graph edges (connections between sources sharing narratives)
        graph_edges = []
        node_ids = [node["id"] for node in nodes]
        
        for i, source_id_1 in enumerate(node_ids):
            for j in range(i + 1, len(node_ids)):
                source_id_2 = node_ids[j]
                
                # Find shared narratives
                narratives_1 = source_connections[source_id_1]
                narratives_2 = source_connections[source_id_2]
                shared = narratives_1.intersection(narratives_2)
                
                if shared:
                    graph_edges.append({
                        "source": source_id_1,
                        "target": source_id_2,
                        "weight": len(shared),
                        "shared_narratives": list(shared)
                    })
        
        # Find communities (this would use NetworkX in a real implementation)
        communities = self._simple_community_detection(nodes, graph_edges)
        
        # Generate mock network chart HTML
        chart_html = self._generate_mock_network_chart(nodes, graph_edges, communities)
        
        return {
            "nodes": nodes,
            "edges": graph_edges,
            "communities": communities,
            "chart_html": chart_html
        }
        
    # Helper methods
    
    def _simple_word_frequency(self, text: str) -> List[tuple]:
        """Perform simple word frequency analysis."""
        # Split text into words
        words = text.lower().split()
        
        # Remove common stopwords
        stopwords = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are"}
        words = [word for word in words if word not in stopwords]
        
        # Count frequencies
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1
            
        # Return sorted by frequency
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    def _analyze_lag(self, series1: List[float], series2: List[float]) -> Dict[str, Any]:
        """Analyze if one series leads or lags another."""
        # This would use cross-correlation analysis in a real implementation
        # For now, return a simple mock result
        return {
            "max_correlation": 0.65,
            "lag": 2,  # Positive means series2 lags series1 (series1 leads)
            "direction": "lead",
            "significance": "medium"
        }
    
    def _calculate_shared_sources(self, narrative_id_1: int, narrative_id_2: int) -> Dict[str, Any]:
        """Calculate shared sources between two narratives."""
        # Get sources for each narrative
        sources1 = set()
        instances1 = NarrativeInstance.query.filter_by(narrative_id=narrative_id_1).all()
        for instance in instances1:
            if instance.source_id:
                sources1.add(instance.source_id)
                
        sources2 = set()
        instances2 = NarrativeInstance.query.filter_by(narrative_id=narrative_id_2).all()
        for instance in instances2:
            if instance.source_id:
                sources2.add(instance.source_id)
                
        # Find intersection
        shared = sources1.intersection(sources2)
        
        # Get source details
        shared_sources = []
        for source_id in shared:
            source = DataSource.query.get(source_id)
            if source:
                shared_sources.append({
                    "id": source.id,
                    "name": source.name,
                    "type": source.type
                })
                
        return {
            "count": len(shared),
            "sources": shared_sources,
            "percentage1": len(shared) / len(sources1) if sources1 else 0,
            "percentage2": len(shared) / len(sources2) if sources2 else 0
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret the correlation coefficient."""
        abs_corr = abs(correlation)
        if abs_corr < 0.2:
            return "Very Weak"
        elif abs_corr < 0.4:
            return "Weak"
        elif abs_corr < 0.6:
            return "Moderate"
        elif abs_corr < 0.8:
            return "Strong"
        else:
            return "Very Strong"
    
    def _simple_community_detection(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Perform simple community detection on the network."""
        # This would use NetworkX's community detection in a real implementation
        # For now, create a simple mock result
        if not nodes:
            return []
            
        communities = []
        # Organize communities by source type
        source_types = set(node["type"] for node in nodes if "type" in node)
        
        for i, source_type in enumerate(source_types):
            community = {
                "id": i + 1,
                "name": f"Community {i + 1}: {source_type}",
                "source_type": source_type,
                "nodes": [node["id"] for node in nodes if node.get("type") == source_type],
                "modularity": 0.5 + (i * 0.1) % 0.5  # Mock value between 0.5 and 1.0
            }
            communities.append(community)
            
        return communities
    
    def _generate_mock_chart(self, data: Dict[str, Any]) -> str:
        """Generate a mock chart HTML for visualization."""
        # In a real implementation, this would create a Plotly chart
        # For now, return a simple HTML scaffold for the chart
        html = """
        <div class="chart-container">
            <h4>Comparative Analysis Chart</h4>
            <div class="alert alert-info">
                Visualization would appear here with Plotly.js
            </div>
            <div class="chart-description">
                <p>This chart would show a side-by-side comparison of the selected narratives across dimensions.</p>
            </div>
        </div>
        """
        return html
    
    def _generate_mock_growth_chart(self, data: Dict[str, Any]) -> str:
        """Generate a mock growth rate chart HTML."""
        # In a real implementation, this would create a Plotly chart
        html = """
        <div class="chart-container">
            <h4>Growth Rate Analysis</h4>
            <div class="alert alert-info">
                Growth rate visualization would appear here with Plotly.js
            </div>
            <div class="chart-description">
                <p>This chart would show the growth trajectory of the narrative over time.</p>
            </div>
        </div>
        """
        return html
    
    def _generate_mock_correlation_chart(self, data: Dict[str, Any]) -> str:
        """Generate a mock correlation chart HTML."""
        # In a real implementation, this would create a Plotly chart
        correlation = data.get("correlation", 0)
        strength = data.get("correlation_strength", "Unknown")
        
        html = f"""
        <div class="chart-container">
            <h4>Correlation Analysis</h4>
            <div class="alert alert-info">
                Correlation visualization would appear here with Plotly.js
            </div>
            <div class="chart-description">
                <p>Correlation coefficient: <strong>{correlation:.2f}</strong> ({strength})</p>
                <p>This chart would show the temporal correlation between the two narratives.</p>
            </div>
        </div>
        """
        return html
    
    def _generate_mock_theme_chart(self, themes: List[Dict]) -> str:
        """Generate a mock theme chart HTML."""
        # In a real implementation, this would create a Plotly chart
        html = """
        <div class="chart-container">
            <h4>Shared Theme Analysis</h4>
            <div class="alert alert-info">
                Theme visualization would appear here with Plotly.js
            </div>
            <div class="chart-description">
                <p>This chart would show the distribution of shared themes across the selected narratives.</p>
            </div>
        </div>
        """
        return html
    
    def _generate_mock_network_chart(self, nodes: List[Dict], edges: List[Dict], communities: List[Dict]) -> str:
        """Generate a mock network chart HTML."""
        # In a real implementation, this would create a Plotly or vis.js network graph
        html = f"""
        <div class="chart-container">
            <h4>Source Coordination Network</h4>
            <div class="alert alert-info">
                Network visualization would appear here with Plotly.js or vis.js
            </div>
            <div class="chart-description">
                <p>This network graph would show the coordination between {len(nodes)} sources with {len(edges)} connections across {len(communities)} communities.</p>
            </div>
        </div>
        """
        return html