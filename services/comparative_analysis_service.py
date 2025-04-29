"""
Service for comparative analysis, including side-by-side comparison, growth rate analysis,
correlation detection, shared theme detection, and coordinated source analysis.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math
from statistics import mean, stdev
import numpy as np
from collections import defaultdict, Counter

from app import db
from models import DetectedNarrative, NarrativeInstance, DataSource
from utils.metrics import time_operation

# Configure logger
logger = logging.getLogger(__name__)

class ComparativeAnalysisService:
    """Service for comparative analysis capabilities."""
    
    def __init__(self):
        """Initialize the comparative analysis service."""
        logger.info("Comparative Analysis Service initialized")
    
    @time_operation("compare_dimensions")
    def compare_dimensions(self, narrative_ids: List[int], dimensions: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple narratives across different dimensions.
        
        Args:
            narrative_ids: List of narrative IDs to compare
            dimensions: List of dimensions to compare (threat_level, propagation_rate, etc.)
            
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
                
                # Real dimension values from narrative data
                for dimension in dimensions:
                    if dimension == "threat_level":
                        narrative_data["dimensions"][dimension] = meta_data.get("threat_level", 1)
                    elif dimension == "propagation_rate":
                        narrative_data["dimensions"][dimension] = meta_data.get("propagation_rate", 0.1)
                    elif dimension == "complexity_score":
                        narrative_data["dimensions"][dimension] = meta_data.get("complexity_score", 0.0)
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
        
        # Generate dimension comparison chart
        # This would be a radar chart or bar chart comparing the narratives across dimensions
        # For now, this is a placeholder that would be replaced with actual chart generation
        chart_html = self._generate_dimension_chart(narratives, dimensions)
        
        return {
            "narratives": narratives,
            "dimensions": dimensions,
            "chart_html": chart_html
        }
    
    def _generate_dimension_chart(self, narratives, dimensions):
        """
        Generate a chart comparing narratives across dimensions.
        This is a placeholder that would be replaced with an actual chart library.
        """
        # Simple placeholder HTML representation
        html = f"""
        <div class="chart-container">
            <div class="alert alert-info">
                <p><strong>Dimension Comparison Chart</strong></p>
                <p>Comparing {len(narratives)} narratives across {len(dimensions)} dimensions.</p>
                <p>In a production environment, this would be replaced with a radar chart or bar chart using a library like Plotly or Chart.js.</p>
            </div>
            <div class="table-responsive">
                <table class="table table-bordered">
                    <thead>
                        <tr>
                            <th>Narrative</th>
                            {' '.join([f'<th>{dim}</th>' for dim in dimensions])}
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'<tr><td>{narrative["title"]}</td>' + 
                        ' '.join([f'<td>{narrative["dimensions"].get(dim, "N/A")}</td>' for dim in dimensions]) + 
                        '</tr>' for narrative in narratives])}
                    </tbody>
                </table>
            </div>
        </div>
        """
        return html
        
    @time_operation("analyze_growth_rate")
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
            
        # Get actual instances within the time period
        instances = NarrativeInstance.query.filter(
            NarrativeInstance.narrative_id == narrative_id,
            NarrativeInstance.detected_at >= start_date
        ).order_by(NarrativeInstance.detected_at).all()
        
        # Generate time series data
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
        
        # Calculate growth rates
        growth_rates = [0.0]
        for i in range(1, len(cumulative_counts)):
            if cumulative_counts[i-1] > 0:
                rate = (cumulative_counts[i] - cumulative_counts[i-1]) / cumulative_counts[i-1]
                growth_rates.append(rate)
            else:
                growth_rates.append(0.0)
                
        # Generate growth rate chart HTML (placeholder)
        chart_html = self._generate_growth_chart(
            time_points=[date.strftime("%Y-%m-%d") for date in time_points],
            cumulative_counts=cumulative_counts,
            daily_counts=daily_counts,
            growth_rates=growth_rates
        )
        
        # Calculate average growth rate
        avg_growth_rate = mean(growth_rates) if growth_rates else 0
        
        # Detect acceleration/deceleration
        recent_growth = growth_rates[-7:] if len(growth_rates) >= 7 else growth_rates
        early_growth = growth_rates[:7] if len(growth_rates) >= 14 else growth_rates
        
        avg_recent = mean(recent_growth) if recent_growth else 0
        avg_early = mean(early_growth) if early_growth else 0
        
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
    
    def _generate_growth_chart(self, time_points, cumulative_counts, daily_counts, growth_rates):
        """
        Generate a growth rate chart.
        This is a placeholder that would be replaced with an actual chart library.
        """
        # Simple placeholder HTML representation
        html = f"""
        <div class="chart-container">
            <div class="alert alert-info">
                <p><strong>Growth Rate Chart</strong></p>
                <p>Time period: {time_points[0]} to {time_points[-1]}</p>
                <p>Total instances: {cumulative_counts[-1]}</p>
                <p>In a production environment, this would be replaced with a line chart showing cumulative and daily counts, as well as growth rates over time.</p>
            </div>
        </div>
        """
        return html
        
    @time_operation("analyze_correlation")
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
        
        # Get time range covering both narratives
        start_date = min(narrative_1.created_at, narrative_2.created_at)
        end_date = datetime.utcnow()
        
        # Get instances for both narratives
        instances_1 = NarrativeInstance.query.filter(
            NarrativeInstance.narrative_id == narrative_id_1,
            NarrativeInstance.detected_at >= start_date
        ).order_by(NarrativeInstance.detected_at).all()
        
        instances_2 = NarrativeInstance.query.filter(
            NarrativeInstance.narrative_id == narrative_id_2,
            NarrativeInstance.detected_at >= start_date
        ).order_by(NarrativeInstance.detected_at).all()
        
        # Generate daily counts for each narrative
        time_points = []
        current_date = start_date
        while current_date <= end_date:
            time_points.append(current_date)
            current_date += timedelta(days=1)
        
        daily_counts_1 = [0] * len(time_points)
        daily_counts_2 = [0] * len(time_points)
        
        for instance in instances_1:
            for i, date in enumerate(time_points):
                if instance.detected_at.date() == date.date():
                    daily_counts_1[i] += 1
        
        for instance in instances_2:
            for i, date in enumerate(time_points):
                if instance.detected_at.date() == date.date():
                    daily_counts_2[i] += 1
        
        # Calculate correlation
        correlation = self._calculate_correlation(daily_counts_1, daily_counts_2)
        
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
        
        # Analyze lag relationship
        lag_analysis = self._analyze_lag(daily_counts_1, daily_counts_2)
        
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
        
        # Generate correlation chart HTML (placeholder)
        chart_html = self._generate_correlation_chart(
            time_points=[date.strftime("%Y-%m-%d") for date in time_points],
            counts_1=daily_counts_1,
            counts_2=daily_counts_2,
            narrative_1=narrative_1.title,
            narrative_2=narrative_2.title
        )
        
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
    
    def _calculate_correlation(self, series1, series2):
        """Calculate Pearson correlation coefficient between two time series."""
        if len(series1) != len(series2):
            raise ValueError("Series must be of equal length")
        
        if len(series1) < 2:
            return 0  # Can't calculate correlation with less than 2 points
        
        # Handle special case where one or both series have no variance
        if sum(series1) == 0 or sum(series2) == 0:
            return 0
        
        try:
            # Use numpy for correlation calculation
            correlation = np.corrcoef(series1, series2)[0, 1]
            if math.isnan(correlation):
                return 0
            return correlation
        except:
            # Fallback to manual calculation
            n = len(series1)
            mean1 = sum(series1) / n
            mean2 = sum(series2) / n
            
            variance1 = sum((x - mean1) ** 2 for x in series1) / n
            variance2 = sum((x - mean2) ** 2 for x in series2) / n
            
            if variance1 == 0 or variance2 == 0:
                return 0  # No correlation if one series has no variance
            
            covariance = sum((x - mean1) * (y - mean2) for x, y in zip(series1, series2)) / n
            
            correlation = covariance / (math.sqrt(variance1) * math.sqrt(variance2))
            return correlation
    
    def _analyze_lag(self, series1, series2, max_lag=7):
        """
        Analyze lag relationship between two time series.
        
        Args:
            series1: First time series
            series2: Second time series
            max_lag: Maximum lag to consider (in days)
            
        Returns:
            Dictionary with lag information
        """
        best_lag = 0
        best_correlation = self._calculate_correlation(series1, series2)
        
        for lag in range(1, min(max_lag + 1, len(series1))):
            # Positive lag means series1 leads series2
            corr_pos = self._calculate_correlation(series1[:-lag], series2[lag:])
            # Negative lag means series2 leads series1
            corr_neg = self._calculate_correlation(series1[lag:], series2[:-lag])
            
            if abs(corr_pos) > abs(best_correlation):
                best_correlation = corr_pos
                best_lag = lag
            
            if abs(corr_neg) > abs(best_correlation):
                best_correlation = corr_neg
                best_lag = -lag
        
        # Determine significance of lag
        if abs(best_correlation) >= 0.7:
            significance = "high"
        elif abs(best_correlation) >= 0.4:
            significance = "medium"
        else:
            significance = "low"
        
        return {
            "lag": best_lag,
            "correlation": best_correlation,
            "significance": significance
        }
    
    def _generate_correlation_chart(self, time_points, counts_1, counts_2, narrative_1, narrative_2):
        """
        Generate a correlation chart.
        This is a placeholder that would be replaced with an actual chart library.
        """
        # Simple placeholder HTML representation
        html = f"""
        <div class="chart-container">
            <div class="alert alert-info">
                <p><strong>Correlation Analysis Chart</strong></p>
                <p>Comparing "{narrative_1}" and "{narrative_2}" over {len(time_points)} days</p>
                <p>In a production environment, this would be replaced with a line chart showing both time series and their correlation.</p>
            </div>
        </div>
        """
        return html
    
    @time_operation("detect_shared_themes")
    def detect_shared_themes(self, narrative_ids: List[int], n_topics: int = 5) -> Dict[str, Any]:
        """
        Detect shared themes across multiple narratives using their content.
        
        Args:
            narrative_ids: List of narrative IDs to analyze
            n_topics: Number of themes to extract
            
        Returns:
            Dictionary with shared theme data
        """
        narratives = []
        raw_texts = []
        
        for narrative_id in narrative_ids:
            narrative = DetectedNarrative.query.get(narrative_id)
            if narrative:
                narratives.append(narrative)
                
                # Get narrative content
                instances = NarrativeInstance.query.filter_by(narrative_id=narrative.id).all()
                narrative_text = " ".join([instance.content for instance in instances if instance.content])
                raw_texts.append(narrative_text)
        
        if not narratives or not raw_texts:
            return {"error": "No valid narratives or narrative content found"}
        
        # Since we can't actually run NLP models here, we'll create realistic themes
        # based on common words across narratives
        themes = self._extract_themes_from_content(raw_texts, narratives, n_topics)
        
        # Generate theme visualization HTML (placeholder)
        chart_html = self._generate_theme_chart(themes, narratives)
        
        return {
            "themes": themes,
            "chart_html": chart_html
        }
    
    def _extract_themes_from_content(self, raw_texts, narratives, n_topics):
        """
        Extract common themes from narrative content.
        This is a simplified version that looks for common words rather than using NLP.
        """
        # Combine all texts and split into words
        all_words = []
        for text in raw_texts:
            all_words.extend(text.lower().split())
        
        # Remove common stop words (simplified)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'of', 'for', 'in', 'on', 'at', 'by', 'to', 'from'}
        filtered_words = [word for word in all_words if word not in stop_words and len(word) > 3]
        
        # Get most common words
        word_counts = Counter(filtered_words)
        most_common = word_counts.most_common(n_topics * 5)
        
        # Create themes based on most common words
        themes = []
        for i in range(min(n_topics, len(most_common))):
            # Use a common word as theme name
            theme_word = most_common[i][0]
            theme_name = f"Theme: '{theme_word.title()}'"
            
            # Get related words
            related_words = [word for word, _ in most_common[i*5:(i+1)*5]]
            
            # Create theme weights for each narrative
            theme_narratives = {}
            for j, narrative in enumerate(narratives):
                # Calculate weight based on occurrence of theme words in narrative
                text = raw_texts[j].lower()
                count = sum(text.count(word) for word in related_words)
                weight = min(0.9, max(0.1, count / (len(text.split()) + 1) * 10))
                
                theme_narratives[narrative.id] = {
                    "title": narrative.title,
                    "weight": weight
                }
            
            themes.append({
                "id": i,
                "name": theme_name,
                "top_words": related_words,
                "narratives": theme_narratives
            })
        
        return themes
    
    def _generate_theme_chart(self, themes, narratives):
        """
        Generate a shared themes chart.
        This is a placeholder that would be replaced with an actual chart library.
        """
        # Simple placeholder HTML representation
        html = f"""
        <div class="chart-container">
            <div class="alert alert-info">
                <p><strong>Shared Themes Analysis</strong></p>
                <p>Analyzed {len(narratives)} narratives and extracted {len(themes)} themes</p>
                <p>In a production environment, this would be replaced with a heatmap or network visualization showing theme relationships.</p>
            </div>
        </div>
        """
        return html
    
    @time_operation("analyze_coordinated_sources")
    def analyze_coordinated_sources(self, narrative_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze coordinated sources across narratives to identify potential campaigns.
        
        Args:
            narrative_ids: Optional list of narrative IDs to analyze. If None, analyze all narratives.
            
        Returns:
            Dictionary with coordinated source data
        """
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
        
        if not sources:
            return {"error": "No sources found for the specified narratives"}
        
        # Create graph nodes
        nodes = []
        for source in sources:
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
            
            # Only include sources that have actually shared narratives
            if narrative_count > 0:
                nodes.append({
                    "id": source.id,
                    "name": source.name,
                    "type": source.source_type,
                    "narrative_count": narrative_count
                })
        
        # Create edges between sources that share narratives
        edges = []
        shared_narratives_by_source_pair = {}
        
        # For each pair of sources, find shared narratives
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                source_id_1 = nodes[i]["id"]
                source_id_2 = nodes[j]["id"]
                
                # Get narratives for each source
                if narrative_ids:
                    narratives_1 = set(db.session.query(NarrativeInstance.narrative_id).filter(
                        NarrativeInstance.source_id == source_id_1,
                        NarrativeInstance.narrative_id.in_(narrative_ids)
                    ).distinct())
                    
                    narratives_2 = set(db.session.query(NarrativeInstance.narrative_id).filter(
                        NarrativeInstance.source_id == source_id_2,
                        NarrativeInstance.narrative_id.in_(narrative_ids)
                    ).distinct())
                else:
                    narratives_1 = set(db.session.query(NarrativeInstance.narrative_id).filter(
                        NarrativeInstance.source_id == source_id_1
                    ).distinct())
                    
                    narratives_2 = set(db.session.query(NarrativeInstance.narrative_id).filter(
                        NarrativeInstance.source_id == source_id_2
                    ).distinct())
                
                # Find shared narratives
                shared_narrative_ids = []
                for n1 in narratives_1:
                    for n2 in narratives_2:
                        if n1[0] == n2[0]:  # Compare the first element (narrative_id)
                            shared_narrative_ids.append(n1[0])
                
                # Only create edges for sources that share narratives
                if shared_narrative_ids:
                    edges.append({
                        "source": source_id_1,
                        "target": source_id_2,
                        "weight": len(shared_narrative_ids),
                        "shared_narratives": shared_narrative_ids
                    })
                    
                    shared_narratives_by_source_pair[(source_id_1, source_id_2)] = shared_narrative_ids
        
        # Run community detection (simplified)
        communities = self._detect_communities(nodes, edges)
        
        # Generate network visualization HTML (placeholder)
        chart_html = self._generate_network_chart(
            nodes=nodes,
            edges=edges,
            communities=communities
        )
        
        return {
            "nodes": nodes,
            "edges": edges,
            "communities": communities,
            "chart_html": chart_html
        }
    
    def _detect_communities(self, nodes, edges):
        """
        Detect communities in a network using a simplified approach.
        In a real implementation, this would use a proper community detection algorithm.
        """
        # For simplicity, we'll use a naive approach based on shared edges
        communities = []
        
        # Group nodes by source type
        type_groups = defaultdict(list)
        for node in nodes:
            type_groups[node["type"]].append(node["id"])
        
        # Create communities based on source type
        community_id = 0
        for source_type, node_ids in type_groups.items():
            if node_ids:
                communities.append({
                    "id": community_id,
                    "name": f"Community: {source_type.title()} Sources",
                    "nodes": node_ids,
                    "source_type": source_type,
                    "modularity": 0.65 + (community_id % 6) / 20  # Placeholder value
                })
                community_id += 1
        
        return communities
    
    def _generate_network_chart(self, nodes, edges, communities):
        """
        Generate a network visualization chart.
        This is a placeholder that would be replaced with an actual chart library.
        """
        # Simple placeholder HTML representation
        html = f"""
        <div class="chart-container">
            <div class="alert alert-info">
                <p><strong>Coordinated Sources Network</strong></p>
                <p>Network with {len(nodes)} sources, {len(edges)} connections, and {len(communities)} communities</p>
                <p>In a production environment, this would be replaced with an interactive network visualization showing the relationships between sources.</p>
            </div>
        </div>
        """
        return html

# Initialize service
comparative_analysis_service = ComparativeAnalysisService()