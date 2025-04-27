"""
Alerts service for narrative complexity analysis.
Generates alerts when narratives exceed complexity thresholds or show rapid increases.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from app import db
from models import DetectedNarrative, User

logger = logging.getLogger(__name__)

class ComplexityAlertService:
    """Service to monitor and generate alerts based on narrative complexity."""
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "overall_complexity": 7.5,
        "linguistic_complexity": 8.0,
        "logical_structure": 8.0,
        "rhetorical_techniques": 8.0,
        "emotional_manipulation": 8.0,
        "rate_of_change": 20  # Percentage increase over previous period
    }
    
    @staticmethod
    def check_high_complexity_alerts(days: int = 1) -> List[Dict[str, Any]]:
        """
        Check for narratives that exceed high complexity thresholds.
        
        Args:
            days: Number of days to look back for recently updated narratives
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        try:
            # Calculate time cutoff
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get narratives updated in the specified period
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.last_updated >= cutoff_date,
                DetectedNarrative.status == 'active'
            ).all()
            
            # Check for high complexity alerts
            for narrative in narratives:
                if not narrative.meta_data:
                    continue
                    
                try:
                    metadata = json.loads(narrative.meta_data)
                    complexity_data = metadata.get('complexity_analysis', {})
                    
                    if not complexity_data or 'overall_complexity_score' not in complexity_data:
                        continue
                    
                    # Check overall complexity
                    overall_score = complexity_data.get('overall_complexity_score', 0)
                    if overall_score >= ComplexityAlertService.DEFAULT_THRESHOLDS['overall_complexity']:
                        alerts.append({
                            'type': 'high_complexity',
                            'narrative_id': narrative.id,
                            'title': narrative.title,
                            'score': overall_score,
                            'threshold': ComplexityAlertService.DEFAULT_THRESHOLDS['overall_complexity'],
                            'timestamp': datetime.now().isoformat(),
                            'dimension': 'overall',
                            'message': f"Narrative {narrative.id} exceeds overall complexity threshold with score {overall_score}."
                        })
                    
                    # Check individual dimensions
                    dimensions = [
                        ('linguistic_complexity', 'Linguistic Complexity'),
                        ('logical_structure', 'Logical Structure'),
                        ('rhetorical_techniques', 'Rhetorical Techniques'),
                        ('emotional_manipulation', 'Emotional Manipulation')
                    ]
                    
                    for dim_key, dim_name in dimensions:
                        dim_data = complexity_data.get(dim_key, {})
                        dim_score = dim_data.get('score', 0)
                        
                        if dim_score >= ComplexityAlertService.DEFAULT_THRESHOLDS[dim_key]:
                            alerts.append({
                                'type': 'high_dimension',
                                'narrative_id': narrative.id,
                                'title': narrative.title,
                                'score': dim_score,
                                'threshold': ComplexityAlertService.DEFAULT_THRESHOLDS[dim_key],
                                'timestamp': datetime.now().isoformat(),
                                'dimension': dim_key,
                                'dimension_name': dim_name,
                                'message': f"Narrative {narrative.id} exceeds {dim_name} threshold with score {dim_score}."
                            })
                
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking complexity thresholds: {e}")
            return []
    
    @staticmethod
    def check_rapid_change_alerts(days: int = 7) -> List[Dict[str, Any]]:
        """
        Check for narratives with rapid increases in complexity.
        
        Args:
            days: Number of days to consider for rate-of-change calculation
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        try:
            # Get historical snapshots of narratives from the given timeframe
            historical_data = ComplexityAlertService._get_historical_complexity_data(days)
            
            # Check for rapid complexity increases
            for narrative_id, snapshots in historical_data.items():
                if len(snapshots) < 2:
                    continue  # Need at least two data points
                
                # Sort by timestamp
                snapshots.sort(key=lambda x: x['timestamp'])
                
                # Get earliest and latest snapshots
                earliest = snapshots[0]
                latest = snapshots[-1]
                
                # Calculate rate of change for overall complexity
                if earliest['overall_score'] > 0:  # Avoid division by zero
                    change_pct = ((latest['overall_score'] - earliest['overall_score']) / 
                                 earliest['overall_score']) * 100
                    
                    # Check if change exceeds threshold
                    if change_pct >= ComplexityAlertService.DEFAULT_THRESHOLDS['rate_of_change']:
                        alerts.append({
                            'type': 'rapid_increase',
                            'narrative_id': narrative_id,
                            'title': latest['title'],
                            'initial_score': earliest['overall_score'],
                            'current_score': latest['overall_score'],
                            'change_percentage': round(change_pct, 1),
                            'period_days': days,
                            'threshold': ComplexityAlertService.DEFAULT_THRESHOLDS['rate_of_change'],
                            'timestamp': datetime.now().isoformat(),
                            'message': f"Narrative {narrative_id} shows {round(change_pct, 1)}% increase in complexity over {days} days."
                        })
                
                # Also check individual dimensions
                dimensions = [
                    ('linguistic_score', 'linguistic_complexity', 'Linguistic Complexity'),
                    ('logical_score', 'logical_structure', 'Logical Structure'),
                    ('rhetorical_score', 'rhetorical_techniques', 'Rhetorical Techniques'),
                    ('emotional_score', 'emotional_manipulation', 'Emotional Manipulation')
                ]
                
                for score_key, dim_key, dim_name in dimensions:
                    if earliest[score_key] > 0:  # Avoid division by zero
                        dim_change_pct = ((latest[score_key] - earliest[score_key]) / 
                                         earliest[score_key]) * 100
                        
                        # Check if dimension change exceeds threshold
                        if dim_change_pct >= ComplexityAlertService.DEFAULT_THRESHOLDS['rate_of_change']:
                            alerts.append({
                                'type': 'rapid_dimension_increase',
                                'narrative_id': narrative_id,
                                'title': latest['title'],
                                'initial_score': earliest[score_key],
                                'current_score': latest[score_key],
                                'change_percentage': round(dim_change_pct, 1),
                                'period_days': days,
                                'dimension': dim_key,
                                'dimension_name': dim_name,
                                'threshold': ComplexityAlertService.DEFAULT_THRESHOLDS['rate_of_change'],
                                'timestamp': datetime.now().isoformat(),
                                'message': f"Narrative {narrative_id} shows {round(dim_change_pct, 1)}% increase in {dim_name} over {days} days."
                            })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking complexity rate of change: {e}")
            return []

    @staticmethod
    def check_coordinated_narratives() -> List[Dict[str, Any]]:
        """
        Check for groups of narratives with similar complexity patterns.
        
        Returns:
            List of alert dictionaries for coordinated narrative groups
        """
        alerts = []
        try:
            # Get active narratives with complexity data
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
                        'overall_score': complexity_data.get('overall_complexity_score', 0),
                        'linguistic_score': complexity_data.get('linguistic_complexity', {}).get('score', 0),
                        'logical_score': complexity_data.get('logical_structure', {}).get('score', 0),
                        'rhetorical_score': complexity_data.get('rhetorical_techniques', {}).get('score', 0),
                        'emotional_score': complexity_data.get('emotional_manipulation', {}).get('score', 0),
                    })
                    
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            
            # Find clusters of similar narratives
            clusters = ComplexityAlertService._cluster_similar_narratives(narratives_with_complexity)
            
            # Generate alerts for significant clusters
            for i, cluster in enumerate(clusters):
                if len(cluster) >= 3:  # Only alert for clusters with 3+ narratives
                    narrative_ids = [n['id'] for n in cluster]
                    narrative_titles = [n['title'] for n in cluster]
                    
                    alerts.append({
                        'type': 'coordinated_narratives',
                        'cluster_id': i + 1,
                        'narrative_ids': narrative_ids,
                        'narrative_titles': narrative_titles[:3] + (["..."] if len(narrative_titles) > 3 else []),
                        'cluster_size': len(cluster),
                        'timestamp': datetime.now().isoformat(),
                        'message': f"Cluster of {len(cluster)} narratives with similar complexity patterns detected."
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking for coordinated narratives: {e}")
            return []
    
    @staticmethod
    def get_all_active_alerts() -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all active alerts for the narrative complexity system.
        
        Returns:
            Dictionary with categorized alerts
        """
        try:
            # Get alerts from all alert types
            high_complexity_alerts = ComplexityAlertService.check_high_complexity_alerts()
            rapid_change_alerts = ComplexityAlertService.check_rapid_change_alerts()
            coordinated_alerts = ComplexityAlertService.check_coordinated_narratives()
            
            return {
                'high_complexity': high_complexity_alerts,
                'rapid_change': rapid_change_alerts,
                'coordinated_narratives': coordinated_alerts,
                'total_count': len(high_complexity_alerts) + len(rapid_change_alerts) + len(coordinated_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving active alerts: {e}")
            return {
                'high_complexity': [],
                'rapid_change': [],
                'coordinated_narratives': [],
                'total_count': 0,
                'error': str(e)
            }
    
    @staticmethod
    def _get_historical_complexity_data(days: int) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get historical complexity data for all narratives in the given timeframe.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary mapping narrative IDs to lists of historical snapshots
        """
        historical_data = {}
        
        # Calculate time cutoff
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get narratives updated in the specified period
        narratives = DetectedNarrative.query.filter(
            DetectedNarrative.last_updated >= cutoff_date
        ).all()
        
        for narrative in narratives:
            if not narrative.meta_data:
                continue
                
            try:
                metadata = json.loads(narrative.meta_data)
                complexity_data = metadata.get('complexity_analysis', {})
                
                if not complexity_data or 'overall_complexity_score' not in complexity_data:
                    continue
                
                # Create a snapshot of complexity data
                snapshot = {
                    'narrative_id': narrative.id,
                    'title': narrative.title,
                    'timestamp': narrative.last_updated.timestamp(),
                    'overall_score': complexity_data.get('overall_complexity_score', 0),
                    'linguistic_score': complexity_data.get('linguistic_complexity', {}).get('score', 0),
                    'logical_score': complexity_data.get('logical_structure', {}).get('score', 0),
                    'rhetorical_score': complexity_data.get('rhetorical_techniques', {}).get('score', 0),
                    'emotional_score': complexity_data.get('emotional_manipulation', {}).get('score', 0),
                }
                
                # Add to historical data
                if narrative.id not in historical_data:
                    historical_data[narrative.id] = []
                
                historical_data[narrative.id].append(snapshot)
                
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        
        return historical_data
    
    @staticmethod
    def _cluster_similar_narratives(narratives: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group narratives with similar complexity patterns.
        
        Args:
            narratives: List of narrative dictionaries with complexity scores
            
        Returns:
            List of clusters (each cluster is a list of narratives)
        """
        if not narratives:
            return []
            
        # Simple clustering algorithm based on complexity profile similarity
        clusters = []
        unclustered = narratives.copy()
        
        while unclustered:
            # Start a new cluster with the first unclustered narrative
            current_cluster = [unclustered.pop(0)]
            
            # Check all remaining narratives for similarity
            i = 0
            while i < len(unclustered):
                narrative = unclustered[i]
                
                # Check if narrative is similar to the cluster
                if ComplexityAlertService._is_similar_profile(narrative, current_cluster[0]):
                    current_cluster.append(narrative)
                    unclustered.pop(i)
                else:
                    i += 1
            
            # Add the cluster to results
            clusters.append(current_cluster)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        return clusters
    
    @staticmethod
    def _is_similar_profile(narrative1: Dict[str, Any], narrative2: Dict[str, Any]) -> bool:
        """
        Check if two narratives have similar complexity profiles.
        
        Args:
            narrative1: First narrative dictionary
            narrative2: Second narrative dictionary
            
        Returns:
            True if profiles are similar, False otherwise
        """
        # Calculate Euclidean distance between complexity profiles
        dimensions = [
            'overall_score',
            'linguistic_score',
            'logical_score',
            'rhetorical_score',
            'emotional_score'
        ]
        
        # Calculate sum of squared differences
        squared_diff_sum = 0
        for dim in dimensions:
            score1 = narrative1.get(dim, 0)
            score2 = narrative2.get(dim, 0)
            squared_diff_sum += (score1 - score2) ** 2
        
        # Calculate Euclidean distance
        distance = squared_diff_sum ** 0.5
        
        # Use a threshold to determine similarity
        # For scores on a 0-10 scale, a distance of 1.5 or less suggests similarity
        return distance <= 1.5