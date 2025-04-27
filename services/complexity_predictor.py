"""
Predictive modeling service for narrative complexity analysis.
Uses historical data to predict future complexity trends.
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from app import db
from models import DetectedNarrative
from services.complexity_alerts import ComplexityAlertService

logger = logging.getLogger(__name__)

class ComplexityPredictor:
    """Service to predict future complexity trends based on historical data."""
    
    @staticmethod
    def predict_narrative_evolution(narrative_id: int, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Predict how a narrative's complexity might evolve in the future.
        
        Args:
            narrative_id: ID of the narrative to predict
            days_ahead: Number of days to predict into the future
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": f"Narrative {narrative_id} not found"}
            
            # Get historical complexity data
            historical_data = ComplexityPredictor._get_historical_data(narrative_id)
            
            if not historical_data:
                logger.warning(f"No historical complexity data for narrative {narrative_id}")
                return {"error": "Insufficient historical data for prediction"}
            
            # Make predictions
            predictions = ComplexityPredictor._generate_predictions(historical_data, days_ahead)
            
            # Format results
            dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                    for i in range(1, days_ahead + 1)]
            
            result = {
                'narrative_id': narrative_id,
                'title': narrative.title,
                'current_complexity': predictions['current_values'],
                'prediction_dates': dates,
                'predicted_complexity': predictions['predicted_values'],
                'confidence': predictions['confidence'],
                'trend_direction': predictions['trend_direction'],
                'potential_peak_date': predictions['potential_peak_date'],
                'factors': predictions['factors']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting narrative evolution: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_similar_trajectories(narrative_id: int) -> Dict[str, Any]:
        """
        Find narratives with similar complexity evolution patterns.
        
        Args:
            narrative_id: ID of the narrative to compare with others
            
        Returns:
            Dictionary with similar narratives and their trajectories
        """
        try:
            # Get narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": f"Narrative {narrative_id} not found"}
            
            # Get historical data for this narrative
            target_history = ComplexityPredictor._get_historical_data(narrative_id)
            
            if not target_history:
                logger.warning(f"No historical complexity data for narrative {narrative_id}")
                return {"error": "Insufficient historical data for comparison"}
            
            # Get other narratives with complexity data
            similar_narratives = []
            
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.id != narrative_id,
                DetectedNarrative.status == 'active'
            ).all()
            
            for other_narrative in narratives:
                other_history = ComplexityPredictor._get_historical_data(other_narrative.id)
                
                if not other_history:
                    continue
                
                # Calculate similarity
                similarity = ComplexityPredictor._calculate_trajectory_similarity(
                    target_history, other_history
                )
                
                if similarity > 0.6:  # Threshold for similarity
                    similar_narratives.append({
                        'id': other_narrative.id,
                        'title': other_narrative.title,
                        'similarity': similarity,
                        'status': other_narrative.status,
                        'trajectory': other_history
                    })
            
            # Sort by similarity (highest first)
            similar_narratives.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'narrative_id': narrative_id,
                'title': narrative.title,
                'trajectory': target_history,
                'similar_narratives': similar_narratives[:5]  # Top 5 most similar
            }
            
        except Exception as e:
            logger.error(f"Error finding similar narratives: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_trending_narratives(days: int = 30, limit: int = 10) -> Dict[str, Any]:
        """
        Identify narratives with the strongest upward complexity trends.
        
        Args:
            days: Number of days to analyze for trends
            limit: Maximum number of trending narratives to return
            
        Returns:
            Dictionary with trending narratives and their metrics
        """
        try:
            # Get active narratives
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.status == 'active'
            ).all()
            
            # Calculate trends for each narrative
            trending_narratives = []
            
            for narrative in narratives:
                # Get historical data
                history = ComplexityPredictor._get_historical_data(narrative.id)
                
                if not history or len(history['timestamps']) < 2:
                    continue
                
                # Calculate trend
                trend_metrics = ComplexityPredictor._calculate_trend_metrics(history)
                
                if trend_metrics['overall_slope'] > 0.05:  # Positive trend threshold
                    trending_narratives.append({
                        'id': narrative.id,
                        'title': narrative.title,
                        'current_complexity': trend_metrics['current_value'],
                        'trend_rate': trend_metrics['overall_slope'],
                        'acceleration': trend_metrics['acceleration'],
                        'days_to_threshold': trend_metrics['days_to_threshold'],
                        'dimension_trends': trend_metrics['dimension_trends']
                    })
            
            # Sort by trend rate (highest first)
            trending_narratives.sort(key=lambda x: x['trend_rate'], reverse=True)
            
            return {
                'trending_narratives': trending_narratives[:limit],
                'analysis_period_days': days,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error identifying trending narratives: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _get_historical_data(narrative_id: int) -> Optional[Dict[str, Any]]:
        """
        Get historical complexity data for a narrative.
        
        Args:
            narrative_id: ID of the narrative to get data for
            
        Returns:
            Dictionary with historical complexity data or None
        """
        # Leverage the alerts service method to get historical data
        historical_data = ComplexityAlertService._get_historical_complexity_data(30)  # Last 30 days
        
        if narrative_id not in historical_data or len(historical_data[narrative_id]) < 2:
            return None
        
        # Format the data for prediction
        snapshots = historical_data[narrative_id]
        
        # Sort by timestamp
        snapshots.sort(key=lambda x: x['timestamp'])
        
        # Extract complexity values over time
        data = {
            'timestamps': [s['timestamp'] for s in snapshots],
            'overall_scores': [s['overall_score'] for s in snapshots],
            'linguistic_scores': [s['linguistic_score'] for s in snapshots],
            'logical_scores': [s['logical_score'] for s in snapshots],
            'rhetorical_scores': [s['rhetorical_score'] for s in snapshots],
            'emotional_scores': [s['emotional_score'] for s in snapshots],
            'title': snapshots[0]['title']
        }
        
        return data
    
    @staticmethod
    def _generate_predictions(
        historical_data: Dict[str, Any], 
        days_ahead: int
    ) -> Dict[str, Any]:
        """
        Generate complexity predictions based on historical data.
        
        Args:
            historical_data: Dictionary with historical complexity data
            days_ahead: Number of days to predict into the future
            
        Returns:
            Dictionary with prediction results
        """
        # Extract time series data
        timestamps = historical_data['timestamps']
        overall_scores = historical_data['overall_scores']
        linguistic_scores = historical_data['linguistic_scores']
        logical_scores = historical_data['logical_scores']
        rhetorical_scores = historical_data['rhetorical_scores']
        emotional_scores = historical_data['emotional_scores']
        
        # Convert timestamps to days from first observation for regression
        days_from_start = [(t - timestamps[0]) / 86400 for t in timestamps]  # Convert to days
        
        # Add last observation as day 0 for prediction
        days_for_prediction = list(range(1, days_ahead + 1))
        
        # For this simple implementation, we'll use linear regression
        # In a real-world scenario, more sophisticated time series models would be used
        
        # Function for linear regression prediction
        def predict_linear_trend(values):
            if len(values) < 2:
                return [values[-1]] * days_ahead, 0, 0  # Flat prediction
            
            # Calculate slope
            x = np.array(days_from_start)
            y = np.array(values)
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            intercept = (np.sum(y) - slope * np.sum(x)) / n
            
            # Calculate R-squared for confidence
            y_pred = slope * x + intercept
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            # Generate predictions
            last_day = days_from_start[-1]
            predictions = [max(0, min(10, slope * (last_day + d) + intercept)) for d in days_for_prediction]
            
            return predictions, slope, r_squared
        
        # Make predictions for each dimension
        overall_predictions, overall_slope, overall_conf = predict_linear_trend(overall_scores)
        linguistic_predictions, linguistic_slope, linguistic_conf = predict_linear_trend(linguistic_scores)
        logical_predictions, logical_slope, logical_conf = predict_linear_trend(logical_scores)
        rhetorical_predictions, rhetorical_slope, rhetorical_conf = predict_linear_trend(rhetorical_scores)
        emotional_predictions, emotional_slope, emotional_conf = predict_linear_trend(emotional_scores)
        
        # Determine trend direction
        if overall_slope > 0.1:
            trend_direction = "strong_increase"
        elif overall_slope > 0.05:
            trend_direction = "moderate_increase"
        elif overall_slope > 0.01:
            trend_direction = "slight_increase"
        elif overall_slope < -0.1:
            trend_direction = "strong_decrease"
        elif overall_slope < -0.05:
            trend_direction = "moderate_decrease"
        elif overall_slope < -0.01:
            trend_direction = "slight_decrease"
        else:
            trend_direction = "stable"
        
        # Determine potential peak date (if trending up)
        potential_peak_date = None
        if overall_slope > 0 and overall_scores[-1] < 8.5:
            days_to_threshold = (8.5 - overall_scores[-1]) / overall_slope if overall_slope > 0 else None
            if days_to_threshold and days_to_threshold < 30:  # Within a month
                peak_date = datetime.fromtimestamp(timestamps[-1]) + timedelta(days=days_to_threshold)
                potential_peak_date = peak_date.strftime('%Y-%m-%d')
        
        # Identify most significant factors
        factors = []
        slopes = [
            ("linguistic_complexity", linguistic_slope),
            ("logical_structure", logical_slope),
            ("rhetorical_techniques", rhetorical_slope),
            ("emotional_manipulation", emotional_slope)
        ]
        
        # Sort by absolute slope value, descending
        slopes.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Add factors based on slopes
        for dimension, slope in slopes:
            if abs(slope) > 0.05:
                direction = "increasing" if slope > 0 else "decreasing"
                factors.append(f"{dimension} is {direction} significantly")
        
        if not factors:
            factors.append("complexity is relatively stable across all dimensions")
        
        # Compile prediction results
        return {
            'current_values': {
                'overall': overall_scores[-1],
                'linguistic': linguistic_scores[-1],
                'logical': logical_scores[-1],
                'rhetorical': rhetorical_scores[-1],
                'emotional': emotional_scores[-1]
            },
            'predicted_values': {
                'overall': overall_predictions,
                'linguistic': linguistic_predictions,
                'logical': logical_predictions,
                'rhetorical': rhetorical_predictions,
                'emotional': emotional_predictions
            },
            'confidence': {
                'overall': overall_conf,
                'linguistic': linguistic_conf,
                'logical': logical_conf,
                'rhetorical': rhetorical_conf,
                'emotional': emotional_conf
            },
            'trend_direction': trend_direction,
            'potential_peak_date': potential_peak_date,
            'factors': factors
        }
    
    @staticmethod
    def _calculate_trajectory_similarity(
        history1: Dict[str, Any], 
        history2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two complexity trajectories.
        
        Args:
            history1: Historical data for first narrative
            history2: Historical data for second narrative
            
        Returns:
            Similarity score (0-1)
        """
        # For simplicity, we'll use correlation between overall scores
        # In a real implementation, more sophisticated similarity metrics would be used
        
        # Get overall scores
        scores1 = history1['overall_scores']
        scores2 = history2['overall_scores']
        
        # Ensure we have enough data points
        if len(scores1) < 2 or len(scores2) < 2:
            return 0
        
        # Normalize lengths by using the most recent data points
        min_length = min(len(scores1), len(scores2))
        scores1 = scores1[-min_length:]
        scores2 = scores2[-min_length:]
        
        # Calculate correlation
        x = np.array(scores1)
        y = np.array(scores2)
        
        # Calculate normalized correlation
        x_normalized = (x - np.mean(x)) / (np.std(x) if np.std(x) > 0 else 1)
        y_normalized = (y - np.mean(y)) / (np.std(y) if np.std(y) > 0 else 1)
        
        correlation = np.mean(x_normalized * y_normalized)
        
        # Transform to similarity score (0-1)
        similarity = (correlation + 1) / 2
        
        return similarity
    
    @staticmethod
    def _calculate_trend_metrics(history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trend metrics for a narrative's complexity.
        
        Args:
            history: Historical complexity data
            
        Returns:
            Dictionary with trend metrics
        """
        # Extract complexity scores
        overall_scores = history['overall_scores']
        linguistic_scores = history['linguistic_scores']
        logical_scores = history['logical_scores']
        rhetorical_scores = history['rhetorical_scores']
        emotional_scores = history['emotional_scores']
        
        # Get last values
        current_value = overall_scores[-1]
        
        # Calculate slopes for linear trends
        def calculate_slope(values):
            if len(values) < 2:
                return 0
            
            # Simple linear regression slope
            x = np.array(range(len(values)))
            y = np.array(values)
            n = len(x)
            return (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        overall_slope = calculate_slope(overall_scores)
        linguistic_slope = calculate_slope(linguistic_scores)
        logical_slope = calculate_slope(logical_scores)
        rhetorical_slope = calculate_slope(rhetorical_scores)
        emotional_slope = calculate_slope(emotional_scores)
        
        # Calculate acceleration (second derivative)
        def calculate_acceleration(values):
            if len(values) < 3:
                return 0
            
            # Calculate first differences (velocities)
            velocities = [values[i+1] - values[i] for i in range(len(values)-1)]
            
            # Calculate slope of velocities (acceleration)
            return calculate_slope(velocities)
        
        acceleration = calculate_acceleration(overall_scores)
        
        # Calculate days to threshold (8.5 overall complexity)
        days_to_threshold = None
        if overall_slope > 0 and current_value < 8.5:
            days_to_threshold = (8.5 - current_value) / overall_slope
        
        return {
            'current_value': current_value,
            'overall_slope': overall_slope,
            'acceleration': acceleration,
            'days_to_threshold': days_to_threshold,
            'dimension_trends': {
                'linguistic': linguistic_slope,
                'logical': logical_slope,
                'rhetorical': rhetorical_slope,
                'emotional': emotional_slope
            }
        }