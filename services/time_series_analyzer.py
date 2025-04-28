"""
Time-series analysis service for narrative complexity.
Analyzes evolving patterns and trends in misinformation complexity.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

from app import db
from models import DetectedNarrative

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """Service to analyze time-series data for narrative complexity patterns."""
    
    @staticmethod
    def analyze_narrative_time_series(narrative_id: int, period: int = 30) -> Dict[str, Any]:
        """
        Perform time-series decomposition on a narrative's complexity data.
        
        Args:
            narrative_id: ID of the narrative to analyze
            period: Period for seasonality decomposition (e.g., 7 for weekly patterns)
            
        Returns:
            Dictionary with decomposition results (trend, seasonal, residual components)
        """
        try:
            # Get narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": f"Narrative {narrative_id} not found"}
            
            # Get historical complexity data
            historical_data = TimeSeriesAnalyzer._get_historical_complexity_data(narrative_id)
            
            if not historical_data or len(historical_data['timestamps']) < period * 2:
                logger.warning(f"Insufficient historical data for narrative {narrative_id}. Need at least {period * 2} data points.")
                return {"error": "Insufficient historical data for time-series decomposition"}
            
            # Prepare the time series data
            ts_data = pd.DataFrame({
                'date': [datetime.fromtimestamp(ts) for ts in historical_data['timestamps']],
                'overall': historical_data['overall_scores'],
                'linguistic': historical_data['linguistic_scores'],
                'logical': historical_data['logical_scores'],
                'rhetorical': historical_data['rhetorical_scores'],
                'emotional': historical_data['emotional_scores']
            })
            
            # Set date as index
            ts_data.set_index('date', inplace=True)
            
            # Perform decomposition for each dimension
            decomposition_results = {}
            for dimension in ['overall', 'linguistic', 'logical', 'rhetorical', 'emotional']:
                try:
                    # Ensure we have a valid time series
                    if ts_data[dimension].isna().any() or len(ts_data[dimension]) < period * 2:
                        continue
                        
                    # Perform decomposition
                    decomposition = seasonal_decompose(
                        ts_data[dimension], 
                        model='additive', 
                        period=period,
                        extrapolate_trend='freq'
                    )
                    
                    # Format results
                    decomposition_results[dimension] = {
                        'trend': decomposition.trend.tolist(),
                        'seasonal': decomposition.seasonal.tolist(),
                        'residual': decomposition.resid.tolist(),
                        'observed': decomposition.observed.tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to decompose {dimension} dimension: {e}")
                    continue
            
            if not decomposition_results:
                return {"error": "Failed to perform time-series decomposition on any dimension"}
            
            # Prepare response
            result = {
                'narrative_id': narrative_id,
                'title': narrative.title,
                'dates': [d.strftime('%Y-%m-%d') for d in ts_data.index],
                'period': period,
                'decomposition': decomposition_results
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in time-series analysis: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_trend_metrics(narrative_id: int) -> Dict[str, Any]:
        """
        Calculate trend metrics for a narrative's complexity evolution.
        
        Args:
            narrative_id: ID of the narrative to analyze
            
        Returns:
            Dictionary with trend metrics (slope, acceleration, etc.)
        """
        try:
            # Get narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": f"Narrative {narrative_id} not found"}
            
            # Get historical complexity data
            historical_data = TimeSeriesAnalyzer._get_historical_complexity_data(narrative_id)
            
            if not historical_data or len(historical_data['timestamps']) < 3:
                logger.warning(f"Insufficient historical data for narrative {narrative_id}")
                return {"error": "Insufficient historical data for trend analysis"}
            
            # Calculate days from first observation
            first_ts = historical_data['timestamps'][0]
            days_from_start = [(ts - first_ts) / 86400 for ts in historical_data['timestamps']]
            
            # Calculate trend metrics for each dimension
            dimensions = ['overall', 'linguistic', 'logical', 'rhetorical', 'emotional']
            dimension_keys = [f"{d}_scores" for d in dimensions]
            
            trend_metrics = {}
            for dimension, key in zip(dimensions, dimension_keys):
                # Get scores for this dimension
                scores = historical_data[key]
                
                # Calculate metrics
                metrics = TimeSeriesAnalyzer._calculate_dimension_trend_metrics(days_from_start, scores)
                trend_metrics[dimension] = metrics
            
            # Calculate overall metrics
            overall_metrics = trend_metrics['overall']
            
            # Determine most significant dimensions
            dimensions_by_impact = sorted(
                [d for d in dimensions if d != 'overall'],
                key=lambda d: abs(trend_metrics[d]['slope']),
                reverse=True
            )
            
            significant_dimensions = []
            for dim in dimensions_by_impact:
                if abs(trend_metrics[dim]['slope']) > 0.02:  # Threshold for significance
                    direction = "increasing" if trend_metrics[dim]['slope'] > 0 else "decreasing"
                    significant_dimensions.append({
                        "dimension": dim,
                        "direction": direction,
                        "slope": trend_metrics[dim]['slope']
                    })
            
            # Prepare response
            result = {
                'narrative_id': narrative_id,
                'title': narrative.title,
                'data_points': len(historical_data['timestamps']),
                'first_observation': datetime.fromtimestamp(first_ts).strftime('%Y-%m-%d'),
                'last_observation': datetime.fromtimestamp(historical_data['timestamps'][-1]).strftime('%Y-%m-%d'),
                'overall_trend': {
                    'slope': overall_metrics['slope'],
                    'intercept': overall_metrics['intercept'],
                    'r_squared': overall_metrics['r_squared'],
                    'current_value': overall_metrics['current_value'],
                    'acceleration': overall_metrics['acceleration'],
                    'recent_change_rate': overall_metrics['recent_change_rate']
                },
                'significant_dimensions': significant_dimensions,
                'dimension_trends': {d: trend_metrics[d] for d in dimensions}
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating trend metrics: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def generate_ai_insights(narrative_id: int) -> Dict[str, Any]:
        """
        Generate AI-based insights about the narrative's complexity evolution.
        
        Args:
            narrative_id: ID of the narrative to analyze
            
        Returns:
            Dictionary with AI-generated insights
        """
        try:
            # Get trend metrics
            trend_metrics = TimeSeriesAnalyzer.calculate_trend_metrics(narrative_id)
            
            if "error" in trend_metrics:
                return {"error": trend_metrics["error"]}
            
            # Get time-series decomposition
            decomposition = TimeSeriesAnalyzer.analyze_narrative_time_series(narrative_id)
            
            # Prepare insights based on trend metrics and decomposition
            insights = []
            
            # Overall trend insight
            overall_slope = trend_metrics['overall_trend']['slope']
            overall_r_squared = trend_metrics['overall_trend']['r_squared']
            
            if abs(overall_slope) < 0.02:
                insights.append("The narrative's complexity has remained relatively stable.")
            elif overall_slope > 0:
                confidence = "strong" if overall_r_squared > 0.7 else "moderate" if overall_r_squared > 0.4 else "weak"
                insights.append(f"The narrative shows a {confidence} upward trend in complexity.")
            else:
                confidence = "strong" if overall_r_squared > 0.7 else "moderate" if overall_r_squared > 0.4 else "weak"
                insights.append(f"The narrative shows a {confidence} downward trend in complexity.")
            
            # Significant dimensions insight
            if trend_metrics['significant_dimensions']:
                top_dimension = trend_metrics['significant_dimensions'][0]
                insights.append(f"The {top_dimension['dimension']} complexity is {top_dimension['direction']} most significantly.")
            
            # Acceleration insight
            acceleration = trend_metrics['overall_trend']['acceleration']
            if abs(acceleration) > 0.005:
                direction = "accelerating" if acceleration > 0 else "decelerating"
                insights.append(f"The rate of complexity change is {direction}.")
            
            # Seasonality insight (if decomposition was successful)
            if "error" not in decomposition:
                for dimension, decomp_data in decomposition['decomposition'].items():
                    seasonal = decomp_data.get('seasonal', [])
                    if seasonal and max(seasonal) - min(seasonal) > 1.0:
                        insights.append(f"The {dimension} complexity shows cyclical patterns.")
                        break
            
            # Generate overall assessment
            if overall_slope > 0.05 and trend_metrics['overall_trend']['current_value'] > 7:
                assessment = "This narrative requires close monitoring due to its high complexity and upward trend."
            elif overall_slope > 0.05:
                assessment = "This narrative shows increasing sophistication and should be monitored."
            elif trend_metrics['overall_trend']['current_value'] > 8:
                assessment = "This narrative has high complexity but appears stable at present."
            else:
                assessment = "This narrative does not currently exhibit concerning complexity patterns."
            
            return {
                'narrative_id': narrative_id,
                'title': trend_metrics['title'],
                'insights': insights,
                'assessment': assessment,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _get_historical_complexity_data(narrative_id: int) -> Optional[Dict[str, Any]]:
        """
        Get historical complexity data for a narrative.
        
        Args:
            narrative_id: ID of the narrative to get data for
            
        Returns:
            Dictionary with historical complexity data or None
        """
        try:
            # Get narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative or not narrative.meta_data:
                return None
            
            # Parse metadata
            try:
                metadata = json.loads(narrative.meta_data)
            except (json.JSONDecodeError, TypeError):
                return None
            
            # Look for complexity snapshots
            snapshots = metadata.get('complexity_snapshots', [])
            if not snapshots:
                # If no snapshots, check if we have a single analysis
                analysis = metadata.get('complexity_analysis')
                if not analysis or 'overall_complexity_score' not in analysis:
                    return None
                
                # Create a single snapshot from the current analysis
                timestamp = analysis.get('analyzed_at', int(datetime.now().timestamp()))
                snapshots = [{
                    'timestamp': timestamp,
                    'overall_score': analysis.get('overall_complexity_score', 0),
                    'linguistic_score': analysis.get('linguistic_complexity', {}).get('score', 0),
                    'logical_score': analysis.get('logical_structure', {}).get('score', 0),
                    'rhetorical_score': analysis.get('rhetorical_techniques', {}).get('score', 0),
                    'emotional_score': analysis.get('emotional_manipulation', {}).get('score', 0),
                    'title': narrative.title
                }]
            
            if not snapshots or len(snapshots) < 2:
                return None
            
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
                'title': narrative.title
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical complexity data: {e}")
            return None
    
    @staticmethod
    def _calculate_dimension_trend_metrics(days_from_start: List[float], scores: List[float]) -> Dict[str, Any]:
        """
        Calculate trend metrics for a dimension.
        
        Args:
            days_from_start: List of days from first observation
            scores: List of complexity scores
            
        Returns:
            Dictionary with trend metrics
        """
        # Ensure we have enough data points
        if len(days_from_start) < 2 or len(scores) < 2:
            return {
                'slope': 0,
                'intercept': scores[0] if scores else 0,
                'r_squared': 0,
                'current_value': scores[-1] if scores else 0,
                'acceleration': 0,
                'recent_change_rate': 0
            }
        
        # Convert to numpy arrays
        x = np.array(days_from_start)
        y = np.array(scores)
        
        # Linear regression for overall trend
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate R-squared
        y_pred = model.predict(x.reshape(-1, 1))
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # Calculate acceleration (change in slope)
        acceleration = 0
        if len(days_from_start) >= 6:
            # Compare first half and second half slopes
            mid_point = len(days_from_start) // 2
            first_half_x = x[:mid_point]
            first_half_y = y[:mid_point]
            second_half_x = x[mid_point:]
            second_half_y = y[mid_point:]
            
            # Calculate slopes for both halves
            if len(first_half_x) >= 2 and len(second_half_x) >= 2:
                first_model = LinearRegression()
                first_model.fit(first_half_x.reshape(-1, 1), first_half_y)
                first_slope = first_model.coef_[0]
                
                second_model = LinearRegression()
                second_model.fit(second_half_x.reshape(-1, 1), second_half_y)
                second_slope = second_model.coef_[0]
                
                acceleration = second_slope - first_slope
        
        # Calculate recent change rate (last few points)
        recent_change_rate = 0
        if len(scores) >= 3:
            # Use last 3 points
            recent_points = scores[-3:]
            if len(recent_points) >= 2:
                recent_change_rate = (recent_points[-1] - recent_points[0]) / (len(recent_points) - 1)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'current_value': scores[-1] if scores else 0,
            'acceleration': acceleration,
            'recent_change_rate': recent_change_rate
        }