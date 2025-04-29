"""
Predictive Modeling Service for the CIVILIAN system.

This service provides predictive modeling capabilities for narratives, including:
- Trajectory forecasting for narrative spread and impact
- Anomaly detection for identifying unusual patterns
- Trend prediction for counter-narrative effectiveness
"""

import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union

from sqlalchemy import desc, func
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
import uuid

from app import db
from models import (
    DetectedNarrative, 
    NarrativeInstance, 
    CounterMessage
)
from utils.time_series import TimeSeriesAnalyzer
from utils.concurrency import run_in_thread
from services.narrative_network import NarrativeNetworkService

# Configure logging
logger = logging.getLogger(__name__)

class PredictiveModeling:
    """Service for predictive modeling of narrative trajectories and impacts."""
    
    def __init__(self):
        """Initialize the predictive modeling service."""
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.narrative_network = NarrativeNetworkService()
        
        # Default forecast settings
        self.default_horizon = 7  # Days
        self.default_confidence = 0.8  # 80% confidence interval
        self.anomaly_threshold = 0.95  # Threshold for anomaly detection
        
        # Model configurations
        self.model_configs = {
            'arima': {
                'order': (2, 1, 2),
                'seasonal_order': (1, 1, 1, 7),  # Weekly seasonality
                'trend': 'c',
                'enforce_stationarity': False,
                'enforce_invertibility': False
            },
            'prophet': {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'multiplicative',
                'daily_seasonality': False,
                'weekly_seasonality': True,
                'yearly_seasonality': False
            },
            'isolation_forest': {
                'contamination': 'auto',
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42
            }
        }
        
        # Import InformationSource model here to avoid circular imports
        from models import InformationSource
        self.InformationSource = InformationSource
    
    def get_narrative_time_series(self, narrative_id: int, days: int = 30) -> pd.DataFrame:
        """
        Get time series data for a narrative.
        
        Args:
            narrative_id: The ID of the narrative
            days: Number of days of history to include
            
        Returns:
            DataFrame with dates and counts
        """
        # Get current date in UTC
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query narrative instances grouped by day
        instances = db.session.query(
            func.date(NarrativeInstance.detected_at).label('date'),
            func.count(NarrativeInstance.id).label('count')
        ).filter(
            NarrativeInstance.narrative_id == narrative_id,
            NarrativeInstance.detected_at >= start_date,
            NarrativeInstance.detected_at <= end_date
        ).group_by(
            func.date(NarrativeInstance.detected_at)
        ).order_by(
            func.date(NarrativeInstance.detected_at)
        ).all()
        
        # Create DataFrame
        df = pd.DataFrame([(i.date, i.count) for i in instances], columns=['date', 'count'])
        
        # Fill missing dates with zeros
        date_range = pd.date_range(start=start_date.date(), end=end_date.date())
        date_df = pd.DataFrame({'date': date_range})
        df = pd.merge(date_df, df, on='date', how='left').fillna(0)
        
        return df
    
    def get_counter_message_time_series(self, counter_id: int, days: int = 30) -> pd.DataFrame:
        """
        Get time series data for counter-narrative effectiveness.
        
        Args:
            counter_id: The ID of the counter message
            days: Number of days of history to include
            
        Returns:
            DataFrame with dates and effectiveness metrics
        """
        # This is a placeholder for actual effectiveness metrics
        # In a real implementation, this would come from monitoring systems
        # that track engagement with counter-narratives
        
        # Get counter message
        counter = CounterMessage.query.get(counter_id)
        if not counter or not counter.narrative_id:
            logger.warning(f"Counter message {counter_id} not found or has no narrative")
            return pd.DataFrame(columns=['date', 'effectiveness'])
        
        # Get narrative instances for the same narrative, as a proxy for effectiveness
        # (the idea is that if counter-narrative is effective, instances should decrease)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get counter message creation date
        counter_created = counter.created_at
        
        # Get instances before and after counter message
        instances_before = db.session.query(
            func.date(NarrativeInstance.detected_at).label('date'),
            func.count(NarrativeInstance.id).label('count')
        ).filter(
            NarrativeInstance.narrative_id == counter.narrative_id,
            NarrativeInstance.detected_at >= start_date,
            NarrativeInstance.detected_at < counter_created
        ).group_by(
            func.date(NarrativeInstance.detected_at)
        ).order_by(
            func.date(NarrativeInstance.detected_at)
        ).all()
        
        instances_after = db.session.query(
            func.date(NarrativeInstance.detected_at).label('date'),
            func.count(NarrativeInstance.id).label('count')
        ).filter(
            NarrativeInstance.narrative_id == counter.narrative_id,
            NarrativeInstance.detected_at >= counter_created,
            NarrativeInstance.detected_at <= end_date
        ).group_by(
            func.date(NarrativeInstance.detected_at)
        ).order_by(
            func.date(NarrativeInstance.detected_at)
        ).all()
        
        # Create DataFrames
        df_before = pd.DataFrame([(i.date, i.count) for i in instances_before], columns=['date', 'count'])
        df_after = pd.DataFrame([(i.date, i.count) for i in instances_after], columns=['date', 'count'])
        
        # Calculate baseline (average before counter message)
        baseline = df_before['count'].mean() if not df_before.empty else 0
        
        # Calculate effectiveness (reduction from baseline)
        df_after['effectiveness'] = df_after['count'].apply(lambda x: max(0, 1 - (x / baseline)) if baseline > 0 else 0)
        
        # Fill missing dates
        date_range = pd.date_range(start=counter_created.date(), end=end_date.date())
        date_df = pd.DataFrame({'date': date_range})
        df = pd.merge(date_df, df_after[['date', 'effectiveness']], on='date', how='left').fillna(0)
        
        return df[['date', 'effectiveness']]
    
    def forecast_narrative_trajectory(
        self, 
        narrative_id: int, 
        days_horizon: int = None,
        model_type: str = 'arima',
        training_days: int = 30
    ) -> Dict[str, Any]:
        """
        Forecast the future trajectory of a narrative.
        
        Args:
            narrative_id: The ID of the narrative
            days_horizon: Number of days to forecast
            model_type: Type of model to use ('arima' or 'prophet')
            training_days: Number of days of historical data to use
            
        Returns:
            Dictionary with forecast results
        """
        # Set default horizon if not provided
        if days_horizon is None:
            days_horizon = self.default_horizon
        
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            logger.error(f"Narrative {narrative_id} not found")
            return {
                "success": False,
                "error": f"Narrative {narrative_id} not found"
            }
        
        # Get time series data
        df = self.get_narrative_time_series(narrative_id, days=training_days)
        if df.empty or df['count'].sum() == 0:
            logger.warning(f"Insufficient data for narrative {narrative_id}")
            return {
                "success": False,
                "error": "Insufficient data for forecasting"
            }
        
        forecast_data = {}
        
        try:
            if model_type == 'arima':
                forecast_data = self._forecast_arima(df, days_horizon)
            elif model_type == 'prophet':
                forecast_data = self._forecast_prophet(df, days_horizon)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return {
                    "success": False,
                    "error": f"Unknown model type: {model_type}"
                }
            
            # Instead of saving to database, store prediction info in narrative metadata
            prediction_id = str(uuid.uuid4())
            
            # Add prediction ID to results
            forecast_data['prediction_id'] = prediction_id
            forecast_data['success'] = True
            
            # Update narrative metadata with prediction info
            meta_data = narrative.get_meta_data() or {}
            meta_data['latest_prediction'] = {
                'id': prediction_id,
                'date': datetime.utcnow().isoformat(),
                'model': model_type,
                'horizon_days': days_horizon,
                'peak_day': forecast_data.get('peak_day'),
                'peak_value': forecast_data.get('peak_value'),
                'trend_direction': forecast_data.get('trend_direction')
            }
            narrative.set_meta_data(meta_data)
            db.session.commit()
            
            return forecast_data
            
        except Exception as e:
            logger.exception(f"Error forecasting narrative {narrative_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Error in forecasting: {str(e)}"
            }
    
    def _forecast_arima(self, df: pd.DataFrame, days_horizon: int) -> Dict[str, Any]:
        """
        Forecast using ARIMA/SARIMAX model.
        
        Args:
            df: DataFrame with date and count columns
            days_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Check if we have enough data for ARIMA
        if len(df) < 7:
            # Fall back to simple moving average
            avg = df['count'].mean()
            trend = df['count'].iloc[-1] - df['count'].iloc[0] if len(df) > 1 else 0
            trend_per_day = trend / (len(df) - 1) if len(df) > 1 else 0
            
            forecast_dates = [
                (df['date'].iloc[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                for i in range(days_horizon)
            ]
            forecast_values = [max(0, avg + trend_per_day * i) for i in range(1, days_horizon + 1)]
            
            return {
                "dates": forecast_dates,
                "values": forecast_values,
                "lower_bound": [max(0, val * 0.7) for val in forecast_values],
                "upper_bound": [val * 1.3 for val in forecast_values],
                "model": "simple_trend",
                "peak_day": forecast_dates[-1] if trend_per_day > 0 else forecast_dates[0],
                "peak_value": max(forecast_values),
                "trend_direction": "up" if trend_per_day > 0 else "down" if trend_per_day < 0 else "stable"
            }
        
        # Prepare data for ARIMA
        y = df['count'].values
        
        try:
            # Try SARIMAX first with weekly seasonality
            config = self.model_configs['arima']
            model = SARIMAX(
                y,
                order=config['order'],
                seasonal_order=config['seasonal_order'],
                trend=config['trend'],
                enforce_stationarity=config['enforce_stationarity'],
                enforce_invertibility=config['enforce_invertibility']
            )
            fit_model = model.fit(disp=False)
            
            # Make forecast
            forecast = fit_model.get_forecast(steps=days_horizon)
            forecast_values = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=1-self.default_confidence)
            
            # Generate forecast dates
            last_date = df['date'].iloc[-1]
            forecast_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                for i in range(days_horizon)
            ]
            
            # Calculate metrics
            peak_idx = forecast_values.argmax()
            peak_day = forecast_dates[peak_idx]
            peak_value = forecast_values[peak_idx]
            
            # Determine trend direction
            if len(forecast_values) > 1:
                first_value = forecast_values[0]
                last_value = forecast_values[-1]
                if last_value > first_value * 1.1:  # 10% increase
                    trend_direction = "up"
                elif last_value < first_value * 0.9:  # 10% decrease
                    trend_direction = "down"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
            
            return {
                "dates": forecast_dates,
                "values": forecast_values.tolist(),
                "lower_bound": conf_int[:, 0].tolist(),
                "upper_bound": conf_int[:, 1].tolist(),
                "model": "sarimax",
                "peak_day": peak_day,
                "peak_value": float(peak_value),
                "trend_direction": trend_direction
            }
            
        except Exception as e:
            logger.warning(f"SARIMAX failed, falling back to ARIMA: {str(e)}")
            
            try:
                # Fall back to simpler ARIMA model
                model = ARIMA(y, order=(1, 1, 0))
                fit_model = model.fit()
                
                # Make forecast
                forecast = fit_model.get_forecast(steps=days_horizon)
                forecast_values = forecast.predicted_mean
                conf_int = forecast.conf_int(alpha=1-self.default_confidence)
                
                # Generate forecast dates
                last_date = df['date'].iloc[-1]
                forecast_dates = [
                    (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                    for i in range(days_horizon)
                ]
                
                # Calculate metrics
                peak_idx = forecast_values.argmax()
                peak_day = forecast_dates[peak_idx]
                peak_value = forecast_values[peak_idx]
                
                # Determine trend direction
                if len(forecast_values) > 1:
                    first_value = forecast_values[0]
                    last_value = forecast_values[-1]
                    if last_value > first_value * 1.1:  # 10% increase
                        trend_direction = "up"
                    elif last_value < first_value * 0.9:  # 10% decrease
                        trend_direction = "down"
                    else:
                        trend_direction = "stable"
                else:
                    trend_direction = "stable"
                
                return {
                    "dates": forecast_dates,
                    "values": forecast_values.tolist(),
                    "lower_bound": conf_int[:, 0].tolist(),
                    "upper_bound": conf_int[:, 1].tolist(),
                    "model": "arima",
                    "peak_day": peak_day,
                    "peak_value": float(peak_value),
                    "trend_direction": trend_direction
                }
                
            except Exception as e2:
                logger.error(f"ARIMA also failed: {str(e2)}")
                # Fall back to simple moving average
                return self._forecast_simple(df, days_horizon)
    
    def _forecast_prophet(self, df: pd.DataFrame, days_horizon: int) -> Dict[str, Any]:
        """
        Forecast using Facebook Prophet model.
        
        Args:
            df: DataFrame with date and count columns
            days_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Prepare data for Prophet
            prophet_df = df.rename(columns={'date': 'ds', 'count': 'y'})
            
            # Initialize and fit Prophet model
            config = self.model_configs['prophet']
            model = Prophet(
                changepoint_prior_scale=config['changepoint_prior_scale'],
                seasonality_prior_scale=config['seasonality_prior_scale'],
                seasonality_mode=config['seasonality_mode'],
                daily_seasonality=config['daily_seasonality'],
                weekly_seasonality=config['weekly_seasonality'],
                yearly_seasonality=config['yearly_seasonality']
            )
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_horizon)
            
            # Make forecast
            forecast = model.predict(future)
            
            # Extract results
            last_date_idx = len(prophet_df)
            forecast_result = forecast.iloc[last_date_idx:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            # Format dates
            forecast_dates = forecast_result['ds'].dt.strftime('%Y-%m-%d').tolist()
            forecast_values = forecast_result['yhat'].tolist()
            lower_bound = forecast_result['yhat_lower'].tolist()
            upper_bound = forecast_result['yhat_upper'].tolist()
            
            # Ensure non-negative values
            forecast_values = [max(0, x) for x in forecast_values]
            lower_bound = [max(0, x) for x in lower_bound]
            
            # Calculate metrics
            peak_idx = forecast_values.index(max(forecast_values)) if forecast_values else 0
            peak_day = forecast_dates[peak_idx] if forecast_dates else None
            peak_value = forecast_values[peak_idx] if forecast_values else None
            
            # Determine trend direction
            if forecast_values and len(forecast_values) > 1:
                first_value = forecast_values[0]
                last_value = forecast_values[-1]
                if last_value > first_value * 1.1:  # 10% increase
                    trend_direction = "up"
                elif last_value < first_value * 0.9:  # 10% decrease
                    trend_direction = "down"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
            
            return {
                "dates": forecast_dates,
                "values": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "model": "prophet",
                "peak_day": peak_day,
                "peak_value": peak_value,
                "trend_direction": trend_direction
            }
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {str(e)}")
            # Fall back to ARIMA or simple forecast
            try:
                return self._forecast_arima(df, days_horizon)
            except:
                return self._forecast_simple(df, days_horizon)
    
    def forecast_narrative(self, narrative_id: int, metric: str = 'complexity', 
                      model_type: str = 'arima', force_refresh: bool = False) -> Dict[str, Any]:
        """
        Generate forecast for a narrative with the specified metric.
        
        Args:
            narrative_id: ID of the narrative
            metric: Metric to forecast ('complexity', 'spread', etc.)
            model_type: Type of model to use
            force_refresh: Whether to force a refresh of the forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Get the narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                return {"error": f"Narrative {narrative_id} not found"}
                
            # Check if we have a recent forecast in metadata
            meta_data = narrative.get_meta_data() or {}
            if not force_refresh and 'latest_prediction' in meta_data:
                latest = meta_data['latest_prediction']
                # Use cached forecast if it's recent (within 24 hours) and for the same model
                prediction_date = datetime.fromisoformat(latest['date'])
                if (latest.get('model') == model_type and 
                    (datetime.utcnow() - prediction_date).total_seconds() < 24 * 3600):
                    logger.info(f"Using cached forecast for narrative {narrative_id}")
                    
                    # Check if we have the full forecast details in metadata
                    if 'forecast_data' in meta_data:
                        return meta_data['forecast_data']
            
            # Get time series data based on metric
            if metric == 'complexity':
                # Use narrative complexity metrics for forecasting
                df = self._get_complexity_time_series(narrative_id)
            else:
                # Default to instance count
                df = self.get_narrative_time_series(narrative_id)
                
            if df.empty:
                return {"error": f"Insufficient data for narrative {narrative_id}"}
                
            # Generate forecast based on model type
            if model_type == 'arima':
                forecast_data = self._forecast_arima(df, self.default_horizon)
            elif model_type == 'prophet':
                # Since we're not using Prophet now, fall back to ARIMA
                forecast_data = self._forecast_arima(df, self.default_horizon)
            else:
                # Default to simple forecast
                forecast_data = self._forecast_simple(df, self.default_horizon)
                
            # Add narrative info
            forecast_data['narrative_id'] = narrative_id
            forecast_data['narrative_title'] = narrative.title
            forecast_data['metric'] = metric
            forecast_data['generated_at'] = datetime.utcnow().isoformat()
            forecast_data['success'] = True
            
            # Store forecast in metadata
            meta_data['latest_prediction'] = {
                'id': str(uuid.uuid4()),
                'date': datetime.utcnow().isoformat(),
                'model': model_type,
                'metric': metric,
                'horizon_days': self.default_horizon,
                'peak_day': forecast_data.get('peak_day'),
                'peak_value': forecast_data.get('peak_value'),
                'trend_direction': forecast_data.get('trend_direction')
            }
            meta_data['forecast_data'] = forecast_data
            narrative.set_meta_data(meta_data)
            db.session.commit()
            
            return forecast_data
            
        except Exception as e:
            logger.exception(f"Error forecasting narrative {narrative_id}: {str(e)}")
            return {"error": f"Error generating forecast: {str(e)}"}
            
    def _get_complexity_time_series(self, narrative_id: int, days: int = 30) -> pd.DataFrame:
        """
        Get time series of complexity metrics for a narrative.
        
        Args:
            narrative_id: ID of the narrative
            days: Number of days of history
            
        Returns:
            DataFrame with dates and complexity values
        """
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return pd.DataFrame()
            
        # For now, use a simple approach based on instance count
        # In a real implementation, this would use actual complexity metrics
        df = self.get_narrative_time_series(narrative_id, days)
        
        # Apply a transformation to convert count to complexity
        # This is a placeholder for actual complexity calculation
        if not df.empty:
            # Scale to 0-1 range and apply logarithmic growth
            max_count = df['count'].max()
            if max_count > 0:
                df['complexity'] = df['count'].apply(lambda x: min(1.0, 0.3 + 0.7 * np.log1p(x) / np.log1p(max_count)))
            else:
                df['complexity'] = 0
                
            # Drop the count column
            df = df[['date', 'complexity']]
            df = df.rename(columns={'complexity': 'count'})  # Rename for compatibility with forecast methods
            
        return df
        
    def find_threshold_crossing(self, forecast_data: Dict[str, Any], 
                               threshold_value: float, direction: str = 'above') -> Dict[str, Any]:
        """
        Find when a forecast crosses a threshold.
        
        Args:
            forecast_data: Forecast data from forecast_narrative
            threshold_value: Threshold value to check
            direction: Direction of crossing ('above' or 'below')
            
        Returns:
            Dictionary with threshold crossing data
        """
        result = {
            "threshold_value": threshold_value,
            "direction": direction,
            "crosses_threshold": False,
            "crossing_point": None,
            "crossing_date": None,
            "days_until_crossing": None
        }
        
        if 'error' in forecast_data or 'values' not in forecast_data or 'dates' not in forecast_data:
            result["error"] = "Invalid forecast data"
            return result
            
        values = forecast_data['values']
        dates = forecast_data['dates']
        
        # Find threshold crossing
        for i, value in enumerate(values):
            if (direction == 'above' and value >= threshold_value) or \
               (direction == 'below' and value <= threshold_value):
                result["crosses_threshold"] = True
                result["crossing_point"] = i
                result["crossing_date"] = dates[i]
                result["days_until_crossing"] = i + 1  # +1 since forecast starts tomorrow
                result["crossing_value"] = value
                break
                
        return result
        
    def analyze_key_factors(self, narrative_id: int, metric: str = 'complexity') -> Dict[str, Any]:
        """
        Analyze key factors influencing a narrative metric.
        
        Args:
            narrative_id: ID of the narrative
            metric: Metric to analyze
            
        Returns:
            Dictionary with key factors analysis
        """
        try:
            # Get the narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                return {"error": f"Narrative {narrative_id} not found"}
                
            # Get narrative data
            narrative_data = narrative.to_dict()
            
            # Get network influence
            network_data = self.narrative_network.analyze_narrative_influence(narrative_id)
            
            # Get temporal patterns
            temporal_data = self._analyze_temporal_patterns(narrative_id)
            
            # Combine factors
            factors = []
            
            # Add network factors
            factors.append({
                "name": "Network Centrality",
                "value": network_data.get("centrality", 0),
                "impact": self._calculate_impact(network_data.get("centrality", 0), 0.3, 0.8),
                "description": "How central this narrative is in the overall belief network"
            })
            
            factors.append({
                "name": "Connection Strength",
                "value": network_data.get("avg_connection_strength", 0),
                "impact": self._calculate_impact(network_data.get("avg_connection_strength", 0), 0.4, 0.9),
                "description": "Average strength of connections to other narratives"
            })
            
            # Add temporal factors
            factors.append({
                "name": "Consistency",
                "value": temporal_data.get("consistency", 0),
                "impact": self._calculate_impact(temporal_data.get("consistency", 0), 0.5, 0.7),
                "description": "How consistent the narrative appearances are over time"
            })
            
            factors.append({
                "name": "Growth Rate",
                "value": temporal_data.get("growth_rate", 0),
                "impact": self._calculate_impact(abs(temporal_data.get("growth_rate", 0)), 0.1, 0.5) * 
                          (1 if temporal_data.get("growth_rate", 0) > 0 else -1),
                "description": "Rate of growth in narrative instances over time"
            })
            
            # Add content factors (from narrative metadata)
            meta_data = narrative.get_meta_data() or {}
            
            factors.append({
                "name": "Source Diversity",
                "value": meta_data.get("source_diversity", 0),
                "impact": self._calculate_impact(meta_data.get("source_diversity", 0), 0.2, 0.6),
                "description": "Diversity of sources spreading this narrative"
            })
            
            # Sort factors by absolute impact
            factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
            
            return {
                "narrative_id": narrative_id,
                "narrative_title": narrative.title,
                "metric": metric,
                "factors": factors,
                "total_score": sum(factor["impact"] for factor in factors),
                "network_score": network_data.get("influence_score", 0),
                "temporal_score": temporal_data.get("temporal_score", 0)
            }
            
        except Exception as e:
            logger.exception(f"Error analyzing key factors for narrative {narrative_id}: {str(e)}")
            return {"error": f"Error analyzing key factors: {str(e)}"}
            
    def _calculate_impact(self, value: float, min_threshold: float, max_threshold: float) -> float:
        """Calculate normalized impact score."""
        if value <= min_threshold:
            return 0.0
        elif value >= max_threshold:
            return 1.0
        else:
            return (value - min_threshold) / (max_threshold - min_threshold)
            
    def _analyze_temporal_patterns(self, narrative_id: int) -> Dict[str, Any]:
        """Analyze temporal patterns for a narrative."""
        # Get time series data
        df = self.get_narrative_time_series(narrative_id)
        
        if df.empty:
            return {
                "consistency": 0,
                "growth_rate": 0,
                "temporal_score": 0
            }
            
        # Calculate consistency (low variance = high consistency)
        mean = df['count'].mean()
        if mean > 0:
            consistency = 1.0 - min(1.0, df['count'].std() / mean)
        else:
            consistency = 0
            
        # Calculate growth rate
        if len(df) > 1:
            first_half = df.iloc[:len(df)//2]['count'].mean()
            second_half = df.iloc[len(df)//2:]['count'].mean()
            
            if first_half > 0:
                growth_rate = (second_half - first_half) / first_half
            else:
                growth_rate = 1.0 if second_half > 0 else 0.0
        else:
            growth_rate = 0
            
        # Calculate overall temporal score
        temporal_score = 0.5 * consistency + 0.5 * min(1.0, max(0, growth_rate))
        
        return {
            "consistency": consistency,
            "growth_rate": growth_rate,
            "temporal_score": temporal_score
        }
        
    def simulate_scenario(self, narrative_id: int, interventions: Dict[str, Any], 
                         metric: str = 'complexity', model_type: str = 'arima') -> Dict[str, Any]:
        """
        Simulate a what-if scenario with interventions.
        
        Args:
            narrative_id: ID of the narrative
            interventions: Dictionary of interventions to apply
            metric: Metric to forecast
            model_type: Type of model to use
            
        Returns:
            Dictionary with scenario results
        """
        try:
            # Generate baseline forecast first
            baseline = self.forecast_narrative(narrative_id, metric, model_type)
            
            if 'error' in baseline:
                return baseline
                
            # Apply interventions to create modified forecast
            modified = self._apply_interventions(baseline, interventions)
            
            return {
                "narrative_id": narrative_id,
                "metric": metric,
                "baseline": baseline,
                "modified": modified,
                "interventions": interventions,
                "difference": {
                    "values": [m - b for m, b in zip(modified['values'], baseline['values'])],
                    "peak_change": modified['peak_value'] - baseline['peak_value'],
                    "trend_change": modified['trend_direction'] != baseline['trend_direction'],
                    "percent_change": (modified['peak_value'] - baseline['peak_value']) / max(0.001, baseline['peak_value']) * 100
                }
            }
            
        except Exception as e:
            logger.exception(f"Error simulating scenario for narrative {narrative_id}: {str(e)}")
            return {"error": f"Error simulating scenario: {str(e)}"}
            
    def _apply_interventions(self, baseline: Dict[str, Any], 
                            interventions: Dict[str, Any]) -> Dict[str, Any]:
        """Apply interventions to a baseline forecast."""
        # Create a copy of the baseline
        result = baseline.copy()
        result['values'] = baseline['values'].copy()
        
        # Apply each intervention
        for intervention_key, params in interventions.items():
            if params['type'] == 'step':
                # Apply step change at a specific date
                try:
                    step_date = params['date']
                    step_value = float(params['value'])
                    
                    # Find index for step date
                    step_idx = baseline['dates'].index(step_date) if step_date in baseline['dates'] else 0
                    
                    # Apply step change
                    for i in range(step_idx, len(result['values'])):
                        result['values'][i] += step_value
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error applying step intervention: {e}")
                    
            elif params['type'] == 'trend':
                # Apply trend factor
                try:
                    factor = float(params['factor'])
                    start_date = params.get('start_date')
                    
                    # Find index for start date
                    start_idx = baseline['dates'].index(start_date) if start_date in baseline['dates'] else 0
                    
                    # Apply trend factor
                    for i in range(start_idx, len(result['values'])):
                        result['values'][i] *= factor
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error applying trend intervention: {e}")
                    
            elif params['type'] == 'counter_message':
                # Apply counter-message effect (initial impact with decay)
                try:
                    counter_date = params['date']
                    impact = float(params['impact'])
                    decay = float(params['decay'])
                    
                    # Find index for counter date
                    counter_idx = baseline['dates'].index(counter_date) if counter_date in baseline['dates'] else 0
                    
                    # Apply counter-message effect with decay
                    for i in range(counter_idx, len(result['values'])):
                        # Calculate time-based decay
                        time_factor = decay ** (i - counter_idx)
                        
                        # Apply impact
                        result['values'][i] += impact * time_factor * baseline['values'][i]
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error applying counter-message intervention: {e}")
                    
        # Ensure non-negative values
        result['values'] = [max(0, v) for v in result['values']]
        
        # Recalculate metrics
        peak_idx = result['values'].index(max(result['values']))
        result['peak_day'] = result['dates'][peak_idx]
        result['peak_value'] = result['values'][peak_idx]
        
        # Determine trend direction
        if len(result['values']) > 1:
            first_value = result['values'][0]
            last_value = result['values'][-1]
            if last_value > first_value * 1.1:  # 10% increase
                result['trend_direction'] = "up"
            elif last_value < first_value * 0.9:  # 10% decrease
                result['trend_direction'] = "down"
            else:
                result['trend_direction'] = "stable"
        else:
            result['trend_direction'] = "stable"
            
        return result
    
    def _forecast_simple(self, df: pd.DataFrame, days_horizon: int) -> Dict[str, Any]:
        """
        Simple forecasting method as a fallback.
        
        Args:
            df: DataFrame with date and count columns
            days_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Use simple moving average and trend
        window = min(7, len(df))
        if window < 2:
            # Just use the last value if we only have one data point
            last_value = df['count'].iloc[-1]
            forecast_values = [last_value] * days_horizon
            
            # Generate forecast dates
            last_date = df['date'].iloc[-1]
            forecast_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                for i in range(days_horizon)
            ]
            
            return {
                "dates": forecast_dates,
                "values": forecast_values,
                "lower_bound": [max(0, v * 0.7) for v in forecast_values],
                "upper_bound": [v * 1.3 for v in forecast_values],
                "model": "constant",
                "peak_day": forecast_dates[0],
                "peak_value": last_value,
                "trend_direction": "stable"
            }
        
        # Calculate recent average and trend
        recent_df = df.iloc[-window:]
        recent_avg = recent_df['count'].mean()
        
        # Calculate trend direction
        first_value = recent_df['count'].iloc[0]
        last_value = recent_df['count'].iloc[-1]
        trend_per_day = (last_value - first_value) / (window - 1) if window > 1 else 0
        
        # Generate forecast
        forecast_values = [max(0, recent_avg + trend_per_day * i) for i in range(1, days_horizon + 1)]
        
        # Generate forecast dates
        last_date = df['date'].iloc[-1]
        forecast_dates = [
            (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
            for i in range(days_horizon)
        ]
        
        # Calculate confidence intervals
        variance = recent_df['count'].var() if len(recent_df) > 1 else recent_avg * 0.1
        std_dev = np.sqrt(variance) if variance > 0 else recent_avg * 0.1
        z_score = 1.96  # 95% confidence
        
        lower_bound = [max(0, val - z_score * std_dev) for val in forecast_values]
        upper_bound = [val + z_score * std_dev for val in forecast_values]
        
        # Determine peak
        peak_idx = forecast_values.index(max(forecast_values)) if forecast_values else 0
        peak_day = forecast_dates[peak_idx] if forecast_dates else None
        peak_value = forecast_values[peak_idx] if forecast_values else None
        
        # Determine overall trend direction
        if trend_per_day > 0.1:  # Arbitrary threshold
            trend_direction = "up"
        elif trend_per_day < -0.1:  # Arbitrary threshold
            trend_direction = "down"
        else:
            trend_direction = "stable"
        
        return {
            "dates": forecast_dates,
            "values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model": "simple_trend",
            "peak_day": peak_day,
            "peak_value": peak_value,
            "trend_direction": trend_direction
        }
    
    def detect_anomalies(self, narrative_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Detect anomalies in narrative patterns.
        
        Args:
            narrative_id: The ID of the narrative
            days: Number of days of historical data to analyze
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            logger.error(f"Narrative {narrative_id} not found")
            return {
                "success": False,
                "error": f"Narrative {narrative_id} not found"
            }
        
        # Get time series data
        df = self.get_narrative_time_series(narrative_id, days=days)
        if df.empty or df['count'].sum() == 0 or len(df) < 5:
            logger.warning(f"Insufficient data for anomaly detection on narrative {narrative_id}")
            return {
                "success": False,
                "error": "Insufficient data for anomaly detection"
            }
        
        try:
            # Use Isolation Forest for anomaly detection
            config = self.model_configs['isolation_forest']
            detector = IsolationForest(
                contamination=config['contamination'],
                n_estimators=config['n_estimators'],
                max_samples=config['max_samples'],
                random_state=config['random_state']
            )
            
            # Prepare data
            X = df['count'].values.reshape(-1, 1)
            
            # Fit and predict
            anomaly_scores = detector.fit_predict(X)
            
            # Convert predictions to actual labels (-1 for anomalies, 1 for normal)
            anomalies = (anomaly_scores == -1)
            
            # Get anomaly dates and values
            anomaly_dates = df.loc[anomalies, 'date'].dt.strftime('%Y-%m-%d').tolist()
            anomaly_values = df.loc[anomalies, 'count'].tolist()
            
            # Calculate decision function scores (negative values are more anomalous)
            decision_scores = detector.decision_function(X)
            anomaly_scores_output = (-decision_scores).tolist()  # Negate so higher = more anomalous
            
            # Update narrative metadata with anomaly info
            meta_data = narrative.get_meta_data() or {}
            meta_data['anomaly_detection'] = {
                'date': datetime.utcnow().isoformat(),
                'anomaly_count': sum(anomalies),
                'days_analyzed': days,
                'last_anomaly_date': anomaly_dates[-1] if anomaly_dates else None,
                'last_anomaly_value': anomaly_values[-1] if anomaly_values else None
            }
            narrative.set_meta_data(meta_data)
            db.session.commit()
            
            return {
                "success": True,
                "anomaly_dates": anomaly_dates,
                "anomaly_values": anomaly_values,
                "anomaly_scores": anomaly_scores_output,
                "total_anomalies": sum(anomalies),
                "days_analyzed": days
            }
            
        except Exception as e:
            logger.exception(f"Error detecting anomalies for narrative {narrative_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Error in anomaly detection: {str(e)}"
            }
    
    def predict_counter_effectiveness(
        self, 
        counter_id: int, 
        days_horizon: int = None
    ) -> Dict[str, Any]:
        """
        Predict the effectiveness of a counter-narrative over time.
        
        Args:
            counter_id: The ID of the counter message
            days_horizon: Number of days to forecast
            
        Returns:
            Dictionary with prediction results
        """
        # Set default horizon if not provided
        if days_horizon is None:
            days_horizon = self.default_horizon
        
        # Get the counter message
        counter = CounterMessage.query.get(counter_id)
        if not counter:
            logger.error(f"Counter message {counter_id} not found")
            return {
                "success": False,
                "error": f"Counter message {counter_id} not found"
            }
        
        # Get time series data
        df = self.get_counter_message_time_series(counter_id)
        if df.empty or df['effectiveness'].sum() == 0:
            logger.warning(f"Insufficient data for counter message {counter_id}")
            return {
                "success": False,
                "error": "Insufficient data for prediction"
            }
        
        try:
            # Use simple model for effectiveness prediction
            # In a real implementation, this could use more sophisticated models
            
            # Calculate current trend
            window = min(7, len(df))
            if window < 2:
                # Just use the last value if we only have one data point
                last_value = df['effectiveness'].iloc[-1]
                forecast_values = [last_value] * days_horizon
                
                # Generate forecast dates
                last_date = df['date'].iloc[-1]
                forecast_dates = [
                    (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                    for i in range(days_horizon)
                ]
                
                return {
                    "success": True,
                    "dates": forecast_dates,
                    "values": forecast_values,
                    "lower_bound": [max(0, v * 0.7) for v in forecast_values],
                    "upper_bound": [min(1.0, v * 1.3) for v in forecast_values],
                    "peak_day": forecast_dates[0],
                    "peak_value": last_value,
                    "trend_direction": "stable"
                }
            
            # Calculate exponentially weighted moving average
            ewma = df['effectiveness'].ewm(span=window).mean()
            last_ewma = ewma.iloc[-1]
            
            # Calculate trend as slope of recent values
            recent_df = df.iloc[-window:]
            X = np.arange(len(recent_df)).reshape(-1, 1)
            y = recent_df['effectiveness'].values
            
            # Simple linear regression
            if len(X) > 1:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
            else:
                slope = 0
            
            # Generate forecast
            forecast_values = []
            for i in range(1, days_horizon + 1):
                # Apply diminishing trend (effectiveness will plateau)
                trend_factor = 1.0 / (i * 0.2 + 1)
                value = last_ewma + slope * i * trend_factor
                # Ensure value is between 0 and 1
                value = max(0, min(1, value))
                forecast_values.append(value)
            
            # Generate forecast dates
            last_date = df['date'].iloc[-1]
            forecast_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                for i in range(days_horizon)
            ]
            
            # Calculate confidence intervals
            variance = recent_df['effectiveness'].var() if len(recent_df) > 1 else last_ewma * 0.1
            std_dev = np.sqrt(variance) if variance > 0 else last_ewma * 0.1
            z_score = 1.96  # 95% confidence
            
            lower_bound = [max(0, val - z_score * std_dev) for val in forecast_values]
            upper_bound = [min(1, val + z_score * std_dev) for val in forecast_values]
            
            # Determine peak
            peak_idx = forecast_values.index(max(forecast_values)) if forecast_values else 0
            peak_day = forecast_dates[peak_idx] if forecast_dates else None
            peak_value = forecast_values[peak_idx] if forecast_values else None
            
            # Determine overall trend direction
            if slope > 0.01:  # Arbitrary threshold
                trend_direction = "up"
            elif slope < -0.01:  # Arbitrary threshold
                trend_direction = "down"
            else:
                trend_direction = "stable"
            
            # Update counter message metadata with prediction info
            meta_data = counter.get_meta_data() or {}
            meta_data['effectiveness_prediction'] = {
                'date': datetime.utcnow().isoformat(),
                'horizon_days': days_horizon,
                'current_effectiveness': float(last_ewma),
                'predicted_peak': float(peak_value) if peak_value is not None else None,
                'trend_direction': trend_direction
            }
            counter.set_meta_data(meta_data)
            db.session.commit()
            
            return {
                "success": True,
                "dates": forecast_dates,
                "values": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "peak_day": peak_day,
                "peak_value": peak_value,
                "trend_direction": trend_direction,
                "current_effectiveness": float(last_ewma)
            }
            
        except Exception as e:
            logger.exception(f"Error predicting effectiveness for counter message {counter_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Error in effectiveness prediction: {str(e)}"
            }
    
    def get_predictions_for_narrative(self, narrative_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent predictions for a narrative.
        
        Args:
            narrative_id: The ID of the narrative
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction data dictionaries
        """
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            logger.error(f"Narrative {narrative_id} not found")
            return []
        
        # Query recent predictions
        predictions = NarrativePrediction.query.filter_by(
            narrative_id=narrative_id
        ).order_by(
            desc(NarrativePrediction.prediction_date)
        ).limit(limit).all()
        
        # Format prediction data
        result = []
        for prediction in predictions:
            try:
                forecast_data = json.loads(prediction.forecast_data)
                result.append({
                    "id": prediction.id,
                    "date": prediction.prediction_date.isoformat(),
                    "model_type": prediction.model_type,
                    "horizon_days": prediction.horizon_days,
                    "forecast": forecast_data
                })
            except:
                # Skip malformed data
                continue
        
        return result
    
    def get_all_active_models(self) -> List[Dict[str, Any]]:
        """
        Get all active prediction models in the system.
        
        Returns:
            List of model data dictionaries
        """
        # Query active models
        models = PredictionModel.query.filter_by(
            is_active=True
        ).order_by(
            PredictionModel.name
        ).all()
        
        # Format model data
        result = []
        for model in models:
            result.append({
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "type": model.model_type,
                "target": model.target_type,
                "parameters": model.get_parameters(),
                "created_at": model.created_at.isoformat(),
                "last_updated": model.last_updated.isoformat(),
                "accuracy": model.accuracy,
                "prediction_count": model.prediction_count
            })
        
        return result
    
    def register_model(
        self,
        name: str,
        model_type: str,
        target_type: str,
        description: str = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Register a new prediction model in the system.
        
        Args:
            name: Name of the model
            model_type: Type of model (e.g., 'arima', 'prophet')
            target_type: Type of prediction target (e.g., 'trajectory', 'effectiveness')
            description: Optional description of the model
            parameters: Optional model parameters as a dictionary
            
        Returns:
            Dictionary with registration results
        """
        try:
            # Check if model with same name already exists
            existing = PredictionModel.query.filter_by(name=name).first()
            if existing:
                logger.warning(f"Model with name '{name}' already exists")
                return {
                    "success": False,
                    "error": f"Model with name '{name}' already exists"
                }
            
            # Create new model
            model = PredictionModel(
                name=name,
                model_type=model_type,
                target_type=target_type,
                description=description or f"{model_type.capitalize()} model for {target_type}",
                parameters=json.dumps(parameters) if parameters else None,
                is_active=True,
                accuracy=0.0,
                prediction_count=0
            )
            
            db.session.add(model)
            db.session.commit()
            
            return {
                "success": True,
                "model_id": model.id,
                "name": model.name,
                "message": f"Model '{name}' registered successfully"
            }
            
        except Exception as e:
            logger.exception(f"Error registering model '{name}': {str(e)}")
            return {
                "success": False,
                "error": f"Error registering model: {str(e)}"
            }
    
    @run_in_thread
    def run_batch_predictions(
        self, 
        model_id: int = None, 
        narrative_limit: int = 20,
        days_horizon: int = None
    ) -> Dict[str, Any]:
        """
        Run batch predictions for multiple narratives.
        
        Args:
            model_id: Optional ID of the model to use
            narrative_limit: Maximum number of narratives to process
            days_horizon: Number of days to forecast
            
        Returns:
            Dictionary with batch prediction results
        """
        # Set default horizon if not provided
        if days_horizon is None:
            days_horizon = self.default_horizon
        
        # Get model if specified
        model_type = 'arima'  # Default
        if model_id:
            model = PredictionModel.query.get(model_id)
            if model and model.is_active:
                model_type = model.model_type
            else:
                logger.warning(f"Model {model_id} not found or not active, using default")
        
        # Get active narratives with sufficient instances
        narratives = db.session.query(
            DetectedNarrative,
            func.count(NarrativeInstance.id).label('instance_count')
        ).join(
            NarrativeInstance,
            DetectedNarrative.id == NarrativeInstance.narrative_id
        ).filter(
            DetectedNarrative.status.in_(['active', 'confirmed'])
        ).group_by(
            DetectedNarrative.id
        ).having(
            func.count(NarrativeInstance.id) >= 5  # Minimum instances for prediction
        ).order_by(
            desc('instance_count')
        ).limit(narrative_limit).all()
        
        # Run predictions
        successful = 0
        failed = 0
        results = []
        
        for narrative, _ in narratives:
            try:
                result = self.forecast_narrative_trajectory(
                    narrative_id=narrative.id,
                    days_horizon=days_horizon,
                    model_type=model_type
                )
                
                if result.get('success', False):
                    successful += 1
                    results.append({
                        "narrative_id": narrative.id,
                        "title": narrative.title,
                        "prediction_id": result.get('prediction_id'),
                        "trend_direction": result.get('trend_direction')
                    })
                else:
                    failed += 1
                    logger.warning(f"Prediction failed for narrative {narrative.id}: {result.get('error')}")
            
            except Exception as e:
                failed += 1
                logger.exception(f"Error in batch prediction for narrative {narrative.id}: {str(e)}")
        
        # Update model stats if using a registered model
        if model_id:
            try:
                model = PredictionModel.query.get(model_id)
                if model:
                    model.prediction_count += successful
                    db.session.commit()
            except:
                pass
        
        return {
            "success": True,
            "total_processed": successful + failed,
            "successful": successful,
            "failed": failed,
            "model_type": model_type,
            "results": results
        }
            
    def analyze_narrative_clusters(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze narrative clusters for prediction patterns.
        
        Args:
            days: Number of days of historical data to analyze
            
        Returns:
            Dictionary with cluster analysis results
        """
        try:
            # Get narratives with cluster assignments
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.status.in_(['active', 'confirmed']),
                DetectedNarrative.last_updated >= datetime.utcnow() - timedelta(days=days)
            ).all()
            
            # Group by clusters
            stream_clusters = {}
            temporal_clusters = {}
            sequence_clusters = {}
            
            for narrative in narratives:
                meta_data = narrative.get_meta_data() or {}
                
                # Get cluster assignments
                stream_cluster = meta_data.get('stream_cluster')
                temporal_cluster = meta_data.get('temporal_cluster')
                sequence_cluster = meta_data.get('sequence_cluster')
                
                # Skip if no cluster assignments
                if stream_cluster is None and temporal_cluster is None and sequence_cluster is None:
                    continue
                
                # Add to stream clusters
                if stream_cluster is not None:
                    if stream_cluster not in stream_clusters:
                        stream_clusters[stream_cluster] = []
                    stream_clusters[stream_cluster].append(narrative.id)
                
                # Add to temporal clusters
                if temporal_cluster is not None:
                    if temporal_cluster not in temporal_clusters:
                        temporal_clusters[temporal_cluster] = []
                    temporal_clusters[temporal_cluster].append(narrative.id)
                
                # Add to sequence clusters
                if sequence_cluster is not None:
                    if sequence_cluster not in sequence_clusters:
                        sequence_clusters[sequence_cluster] = []
                    sequence_clusters[sequence_cluster].append(narrative.id)
            
            # Analyze growth patterns per cluster
            cluster_growth = {}
            
            # Focus on sequence clusters for growth patterns
            for cluster_id, narrative_ids in sequence_clusters.items():
                # Skip if too few narratives
                if len(narrative_ids) < 3:
                    continue
                
                # Get recent growth rates
                growth_rates = []
                for narrative_id in narrative_ids:
                    # Get time series data
                    df = self.get_narrative_time_series(narrative_id, days=14)  # Last 2 weeks
                    if df.empty or df['count'].sum() == 0 or len(df) < 3:
                        continue
                    
                    # Calculate growth rate
                    first_week = df.iloc[:7]['count'].sum()
                    second_week = df.iloc[7:]['count'].sum()
                    if first_week > 0:
                        growth_rate = (second_week - first_week) / first_week
                    else:
                        growth_rate = 0 if second_week == 0 else 1
                    
                    growth_rates.append(growth_rate)
                
                # Calculate average growth rate
                if growth_rates:
                    avg_growth = sum(growth_rates) / len(growth_rates)
                    cluster_growth[cluster_id] = {
                        "average_growth_rate": avg_growth,
                        "narrative_count": len(narrative_ids),
                        "growth_trend": "up" if avg_growth > 0.1 else "down" if avg_growth < -0.1 else "stable"
                    }
            
            # Get most active clusters
            most_active = sorted(
                [(k, len(v)) for k, v in sequence_clusters.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Get fastest growing clusters
            fastest_growing = sorted(
                [(k, v["average_growth_rate"]) for k, v in cluster_growth.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "success": True,
                "stream_clusters": {k: len(v) for k, v in stream_clusters.items()},
                "temporal_clusters": {k: len(v) for k, v in temporal_clusters.items()},
                "sequence_clusters": {k: len(v) for k, v in sequence_clusters.items()},
                "most_active_clusters": most_active,
                "fastest_growing_clusters": fastest_growing,
                "cluster_growth_patterns": cluster_growth
            }
            
        except Exception as e:
            logger.exception(f"Error analyzing narrative clusters: {str(e)}")
            return {
                "success": False,
                "error": f"Error in cluster analysis: {str(e)}"
            }
    
    def get_trending_narratives(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending narratives based on recent growth.
        
        Args:
            days: Number of days to consider for trending analysis
            limit: Maximum number of narratives to return
            
        Returns:
            List of trending narrative data
        """
        try:
            # Get active narratives
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.status.in_(['active', 'confirmed'])
            ).all()
            
            # Calculate growth for each narrative
            narrative_growth = []
            
            for narrative in narratives:
                # Get time series data
                df = self.get_narrative_time_series(narrative.id, days=days*2)  # Double the days for comparison
                if df.empty or df['count'].sum() == 0 or len(df) < days:
                    continue
                
                # Calculate growth rate
                current_period = df.iloc[-days:]['count'].sum()
                previous_period = df.iloc[-days*2:-days]['count'].sum() if len(df) >= days*2 else 0
                
                if previous_period > 0:
                    growth_rate = (current_period - previous_period) / previous_period
                else:
                    growth_rate = 1 if current_period > 0 else 0
                
                # Add to results if there's any activity
                if current_period > 0:
                    narrative_growth.append({
                        "id": narrative.id,
                        "title": narrative.title,
                        "current_activity": int(current_period),
                        "previous_activity": int(previous_period),
                        "growth_rate": growth_rate,
                        "last_updated": narrative.last_updated.isoformat(),
                        "confidence_score": narrative.confidence_score
                    })
            
            # Sort by growth rate
            trending = sorted(narrative_growth, key=lambda x: x["growth_rate"], reverse=True)[:limit]
            
            return trending
            
        except Exception as e:
            logger.exception(f"Error getting trending narratives: {str(e)}")
            return []
            
    def get_top_risk_sources(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top risk information sources based on reliability metrics.
        
        Args:
            limit: Maximum number of sources to return
            
        Returns:
            List of dictionaries containing source information
        """
        try:
            # Query sources with lower reliability scores
            sources = self.InformationSource.query.filter_by(
                status='active'
            ).order_by(
                self.InformationSource.reliability_score.asc()
            ).limit(limit).all()
            
            result = []
            for source in sources:
                # Get recent misinformation instances
                recent_instances = db.session.query(
                    func.count(NarrativeInstance.id).label('count')
                ).filter(
                    NarrativeInstance.source_id == source.id,
                    NarrativeInstance.is_misinformation == True,
                    NarrativeInstance.detected_at >= datetime.utcnow() - timedelta(days=30)
                ).scalar()
                
                # Generate risk level based on reliability score
                if source.reliability_score < 0.3:
                    risk_level = "High"
                elif source.reliability_score < 0.7:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                # Determine trend based on historical reliability (mock for now)
                trend = "stable"  # Default if we can't determine
                meta_data = {}
                if hasattr(source, 'get_meta_data'):
                    meta_data = source.get_meta_data() or {}
                
                if 'historical_reliability' in meta_data:
                    history = meta_data['historical_reliability']
                    if len(history) >= 2:
                        current = history[-1]
                        previous = history[-2]
                        if current > previous * 1.05:
                            trend = "up"
                        elif current < previous * 0.95:
                            trend = "down"
                
                result.append({
                    'id': source.id,
                    'name': source.name,
                    'url': source.url,
                    'source_type': source.source_type,
                    'reliability_score': source.reliability_score,
                    'recent_misinfo_count': recent_instances,
                    'risk_trend': trend,
                    'projected_risk': risk_level
                })
            
            return result
            
        except Exception as e:
            logger.exception(f"Error getting top risk sources: {str(e)}")
            return []
    
    def predict_source_reliability(
        self,
        source_id: int,
        days_history: int = 90,
        days_horizon: int = 30,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Predict future reliability of an information source.
        
        Args:
            source_id: ID of the information source
            days_history: Number of days of historical data to use
            days_horizon: Number of days to forecast
            force_refresh: Force refresh of prediction even if recent one exists
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Get the source
            source = self.InformationSource.query.get(source_id)
            if not source:
                logger.error(f"Information source {source_id} not found")
                return {
                    "success": False,
                    "error": f"Information source {source_id} not found"
                }
            
            # Check if we have a recent forecast in cache and not forcing refresh
            meta_data = {}
            if hasattr(source, 'get_meta_data'):
                meta_data = source.get_meta_data() or {}
            
            if not force_refresh and 'reliability_forecast' in meta_data:
                forecast = meta_data['reliability_forecast']
                generated_at = datetime.fromisoformat(forecast['generated_at']) if 'generated_at' in forecast else None
                
                # If forecast is less than 24 hours old, return cached version
                if generated_at and (datetime.utcnow() - generated_at).total_seconds() < 86400:
                    forecast['success'] = True
                    forecast['from_cache'] = True
                    return forecast
            
            # Get historical reliability data
            if 'historical_reliability' not in meta_data or not meta_data['historical_reliability']:
                # If no historical data, return error
                logger.warning(f"Insufficient historical data for source {source_id}")
                return {
                    "success": False,
                    "error": "Insufficient historical data for forecasting"
                }
            
            # Extract historical data (assume format is list of [timestamp, score] pairs)
            history = meta_data['historical_reliability']
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(history, columns=['timestamp', 'reliability'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Check if we have enough data
            if len(df) < 5:  # Need at least 5 data points
                logger.warning(f"Insufficient historical data points for source {source_id}")
                return {
                    "success": False,
                    "error": "Insufficient historical data points (need at least 5)"
                }
            
            # Forecast using ARIMA model
            try:
                # Create time series model
                model = ARIMA(df['reliability'], order=(2, 1, 2))
                model_fit = model.fit()
                
                # Forecast
                forecast_values = model_fit.forecast(steps=days_horizon)
                forecast_index = pd.date_range(
                    start=df.index[-1] + pd.Timedelta(days=1), 
                    periods=days_horizon
                )
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': forecast_index.strftime('%Y-%m-%d'),
                    'reliability': forecast_values.tolist()
                })
                
                # Ensure reliability values are between 0 and 1
                forecast_df['reliability'] = forecast_df['reliability'].apply(
                    lambda x: max(0, min(1, x))
                )
                
                # Get confidence intervals
                ci = model_fit.get_forecast(steps=days_horizon).conf_int()
                lower_bound = ci.iloc[:, 0].apply(lambda x: max(0, min(1, x))).tolist()
                upper_bound = ci.iloc[:, 1].apply(lambda x: max(0, min(1, x))).tolist()
                
                # Calculate trend
                current_value = df['reliability'].iloc[-1]
                future_value = forecast_df['reliability'].iloc[-1]
                
                if future_value > current_value * 1.1:
                    trend = "improving"
                elif future_value < current_value * 0.9:
                    trend = "deteriorating"
                else:
                    trend = "stable"
                
                # Determine risk level
                risk_level = "Low"
                if future_value < 0.3:
                    risk_level = "High"
                elif future_value < 0.7:
                    risk_level = "Medium"
                
                # Create result
                result = {
                    "success": True,
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_id": source_id,
                    "source_name": source.name,
                    "current_reliability": float(current_value),
                    "forecast": {
                        "dates": forecast_df['date'].tolist(),
                        "values": forecast_df['reliability'].tolist(),
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound
                    },
                    "trend": trend,
                    "risk_level": risk_level,
                    "days_history": days_history,
                    "days_horizon": days_horizon,
                    "model": "arima"
                }
                
                # Cache the result in source metadata
                meta_data['reliability_forecast'] = result
                if hasattr(source, 'set_meta_data'):
                    source.set_meta_data(meta_data)
                    db.session.commit()
                
                return result
                
            except Exception as e:
                logger.exception(f"Error forecasting reliability for source {source_id}: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error in forecasting: {str(e)}"
                }
            
        except Exception as e:
            logger.exception(f"Error in source reliability prediction: {str(e)}")
            return {
                "success": False,
                "error": f"Error in prediction: {str(e)}"
            }