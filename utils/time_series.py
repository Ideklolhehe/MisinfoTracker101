"""
Time Series Analysis Utilities for CIVILIAN.

This module provides tools for analyzing time series data of narratives,
including trend detection, seasonality analysis, and anomaly detection.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """Utility class for time series analysis of narrative data."""
    
    def __init__(self):
        """Initialize the time series analyzer."""
        self.smoothing_window = 3  # Default window for moving averages
        self.trend_threshold = 0.1  # 10% change threshold for trend detection
        
    def smooth_series(self, series: pd.Series, window: int = None) -> pd.Series:
        """
        Apply smoothing to a time series using moving average.
        
        Args:
            series: Time series data
            window: Window size for moving average (default: self.smoothing_window)
            
        Returns:
            Smoothed series
        """
        if window is None:
            window = self.smoothing_window
            
        if len(series) < window:
            return series
            
        return series.rolling(window=window, center=True).mean().fillna(series)
    
    def detect_trend(self, series: pd.Series) -> str:
        """
        Detect trend direction in a time series.
        
        Args:
            series: Time series data
            
        Returns:
            Trend direction: "up", "down", or "stable"
        """
        if len(series) < 2:
            return "stable"
            
        # Use linear regression to determine trend
        x = np.arange(len(series))
        y = series.values
        
        # Handle cases with all zeros
        if np.all(y == 0):
            return "stable"
            
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope by mean value to get relative change
        mean_value = np.mean(y)
        if mean_value == 0:
            return "stable"
            
        relative_change = slope * len(series) / mean_value
        
        if relative_change > self.trend_threshold:
            return "up"
        elif relative_change < -self.trend_threshold:
            return "down"
        else:
            return "stable"
    
    def detect_anomalies(self, series: pd.Series, threshold: float = 2.0) -> List[int]:
        """
        Detect anomalies in time series using simple z-score method.
        
        Args:
            series: Time series data
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of indices where anomalies occur
        """
        if len(series) < 4:  # Need enough data points
            return []
            
        # Calculate rolling mean and std
        rolling_mean = series.rolling(window=3, center=True).mean()
        rolling_std = series.rolling(window=3, center=True).std()
        
        # Handle beginning and end where rolling values are NaN
        rolling_mean.fillna(series.mean(), inplace=True)
        rolling_std.fillna(series.std(), inplace=True)
        
        # Handle case where std is 0
        rolling_std = rolling_std.replace(0, series.std() if series.std() > 0 else 1)
        
        # Calculate z-scores
        z_scores = abs((series - rolling_mean) / rolling_std)
        
        # Find anomalies
        return list(np.where(z_scores > threshold)[0])
    
    def forecast_naive(self, series: pd.Series, horizon: int = 7) -> Tuple[List[float], List[float], List[float]]:
        """
        Simple naive forecasting using trend-based projection.
        
        Args:
            series: Time series data
            horizon: Number of periods to forecast
            
        Returns:
            Tuple of (forecast values, lower bounds, upper bounds)
        """
        if len(series) < 2:
            # If only one point, just repeat it
            value = series.iloc[0] if len(series) == 1 else 0
            return [value] * horizon, [value * 0.7] * horizon, [value * 1.3] * horizon
            
        # Use simple trend projection
        x = np.arange(len(series))
        y = series.values
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Project forward
        future_x = np.arange(len(series), len(series) + horizon)
        forecast = intercept + slope * future_x
        
        # Force non-negative values
        forecast = np.maximum(forecast, 0)
        
        # Calculate prediction intervals (simple approach)
        std_dev = series.std() if len(series) > 1 else series.mean() * 0.1
        lower_bound = np.maximum(forecast - 1.96 * std_dev, 0)
        upper_bound = forecast + 1.96 * std_dev
        
        return forecast.tolist(), lower_bound.tolist(), upper_bound.tolist()
    
    def calculate_growth_rate(self, series: pd.Series, periods: int = 7) -> float:
        """
        Calculate growth rate over the specified periods.
        
        Args:
            series: Time series data
            periods: Number of periods to consider
            
        Returns:
            Growth rate as a decimal (e.g., 0.05 for 5% growth)
        """
        if len(series) < 2:
            return 0.0
            
        # Use data from the most recent periods
        recent_data = series.iloc[-min(periods, len(series)):]
        
        if recent_data.iloc[0] == 0:
            # Avoid division by zero
            return 0.0 if recent_data.iloc[-1] == 0 else 1.0
            
        # Calculate relative change
        return (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
    
    def compare_series(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """
        Compare two time series and return similarity metrics.
        
        Args:
            series1: First time series
            series2: Second time series
            
        Returns:
            Dictionary with comparison metrics
        """
        # Ensure series are aligned and have the same length
        min_len = min(len(series1), len(series2))
        if min_len < 2:
            return {
                "correlation": 0,
                "similar_trend": False,
                "mean_difference": 0,
                "percentage_difference": 0
            }
            
        s1 = series1.iloc[-min_len:].reset_index(drop=True)
        s2 = series2.iloc[-min_len:].reset_index(drop=True)
        
        # Calculate correlation
        correlation = s1.corr(s2)
        
        # Compare trends
        trend1 = self.detect_trend(s1)
        trend2 = self.detect_trend(s2)
        similar_trend = trend1 == trend2
        
        # Calculate differences
        mean_difference = (s2 - s1).mean()
        
        # Percentage difference relative to first series
        s1_mean = s1.mean()
        if s1_mean == 0:
            percentage_difference = 0 if s2.mean() == 0 else 1
        else:
            percentage_difference = mean_difference / s1_mean
            
        return {
            "correlation": correlation,
            "similar_trend": similar_trend,
            "mean_difference": mean_difference,
            "percentage_difference": percentage_difference
        }