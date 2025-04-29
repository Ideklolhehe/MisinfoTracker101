"""
Time series analysis utilities for the CIVILIAN predictive modeling system.

This module provides functions and classes for time series analysis, including trend detection,
seasonality detection, smoothing, and decomposition.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import signal
from scipy import stats


class TimeSeriesAnalyzer:
    """
    Class for time series analysis and forecasting.
    This class wraps the individual time series functions in this module.
    """
    
    def __init__(self):
        """Initialize the time series analyzer."""
        pass
    
    def smooth_time_series(self, values: List[float], window_size: int = 3, method: str = 'moving_average') -> List[float]:
        """
        Smooth a time series using different methods.
        
        Args:
            values: List of values to smooth
            window_size: Size of the smoothing window
            method: Smoothing method ('moving_average', 'exponential', 'gaussian')
            
        Returns:
            List of smoothed values
        """
        return smooth_time_series(values, window_size, method)
    
    def detect_trends(self, values: List[float], dates: List[datetime], 
                      window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect trends in a time series.
        
        Args:
            values: List of values to analyze
            dates: List of dates corresponding to the values
            window_size: Optional window size for piecewise trend detection
            
        Returns:
            Dictionary with trend information
        """
        return detect_trends(values, dates, window_size)
    
    def detect_seasonality(self, values: List[float], dates: List[datetime], 
                           max_period: int = 30) -> Dict[str, Any]:
        """
        Detect seasonality patterns in a time series.
        
        Args:
            values: List of values to analyze
            dates: List of dates corresponding to the values
            max_period: Maximum period length to consider
            
        Returns:
            Dictionary with seasonality information
        """
        return detect_seasonality(values, dates, max_period)
    
    def decompose_time_series(self, values: List[float], period: int = None, 
                              dates: List[datetime] = None) -> Dict[str, List[float]]:
        """
        Decompose a time series into trend, seasonal, and residual components.
        
        Args:
            values: List of values to decompose
            period: Period length for seasonal decomposition (if None, will attempt to detect)
            dates: Optional list of dates for the time series
            
        Returns:
            Dictionary with decomposed components
        """
        return decompose_time_series(values, period, dates)
    
    def forecast_time_series(self, values: List[float], steps: int = 7, 
                             method: str = 'arima', 
                             period: int = None) -> Dict[str, Any]:
        """
        Forecast future values for a time series.
        
        Args:
            values: List of values to forecast from
            steps: Number of steps to forecast
            method: Forecasting method ('arima', 'exponential_smoothing', 'linear')
            period: Seasonality period (for seasonal models)
            
        Returns:
            Dictionary with forecast results, including predicted values and confidence intervals
        """
        return forecast_time_series(values, steps, method, period)
    
    def detect_anomalies(self, values: List[float], dates: List[datetime] = None, 
                         threshold: float = 2.0, 
                         window_size: int = None) -> Dict[str, Any]:
        """
        Detect anomalies in a time series.
        
        Args:
            values: List of values to analyze
            dates: Optional list of dates corresponding to the values
            threshold: Z-score threshold for anomaly detection
            window_size: Size of the rolling window for local anomaly detection 
                        (if None, global anomalies are detected)
            
        Returns:
            Dictionary with anomaly information
        """
        return detect_anomalies(values, dates, threshold, window_size)
    
    def find_change_points(self, values: List[float], dates: List[datetime] = None,
                          min_size: int = 5, 
                          threshold: float = 2.0) -> Dict[str, Any]:
        """
        Find points where the time series shows significant changes in level or trend.
        
        Args:
            values: List of values to analyze
            dates: Optional list of dates corresponding to the values
            min_size: Minimum segment size for change point detection
            threshold: Threshold for change significance
            
        Returns:
            Dictionary with change point information
        """
        return find_change_points(values, dates, min_size, threshold)


def smooth_time_series(values: List[float], window_size: int = 3, method: str = 'moving_average') -> List[float]:
    """
    Smooth a time series using different methods.

    Args:
        values: List of values to smooth
        window_size: Size of the smoothing window
        method: Smoothing method ('moving_average', 'exponential', 'gaussian')

    Returns:
        List of smoothed values
    """
    if len(values) < window_size:
        return values

    values_array = np.array(values)
    
    if method == 'moving_average':
        # Simple moving average
        weights = np.ones(window_size) / window_size
        smoothed = np.convolve(values_array, weights, mode='same')
        
        # Fix the edges (first and last window_size/2 values)
        half_window = window_size // 2
        for i in range(half_window):
            # Start of array
            window = values_array[:i + half_window + 1]
            smoothed[i] = np.mean(window)
            
            # End of array
            window = values_array[-(i + half_window + 1):]
            smoothed[-(i + 1)] = np.mean(window)
        
    elif method == 'exponential':
        # Exponential smoothing
        alpha = 2 / (window_size + 1)  # Smoothing factor
        smoothed = np.zeros_like(values_array)
        smoothed[0] = values_array[0]
        
        for i in range(1, len(values_array)):
            smoothed[i] = alpha * values_array[i] + (1 - alpha) * smoothed[i-1]
            
    elif method == 'gaussian':
        # Gaussian smoothing
        x = np.arange(-window_size // 2, window_size // 2 + 1)
        kernel = np.exp(-(x**2) / (2 * (window_size / 4)**2))
        kernel = kernel / np.sum(kernel)
        smoothed = np.convolve(values_array, kernel, mode='same')
        
    else:
        # Default to original values if method not recognized
        smoothed = values_array
    
    return smoothed.tolist()


def detect_trends(values: List[float], dates: List[datetime], 
                  window_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Detect trends in a time series.

    Args:
        values: List of values to analyze
        dates: List of dates corresponding to the values
        window_size: Optional window size for piecewise trend detection

    Returns:
        Dictionary with trend information
    """
    if len(values) < 3:
        return {
            'trend': 'no_data',
            'slope': 0.0,
            'p_value': 1.0,
            'strength': 0.0,
            'direction': 'stable',
            'segments': []
        }
    
    # Convert to numpy arrays for easier manipulation
    y = np.array(values)
    x = np.arange(len(y))
    
    # Perform linear regression to detect overall trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Determine trend direction and strength
    trend_strength = abs(r_value)
    
    if p_value < 0.05:  # Statistically significant trend
        if slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
    else:
        trend_direction = 'stable'
    
    result = {
        'trend': 'linear' if p_value < 0.05 else 'no_trend',
        'slope': slope,
        'p_value': p_value,
        'strength': trend_strength,
        'direction': trend_direction,
        'segments': []
    }
    
    # Piecewise trend detection for detecting changing trends
    if window_size and len(values) >= window_size * 2:
        segments = []
        
        # Analyze trends in sliding windows
        for i in range(0, len(values) - window_size + 1, window_size // 2):
            end_idx = min(i + window_size, len(values))
            segment_y = y[i:end_idx]
            segment_x = np.arange(len(segment_y))
            
            # Skip segments with too few points
            if len(segment_y) < 3:
                continue
                
            # Calculate trend for this segment
            seg_slope, seg_intercept, seg_r, seg_p, seg_std_err = stats.linregress(segment_x, segment_y)
            
            # Determine segment trend
            if seg_p < 0.05:
                if seg_slope > 0:
                    seg_trend = 'increasing'
                else:
                    seg_trend = 'decreasing'
            else:
                seg_trend = 'stable'
                
            segments.append({
                'start_index': i,
                'end_index': end_idx - 1,
                'start_date': dates[i].strftime('%Y-%m-%d'),
                'end_date': dates[end_idx - 1].strftime('%Y-%m-%d'),
                'trend': seg_trend,
                'slope': seg_slope,
                'p_value': seg_p,
                'strength': abs(seg_r)
            })
            
        result['segments'] = segments
        
        # Check if there are changing trend directions
        if len(segments) >= 2:
            changing_direction = False
            for i in range(1, len(segments)):
                prev_trend = segments[i-1]['trend']
                curr_trend = segments[i]['trend']
                
                if prev_trend != curr_trend and prev_trend != 'stable' and curr_trend != 'stable':
                    changing_direction = True
                    break
                    
            if changing_direction:
                result['trend'] = 'changing'
    
    return result


def detect_seasonality(values: List[float], dates: List[datetime], 
                       max_period: int = 30) -> Dict[str, Any]:
    """
    Detect seasonality patterns in a time series.

    Args:
        values: List of values to analyze
        dates: List of dates corresponding to the values
        max_period: Maximum period length to consider

    Returns:
        Dictionary with seasonality information
    """
    if len(values) < max_period * 2:
        return {
            'seasonal': False,
            'period': None,
            'strength': 0.0,
            'p_value': 1.0
        }
    
    # Convert to numpy array
    y = np.array(values)
    
    # Calculate date differences to check for regular sampling
    date_diffs = [(dates[i+1] - dates[i]).total_seconds() for i in range(len(dates)-1)]
    regular_sampling = len(set(date_diffs)) <= 1
    
    # Detect potential periods using autocorrelation
    n = len(y)
    max_lag = min(max_period, n // 3)
    
    # Remove trend using differencing to focus on seasonal patterns
    y_diff = np.diff(y)
    
    # Calculate autocorrelation
    autocorr = np.correlate(y_diff, y_diff, mode='full')[n-1:n+max_lag-1]
    autocorr /= np.max(autocorr)
    
    # Find peaks in autocorrelation
    peaks = signal.find_peaks(autocorr, height=0.3, distance=2)[0]
    
    if len(peaks) > 0:
        # Get the most prominent peak
        peak_heights = autocorr[peaks]
        strongest_peak = peaks[np.argmax(peak_heights)]
        
        # Calculate p-value using significance test for autocorrelation
        # (simplified approximation)
        se = 1.0 / np.sqrt(n)
        z_score = autocorr[strongest_peak] / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        result = {
            'seasonal': p_value < 0.05,
            'period': int(strongest_peak + 1),
            'strength': float(autocorr[strongest_peak]),
            'p_value': float(p_value),
            'peaks': [int(p + 1) for p in peaks.tolist()],
            'peak_strengths': [float(autocorr[p]) for p in peaks]
        }
    else:
        result = {
            'seasonal': False,
            'period': None,
            'strength': 0.0,
            'p_value': 1.0,
            'peaks': [],
            'peak_strengths': []
        }
    
    # Check specifically for common periods if we have regular sampling
    if regular_sampling:
        # Potential periods to check (daily, weekly, biweekly, monthly)
        common_periods = {'daily': 1, 'weekly': 7, 'biweekly': 14, 'monthly': 30}
        
        result['common_periods'] = {}
        
        for period_name, period in common_periods.items():
            if period < n // 3:
                strength = autocorr[period - 1] if period - 1 < len(autocorr) else 0
                se = 1.0 / np.sqrt(n)
                z_score = strength / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                result['common_periods'][period_name] = {
                    'period': period,
                    'strength': float(strength),
                    'significant': p_value < 0.05,
                    'p_value': float(p_value)
                }
    
    return result


def decompose_time_series(values: List[float], period: int = None, 
                          dates: List[datetime] = None) -> Dict[str, List[float]]:
    """
    Decompose a time series into trend, seasonal, and residual components.

    Args:
        values: List of values to decompose
        period: Period length for seasonal decomposition (if None, will attempt to detect)
        dates: Optional list of dates for the time series

    Returns:
        Dictionary with decomposed components
    """
    if len(values) < 4:
        return {
            'trend': values,
            'seasonal': [0] * len(values),
            'residual': [0] * len(values)
        }
    
    # Convert to pandas Series for statsmodels
    if dates:
        ts = pd.Series(values, index=pd.DatetimeIndex(dates))
    else:
        ts = pd.Series(values)
    
    # If period not specified, try to detect it
    if period is None and len(values) >= 8:
        if dates:
            seasonality = detect_seasonality(values, dates)
        else:
            # Use dummy dates if not provided
            dummy_dates = [datetime.now() + timedelta(days=i) for i in range(len(values))]
            seasonality = detect_seasonality(values, dummy_dates)
        
        if seasonality['seasonal'] and seasonality['period']:
            period = seasonality['period']
        else:
            # Default to period of 7 (weekly) if no seasonality detected
            period = 7
    elif period is None:
        # Fallback if series too short
        period = 2
    
    # Ensure period is reasonable given the data length
    period = min(period, len(values) // 2)
    period = max(period, 2)  # Must be at least 2
    
    try:
        # Decompose time series
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        
        # Convert components to lists, handling NaN values
        trend = decomposition.trend.fillna(method='bfill').fillna(method='ffill').tolist()
        seasonal = decomposition.seasonal.fillna(0).tolist()
        residual = decomposition.resid.fillna(0).tolist()
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    except Exception as e:
        # Fallback if decomposition fails
        return {
            'trend': smooth_time_series(values, window_size=max(3, len(values)//5)),
            'seasonal': [0] * len(values),
            'residual': [0] * len(values),
            'error': str(e)
        }


def forecast_time_series(values: List[float], steps: int = 7, 
                         method: str = 'arima', 
                         period: int = None) -> Dict[str, Any]:
    """
    Forecast future values for a time series.

    Args:
        values: List of values to forecast from
        steps: Number of steps to forecast
        method: Forecasting method ('arima', 'exponential_smoothing', 'linear')
        period: Seasonality period (for seasonal models)

    Returns:
        Dictionary with forecast results, including predicted values and confidence intervals
    """
    if len(values) < 3:
        # Not enough data for forecasting
        return {
            'forecast': [values[-1] if values else 0] * steps,
            'lower_bound': [values[-1] if values else 0] * steps,
            'upper_bound': [values[-1] if values else 0] * steps,
            'confidence': 0.0,
            'method': 'constant'
        }
    
    # Convert to numpy array
    y = np.array(values)
    
    # Auto-detect seasonality if period not specified
    if period is None and len(values) >= 8:
        # Use dummy dates
        dummy_dates = [datetime.now() + timedelta(days=i) for i in range(len(values))]
        seasonality = detect_seasonality(values, dummy_dates)
        
        if seasonality['seasonal'] and seasonality['period']:
            period = seasonality['period']
    
    try:
        if method == 'arima':
            # Determine appropriate order for ARIMA model
            if period and len(values) >= period * 2:
                # Seasonal ARIMA
                p, d, q = 1, 1, 1
                P, D, Q, s = 1, 1, 1, period
                
                # Use statsmodels ARIMA
                model = ARIMA(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
            else:
                # Non-seasonal ARIMA
                p, d, q = 2, 1, 2
                
                # Use statsmodels ARIMA
                model = ARIMA(y, order=(p, d, q))
                
            # Fit model
            fitted_model = model.fit()
            
            # Make forecast
            forecast_result = fitted_model.forecast(steps)
            forecast_values = forecast_result.tolist()
            
            # Get confidence intervals (prediction intervals)
            conf_int = fitted_model.get_forecast(steps).conf_int(alpha=0.05)
            lower_bound = conf_int.iloc[:, 0].tolist()
            upper_bound = conf_int.iloc[:, 1].tolist()
            
            # Calculate confidence score based on interval width
            interval_width = np.mean(np.array(upper_bound) - np.array(lower_bound))
            max_value = max(abs(np.max(y)), abs(np.min(y)))
            confidence = 1.0 - min(1.0, interval_width / (2 * max_value)) if max_value > 0 else 0.5
            
        elif method == 'exponential_smoothing':
            # Configure Holt-Winters Exponential Smoothing
            if period and len(values) >= period * 2:
                # With seasonality
                model = ExponentialSmoothing(
                    y, 
                    trend='add', 
                    seasonal='add', 
                    seasonal_periods=period
                )
            else:
                # Without seasonality
                model = ExponentialSmoothing(
                    y, 
                    trend='add', 
                    seasonal=None
                )
                
            # Fit model
            fitted_model = model.fit()
            
            # Get forecast
            forecast_result = fitted_model.forecast(steps)
            forecast_values = forecast_result.tolist()
            
            # Approximate confidence intervals
            residuals = fitted_model.resid
            residual_std = np.std(residuals)
            z_value = 1.96  # 95% confidence
            
            lower_bound = [max(0, f - z_value * residual_std) for f in forecast_values]
            upper_bound = [f + z_value * residual_std for f in forecast_values]
            
            # Approximate confidence
            interval_width = np.mean(np.array(upper_bound) - np.array(lower_bound))
            max_value = max(abs(np.max(y)), abs(np.min(y)))
            confidence = 1.0 - min(1.0, interval_width / (2 * max_value)) if max_value > 0 else 0.5
            
        elif method == 'linear':
            # Simple linear regression-based forecast
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Forecast future values
            future_x = np.arange(len(y), len(y) + steps)
            forecast_values = [slope * i + intercept for i in future_x]
            
            # Approximate prediction intervals
            y_pred = slope * x + intercept
            residuals = y - y_pred
            residual_std = np.std(residuals)
            
            # 95% prediction interval
            t_value = stats.t.ppf(0.975, len(y) - 2)
            
            lower_bound = []
            upper_bound = []
            
            for i, fx in enumerate(future_x):
                # Formula for prediction interval
                se_pred = residual_std * np.sqrt(1 + 1/len(y) + 
                                                 (fx - np.mean(x))**2 / 
                                                 np.sum((x - np.mean(x))**2))
                margin = t_value * se_pred
                
                lower_bound.append(forecast_values[i] - margin)
                upper_bound.append(forecast_values[i] + margin)
            
            # Calculate confidence based on R-squared
            confidence = r_value ** 2
            
        else:
            # Default to simple moving average if method not recognized
            forecast_values = [np.mean(y[-3:])] * steps
            std = np.std(y) if len(y) > 1 else 0
            lower_bound = [max(0, f - 1.96 * std) for f in forecast_values]
            upper_bound = [f + 1.96 * std for f in forecast_values]
            confidence = 0.5
            method = 'moving_average'
        
        return {
            'forecast': forecast_values,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': float(confidence),
            'method': method
        }
        
    except Exception as e:
        # Fallback to simple moving average if forecasting fails
        forecast_values = [np.mean(y[-3:])] * steps
        std = np.std(y) if len(y) > 1 else 0
        lower_bound = [max(0, f - 1.96 * std) for f in forecast_values]
        upper_bound = [f + 1.96 * std for f in forecast_values]
        
        return {
            'forecast': forecast_values,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': 0.3,
            'method': 'moving_average',
            'error': str(e)
        }


def detect_anomalies(values: List[float], dates: List[datetime] = None, 
                     threshold: float = 2.0, 
                     window_size: int = None) -> Dict[str, Any]:
    """
    Detect anomalies in a time series.

    Args:
        values: List of values to analyze
        dates: Optional list of dates corresponding to the values
        threshold: Z-score threshold for anomaly detection
        window_size: Size of the rolling window for local anomaly detection 
                    (if None, global anomalies are detected)

    Returns:
        Dictionary with anomaly information
    """
    if len(values) < 4:
        return {
            'anomalies': [],
            'anomaly_indexes': [],
            'method': 'none',
            'threshold': threshold
        }
    
    # Convert to numpy array
    y = np.array(values)
    anomaly_indexes = []
    
    if window_size and window_size < len(y) and window_size >= 4:
        # Local anomaly detection using rolling windows
        anomaly_scores = np.zeros_like(y, dtype=float)
        
        # Pad beginning with NaN
        half_window = window_size // 2
        
        for i in range(len(y)):
            # Define window around current point
            start_idx = max(0, i - half_window)
            end_idx = min(len(y), i + half_window + 1)
            window = y[start_idx:end_idx]
            
            # Skip if window too small
            if len(window) < 4:
                continue
                
            # Skip the current point when calculating mean and std
            window_without_current = np.concatenate([window[:i-start_idx], window[i-start_idx+1:]])
            
            window_mean = np.mean(window_without_current)
            window_std = np.std(window_without_current)
            
            if window_std > 0:
                # Calculate z-score
                z_score = abs((y[i] - window_mean) / window_std)
                anomaly_scores[i] = z_score
                
                if z_score > threshold:
                    anomaly_indexes.append(i)
        
        method = 'rolling_window'
        
    else:
        # Global anomaly detection
        # First, try to decompose the time series to account for trends and seasonality
        if len(y) >= 8:
            try:
                if dates:
                    decomp = decompose_time_series(values, dates=dates)
                else:
                    dummy_dates = [datetime.now() + timedelta(days=i) for i in range(len(values))]
                    decomp = decompose_time_series(values, dates=dummy_dates)
                
                # Analyze residuals
                residuals = np.array(decomp['residual'])
                residual_mean = np.mean(residuals)
                residual_std = np.std(residuals)
                
                if residual_std > 0:
                    # Calculate z-scores for residuals
                    z_scores = np.abs((residuals - residual_mean) / residual_std)
                    
                    # Identify anomalies
                    anomaly_indexes = np.where(z_scores > threshold)[0].tolist()
                    method = 'decomposition'
                else:
                    # Fallback to simple z-score if std is zero
                    mean = np.mean(y)
                    std = np.std(y)
                    
                    if std > 0:
                        z_scores = np.abs((y - mean) / std)
                        anomaly_indexes = np.where(z_scores > threshold)[0].tolist()
                    
                    method = 'global_z_score'
            except:
                # Fallback to simple z-score if decomposition fails
                mean = np.mean(y)
                std = np.std(y)
                
                if std > 0:
                    z_scores = np.abs((y - mean) / std)
                    anomaly_indexes = np.where(z_scores > threshold)[0].tolist()
                
                method = 'global_z_score'
        else:
            # Simple z-score for short series
            mean = np.mean(y)
            std = np.std(y)
            
            if std > 0:
                z_scores = np.abs((y - mean) / std)
                anomaly_indexes = np.where(z_scores > threshold)[0].tolist()
            
            method = 'global_z_score'
    
    # Convert anomalies to structured format
    anomalies = []
    for idx in anomaly_indexes:
        anomaly = {
            'index': idx,
            'value': float(y[idx])
        }
        
        if dates and idx < len(dates):
            anomaly['date'] = dates[idx].strftime('%Y-%m-%d')
        
        anomalies.append(anomaly)
    
    return {
        'anomalies': anomalies,
        'anomaly_indexes': anomaly_indexes,
        'method': method,
        'threshold': threshold
    }


def find_change_points(values: List[float], dates: List[datetime] = None,
                        min_size: int = 5, 
                        threshold: float = 2.0) -> Dict[str, Any]:
    """
    Find points where the time series shows significant changes in level or trend.

    Args:
        values: List of values to analyze
        dates: Optional list of dates corresponding to the values
        min_size: Minimum segment size for change point detection
        threshold: Threshold for change significance

    Returns:
        Dictionary with change point information
    """
    if len(values) < min_size * 2:
        return {
            'change_points': [],
            'change_point_indexes': []
        }
    
    # Convert to numpy array
    y = np.array(values)
    n = len(y)
    
    # Compute difference in means
    change_scores = np.zeros(n - min_size * 2 + 1)
    
    for i in range(min_size, n - min_size + 1):
        # Compare means of segments before and after potential change point
        before_mean = np.mean(y[i-min_size:i])
        after_mean = np.mean(y[i:i+min_size])
        
        # Calculate pooled standard deviation
        before_var = np.var(y[i-min_size:i])
        after_var = np.var(y[i:i+min_size])
        pooled_std = np.sqrt((before_var * min_size + after_var * min_size) / (2 * min_size - 2))
        
        # Calculate change score (t-statistic)
        if pooled_std > 0:
            change_scores[i-min_size] = abs(before_mean - after_mean) / (pooled_std * np.sqrt(2/min_size))
        else:
            change_scores[i-min_size] = 0
    
    # Find change points exceeding threshold
    candidate_indexes = np.where(change_scores > threshold)[0] + min_size
    
    # Filter change points to ensure minimum separation
    change_point_indexes = []
    if len(candidate_indexes) > 0:
        change_point_indexes = [candidate_indexes[0]]
        
        for idx in candidate_indexes[1:]:
            if idx - change_point_indexes[-1] >= min_size:
                change_point_indexes.append(idx)
    
    # Convert to structured format
    change_points = []
    for idx in change_point_indexes:
        change_point = {
            'index': int(idx),
            'value': float(y[idx]),
            'score': float(change_scores[idx-min_size])
        }
        
        # Add before/after means
        change_point['before_mean'] = float(np.mean(y[max(0, idx-min_size):idx]))
        change_point['after_mean'] = float(np.mean(y[idx:min(len(y), idx+min_size)]))
        change_point['change'] = change_point['after_mean'] - change_point['before_mean']
        
        if dates and idx < len(dates):
            change_point['date'] = dates[idx].strftime('%Y-%m-%d')
        
        # Calculate direction and significance
        if change_point['change'] > 0:
            change_point['direction'] = 'increasing'
        else:
            change_point['direction'] = 'decreasing'
            
        change_point['percent_change'] = (change_point['change'] / change_point['before_mean'] * 100) if change_point['before_mean'] != 0 else float('inf')
        
        change_points.append(change_point)
    
    return {
        'change_points': change_points,
        'change_point_indexes': change_point_indexes
    }