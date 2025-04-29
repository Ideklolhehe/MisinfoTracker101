"""
Predictive modeling services for the CIVILIAN system.
Provides trajectory forecasting, threshold projections, and scenario analysis.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import numpy as np
import pandas as pd
from sqlalchemy import func, and_, or_, desc

from utils.app_context import with_app_context
from utils.environment import get_config, get_int_config, get_bool_config
from utils.metrics import time_operation
from utils.concurrency import ThreadSafeDict

from app import db
from models import DetectedNarrative, NarrativeInstance

# Configure module logger
logger = logging.getLogger(__name__)

# Cache for model forecasts
forecast_cache = ThreadSafeDict()

# Helper for float configs
def get_float_config(key: str, default: float) -> float:
    """Get a float configuration value."""
    return float(get_config(key, default=default, cast=float))

# Model configuration
FORECAST_HORIZON = get_int_config('FORECAST_HORIZON', default=30)  # days
PROPHET_CONFIDENCE = get_float_config('PROPHET_CONFIDENCE', default=0.8)  # 80% confidence interval
FORECAST_REFRESH_INTERVAL = get_int_config('FORECAST_REFRESH_INTERVAL', default=86400)  # seconds (1 day)


class TimeSeriesModel:
    """Base class for time series forecasting models."""
    
    def __init__(self, name: str):
        """
        Initialize the model.
        
        Args:
            name: Model name
        """
        self.name = name
        
    def fit(self, data: pd.DataFrame) -> 'TimeSeriesModel':
        """
        Fit the model to data.
        
        Args:
            data: DataFrame with 'ds' (dates) and 'y' (target) columns
            
        Returns:
            Self for chaining
        """
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, periods: int = FORECAST_HORIZON) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecast
        """
        raise NotImplementedError("Subclasses must implement predict()")
        
    def get_components(self) -> Dict[str, Any]:
        """
        Get model components (trend, seasonality, etc.).
        
        Returns:
            Dictionary of components
        """
        return {}


class ProphetModel(TimeSeriesModel):
    """Facebook Prophet forecasting model."""
    
    def __init__(self, name: str = "prophet", interval_width: float = PROPHET_CONFIDENCE):
        """
        Initialize Prophet model.
        
        Args:
            name: Model name
            interval_width: Width of prediction intervals (0-1)
        """
        super().__init__(name)
        self.interval_width = interval_width
        self.model = None
        
    def fit(self, data: pd.DataFrame) -> 'ProphetModel':
        """
        Fit Prophet model to data.
        
        Args:
            data: DataFrame with 'ds' (dates) and 'y' (target) columns
            
        Returns:
            Self for chaining
        """
        try:
            from prophet import Prophet
            
            self.model = Prophet(
                interval_width=self.interval_width,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            # Add additional regressors if available
            if 'propagation' in data.columns:
                self.model.add_regressor('propagation')
                
            if 'threat' in data.columns:
                self.model.add_regressor('threat')
                
            # Fit the model
            with time_operation(f"fit_{self.name}"):
                self.model.fit(data)
                
            logger.info(f"Fitted Prophet model to {len(data)} data points")
            return self
        except ImportError:
            logger.error("Prophet package not available. Install with 'pip install prophet'")
            raise
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
            
    def predict(self, periods: int = FORECAST_HORIZON) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            periods: Number of days to forecast
            
        Returns:
            DataFrame with forecast
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() first")
            raise ValueError("Model not fitted")
            
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Add regressor values if needed
            # For demonstration, we'll use the last value for regressors
            if hasattr(self.model, 'extra_regressors'):
                for regressor in self.model.extra_regressors:
                    name = regressor['name']
                    if name == 'propagation':
                        # Use the last value or a reasonable default
                        future[name] = 0.5
                    elif name == 'threat':
                        # Use the last value or a reasonable default
                        future[name] = 2
                        
            # Generate forecast
            with time_operation(f"predict_{self.name}"):
                forecast = self.model.predict(future)
                
            logger.info(f"Generated {periods} day forecast with Prophet")
            return forecast
        except Exception as e:
            logger.error(f"Error generating Prophet forecast: {e}")
            raise
            
    def get_components(self) -> Dict[str, Any]:
        """
        Get model components (trend, seasonality, etc.).
        
        Returns:
            Dictionary of components
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() first")
            return {}
            
        try:
            components = {}
            forecast = self.predict(periods=FORECAST_HORIZON)
            
            # Extract trend component
            if 'trend' in forecast:
                components['trend'] = forecast[['ds', 'trend']].to_dict(orient='records')
                
            # Extract seasonal components
            for column in forecast.columns:
                if column.startswith(('yearly', 'weekly', 'daily')):
                    components[column] = forecast[['ds', column]].to_dict(orient='records')
                    
            return components
        except Exception as e:
            logger.error(f"Error extracting model components: {e}")
            return {}


class DartsModel(TimeSeriesModel):
    """Darts forecasting model wrapper."""
    
    def __init__(
        self, 
        name: str = "darts_rnn",
        model_type: str = "RNN",
        input_chunk_length: int = 14,
        output_chunk_length: int = 7
    ):
        """
        Initialize Darts model.
        
        Args:
            name: Model name
            model_type: Type of model to use (RNN, ARIMA, etc.)
            input_chunk_length: Input sequence length for RNN models
            output_chunk_length: Output sequence length for RNN models
        """
        super().__init__(name)
        self.model_type = model_type
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model = None
        self.series = None
        
    def fit(self, data: pd.DataFrame) -> 'DartsModel':
        """
        Fit Darts model to data.
        
        Args:
            data: DataFrame with 'ds' (dates) and 'y' (target) columns
            
        Returns:
            Self for chaining
        """
        try:
            from darts import TimeSeries
            from darts.models import RNNModel, AutoARIMA, ExponentialSmoothing
            
            # Convert to Darts TimeSeries
            self.series = TimeSeries.from_dataframe(
                data,
                time_col='ds',
                value_cols='y'
            )
            
            # Choose model based on type
            if self.model_type == 'RNN':
                self.model = RNNModel(
                    model='LSTM',
                    input_chunk_length=self.input_chunk_length,
                    output_chunk_length=self.output_chunk_length,
                    random_state=42,
                    n_epochs=100,
                    force_reset=True
                )
            elif self.model_type == 'ARIMA':
                self.model = AutoARIMA()
            elif self.model_type == 'ETS':
                self.model = ExponentialSmoothing()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
            # Fit the model
            with time_operation(f"fit_{self.name}"):
                self.model.fit(self.series)
                
            logger.info(f"Fitted {self.model_type} model to {len(data)} data points")
            return self
        except ImportError:
            logger.error("Darts package not available. Install with 'pip install darts'")
            raise
        except Exception as e:
            logger.error(f"Error fitting {self.model_type} model: {e}")
            raise
            
    def predict(self, periods: int = FORECAST_HORIZON) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecast
        """
        if self.model is None or self.series is None:
            logger.error("Model not fitted. Call fit() first")
            raise ValueError("Model not fitted")
            
        try:
            # Generate forecast
            with time_operation(f"predict_{self.name}"):
                prediction = self.model.predict(periods)
                
            # Convert back to DataFrame
            forecast = prediction.pd_dataframe().reset_index()
            forecast.columns = ['ds', 'yhat']
            
            # Add current datetime as model doesn't return it
            forecast['forecast_date'] = datetime.now()
            
            logger.info(f"Generated {periods} period forecast with {self.model_type}")
            return forecast
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
            
    def get_components(self) -> Dict[str, Any]:
        """
        Get model components (trend, seasonality, etc.).
        
        Returns:
            Dictionary of components (empty for most Darts models)
        """
        return {}


class DeepARModel(TimeSeriesModel):
    """DeepAR forecasting model using GluonTS."""
    
    def __init__(
        self,
        name: str = "deepar",
        freq: str = "D",
        prediction_length: int = FORECAST_HORIZON,
        epochs: int = 10
    ):
        """
        Initialize DeepAR model.
        
        Args:
            name: Model name
            freq: Time series frequency (D for daily)
            prediction_length: Forecast horizon
            epochs: Training epochs
        """
        super().__init__(name)
        self.freq = freq
        self.prediction_length = prediction_length
        self.epochs = epochs
        self.model = None
        self.last_train_date = None
        
    def fit(self, data: pd.DataFrame) -> 'DeepARModel':
        """
        Fit DeepAR model to data.
        
        Args:
            data: DataFrame with 'ds' (dates) and 'y' (target) columns
            
        Returns:
            Self for chaining
        """
        try:
            from gluonts.dataset.pandas import PandasDataset
            from gluonts.dataset.common import ListDataset
            from gluonts.model.deepar import DeepAREstimator
            from gluonts.mx.trainer import Trainer
            
            # Remember the last training date
            self.last_train_date = data['ds'].max()
            
            # Prepare data in GluonTS format
            train_data = PandasDataset(
                data,
                target='y',
                timestamp='ds',
                freq=self.freq
            )
            
            # Create and train model
            self.model = DeepAREstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                trainer=Trainer(
                    epochs=self.epochs,
                    learning_rate=1e-3,
                    num_batches_per_epoch=100
                )
            )
            
            with time_operation(f"fit_{self.name}"):
                self.predictor = self.model.train(train_data)
                
            logger.info(f"Fitted DeepAR model to {len(data)} data points")
            return self
        except ImportError:
            logger.error("GluonTS package not available. Install with 'pip install gluonts'")
            raise
        except Exception as e:
            logger.error(f"Error fitting DeepAR model: {e}")
            raise
            
    def predict(self, periods: int = None) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            periods: Number of periods to forecast (ignored, uses prediction_length)
            
        Returns:
            DataFrame with forecast
        """
        if self.model is None or self.predictor is None:
            logger.error("Model not fitted. Call fit() first")
            raise ValueError("Model not fitted")
            
        if periods and periods != self.prediction_length:
            logger.warning(
                f"DeepAR uses fixed prediction length ({self.prediction_length}). "
                f"Ignoring requested periods ({periods})"
            )
            
        try:
            # Generate forecast
            with time_operation(f"predict_{self.name}"):
                # This is a simplified example, normally you'd create proper test data
                forecast = list(self.predictor.predict(num_samples=100))
                
            # Convert forecast to DataFrame
            dates = pd.date_range(
                start=self.last_train_date,
                periods=self.prediction_length + 1,
                freq=self.freq
            )[1:]  # Skip the last training point
            
            result = pd.DataFrame({
                'ds': dates,
                'yhat': forecast[0].mean,
                'yhat_lower': forecast[0].quantile(0.1),
                'yhat_upper': forecast[0].quantile(0.9)
            })
            
            logger.info(f"Generated {self.prediction_length} day forecast with DeepAR")
            return result
        except Exception as e:
            logger.error(f"Error generating DeepAR forecast: {e}")
            raise


class PredictiveModeling:
    """
    Service for predictive modeling of narrative complexity and key metrics.
    """
    
    def __init__(self):
        """Initialize the service."""
        self.models = {}
        
    @with_app_context
    def get_narrative_time_series(
        self,
        narrative_id: int,
        metric: str = 'complexity',
        interval: str = 'day'
    ) -> pd.DataFrame:
        """
        Get time series data for a narrative.
        
        Args:
            narrative_id: ID of the narrative
            metric: Metric to retrieve (complexity, propagation, threat)
            interval: Time interval (day, week, month)
            
        Returns:
            DataFrame with time series data
        """
        # Determine date trunc function based on interval
        if interval == 'week':
            date_trunc = func.date_trunc('week', NarrativeInstance.detected_at)
        elif interval == 'month':
            date_trunc = func.date_trunc('month', NarrativeInstance.detected_at)
        else:
            date_trunc = func.date_trunc('day', NarrativeInstance.detected_at)
            
        try:
            # Get narrative to ensure it exists
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.error(f"Narrative {narrative_id} not found")
                return pd.DataFrame()
                
            # Get time series based on metric
            if metric == 'complexity':
                # For complexity, get data from the narrative's historical complexity scores
                # This is stored in meta_data as a JSON array
                if not narrative.meta_data:
                    logger.warning(f"No meta_data for narrative {narrative_id}")
                    return pd.DataFrame()
                    
                meta_data = json.loads(narrative.meta_data)
                if 'complexity_history' not in meta_data:
                    logger.warning(f"No complexity history for narrative {narrative_id}")
                    return pd.DataFrame()
                    
                # Convert to DataFrame
                history = meta_data['complexity_history']
                if not history:
                    return pd.DataFrame()
                    
                df = pd.DataFrame(history)
                df['ds'] = pd.to_datetime(df['date'])
                df['y'] = df['complexity']
                
                return df[['ds', 'y']]
            elif metric in ('propagation', 'threat'):
                # For propagation and threat, query from instances
                query = db.session.query(
                    date_trunc.label('date'),
                    func.avg(func.cast(func.json_extract(
                        NarrativeInstance.meta_data, f'$.{metric}'
                    ), db.Float)).label('value')
                ).filter(
                    NarrativeInstance.narrative_id == narrative_id
                ).group_by(
                    date_trunc
                ).order_by(
                    date_trunc
                )
                
                result = query.all()
                
                # Convert to DataFrame
                df = pd.DataFrame([(r.date, r.value) for r in result], columns=['ds', 'y'])
                df['ds'] = pd.to_datetime(df['ds'])
                
                return df
            else:
                logger.warning(f"Unknown metric: {metric}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving time series for narrative {narrative_id}: {e}")
            return pd.DataFrame()
            
    def forecast_narrative(
        self,
        narrative_id: int,
        metric: str = 'complexity',
        model_type: str = 'prophet',
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a forecast for a narrative metric.
        
        Args:
            narrative_id: ID of the narrative
            metric: Metric to forecast
            model_type: Type of model to use
            force_refresh: Whether to force a refresh of the forecast
            
        Returns:
            Forecast results
        """
        # Check cache first
        cache_key = f"{narrative_id}_{metric}_{model_type}"
        if not force_refresh and cache_key in forecast_cache:
            cached = forecast_cache[cache_key]
            # Check if cache is still valid
            if time.time() - cached.get('timestamp', 0) < FORECAST_REFRESH_INTERVAL:
                logger.info(f"Using cached forecast for {cache_key}")
                return cached
                
        # Get time series data
        df = self.get_narrative_time_series(narrative_id, metric)
        if df.empty:
            logger.warning(f"No time series data for narrative {narrative_id}, metric {metric}")
            return {
                'error': 'No data available for forecasting',
                'narrative_id': narrative_id,
                'metric': metric
            }
            
        # Create and fit model
        try:
            if model_type == 'prophet':
                model = ProphetModel()
            elif model_type.startswith('darts'):
                # Parse model type, e.g. darts_rnn
                parts = model_type.split('_')
                darts_type = parts[1].upper() if len(parts) > 1 else 'RNN'
                model = DartsModel(model_type=darts_type)
            elif model_type == 'deepar':
                model = DeepARModel()
            else:
                logger.error(f"Unknown model type: {model_type}")
                return {
                    'error': f'Unknown model type: {model_type}',
                    'narrative_id': narrative_id,
                    'metric': metric
                }
                
            # Fit model
            model.fit(df)
            
            # Generate forecast
            forecast = model.predict()
            
            # Create result dictionary
            result = {
                'narrative_id': narrative_id,
                'metric': metric,
                'model_type': model_type,
                'forecast_date': datetime.now().isoformat(),
                'timestamp': time.time(),
                'forecast_periods': FORECAST_HORIZON,
                'historical_data': df.to_dict(orient='records'),
                'forecast_data': forecast.to_dict(orient='records'),
                'components': model.get_components()
            }
            
            # Cache the result
            forecast_cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error forecasting for narrative {narrative_id}: {e}")
            return {
                'error': str(e),
                'narrative_id': narrative_id,
                'metric': metric
            }
            
    def find_threshold_crossing(
        self,
        forecast: Dict[str, Any],
        threshold: float,
        direction: str = 'above'
    ) -> Dict[str, Any]:
        """
        Find when a forecast crosses a threshold.
        
        Args:
            forecast: Forecast dictionary from forecast_narrative()
            threshold: Threshold value
            direction: Direction of crossing ('above' or 'below')
            
        Returns:
            Dictionary with crossing information
        """
        if 'error' in forecast:
            return {'error': forecast['error']}
            
        try:
            # Convert forecast data to DataFrame
            df = pd.DataFrame(forecast['forecast_data'])
            
            # Determine which column to use
            value_col = 'yhat'  # Default
            
            # Convert dates
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Find crossing points
            if direction == 'above':
                crossings = df[df[value_col] >= threshold]
            else:
                crossings = df[df[value_col] <= threshold]
                
            if crossings.empty:
                return {
                    'narrative_id': forecast['narrative_id'],
                    'metric': forecast['metric'],
                    'threshold': threshold,
                    'direction': direction,
                    'crosses_threshold': False,
                    'message': f"Forecast does not cross the {threshold} threshold"
                }
                
            # Get the first crossing
            first_crossing = crossings.iloc[0]
            
            # Calculate days until crossing
            days_until = (first_crossing['ds'] - datetime.now()).days
            
            return {
                'narrative_id': forecast['narrative_id'],
                'metric': forecast['metric'],
                'threshold': threshold,
                'direction': direction,
                'crosses_threshold': True,
                'crossing_date': first_crossing['ds'].isoformat(),
                'days_until_crossing': days_until,
                'crossing_value': float(first_crossing[value_col]),
                'message': f"Forecast crosses the {threshold} threshold in {days_until} days"
            }
        except Exception as e:
            logger.error(f"Error finding threshold crossing: {e}")
            return {
                'error': str(e),
                'narrative_id': forecast['narrative_id'],
                'metric': forecast['metric'],
                'threshold': threshold
            }
            
    def analyze_key_factors(
        self,
        narrative_id: int,
        metric: str = 'complexity'
    ) -> Dict[str, Any]:
        """
        Analyze key factors influencing a metric for a narrative.
        
        Args:
            narrative_id: ID of the narrative
            metric: Metric to analyze
            
        Returns:
            Dictionary with key factors
        """
        try:
            # Get related metrics data
            complexity_df = self.get_narrative_time_series(narrative_id, 'complexity')
            propagation_df = self.get_narrative_time_series(narrative_id, 'propagation')
            threat_df = self.get_narrative_time_series(narrative_id, 'threat')
            
            # Merge dataframes
            if complexity_df.empty or propagation_df.empty or threat_df.empty:
                logger.warning(f"Missing data for key factor analysis of narrative {narrative_id}")
                return {
                    'narrative_id': narrative_id,
                    'metric': metric,
                    'error': 'Insufficient data for key factor analysis'
                }
                
            # Perform simple correlation analysis
            merged = complexity_df.merge(
                propagation_df.rename(columns={'y': 'propagation'}),
                on='ds',
                how='inner'
            ).merge(
                threat_df.rename(columns={'y': 'threat'}),
                on='ds',
                how='inner'
            )
            
            if merged.empty:
                return {
                    'narrative_id': narrative_id,
                    'metric': metric,
                    'error': 'No overlapping data points for correlation analysis'
                }
                
            # Calculate correlations
            corr = merged.corr()
            
            # Get correlations with target metric
            target_col = 'y' if metric == 'complexity' else metric
            correlations = corr[target_col].drop(target_col).to_dict()
            
            # Sort by absolute correlation
            sorted_corr = sorted(
                correlations.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Build result
            result = {
                'narrative_id': narrative_id,
                'metric': metric,
                'correlations': [
                    {'factor': k, 'correlation': v}
                    for k, v in sorted_corr
                ],
                'sample_size': len(merged)
            }
            
            # Try to use more advanced feature importance if available
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.inspection import permutation_importance
                
                # Prepare data for random forest
                X = merged.drop(['ds', target_col], axis=1)
                y = merged[target_col]
                
                if len(X) >= 10:  # Minimum samples for meaningful analysis
                    # Fit random forest
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    
                    # Calculate feature importance
                    perm_importance = permutation_importance(
                        rf, X, y, n_repeats=10, random_state=42
                    )
                    
                    # Sort features by importance
                    feature_importance = {
                        feature: float(importance)
                        for feature, importance in zip(
                            X.columns,
                            perm_importance.importances_mean
                        )
                    }
                    
                    sorted_importance = sorted(
                        feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    result['feature_importance'] = [
                        {'factor': k, 'importance': v}
                        for k, v in sorted_importance
                    ]
            except Exception as e:
                logger.warning(f"Could not perform advanced feature importance: {e}")
                
            return result
        except Exception as e:
            logger.error(f"Error analyzing key factors for narrative {narrative_id}: {e}")
            return {
                'error': str(e),
                'narrative_id': narrative_id,
                'metric': metric
            }
            
    def simulate_scenario(
        self,
        narrative_id: int,
        interventions: Dict[str, Any],
        metric: str = 'complexity',
        model_type: str = 'prophet'
    ) -> Dict[str, Any]:
        """
        Simulate a what-if scenario with interventions.
        
        Args:
            narrative_id: ID of the narrative
            interventions: Dictionary of interventions
            metric: Metric to forecast
            model_type: Type of model to use
            
        Returns:
            Dictionary with scenario results
        """
        try:
            # Get baseline forecast
            baseline = self.forecast_narrative(narrative_id, metric, model_type)
            if 'error' in baseline:
                return baseline
                
            # Get time series data
            df = self.get_narrative_time_series(narrative_id, metric)
            if df.empty:
                logger.warning(f"No time series data for narrative {narrative_id}, metric {metric}")
                return {
                    'error': 'No data available for scenario modeling',
                    'narrative_id': narrative_id,
                    'metric': metric
                }
                
            # Apply interventions to create modified dataframe
            modified_df = df.copy()
            
            # Example intervention types:
            # 1. 'step': Add a one-time step change at a specific date
            # 2. 'trend': Change the overall trend by a factor
            # 3. 'counter_message': Simulate effect of a counter-message
            
            intervention_effects = []
            
            for name, intervention in interventions.items():
                intervention_type = intervention.get('type')
                
                if intervention_type == 'step':
                    # Add a step change at a specific date
                    date = pd.to_datetime(intervention.get('date'))
                    value = float(intervention.get('value', 0))
                    
                    # Add a step change column
                    modified_df[f'step_{name}'] = (modified_df['ds'] >= date).astype(float)
                    
                    intervention_effects.append({
                        'name': name,
                        'type': 'step',
                        'date': date.isoformat(),
                        'value': value,
                        'description': f"Step change of {value} on {date.strftime('%Y-%m-%d')}"
                    })
                    
                elif intervention_type == 'trend':
                    # Change trend by a factor
                    factor = float(intervention.get('factor', 1.0))
                    
                    # Add a trend modifier column
                    start_date = pd.to_datetime(intervention.get('start_date', df['ds'].min()))
                    modified_df[f'trend_{name}'] = (
                        (modified_df['ds'] - start_date).dt.days * (factor - 1) / 30
                    )
                    
                    intervention_effects.append({
                        'name': name,
                        'type': 'trend',
                        'factor': factor,
                        'start_date': start_date.isoformat(),
                        'description': f"Trend change by factor {factor} from {start_date.strftime('%Y-%m-%d')}"
                    })
                    
                elif intervention_type == 'counter_message':
                    # Simulate counter-message effect
                    date = pd.to_datetime(intervention.get('date'))
                    impact = float(intervention.get('impact', -0.1))
                    decay = float(intervention.get('decay', 0.9))
                    
                    # Add counter-message effect column
                    days_since = (modified_df['ds'] - date).dt.days
                    effect = np.zeros(len(modified_df))
                    mask = days_since >= 0
                    effect[mask] = impact * (decay ** days_since[mask])
                    modified_df[f'counter_{name}'] = effect
                    
                    intervention_effects.append({
                        'name': name,
                        'type': 'counter_message',
                        'date': date.isoformat(),
                        'impact': impact,
                        'decay': decay,
                        'description': f"Counter-message with initial impact {impact} on {date.strftime('%Y-%m-%d')}"
                    })
                    
            # Create and fit model with modified data
            if model_type == 'prophet':
                model = ProphetModel(name=f"scenario_{narrative_id}")
                
                # Add intervention regressors
                for name in interventions.keys():
                    intervention_type = interventions[name].get('type')
                    if intervention_type == 'step':
                        model.model.add_regressor(f'step_{name}')
                    elif intervention_type == 'trend':
                        model.model.add_regressor(f'trend_{name}')
                    elif intervention_type == 'counter_message':
                        model.model.add_regressor(f'counter_{name}')
                        
                # Fit model
                model.fit(modified_df)
                
                # Generate forecast
                forecast = model.predict()
                
                # Create result dictionary
                result = {
                    'narrative_id': narrative_id,
                    'metric': metric,
                    'model_type': model_type,
                    'scenario_name': 'Custom Scenario',
                    'forecast_date': datetime.now().isoformat(),
                    'interventions': intervention_effects,
                    'baseline_forecast': baseline['forecast_data'],
                    'scenario_forecast': forecast.to_dict(orient='records'),
                }
                
                return result
            else:
                # Other model types not fully implemented for scenario modeling
                return {
                    'error': f'Scenario modeling not implemented for model type: {model_type}',
                    'narrative_id': narrative_id,
                    'metric': metric
                }
        except Exception as e:
            logger.error(f"Error simulating scenario for narrative {narrative_id}: {e}")
            return {
                'error': str(e),
                'narrative_id': narrative_id,
                'metric': metric
            }


# Create singleton instance
predictive_modeling_service = PredictiveModeling()


# Function moved to the top of the file