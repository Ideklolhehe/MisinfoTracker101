"""
Predictive modeling service for misinformation complexity trajectory.
Uses linear regression and statistical analysis to forecast complexity evolution.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from app import db
from models import DetectedNarrative
from services.time_series_analyzer import TimeSeriesAnalyzer

logger = logging.getLogger(__name__)

class PredictiveModeling:
    """Service for predictive modeling of misinformation complexity trajectory."""
    
    def __init__(self):
        """Initialize the predictive modeling service."""
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_trained = False
    
    def train_model(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        test_size: float = 0.2, 
        random_state: int = 42
    ):
        """
        Trains the linear regression model.
        
        Args:
            data: The input dataframe containing features and target variable.
            target_column: The name of the column to be predicted.
            feature_columns: A list of column names to be used as features.
            test_size: The proportion of the dataset to include in the test split.
            random_state: Random state for reproducibility.
            
        Raises:
            ValueError: If the target column or feature columns are not found in the data.
        """
        try:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the data.")
            
            for col in feature_columns:
                if col not in data.columns:
                    raise ValueError(f"Feature column '{col}' not found in the data.")
            
            X = data[feature_columns]
            y = data[target_column]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            self.model.fit(self.X_train, self.y_train)
            self.model_trained = True
            
        except ValueError as e:
            logger.error(f"Error during model training: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during model training: {e}")
            raise Exception(f"Error during model training: {e}")
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Predicts the target variable for new data.
        
        Args:
            new_data: The input dataframe containing the same features used for training.
            
        Returns:
            The predicted values.
            
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.model_trained:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        try:
            predictions = self.model.predict(new_data)
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise Exception(f"Error during prediction: {e}")
    
    def calculate_confidence_interval(
        self, new_data: pd.DataFrame, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the confidence interval for the predictions.
        
        Args:
            new_data: The input dataframe for which to calculate confidence intervals.
            alpha: The significance level for the confidence interval.
            
        Returns:
            A tuple containing the lower and upper bounds of the confidence interval.
            
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.model_trained:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        try:
            X = self.X_train
            y = self.y_train
            
            # Add a constant for the intercept
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            new_data_with_const = sm.add_constant(new_data)
            predictions = model.get_prediction(new_data_with_const)
            lower, upper = predictions.conf_int(alpha=alpha).T
            
            return lower, upper
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            raise Exception(f"Error calculating confidence interval: {e}")
    
    def evaluate_model(self) -> float:
        """
        Evaluates the model on the test set using Mean Squared Error.
        
        Returns:
            The Mean Squared Error.
            
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.model_trained:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        try:
            y_pred = self.model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            return mse
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise Exception(f"Error evaluating model: {e}")
    
    def identify_key_factors(self) -> Dict[str, float]:
        """
        Identifies key factors driving complexity based on model coefficients.
        
        Returns:
            A dictionary of feature names and their corresponding coefficients.
            
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.model_trained:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        try:
            coefficients = self.model.coef_
            feature_names = self.X_train.columns
            factor_importance = dict(zip(feature_names, coefficients))
            return factor_importance
        except Exception as e:
            logger.error(f"Error identifying key factors: {e}")
            raise Exception(f"Error identifying key factors: {e}")
    
    def what_if_scenario(
        self, baseline_data: pd.DataFrame, intervention_changes: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Models 'what-if' scenarios by changing feature values and predicting the outcome.
        
        Args:
            baseline_data: The baseline data representing the current state.
            intervention_changes: A dictionary of feature names and their corresponding changes.
            
        Returns:
            A tuple containing the predicted values for the baseline and the intervention scenario.
            
        Raises:
            ValueError: If the model has not been trained yet.
            ValueError: If the intervention change keys are not found in the baseline data.
        """
        if not self.model_trained:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        try:
            for feature in intervention_changes:
                if feature not in baseline_data.columns:
                    raise ValueError(f"Intervention feature '{feature}' not found in the baseline data.")
            
            intervention_data = baseline_data.copy()
            for feature, change in intervention_changes.items():
                intervention_data[feature] = intervention_data[feature] + change
            
            baseline_prediction = self.predict(baseline_data)
            intervention_prediction = self.predict(intervention_data)
            
            return baseline_prediction, intervention_prediction
        except ValueError as e:
            logger.error(f"Error in what-if scenario: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in what-if scenario: {e}")
            raise Exception(f"Error running what-if scenario: {e}")

class ComplexityPredictionService:
    """Service for predicting future complexity of misinformation narratives."""
    
    @staticmethod
    def predict_narrative_complexity(
        narrative_id: int, days_ahead: int = 7, confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Predict future complexity for a specific narrative.
        
        Args:
            narrative_id: ID of the narrative to predict
            days_ahead: Number of days to predict into the future
            confidence_level: Confidence level for prediction intervals (0-1)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get the narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": f"Narrative {narrative_id} not found"}
            
            # Get historical complexity data
            historical_data = TimeSeriesAnalyzer._get_historical_complexity_data(narrative_id)
            
            if not historical_data or len(historical_data['timestamps']) < 3:
                logger.warning(f"Insufficient historical data for narrative {narrative_id}")
                return {"error": "Insufficient historical data for prediction"}
            
            # Prepare data for modeling
            data = ComplexityPredictionService._prepare_prediction_data(historical_data)
            
            if data is None or data.empty:
                return {"error": "Failed to prepare data for prediction"}
            
            # Train model for overall complexity
            predictor = PredictiveModeling()
            try:
                predictor.train_model(
                    data=data,
                    target_column='overall_score',
                    feature_columns=['days_since_start']
                )
            except Exception as e:
                logger.error(f"Error training prediction model: {e}")
                return {"error": f"Failed to train prediction model: {e}"}
            
            # Generate future dates for prediction
            future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                           for i in range(1, days_ahead + 1)]
            
            # Prepare prediction data
            last_day = data['days_since_start'].max()
            future_days = pd.DataFrame({
                'days_since_start': [last_day + i for i in range(1, days_ahead + 1)]
            })
            
            # Make predictions
            predictions = predictor.predict(future_days)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            try:
                lower_bound, upper_bound = predictor.calculate_confidence_interval(future_days, alpha)
            except Exception as e:
                logger.warning(f"Error calculating confidence intervals: {e}")
                # Fallback: Use simple heuristic for confidence intervals
                prediction_std = np.std(data['overall_score']) 
                z_score = 1.96  # Approx. 95% confidence
                lower_bound = predictions - z_score * prediction_std
                upper_bound = predictions + z_score * prediction_std
            
            # Format predictions to be within valid range
            predictions = np.clip(predictions, 0, 10)
            lower_bound = np.clip(lower_bound, 0, 10)
            upper_bound = np.clip(upper_bound, 0, 10)
            
            # Calculate trend direction and key insights
            current_complexity = data['overall_score'].iloc[-1]
            final_prediction = predictions[-1]
            
            if final_prediction > current_complexity * 1.2:
                trend = "strong_increase"
                insight = "Significant increase in complexity projected."
            elif final_prediction > current_complexity * 1.05:
                trend = "moderate_increase"
                insight = "Moderate increase in complexity projected."
            elif final_prediction < current_complexity * 0.8:
                trend = "strong_decrease"
                insight = "Significant decrease in complexity projected."
            elif final_prediction < current_complexity * 0.95:
                trend = "moderate_decrease"
                insight = "Moderate decrease in complexity projected."
            else:
                trend = "stable"
                insight = "Complexity level projected to remain relatively stable."
            
            # Prepare response
            result = {
                'narrative_id': narrative_id,
                'title': narrative.title,
                'current_complexity': float(current_complexity),
                'dates': future_dates,
                'predicted_complexity': predictions.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist(),
                'trend_direction': trend,
                'main_insight': insight,
                'confidence_level': confidence_level,
                'data_points_used': len(data),
                'prediction_date': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting narrative complexity: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def predict_multiple_narratives(
        days_ahead: int = 7, min_data_points: int = 3, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Predict complexity for multiple narratives.
        
        Args:
            days_ahead: Number of days to predict into the future
            min_data_points: Minimum number of historical data points required
            limit: Maximum number of narratives to predict
            
        Returns:
            Dictionary with prediction results for multiple narratives
        """
        try:
            # Get active narratives
            narratives = DetectedNarrative.query.filter_by(status='active').all()
            
            predictions = []
            for narrative in narratives[:limit]:
                # Check if we have sufficient historical data
                historical_data = TimeSeriesAnalyzer._get_historical_complexity_data(narrative.id)
                
                if historical_data and len(historical_data['timestamps']) >= min_data_points:
                    # Make prediction
                    prediction = ComplexityPredictionService.predict_narrative_complexity(
                        narrative.id, days_ahead
                    )
                    
                    if "error" not in prediction:
                        predictions.append(prediction)
            
            if not predictions:
                return {"error": "No narratives with sufficient data for prediction"}
            
            # Sort by prediction certainty (inverse of confidence interval width)
            for pred in predictions:
                if 'lower_bound' in pred and 'upper_bound' in pred:
                    avg_interval = np.mean([u - l for u, l in zip(pred['upper_bound'], pred['lower_bound'])])
                    pred['certainty_score'] = 1.0 / (1.0 + avg_interval)
                else:
                    pred['certainty_score'] = 0
            
            # Sort by certainty (highest first)
            predictions.sort(key=lambda x: x.get('certainty_score', 0), reverse=True)
            
            return {
                'narrative_count': len(predictions),
                'predictions': predictions,
                'prediction_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting multiple narratives: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def what_if_analysis(
        narrative_id: int, intervention_scenario: str, days_ahead: int = 14
    ) -> Dict[str, Any]:
        """
        Perform 'what-if' analysis for different intervention scenarios.
        
        Args:
            narrative_id: ID of the narrative to analyze
            intervention_scenario: Type of intervention to model ('counter_narrative', 
                                  'debunking', 'visibility_reduction')
            days_ahead: Number of days to predict into the future
            
        Returns:
            Dictionary with what-if analysis results
        """
        try:
            # Get the narrative
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative:
                logger.warning(f"Narrative {narrative_id} not found")
                return {"error": f"Narrative {narrative_id} not found"}
            
            # Get historical complexity data
            historical_data = TimeSeriesAnalyzer._get_historical_complexity_data(narrative_id)
            
            if not historical_data or len(historical_data['timestamps']) < 3:
                logger.warning(f"Insufficient historical data for narrative {narrative_id}")
                return {"error": "Insufficient historical data for what-if analysis"}
            
            # Prepare data for modeling
            data = ComplexityPredictionService._prepare_prediction_data(historical_data)
            
            if data is None or data.empty:
                return {"error": "Failed to prepare data for what-if analysis"}
            
            # Add additional features for intervention modeling
            data['counter_narrative_active'] = 0
            data['debunking_active'] = 0
            data['visibility_reduced'] = 0
            
            # Train base model with all features
            feature_columns = [
                'days_since_start', 
                'counter_narrative_active', 
                'debunking_active', 
                'visibility_reduced'
            ]
            
            predictor = PredictiveModeling()
            try:
                predictor.train_model(
                    data=data,
                    target_column='overall_score',
                    feature_columns=feature_columns
                )
            except Exception as e:
                logger.error(f"Error training what-if model: {e}")
                return {"error": f"Failed to train what-if model: {e}"}
            
            # Prepare baseline prediction data
            last_day = data['days_since_start'].max()
            baseline_data = pd.DataFrame({
                'days_since_start': [last_day + i for i in range(1, days_ahead + 1)],
                'counter_narrative_active': [0] * days_ahead,
                'debunking_active': [0] * days_ahead,
                'visibility_reduced': [0] * days_ahead
            })
            
            # Prepare intervention data
            intervention_data = baseline_data.copy()
            
            # Set intervention parameters based on scenario
            if intervention_scenario == 'counter_narrative':
                intervention_data['counter_narrative_active'] = 1
                scenario_name = "Counter-narrative Campaign"
                scenario_description = "Publication of counter-narratives that directly address this misinformation"
            elif intervention_scenario == 'debunking':
                intervention_data['debunking_active'] = 1
                scenario_name = "Fact-checking & Debunking"
                scenario_description = "Systematic fact-checking and debunking of claims in this narrative"
            elif intervention_scenario == 'visibility_reduction':
                intervention_data['visibility_reduced'] = 1
                scenario_name = "Visibility Reduction"
                scenario_description = "Reduction in visibility and reach of this narrative on platforms"
            else:
                return {"error": f"Unknown intervention scenario: {intervention_scenario}"}
            
            # Make predictions
            baseline_predictions = predictor.predict(baseline_data)
            intervention_predictions = predictor.predict(intervention_data)
            
            # Format predictions to be within valid range
            baseline_predictions = np.clip(baseline_predictions, 0, 10)
            intervention_predictions = np.clip(intervention_predictions, 0, 10)
            
            # Generate dates
            future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                           for i in range(1, days_ahead + 1)]
            
            # Calculate impact metrics
            baseline_final = baseline_predictions[-1]
            intervention_final = intervention_predictions[-1]
            absolute_reduction = baseline_final - intervention_final
            relative_reduction = (baseline_final - intervention_final) / baseline_final if baseline_final > 0 else 0
            
            # Prepare response
            result = {
                'narrative_id': narrative_id,
                'title': narrative.title,
                'dates': future_dates,
                'baseline_predictions': baseline_predictions.tolist(),
                'intervention_predictions': intervention_predictions.tolist(),
                'scenario': scenario_name,
                'scenario_description': scenario_description,
                'impact_metrics': {
                    'absolute_reduction': float(absolute_reduction),
                    'relative_reduction': float(relative_reduction) * 100,  # as percentage
                },
                'analysis_date': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in what-if analysis: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _prepare_prediction_data(historical_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Prepare historical data for prediction modeling.
        
        Args:
            historical_data: Dictionary with historical complexity data
            
        Returns:
            DataFrame prepared for prediction modeling or None if preparation fails
        """
        try:
            # Extract data
            timestamps = historical_data['timestamps']
            overall_scores = historical_data['overall_scores']
            
            # Convert to DataFrame
            data = pd.DataFrame({
                'timestamp': timestamps,
                'overall_score': overall_scores
            })
            
            # Convert timestamps to datetime
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
            
            # Calculate days since first observation
            first_date = data['datetime'].min()
            data['days_since_start'] = (data['datetime'] - first_date).dt.total_seconds() / (24 * 3600)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None