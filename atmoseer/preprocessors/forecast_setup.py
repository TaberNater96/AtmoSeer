import torch
import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass

@dataclass
class ForecastResult:
    """Container for a single forecast value with confidence bounds."""
    date: date
    ppm: float
    lower_bound: float
    upper_bound: float

class CO2ForecastHelper:
    """Helper class to simplify forecasting with AtmoSeer models."""
    
    CUTOFF_DATE = date(2024, 5, 31)  # Last available measurement
    BASE_UNCERTAINTY = 1.0  # Starting uncertainty in ppm
    UNCERTAINTY_GROWTH = 0.02  # Daily growth rate in uncertainty
    MIN_YEARLY_INCREASE = 2.0  # Minimum yearly CO2 increase in ppm
    
    def __init__(self, model, data: pd.DataFrame):
        """Initialize forecaster with a trained model and its corresponding data."""
        self.model = model
        self.sequence_length = model.model_config.sequence_length
        self.preprocessor = None
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date']).dt.date
        self._initialize_preprocessor()
        self._calculate_baseline_metrics()
    
    def _initialize_preprocessor(self) -> None:
        """Setup the preprocessor with the training data."""
        from atmoseer.preprocessors.atmoseer_preprocessor import AtmoSeerPreprocessor
        self.preprocessor = AtmoSeerPreprocessor()
        prepared_data = self.preprocessor.prepare_data(self.data)
        self.input_dim = prepared_data['input_dim']
    
    def _calculate_baseline_metrics(self) -> None:
        """Calculate baseline metrics from historical data for validation."""
        recent_year = self.data[self.data['date'] > (self.CUTOFF_DATE - timedelta(days=365))]
        
        # Calculate average yearly increase
        year_ago = recent_year.iloc[0]['ppm']
        latest = recent_year.iloc[-1]['ppm']
        self.yearly_increase = latest - year_ago
        
        # Calculate seasonal patterns
        monthly_means = self.data.groupby(self.data['date'].map(lambda x: x.month))['ppm'].mean()
        self.seasonal_amplitude = (monthly_means.max() - monthly_means.min()) / 2
        self.max_month = monthly_means.idxmax()
        self.min_month = monthly_means.idxmin()
    
    def _get_recent_data(self) -> pd.DataFrame:
        """Get the most recent sequence_length days of data for prediction."""
        return self.data.sort_values('date').tail(self.sequence_length).copy()
    
    def _prepare_sequence(self, sequence_data: pd.DataFrame) -> torch.Tensor:
        """Prepare a sequence for forecasting using the preprocessor's components."""
        data = sequence_data.copy()
        
        # Create cyclic features for month
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Encode site
        data['site'] = self.preprocessor.site_encoder.transform(data['site'])
        
        # Get feature columns in correct order
        feature_columns = (
            self.preprocessor.numeric_features +
            self.preprocessor.temporal_features +
            self.preprocessor.lag_features +
            ['site', 'month_sin', 'month_cos']
        )
        
        # Create numpy array and normalize
        sequence = data[feature_columns].values
        
        # Apply normalization using preprocessor's fitted scalers
        numeric_idx = [feature_columns.index(f) for f in self.preprocessor.numeric_features]
        temporal_idx = [feature_columns.index(f) for f in self.preprocessor.temporal_features]
        lag_idx = [feature_columns.index(f) for f in self.preprocessor.lag_features]
        
        if numeric_idx:
            sequence[:, numeric_idx] = self.preprocessor.scaler.transform(sequence[:, numeric_idx])
        if temporal_idx:
            sequence[:, temporal_idx] = self.preprocessor.temporal_scaler.transform(sequence[:, temporal_idx])
        if lag_idx:
            sequence[:, lag_idx] = self.preprocessor.lag_scaler.transform(sequence[:, lag_idx])
        
        return torch.FloatTensor(sequence).unsqueeze(0)
    
    def _update_sequence_features(
        self, 
        sequence: pd.DataFrame, 
        prediction_date: date,
        recent_predictions: List[float]
    ) -> pd.DataFrame:
        """Update sequence features for the next prediction."""
        updated = sequence.copy()
        
        # Shift the sequence window forward
        updated = updated.iloc[1:].copy()
        new_row = updated.iloc[-1:].copy()
        new_row['date'] = prediction_date
        
        # Update temporal features
        new_row['year'] = prediction_date.year
        new_row['month'] = prediction_date.month
        new_row['month_sin'] = np.sin(2 * np.pi * prediction_date.month / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * prediction_date.month / 12)
        
        # Update lag features using actual predictions
        if len(recent_predictions) >= 1:
            new_row['ppm_lag_14'] = recent_predictions[-1]
        if len(recent_predictions) >= 30:
            new_row['ppm_lag_30'] = recent_predictions[-30]
        if len(recent_predictions) >= 365:
            new_row['ppm_lag_365'] = recent_predictions[-365]
        
        # Calculate realistic co2_change_rate
        if len(recent_predictions) >= 2:
            change = recent_predictions[-1] - recent_predictions[-2]
            # Limit extreme changes
            change = np.clip(change, -0.5, 0.5)
            new_row['co2_change_rate'] = change
        
        updated = pd.concat([updated, new_row], ignore_index=True)
        return updated
    
    def _validate_prediction(
        self,
        prediction: float,
        date: date,
        previous_prediction: float = None
    ) -> float:
        """Validate and adjust prediction to maintain realistic patterns."""
        # Calculate expected seasonal component
        days_from_start = (date - self.CUTOFF_DATE).days
        yearly_position = 2 * np.pi * date.month / 12
        seasonal_effect = self.seasonal_amplitude * np.sin(yearly_position)
        
        # Calculate minimum expected value based on yearly increase
        min_increase = (self.MIN_YEARLY_INCREASE / 365) * days_from_start
        min_expected = self.data.iloc[-1]['ppm'] + min_increase + seasonal_effect
        
        # Enforce minimum value and limit day-to-day changes
        if previous_prediction is not None:
            max_daily_change = 0.5  # Maximum allowed daily change in ppm
            prediction = np.clip(
                prediction,
                previous_prediction - max_daily_change,
                previous_prediction + max_daily_change
            )
        
        return max(prediction, min_expected)
    
    def _validate_forecast_date(self, forecast_date: date) -> None:
        """Ensure the forecast date is valid."""
        if not isinstance(forecast_date, date):
            raise ValueError("forecast_date must be a date object")
            
        if forecast_date <= self.CUTOFF_DATE:
            raise ValueError(
                f"Cannot forecast for {forecast_date}. Date must be after {self.CUTOFF_DATE}"
            )
            
        max_forecast = self.CUTOFF_DATE + timedelta(days=365)
        if forecast_date > max_forecast:
            raise ValueError(
                f"Cannot forecast beyond {max_forecast} (1 year from last measurement)"
            )
    
    def predict_for_date(
        self, 
        forecast_date: Union[str, date, datetime]
    ) -> Tuple[ForecastResult, Dict[date, ForecastResult]]:
        """Generate a forecast for a specific date and all dates leading up to it."""
        # Convert to date object if necessary
        if isinstance(forecast_date, str):
            forecast_date = pd.to_datetime(forecast_date).date()
        elif isinstance(forecast_date, datetime):
            forecast_date = forecast_date.date()
            
        # Validate the date
        self._validate_forecast_date(forecast_date)
        
        # Calculate how many days to forecast
        days_to_forecast = (forecast_date - self.CUTOFF_DATE).days
        
        # Get initial sequence and prepare for predictions
        current_sequence_data = self._get_recent_data()
        current_sequence = self._prepare_sequence(current_sequence_data)
        
        # Track predictions and dates
        predictions = []
        dates = []
        current_date = self.CUTOFF_DATE
        
        # Generate predictions one day at a time
        with torch.no_grad():
            for day in range(days_to_forecast):
                # Advance to next date
                current_date += timedelta(days=1)
                dates.append(current_date)
                
                # Make prediction
                pred = self.model(current_sequence)
                pred_value = float(pred.cpu().numpy()[0][0])
                
                # Convert to original scale
                pred_value = float(self.preprocessor.target_scaler.inverse_transform(
                    [[pred_value]]
                )[0][0])
                
                # Validate prediction
                if predictions:
                    pred_value = self._validate_prediction(
                        pred_value,
                        current_date,
                        predictions[-1]
                    )
                else:
                    pred_value = self._validate_prediction(
                        pred_value,
                        current_date
                    )
                
                predictions.append(pred_value)
                
                # Update sequence for next prediction
                if day < days_to_forecast - 1:
                    current_sequence_data = self._update_sequence_features(
                        current_sequence_data,
                        current_date + timedelta(days=1),
                        predictions
                    )
                    current_sequence = self._prepare_sequence(current_sequence_data)
        
        # Calculate results with increasing uncertainty
        all_results = {}
        for i, (d, pred) in enumerate(zip(dates, predictions)):
            # Calculate uncertainty that grows with time
            days_out = i + 1
            uncertainty = self.BASE_UNCERTAINTY + (self.UNCERTAINTY_GROWTH * days_out)
            
            # Add seasonal component to uncertainty
            seasonal_uncertainty = abs(np.sin(2 * np.pi * d.month / 12)) * uncertainty
            total_uncertainty = uncertainty + seasonal_uncertainty
            
            result = ForecastResult(
                date=d,
                ppm=pred,
                lower_bound=pred - (2 * total_uncertainty),
                upper_bound=pred + (2 * total_uncertainty)
            )
            all_results[d] = result
        
        return all_results[forecast_date], all_results