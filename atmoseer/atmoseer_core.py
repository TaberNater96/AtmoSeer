import torch
import torch.nn as nn
import random
from configs.atmoseer_config import ModelConfig, TrainConfig, BayesianTunerConfig
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
from datetime import datetime
from pathlib import Path
import gc
from typing import Dict, Any, Optional, Tuple, Union
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Set random seeds for reproducibility, each package must be individually addressed to lock in randomized settings under the hood
random.seed(10) # standard python
np.random.seed(10) # numpy
torch.manual_seed(10) # torch
if torch.cuda.is_available(): # GPU
    torch.cuda.manual_seed_all(10)

class AtmoSeer(nn.Module):
    """
    A bidirectional LSTM model with attention mechanism for greenhouse gas prediction.
    
    This model is designed specifically for time series forecasting of atmospheric gas concentrations.
    It combines a BiLSTM architecture with an attention mechanism to capture both long-term dependencies
    and seasonal patterns in greenhouse gas measurements. The model can handle multiple input features
    including temporal, spatial, and environmental variables. This model uses a sequence-to-one architecture, 
    where it takes a sequence of historical measurements and predicts a single future value. This design choice 
    was made because atmospheric gas concentrations show strong autocorrelation and seasonal patterns that
    benefit from looking at extended historical contexts.
    """
    def __init__(
        self: "AtmoSeer",
        model_config: ModelConfig = ModelConfig(),
    ) -> None:
        """
        Initialize AtmoSeer with the specified configuration.
        
        Args:
            model_config (ModelConfig): Configuration dataclass containing model hyperparameters.
                                        If not provided, uses default values from ModelConfig.
        """
        # Initialize parent nn.Module class to enable PyTorch functionality
        super().__init__()
        self.model_config = model_config        
        self.input_norm = nn.LayerNorm(model_config.input_dim)
        
        self.lstm = nn.LSTM(
            input_size=model_config.input_dim,  # Use input_dim directly
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            batch_first=model_config.batch_first,
            bidirectional=model_config.bidirectional,
            dropout=model_config.dropout if model_config.num_layers > 1 else 0
        )
        
        # Calculate output dimension of LSTM, double if bidirectional
        lstm_out_dim = model_config.hidden_dim * 2 if model_config.bidirectional else model_config.hidden_dim
        
        # Attention mechanism to weight different time steps, uses a two-layer neural network to compute attention scores
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.LayerNorm(lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1)
        )
        
        # Final fully connected layers for prediction, gradually reduce dimensions to single output value
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.LayerNorm(lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(lstm_out_dim // 2, lstm_out_dim // 4),
            nn.LayerNorm(lstm_out_dim // 4),
            nn.ReLU(),
            nn.Linear(lstm_out_dim // 4, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better gradient flow."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Use orthogonal initialization for LSTM weights
                    nn.init.orthogonal_(param)
                elif len(param.shape) > 1:
                    # Use Xavier/Glorot for other weight matrices
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize bias terms with zeros
                nn.init.zeros_(param)
        
    def forward(
        self: "AtmoSeer",
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform forward pass through AtmoSeer.
                
        This method implements the core prediction logic using a bidirectional LSTM with attention. The input sequence is 
        first processed through bidirectional LSTM layers to capture temporal patterns, then passed through an attention 
        mechanism that learns to weight different time steps. The attention mechanism allows the model to focus on relevant 
        parts of the input sequence when making predictions, particularly useful for capturing seasonal patterns and important 
        historical events in the time series. These weighted states are combined into a context vector that captures the most 
        relevant temporal information, which is finally passed through fully connected layers to generate the prediction.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
                            containing the feature sequences for prediction.
                            - batch_size: Number of samples in batch
                            - sequence_length: Number of time steps (specified in model_config)
                            - input_dim: Number of input features (specified in model_config)
        
        Returns:
            torch.Tensor: Predicted values tensor of shape (batch_size, 1)
                          representing the predicted gas concentration for each sample.
        """
        # Apply input normalization
        x = self.input_norm(x)
        
        # x shape: [batch_size, seq_length, input_dim]
        lstm_out, _ = self.lstm(x) # shape: [batch_size, seq_length, hidden_dim*2] (if bidirectional)
        
        # Calculate attention scores for each time step and apply to only the last sequence
        attention_weights = self.attention(lstm_out)
        
        # Normalize attention weights using softmax to ensure weights sum to 1 for each sequence
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Create context vector by weighted sum of LSTM outputs, the multiplication broadcasts the weights across the hidden dimensions
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) # higher attention weights will have more influence
        
        # Generate final prediction through fully connected layers
        output = self.fc(context_vector) # shape: [batch_size, 1]
        return output
    
    def train_model(
        self: "AtmoSeer",
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_config: TrainConfig = TrainConfig()
    ) -> dict[str, list[float]]:
        """
        Train AtmoSeer using the data loaders.

        This method implements a robust training loop that uses gradient accumulation to handle large batch sizes, 
        allowing for training by accumulating gradients over multiple forward passes before updating model parameters. 
        The training process includes early stopping to prevent overfitting and gradient clipping to handle exploding 
        gradients, a common consequence in RNNs. Throughout training, the process monitors validation loss and maintains 
        checkpoints of the best model state, ensuring optimal model selection based on validation performance.
        
        Args:
            train_loader (DataLoader): PyTorch DataLoader containing training data.
                                       Expected to yield tuples of (features, targets).
            val_loader (DataLoader): PyTorch DataLoader containing validation data.
                                     Used for early stopping and model selection.
            train_config (TrainConfig): Configuration object containing training parameters.
                                        Uses default values if not provided.

        Returns:
            dict[str, list[float]]: Dictionary containing training history with keys:
                - 'train_loss': List of training losses for each epoch
                - 'val_loss': List of validation losses for each epoch
        """
        self.to(train_config.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=train_config.learning_rate)
        criterion = nn.MSELoss()
        
        # Track best validation loss for model checkpointing
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0
        early_stopping_counter = 0
        min_delta = train_config.min_delta         # minimum change in validation loss to qualify as improvement
        training_history = {'train_loss': [], 'val_loss': []}
        best_state = None
        
        # Main training loop over epochs
        for epoch in range(train_config.num_epochs):
            self.train()
            total_loss = 0
            optimizer.zero_grad() # reset the gradients at the start of each epoch
            
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(train_config.device)
                y_batch = y_batch.to(train_config.device)
                
                # Forward pass and loss calculation
                predictions = self(x_batch)
                loss = criterion(predictions, y_batch)
                
                # Scale loss for gradient accumulation, this increases batch size without memory overhead
                loss = loss / train_config.gradient_accumulation_steps
                
                # Backward pass to compute gradients
                loss.backward()
                
                # Update weights only after accumulating gradients
                if (i + 1) % train_config.gradient_accumulation_steps == 0:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters(), train_config.gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad() # reset gradients
                
                total_loss += loss.item() * train_config.gradient_accumulation_steps
            
            avg_train_loss = total_loss / len(train_loader)
            val_loss = self._validate(val_loader, criterion, train_config.device)
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_loss)
            
            # Early stopping and model checkpoint logic
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                best_train_loss = avg_train_loss
                best_epoch = epoch
                best_state = self.state_dict()
                early_stopping_counter = 0
                print(f"Epoch: {epoch} \nNew best validation loss: {val_loss:.6f} for current trial")
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= train_config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with validation loss {best_val_loss:.6f}")
                break
            
        # Only try to load best_state if we found one
        if best_state is not None:
            self.load_state_dict(best_state)
        else:
            print("Warning: No best state was found during training")
        
        # Return extended history including best epoch information
        return {
            'train_loss': training_history['train_loss'],
            'val_loss': training_history['val_loss'],
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': best_train_loss
        }
    
    def _validate(
        self: "AtmoSeer",
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> float:
        """
        Evaluate the model's performance on the validation dataset. 
        
        This method computes the average loss across all batches in the validation set, providing a metric 
        for model performance on unseen data. The evaluation is performed with gradient computation disabled 
        to save memory and computation time during validation.

        Args:
            val_loader (DataLoader): PyTorch DataLoader containing validation data,
                                     expected to yield tuples of (features, targets).
            criterion (nn.Module): Loss function module used to compute validation loss,
                                   typically the same criterion used in training.
            device (torch.device): Device (CPU/GPU) where the computation will be performed.

        Returns:
            float: Average validation loss across all batches in the validation set.
        """
        # Set model to evaluation mode - disables dropout and batch normalization
        self.eval()
        total_val_loss = 0
        
        # Disable gradient computation for validation to reduce memory usage and speed up validation
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = self(x_batch)
                loss = criterion(predictions, y_batch)
                total_val_loss += loss.item()
        
        # Return average loss across all batches
        return total_val_loss / len(val_loader)

    def save_model(
        self: "AtmoSeer",
        path: str
    ) -> None:
        """
        Save the model's state and configuration to disk. 

        Args:
            path (str): File path where the model should be saved. 
        """
        torch.save({
            'model_state_dict': self.state_dict(), # learned parameters
            'model_config': self.model_config      # architecture configuration
        }, path)

    @classmethod
    def load_model(
        cls,
        path: str,
        device: torch.device | None = None
    ) -> "AtmoSeer":
        """
        Load a saved AtmoSeer model from disk. 

        Args:
            path (str): Path to the saved model file created by save_model().
            device (torch.device | None): Device where the model should be loaded. If None, the model 
                                          will be loaded to the device it was saved from.

        Returns:
            AtmoSeer: A reconstructed model instance with restored parameters
                      and configuration, ready for use.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(model_config=checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _update_prediction_history(
        self, 
        target_date: pd.Timestamp, 
        prediction_value: float
    ) -> None:
        """
        Keeps track of the model's predictions over time, building a historical record that helps with future forecasting. When predicting 
        greenhouse gas concentrations, the model often need to reference recent predictions to maintain consistency in forecasts. This 
        method adds each new prediction to the model's memory, allowing it to build upon its own previous predictions.

        Args:
            target_date (pd.Timestamp): The date we're predicting for, which becomes our reference point in the prediction history.
            prediction_value (float): The predicted gas concentration for this date.
        """
        if not hasattr(self, 'prediction_history'):
            self.prediction_history = {}
        
        self.prediction_history[target_date] = prediction_value
        
    def _get_lag_value(
        self, 
        target_date: pd.Timestamp, 
        lag_days: int
    ) -> float:
        """
        Retrieves the gas concentration value for a specified number of days before the target date.
        
        This method implements a hierarchical lookup strategy to obtain historical or predicted values:
        1. Searches for exact matches in the prediction history
        2. Retrieves values from historical data if the date precedes the last known date
        3. Locates nearest predicted values for future dates
        4. Computes trend-based extrapolation when no direct values are available
        
        Args:
            target_date (pd.Timestamp): Reference date from which to calculate the lag
            lag_days (int): Number of days to look backward from the target date
            
        Returns:
            float: Gas concentration value for the lagged date, sourced from either historical data,
                   previous predictions, or trend-based estimates
        """
        lag_date = target_date - pd.Timedelta(days=lag_days)
        
        # First check prediction history for the exact date
        if hasattr(self, 'prediction_history') and lag_date in self.prediction_history:
            return self.prediction_history[lag_date]
        
        # If the lag date is in or before our historical data
        if lag_date <= self.last_known_date:
            days_from_end = (self.last_known_date - lag_date).days
            if days_from_end < len(self.last_known_ppm):
                return self.last_known_ppm[-days_from_end-1]
            return self.last_known_ppm[0]
        
        # For future dates without exact matches, find the closest predicted date
        if hasattr(self, 'prediction_history'):
            closest_date = None
            min_diff = float('inf')
            
            for pred_date in self.prediction_history.keys():
                diff = abs((pred_date - lag_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_date = pred_date
            
            if closest_date is not None:
                return self.prediction_history[closest_date]
        
        # If no suitable prediction found, use trend-adjusted last known value
        days_forward = (lag_date - self.last_known_date).days
        if days_forward > 0:
            # Calculate trend from last year of historical data
            last_year = self.last_known_ppm[-365:]
            daily_trend = (last_year[-1] - last_year[0]) / 365
            trend_adjustment = daily_trend * days_forward
            return self.last_known_ppm[-1] + trend_adjustment
        
        return self.last_known_ppm[-1]

    def prepare_prediction_defaults(
        self: "AtmoSeer",
        last_known_data: pd.DataFrame
    ) -> None:
        """
        Initialize default values for future predictions based on the most recent historical data. This method stores 
        essential information about the last known state of the system, including temporal patterns and location-specific 
        features. These stored values are used as reference points when making predictions beyond the last known date, 
        ensuring continuity in the prediction process and handling cases where actual historical data is not available.

        Args:
            last_known_data (pd.DataFrame): DataFrame containing the most recent historical data.

        Note:
            This method must be called before making predictions to ensure the model has appropriate reference 
            values for location features and recent measurements.
        """
        df = last_known_data.copy()
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Store the most recent date and ppm from the historical data, used as reference point for generating future dates
        self.last_known_date = df.index.max()
        self.last_known_ppm = df['ppm'].iloc[-365:].values
        
        self.daily_trend = (self.last_known_ppm[-1] - self.last_known_ppm[0]) / 365
        self.seasonal_patterns = {}
        
        # Calculate monthly seasonal patterns
        monthly_means = df.resample('M')['ppm'].mean()
        
        yearly_cycle = monthly_means.groupby(monthly_means.index.month).mean()
        yearly_mean = yearly_cycle.mean()
        self.seasonal_patterns = (yearly_cycle - yearly_mean).to_dict()
        
        site_encoder = LabelEncoder()
        encoded_site = site_encoder.fit_transform([df['site'].iloc[0]])[0]
        
        self.default_features = {
            'latitude': float(df['latitude'].mean()),
            'longitude': float(df['longitude'].mean()),
            'altitude': float(df['altitude'].mean()),
            'biomass_density': float(df['biomass_density'].iloc[-1]),
            'site': float(encoded_site)
        }

    def _prepare_features(
        self: "AtmoSeer",
        target_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Generate a complete feature vector for a specific date. 
        
        This method combines temporal features, spatial characteristics, and historical measurements to create the input 
        vector required by the model. The feature generation process handles both historical dates (where actual lag values 
        are available) and future dates (where predicted values must be used for lags).
        
        Args:
            target_date (pd.Timestamp): The date for which to generate features. Can be historical or future date.

        Returns:
            np.ndarray: Array of features in the order expected by the model: [site, latitude, longitude, altitude, year, month, 
                        season, co2_change_rate, month_sin, month_cos, ppm_lag_14, ppm_lag_30, ppm_lag_365, biomass_density]
        """
        # Calculate temporal features, month is encoded both as raw value and cyclical components
        month = target_date.month
        features = {
            'year': float(target_date.year),
            'month_sin': float(np.sin(2 * np.pi * month / 12)),
            'month_cos': float(np.cos(2 * np.pi * month / 12))
        }
        
        features.update(self.default_features)
        
        ppm_lag_14 = self._get_lag_value(target_date, 14)
        ppm_lag_30 = self._get_lag_value(target_date, 30)
        ppm_lag_365 = self._get_lag_value(target_date, 365)
        
        seasonal_factor = self.seasonal_patterns.get(month, 0)
        
        features.update({
            'ppm_lag_14': ppm_lag_14 + seasonal_factor,
            'ppm_lag_30': ppm_lag_30 + seasonal_factor,
            'ppm_lag_365': ppm_lag_365 + seasonal_factor
        })
        
        # Calculate rate of change using available lag values, uses shorter time window to capture recent trends
        features['co2_change_rate'] = float((ppm_lag_14 - ppm_lag_30) / 16)
        
        feature_order = ['site', 'latitude', 'longitude', 'altitude', 'year', 'co2_change_rate', 
                        'month_sin', 'month_cos', 'ppm_lag_14', 'ppm_lag_30', 'ppm_lag_365', 'biomass_density']
        
        return np.array([features[col] for col in feature_order], dtype=np.float32)

    def _prepare_sequence(
        self: "AtmoSeer",
        target_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Generate a sequence of feature vectors leading up to the target prediction date. This method creates a 
        chronological sequence of features by working backwards from the target date, generating feature vectors 
        for each time step in the sequence. The method constructs a sequence of days, where each day's feature vector
        includes temporal features (like month, year, seasonal indicators), spatial features (latitude, longitude, altitude),
        and lagged values from historical measurements. 
        
        Args:
            target_date (pd.Timestamp): The date for which we want to generate a prediction. The sequence will be built
                                        working backwards from this date.

        Returns:
            np.ndarray: Array of shape (sequence_length, input_dim) containing the feature vectors for each time step in 
                        the sequence. The sequence is ordered chronologically with the oldest time step first.
        """
        sequence = []
        
        # Start from target date and work backwards
        current_date = target_date
        
        # Generate features for each time step in the sequence and insert at the beginning to maintain chronological order
        for _ in range(self.model_config.sequence_length):
            features = self._prepare_features(current_date)
            sequence.insert(0, features)  # insert at the start to so that the oldest data is first in the sequence
            current_date = current_date - pd.Timedelta(days=1) # move back one day for next iteration
        
        return np.array(sequence, dtype=np.float32)

    def predict(
        self,
        target_date: str,
        test_loader: torch.utils.data.DataLoader,
        target_scaler: StandardScaler,
        return_confidence: bool = True
    ) -> dict[str, Union[float, tuple[float, float], dict[str, float]]]:
        """
        Generate predictions for greenhouse gas concentrations at a specified future date. This method combines the 
        model's prediction capabilities with uncertainty estimation, providing both a point prediction and confidence 
        intervals. The prediction process involves generating a sequence of features leading up to the target date, then
        using the trained model to forecast the gas concentration. The uncertainty estimation preserves temporal ordering 
        by using the temporal test split, which is crucial for time series forecasting. The error calculation considers the 
        natural temporal dependencies in the data.

        Args:
            target_date (str): Date for which to generate prediction in format 'YYYY/MM'. The day component is automatically 
                               set to the first of the month.
            test_loader (DataLoader): PyTorch DataLoader containing test data that was temporally split (last 10% of time 
                                      series) for unbiased error calculation.
            target_scaler (StandardScaler): Scaler used to transform target values during training.
                                            Used to convert predictions back to original scale.
            return_confidence (bool): If True, includes confidence intervals in the output. Default is True.

        Returns:
            dict[str, float | tuple[float, float]]: Dictionary containing:
                - 'prediction': Predicted gas concentration value
                - 'confidence_interval': Tuple of (lower_bound, upper_bound) Only included if return_confidence is True
        """
        self.eval()
        self.train(False)    # double check the model is not training
        self.device = next(self.parameters()).device
        
        try:
            if '/' in target_date:
                month, year = target_date.split('/')
                month, year = int(month), int(year)
            else:
                raise ValueError("Date must be in MM/YYYY format")
            target_date = pd.Timestamp(year=year, month=month, day=1)
        
            # Generate sequence of features leading up to target date
            sequence = self._prepare_sequence(target_date)
            
            # Disable gradient computation for prediction
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                prediction = self(x)
                
                # Convert prediction back to original scale
                prediction_value = target_scaler.inverse_transform(
                    prediction.cpu().numpy().reshape(-1, 1)
                )[0][0]
                
                self._update_prediction_history(target_date, prediction_value)
            
            result = {'prediction': prediction_value}
            
            # Calculate confidence intervals if requested
            if return_confidence:
                # Calculate errors on test set to maintain temporal structure
                temporal_errors = []
                sequential_errors = []  # track errors in sequence for trend analysis
                
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.device)
                        test_preds = self(X_batch)
                        
                        # Convert predictions and actual values back to original scale
                        test_preds_orig = target_scaler.inverse_transform(
                            test_preds.cpu().numpy().reshape(-1, 1)
                        ).reshape(-1)
                        y_batch_orig = target_scaler.inverse_transform(
                            y_batch.numpy().reshape(-1, 1)
                        ).reshape(-1)
                        
                        # Calculate errors in original scale
                        errors = np.abs(test_preds_orig - y_batch_orig)
                        temporal_errors.extend(errors)
                        sequential_errors.append(errors.mean())
                
                # Calculate base error margin from test set
                error_margin = np.percentile(temporal_errors, 95)
                
                # Analyze error trend in temporal sequence
                error_trend = np.polyfit(range(len(sequential_errors)), sequential_errors, 1)[0]
                
                # Calculate prediction horizon in months
                months_out = abs((target_date - self.last_known_date).days) / 30.44
                
                # Adjust error margin based on observed error trend and horizon
                trend_scaling = max(1.0, 1.0 + (error_trend * months_out))
                final_error_margin = error_margin * trend_scaling
                
                result['confidence_interval'] = (
                    prediction_value - final_error_margin,
                    prediction_value + final_error_margin
                )
                
                result['error_metadata'] = {
                    'base_error_margin': error_margin,
                    'trend_scaling': trend_scaling,
                    'prediction_horizon_months': months_out
                }
            
            return result
        
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
    
class BayesianTuner:
    """
    This class implements a Bayesian optimization approach to find optimal hyperparameters for the AtmoSeer model. It uses Gaussian 
    Process regression to model the relationship between hyperparameters and model performance, guided by an acquisition function to
    balance exploration and exploitation in the search space. The process is designed to be resilient to training failures and memory 
    constraints while maintaining detailed logs for analysis and reproducibility.
    
    Main Components:
    - Automated trial management and logging
    - Memory-aware resource handling
    - Persistent storage of optimization results
    - Recovery capability from interrupted optimization
    - Configurable cleanup policies for trial artifacts
    """
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: BayesianTunerConfig
    ) -> None:
        """        
        This method sets up the optimization environment, initializes tracking metrics, and makes sure the logging directory structure 
        exists. It can resume interrupted optimization runs by loading existing logs.
        
        Args:
            train_loader (DataLoader): PyTorch DataLoader containing training data. Used to evaluate each trial's hyperparameter set.
            val_loader (DataLoader): PyTorch DataLoader containing validation data. Used to compute the optimization objective.
            config (BayesianTunerConfig): Configuration object containing optimization parameters, directory paths, and resource limits.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.best_val_loss = float('inf')
        self.best_trial_id = None
        self.current_trial = 0
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """
        Initialize the logging system for tracking optimization progress.
        
        This method establishes a JSON-based logging system that maintains a complete record of all optimization trials. It creates or 
        loads the optimization log file, which will allow the tuning process to resume from previous runs.
        
        This method performs two main tasks:
        1. Creates/accesses a JSON log file for storing optimization results
        2. Determines the current trial number by counting existing trials
        
        The logging setup supports:
        - Persistent storage of optimization history
        - Recovery from interrupted optimization runs
        - Structured tracking of trial results
        """
        self.log_path = self.config.gas_dir / 'optimization_results.json'
        self.logger = JSONLogger(path=str(self.log_path))
        
        # If log file exists, count previous trials to resume optimization
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                log_data = json.load(f)
                self.current_trial = len(log_data['trials']) # set current trial to continue from last recorded trial
    
    def _cleanup_memory(self) -> None:
        """
        Release GPU and system memory between optimization trials.
        
        This method performs memory cleanup to prevent memory leaks and OOM (Out Of Memory) errors during long optimization runs. It's particularly 
        important when running multiple trials with large models or datasets, as PyTorch can retain memory allocations between trials even after tensors 
        are no longer needed. This is a preventive measure against memory fragmentation, which can occur even when total memory usage appears acceptable. 
        
        The cleanup process has two stages:
        1. GPU memory clearance - forces immediate release of all cached GPU tensors
        2. Python garbage collection - ensures unused Python objects are properly deallocated
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _save_trial(
        self,
        trial_id: int,
        model: AtmoSeer,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Persist optimization trial results and model artifacts to disk.
        
        The saving strategy follows a two-tier approach:
        1. Every trial gets its metrics and parameters saved for analysis and debugging
        2. Best-performing models get additional artifacts saved to enable model recovery
        
        Args:
            trial_id (int): Unique identifier for the trial, used in directory naming
            model (AtmoSeer): The trained model instance to be saved if it's the best
            params (Dict[str, Any]): Hyperparameters used in this trial
            metrics (Dict[str, float]): Performance metrics from the trial
            is_best (bool): Flag indicating if this trial achieved best performance. Defaults to False.
        """
        trial_dir = self.config.trials_dir / f'trial_{trial_id:03d}'
        trial_dir.mkdir(exist_ok=True)
        
        # Save trial metrics and parameters
        trial_info = {
            'trial_id': trial_id,
            'parameters': params,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(trial_dir / 'metrics.json', 'w') as f:
            json.dump(trial_info, f, indent=4)
        
        # Save model if it's among the best
        if is_best:
            model.save_model(trial_dir / 'model.pth')
            
            # Update best model directory with current best performer
            model.save_model(self.config.best_model_dir / 'model.pth')
            with open(self.config.best_model_dir / 'config.json', 'w') as f:
                json.dump(trial_info, f, indent=4)
    
    def _cleanup_old_trials(self) -> None:
        """
        Remove artifacts from suboptimal trials to manage disk space.
        
        This method implements a selective cleanup strategy that preserves only the artifacts from the best-performing trial while removing others. It's 
        designed to prevent disk space exhaustion during long optimization runs while retaining essential information for model deployment and analysis. 
        This preserves the best trial's artifacts for model recovery while removing all other trail directories and their contents.
        """
        if not self.config.cleanup_trials:
            return
            
        # Keep only the best trial
        for trial_dir in self.config.trials_dir.glob('trial_*'):
            trial_id = int(trial_dir.name.split('_')[1])
            # Only keep the best trial, remove all others to conserve disk space
            if trial_id != self.best_trial_id:
                if trial_dir.exists():
                    # Remove all files and then the directory itself
                    for file in trial_dir.glob('*'):
                        file.unlink()
                    trial_dir.rmdir()
    
    def _objective(self, **params) -> float:
        """
        Evaluates AtmoSeer model performance for a given hyperparameter configuration.
        
        The trial evaluation process:
        
        1. Parameter Mapping:
            - Converts continuous Bayesian optimization parameters to appropriate model types
            - Configures both model architecture and training parameters simultaneously
            
        2. Model Evaluation:
            - Initializes AtmoSeer with current hyperparameter set
            - Executes training cycle with current configuration
            - Tracks validation performance for optimization guidance
            
        3. Resource Management:
            - Saves trial artifacts for successful configurations
            - Implements periodic cleanup of suboptimal trials
            - Handles trial failures without disrupting optimization
        
        Args:
            **params: Hyperparameters to evaluate:
                - hidden_dim: LSTM hidden layer dimension
                - num_layers: Number of LSTM layers
                - dropout: Dropout rate for regularization
                - sequence_length: Input sequence length 
                - batch_size: Training batch size 
                - learning_rate: Model optimization rate
                
        Returns:
            float: Negative validation loss for maximization in Bayesian optimization.
                   Returns -inf for failed trials to avoid those parameter regions.
        """
        self.current_trial += 1
        print(f"\nTrial {self.current_trial}/{self.config.n_trials}")
        
        # Convert parameters to appropriate types
        model_params = {
            'hidden_dim': int(params['hidden_dim']),
            'num_layers': int(params['num_layers']),
            'dropout': params['dropout'],
            'sequence_length': int(params['sequence_length'])
        }
        
        train_params = {
            'batch_size': int(params['batch_size']),
            'learning_rate': params['learning_rate']
        }
        
        model_config = ModelConfig(**model_params)
        train_config = TrainConfig(**train_params)        
        model = AtmoSeer(model_config=model_config)
        
        try:
            # Execute training pipeline and capture performance metrics
            history = model.train_model(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                train_config=train_config
            )
            
            val_loss = history['best_val_loss']
            
            # Update best model tracking if current trial shows improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_trial_id = self.current_trial
            
            self._save_trial(
                trial_id=self.current_trial,
                model=model,
                params={**model_params, **train_params},
                metrics=history,
                is_best=is_best
            )
            
            # Release GPU memory between trials
            self._cleanup_memory()
            if self.current_trial % 5 == 0:   # periodic cleanup of older, suboptimal trials
                self._cleanup_old_trials()
            
            return -val_loss  # negative because we want to maximize
            
        except Exception as e:
            print(f"Trial {self.current_trial} failed: {str(e)}")
            self._cleanup_memory()
            return float('-inf')  # return worst possible score on failed configurations
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Execute the Bayesian optimization process to find optimal hyperparameters for the AtmoSeer model.
        
        This method implements a sophisticated hyperparameter optimization strategy using Gaussian Process regression to model the relationship between 
        hyperparameters and model performance. The optimization process follows a specific sequence:
        
        1. Initialization Phase:
        - Starts with random exploration to build initial understanding of parameter space
        - Uses 5 initial random trials to establish baseline performance metrics
        - Creates foundation for Gaussian Process modeling
        
        2. Bayesian Optimization Phase:
        - Applies Expected Improvement (EI) acquisition function to balance exploration/exploitation
        - Updates probability model after each trial to refine search strategy
        - Adapts search based on observed performance patterns
        
        3. Resource Management:
        - Uses trial logging for tracking optimization progress
        - Activates recovery from interrupted optimization runs
        - Manages computational resources through periodic cleanup
        
        Returns:
            Tuple[Dict[str, Any], float]: A tuple containing:
                - Best hyperparameter configuration found during optimization
                - Corresponding validation loss (lower is better)
        """
        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=self.config.param_bounds,
            random_state=self.config.random_state
        )
        
        # Set up logging to track optimization progress and enable run recovery
        optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        
        default_params = {
            'hidden_dim': 256,          # from ModelConfig
            'num_layers': 2,            # from ModelConfig
            'dropout': 0.25,            # from ModelConfig
            'sequence_length': 30,      # from ModelConfig
            'learning_rate': 1e-4,      # from TrainConfig
            'batch_size': 64            # from TrainConfig
        }
    
        # Register the default configuration as the first point
        optimizer.probe(params=default_params, lazy=True)
        
        optimizer.maximize(
            init_points=1,     # 1 more random points (total 2 with default) within parameter bounds
            n_iter=48          # remaining trials use Bayesian optimization for a total of 50
        )
        
        self._cleanup_old_trials()
        
        # The target is negated here because the optimizer maximizes when it needs to minimize loss
        return optimizer.max['params'], -optimizer.max['target'] 
    
    @staticmethod
    def load_best_model(
        gas_type: str,
        models_dir: Path=Path('../atmoseer/models'),
        device: Optional[torch.device] = None
    ) -> AtmoSeer:
        """
        Load the best performing model from a completed optimization process.
        
        Args:
            gas_type (str): Type of greenhouse gas the model was trained for ('co2', 'ch4', 'n2o', 'sf6').
            models_dir (Path): Base directory containing all model artifacts. Expected to have subdirectories for each gas type.
                               Defaults to '../atmoseer/models'.
            device (Optional[torch.device]): Device where the model should be loaded. If None, uses the device from the saved state.
                                             Defaults to None.
        
        Returns:
            AtmoSeer: A fully initialized model instance with the best performing configuration and parameters from the optimization process.
        
        Note:
            The method expects the following directory structure:
            models_dir/
            └── gas_type/
                └── best_model/
                    └── model.pth
        """
        best_model_path = Path(models_dir) / gas_type / 'best_model' / 'model.pth'
        return AtmoSeer.load_model(str(best_model_path), device) # this will restore both model architecture and learned parameters
    
    @staticmethod
    def make_predictions(model, dates_to_predict, test_loader, target_scaler):
        """
        Make predictions for a list of dates with error handling.
        
        Args:
            model: Trained AtmoSeer model
            dates_to_predict: List of dates in MM/YYYY format
            test_loader: PyTorch DataLoader for test data
            target_scaler: StandardScaler for target variable
        """
        model.eval()
        model.train(False)
        parsed_dates = []
        for date in dates_to_predict:
            if '/' in date:
                month, year = map(int, date.split('/'))
                parsed_dates.append((pd.Timestamp(year=year, month=month, day=1), date))
        
        sorted_dates = [date[1] for date in sorted(parsed_dates)]
        
        for date in sorted_dates:
            try:
                prediction = model.predict(
                    target_date=date,
                    test_loader=test_loader,
                    target_scaler=target_scaler,
                    return_confidence=True
                )
                
                print(f"\nPrediction for {date}:")
                print(f"PPM: {prediction['prediction']:.2f}")
                print(f"Confidence Interval: ({prediction['confidence_interval'][0]:.2f}, "
                    f"{prediction['confidence_interval'][1]:.2f})")
                
                print("\nPrediction Metadata:")
                print(f"Base Error Margin: {prediction['error_metadata']['base_error_margin']:.2f}")
                print(f"Trend Scaling: {prediction['error_metadata']['trend_scaling']:.2f}")
                print(f"Prediction Horizon (months): {prediction['error_metadata']['prediction_horizon_months']:.1f}")
                
            except Exception as e:
                print(f"\nError predicting for {date}: {str(e)}")
                continue