import torch
import torch.nn as nn
from configs.atmoseer_config import ModelConfig, TrainConfig
import numpy as np
import pandas as pd

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
        
        self.lstm = nn.LSTM(
            input_size=model_config.input_dim,
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            batch_first=model_config.batch_first,
            bidirectional=model_config.bidirectional,
            dropout=model_config.dropout if model_config.num_layers > 1 else 0 # only apply dropout between LSTM layers, not for single layer
        )
        
        # Calculate output dimension of LSTM, double if bidirectional
        lstm_out_dim = model_config.hidden_dim * 2 if model_config.bidirectional else model_config.hidden_dim
        
        # Attention mechanism to weight different time steps, uses a two-layer neural network to compute attention scores
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1)
        )
        
        # Final fully connected layers for prediction, gradually reduce dimensions to single output value
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(lstm_out_dim // 2, 1)
        )
        
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
        early_stopping_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
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
            
            val_loss = self._validate(val_loader, criterion, train_config.device)
            
            # Early stopping and model checkpoint logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict() # save the best model
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= train_config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
            training_history['train_loss'].append(total_loss / len(train_loader))
            training_history['val_loss'].append(val_loss)
            
        # Restore best model state after training loop  
        self.load_state_dict(best_state)
        return training_history
    
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
        # Store the most recent date and ppm from the historical data, used as reference point for generating future dates
        self.last_known_date = pd.to_datetime(last_known_data['date'].max())
        self.last_known_ppm = last_known_data['ppm'].iloc[-365:].values  # last year of values
        
        # Store average location and biomass values
        self.default_features = {
            'latitude': last_known_data['latitude'].mean(),
            'longitude': last_known_data['longitude'].mean(),
            'altitude': last_known_data['altitude'].mean(),
            'biomass_density': last_known_data['biomass_density'].iloc[-1],  # use last known value
            'site': last_known_data['site'].mode()[0]                        # most common site
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
        features = {
            'year': target_date.year,
            'month': target_date.month,
            'month_sin': np.sin(2 * np.pi * target_date.month / 12),
            'month_cos': np.cos(2 * np.pi * target_date.month / 12),
            'season': (target_date.month % 12 + 3) // 3
        }
        
        # Add default location features
        features.update(self.default_features)
        
        # Handle lag features
        if target_date <= self.last_known_date:
            features.update({
                'ppm_lag_14': self.last_known_ppm[-14],
                'ppm_lag_30': self.last_known_ppm[-30],
                'ppm_lag_365': self.last_known_ppm[-365]
            })
        else:
            # Use most recent predictions for lags
            features.update({
                'ppm_lag_14': self.last_known_ppm[-1],
                'ppm_lag_30': self.last_known_ppm[-1],
                'ppm_lag_365': self.last_known_ppm[-1]
            })
        
        # Calculate rate of change using available lag values, uses shorter time window to capture recent trends
        features['co2_change_rate'] = (features['ppm_lag_14'] - features['ppm_lag_30']) / 16
        
        feature_order = ['site', 'latitude', 'longitude', 'altitude', 'year', 'month', 'season', 'co2_change_rate', 
                         'month_sin', 'month_cos', 'ppm_lag_14', 'ppm_lag_30', 'ppm_lag_365', 'biomass_density']
        
        return np.array([features[col] for col in feature_order])

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
        for i in range(self.model_config.sequence_length):
            features = self._prepare_features(current_date)
            sequence.insert(0, features)  # insert at the start to so that the oldest data is first in the sequence
            current_date = current_date - pd.Timedelta(days=1) # move back one day for next iteration
        
        return np.array(sequence)

    def predict(
        self: "AtmoSeer",
        target_date: str,
        test_loader: torch.utils.data.DataLoader,
        return_confidence: bool = True
    ) -> dict[str, float | tuple[float, float]]:
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
            return_confidence (bool): If True, includes confidence intervals in the output. Default is True.

        Returns:
            dict[str, float | tuple[float, float]]: Dictionary containing:
                - 'prediction': Predicted gas concentration value
                - 'confidence_interval': Tuple of (lower_bound, upper_bound) Only included if return_confidence is True
        """
        self.eval()
        
        # Convert string date to timestamp, setting day to first of month
        target_date = pd.to_datetime(target_date + '/01')
        
        # Generate sequence of features leading up to target date
        sequence = self._prepare_sequence(target_date)
        
        # Disable gradient computation for prediction
        with torch.no_grad():
            # Add batch dimension and move to GPU
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)            
            prediction = self(x)
            prediction_value = prediction.item()
        
        result = {'prediction': prediction_value}
        
        # Calculate confidence intervals if requested
        if return_confidence:
            # Calculate errors on test set to maintain temporal structure
            temporal_errors = []
            sequential_errors = []  # track errors in sequence for trend analysis
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)                    
                    test_preds = self(X_batch)
                    
                    # Calculate absolute errors while preserving temporal order
                    errors = torch.abs(test_preds - y_batch)
                    temporal_errors.extend(errors.cpu().numpy())
                    sequential_errors.append(errors.mean().item())
            
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