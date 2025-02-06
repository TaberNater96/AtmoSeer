import torch
import torch.nn as nn
from configs.atmoseer_config import ModelConfig, TrainConfig
import numpy as np
import pandas as pd

class AtmoSeer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
    ):
        super().__init__()
        self.model_config = model_config
        
        self.lstm = nn.LSTM(
            input_size=model_config.input_dim,
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            batch_first=model_config.batch_first,
            bidirectional=model_config.bidirectional,
            dropout=model_config.dropout if model_config.num_layers > 1 else 0
        )
        
        lstm_out_dim = model_config.hidden_dim * 2 if model_config.bidirectional else model_config.hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(lstm_out_dim // 2, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # x shape: [batch_size, seq_length, input_dim]
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch_size, seq_length, hidden_dim*2] (if bidirectional)
        
        # Get the last sequence hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Apply attention only to the last sequence
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context_vector)
        return output
    
    def train_model(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        train_config: TrainConfig = TrainConfig()
    ) -> dict:
        self.to(train_config.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=train_config.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        early_stopping_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(train_config.num_epochs):
            self.train()
            total_loss = 0
            optimizer.zero_grad()
            
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(train_config.device)
                y_batch = y_batch.to(train_config.device)
                
                predictions = self(x_batch)
                loss = criterion(predictions, y_batch)
                loss = loss / train_config.gradient_accumulation_steps
                loss.backward()
                
                if (i + 1) % train_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), train_config.gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * train_config.gradient_accumulation_steps
            
            # Validation
            val_loss = self._validate(val_loader, criterion, train_config.device)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= train_config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
            training_history['train_loss'].append(total_loss / len(train_loader))
            training_history['val_loss'].append(val_loss)
            
        self.load_state_dict(best_state)
        return training_history
    
    def _validate(
        self, 
        val_loader: torch.utils.data.DataLoader, 
        criterion: nn.Module, 
        device: torch.device
    ) -> float:
        self.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = self(x_batch)
                loss = criterion(predictions, y_batch)
                total_val_loss += loss.item()
        
        return total_val_loss / len(val_loader)

    def save_model(
        self, 
        path: str
    ):
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config
        }, path)

    @classmethod
    def load_model(
        cls, 
        path: str, 
        device: torch.device = None
    ) -> 'AtmoSeer':
        checkpoint = torch.load(path, map_location=device)
        model = cls(model_config=checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def prepare_prediction_defaults(self, last_known_data: pd.DataFrame):
        """Store default values for prediction"""
        self.last_known_date = pd.to_datetime(last_known_data['date'].max())
        self.last_known_ppm = last_known_data['ppm'].iloc[-365:].values  # Last year of values
        
        # Store average location and biomass values
        self.default_features = {
            'latitude': last_known_data['latitude'].mean(),
            'longitude': last_known_data['longitude'].mean(),
            'altitude': last_known_data['altitude'].mean(),
            'biomass_density': last_known_data['biomass_density'].iloc[-1],  # Use last known value
            'site': last_known_data['site'].mode()[0]  # Most common site
        }

    def _prepare_sequence(self, target_date):
        """Prepare sequence of last sequence_length days for prediction"""
        sequence = []
        current_date = target_date
        
        # Create sequence of features for last sequence_length days
        for i in range(self.model_config.sequence_length):
            features = self._prepare_features(current_date)
            sequence.insert(0, features)  # Insert at beginning to maintain chronological order
            current_date = current_date - pd.Timedelta(days=1)
        
        return np.array(sequence)

    def predict(self, target_date: str, return_confidence: bool = True) -> dict:
        """Predict PPM for a given date"""
        self.eval()
        
        # Parse target date
        target_date = pd.to_datetime(target_date + '/01')
        
        # Get sequence of features leading up to target date
        sequence = self._prepare_sequence(target_date)
        
        with torch.no_grad():
            # Add batch dimension and send to device
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Shape: [1, seq_length, num_features]
            prediction = self(x)
            prediction_value = prediction.item()
        
        result = {'prediction': prediction_value}
        
        if return_confidence:
            # Simple confidence based on time distance
            months_out = abs((target_date - self.last_known_date).days) / 30.44  # Approximate months
            base_uncertainty = 0.02  # 2% base uncertainty
            time_uncertainty = 0.001 * months_out  # 0.1% additional per month
            confidence_range = prediction_value * (base_uncertainty + time_uncertainty)
            
            result.update({
                'confidence_interval': (
                    prediction_value - confidence_range,
                    prediction_value + confidence_range
                )
            })
        
        return result

    def _prepare_features(self, target_date: pd.Timestamp) -> np.ndarray:
        """Prepare feature vector for prediction"""
        # Calculate temporal features
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
            # Use actual historical values for lags if available
            # This would require storing historical data - simplified here
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
        
        # Calculate change rate (simplified)
        features['co2_change_rate'] = (features['ppm_lag_14'] - features['ppm_lag_30']) / 16
        
        # Return features in correct order matching training data
        feature_order = ['site', 'latitude', 'longitude', 'altitude', 'year', 
                        'month', 'season', 'co2_change_rate', 'month_sin', 
                        'month_cos', 'ppm_lag_14', 'ppm_lag_30', 'ppm_lag_365', 
                        'biomass_density']
        
        return np.array([features[col] for col in feature_order])