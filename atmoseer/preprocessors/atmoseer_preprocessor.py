import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Union

class AtmoSeerPreprocessor:
    def __init__(self) -> None:
        # Initialize all scalers
        self.scaler = StandardScaler()  # for numeric features
        self.temporal_scaler = StandardScaler()  # for temporal features
        self.lag_scaler = StandardScaler()  # for lag features
        self.target_scaler = StandardScaler()  # for target variable
        self.site_encoder = LabelEncoder()
        
        self.numeric_features = [
            'latitude', 'longitude', 'altitude', 
            'co2_change_rate', 'biomass_density'
        ]
        
        self.temporal_features = ['year']
        
        self.lag_features = [
            'ppm_lag_14', 'ppm_lag_30', 'ppm_lag_365'
        ]

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # For biomass_density, use forward fill then backward fill
        if 'biomass_density' in df.columns:
            df['biomass_density'] = df['biomass_density'].fillna(method='ffill').fillna(method='bfill')
        
        # For numeric features, use forward fill with rolling mean as backup
        for col in self.numeric_features:
            if col in df.columns and df[col].isnull().any():
                # First try forward fill
                df[col] = df[col].fillna(method='ffill')
                
                # If any NaN remain, use rolling mean with 30-day window
                if df[col].isnull().any():
                    rolling_mean = df[col].rolling(window=30, min_periods=1).mean()
                    df[col] = df[col].fillna(rolling_mean)
                
                # If still any NaN (at the start), use backward fill
                df[col] = df[col].fillna(method='bfill')
        
        # For lag features, use the previous available value
        for col in self.lag_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df

    def _validate_data(self, df: pd.DataFrame, check_missing: bool = True) -> None:
        """Validate data integrity with optional missing value check."""
        if check_missing:
            missing = df.isnull().sum()
            if missing.any():
                print("Warning: Missing values detected. Will attempt to handle them.")
                print("Missing value counts:")
                print(missing[missing > 0])
        
        # Check for infinite values
        infinite = np.isinf(df.select_dtypes(include=np.number)).sum()
        if infinite.any():
            raise ValueError(f"Infinite values found in columns: {infinite[infinite > 0].index.tolist()}")

    def _create_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclic features for temporal periodicities."""
        df = df.copy()
        # Convert month to cyclic features (sin and cos)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features while preserving temporal relationships."""
        df = df.copy()
        
        # Scale numeric features
        if len(self.numeric_features) > 0:
            df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])
        
        # Scale temporal features separately to preserve order
        if len(self.temporal_features) > 0:
            df[self.temporal_features] = self.temporal_scaler.fit_transform(df[self.temporal_features])
        
        # Scale lag features together to preserve relationships
        if len(self.lag_features) > 0:
            df[self.lag_features] = self.lag_scaler.fit_transform(df[self.lag_features])
        
        return df

    def _create_dataloader(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        seq_length: int, 
        batch_size: int
    ) -> DataLoader:
        """Create DataLoader with sliding window sequences."""
        if len(X) <= seq_length:
            raise ValueError(f"Data length ({len(X)}) must be greater than sequence length ({seq_length})")
        
        Xs, ys = [], []
        
        # Create sliding windows of sequences while maintaining temporal order
        for i in range(len(X) - seq_length):
            try:
                # Extract sequence and verify its integrity
                sequence = X.iloc[i:i+seq_length].values
                target = y.iloc[i+seq_length]
                
                # Validate sequence
                if len(sequence) != seq_length:
                    continue
                    
                if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
                    continue
                
                Xs.append(sequence)
                ys.append(target)
                
            except Exception as e:
                continue
        
        if not Xs:
            raise ValueError("No valid sequences could be created from the data")
        
        try:
            # Convert to PyTorch tensors with appropriate shapes for LSTM input
            X_tensor = torch.FloatTensor(Xs)  # shape: [num_sequences, seq_length, num_features]
            y_tensor = torch.FloatTensor(ys).reshape(-1, 1)
            
            # Verify tensor shapes
            expected_feature_dim = X.shape[1]
            if X_tensor.shape[2] != expected_feature_dim:
                raise ValueError(f"Feature dimension mismatch. Expected {expected_feature_dim}, got {X_tensor.shape[2]}")
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=min(batch_size, len(dataset)),
                shuffle=False,
                drop_last=False
            )
            
            return dataloader
            
        except Exception as e:
            raise ValueError(f"Error creating DataLoader: {str(e)}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        seq_length: int = 30, 
        batch_size: int = 32,
        validation_split: float = 0.1,
        test_split: float = 0.1
    ) -> Dict[str, Union[DataLoader, StandardScaler]]:
        """Prepare data with improved preprocessing and validation."""
        # First validate data and check for missing values
        self._validate_data(df, check_missing=True)
        
        # Sort by date to ensure temporal ordering
        df = df.sort_values('date').copy()
        
        # Handle missing values before any other preprocessing
        df = self._handle_missing_values(df)
        
        # Create cyclic features
        df = self._create_cyclic_features(df)
        
        # Encode categorical features
        df['site'] = self.site_encoder.fit_transform(df['site'])
        
        # Normalize features by group
        df = self._normalize_features(df)
        
        # Scale target variable (ppm)
        df['ppm'] = self.target_scaler.fit_transform(df[['ppm']])
        
        # Prepare feature matrix X and target y
        feature_columns = (
            self.numeric_features + 
            self.temporal_features + 
            self.lag_features + 
            ['site', 'month_sin', 'month_cos']
        )
        
        X = df[feature_columns]
        y = df['ppm']
        
        # Final validation after all preprocessing
        assert not np.any(np.isnan(X.values)), "NaN values found in features after preprocessing"
        assert not np.any(np.isinf(X.values)), "Infinite values found in features after preprocessing"
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * (1 - validation_split - test_split))
        val_end = int(n * (1 - test_split))
        
        # Split data preserving temporal order
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Create dataloaders
        train_loader = self._create_dataloader(X_train, y_train, seq_length, batch_size)
        val_loader = self._create_dataloader(X_val, y_val, seq_length, batch_size)
        test_loader = self._create_dataloader(X_test, y_test, seq_length, batch_size)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'input_dim': X.shape[1],
            'target_scaler': self.target_scaler
        }