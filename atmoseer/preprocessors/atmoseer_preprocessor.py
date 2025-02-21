import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Union

class AtmoSeerPreprocessor:
    """
    Handles data preprocessing including missing value imputation, feature normalization,
    temporal feature engineering, and sequence preparation for LSTM models. Maintains separate 
    scalers for different feature groups to preserve their relative relationships.
    """
    def __init__(self) -> None:
        """
        Initialize preprocessor parameters.

        Attributes:
            scaler: StandardScaler for numeric features
            temporal_scaler: StandardScaler for temporal features
            lag_scaler: StandardScaler for lag features
            target_scaler: StandardScaler for target variable
            site_encoder: LabelEncoder for measurement sites
            numeric_features: List of numeric feature names
            temporal_features: List of temporal feature names
            lag_features: List of lag feature names
        """
        self.scaler = StandardScaler()           # for numeric features
        self.temporal_scaler = StandardScaler()  # for temporal features
        self.lag_scaler = StandardScaler()       # for lag features
        self.target_scaler = StandardScaler()    # for target variable
        self.site_encoder = LabelEncoder()
        
        self.numeric_features = [
            'latitude', 'longitude', 'altitude', 
            'co2_change_rate', 'biomass_density'
        ]
        
        self.temporal_features = ['year']
        
        self.lag_features = [
            'ppm_lag_14', 'ppm_lag_30', 'ppm_lag_365'
        ]

    def _handle_missing_values(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Impute missing values using temporal-aware strategies for different feature types.

        Applies specialized imputation strategies: forward fill with backup methods for numeric features,
        bi-directional fill for biomass density, and forward-backward fill for lag features.

        Args:
            df: DataFrame containing feature columns with potential missing values

        Returns:
            DataFrame with imputed values replacing all NaN entries
        """
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

    def _validate_data(
        self, 
        df: pd.DataFrame, 
        check_missing: bool = True
    ) -> None:
        """
        Validate data integrity and check for problematic values in the DataFrame.

        Performs checks for missing and infinite values, providing warnings for missing data
        and raising errors for infinite values which could break the preprocessing pipeline.

        Args:
            df: DataFrame to validate
            check_missing: If True, checks and reports missing values. Defaults to True.

        Raises:
            ValueError: If infinite values are found in any numeric columns
        """
        if check_missing:
            missing = df.isnull().sum()
            if missing.any():
                print("Warning: Missing values detected. Will attempt to handle them.")
                print("Missing value counts:")
                print(missing[missing > 0])
        
        # Infinite values cannot be handled by scalers and will break the pipeline
        infinite = np.isinf(df.select_dtypes(include=np.number)).sum()
        if infinite.any():
            raise ValueError(f"Infinite values found in columns: {infinite[infinite > 0].index.tolist()}")

    def _create_cyclic_features(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform monthly data into cyclic features using sine and cosine transformations.

        Converts the month column into two continuous cyclic features that capture the cyclical 
        nature of months, preserving the temporal relationship between December and January.

        Args:
            df: DataFrame containing a 'month' column with values 1-12

        Returns:
            DataFrame with additional 'month_sin' and 'month_cos' columns
        """
        df = df.copy()
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df
    
    def _normalize_features(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize different feature groups using their respective scalers.
        
        Applies separate normalization to numeric, temporal, and lag features to preserve the relative 
        relationships within each feature group. Uses pre-initialized scalers for each group.

        Args:
            df: DataFrame containing features to be normalized

        Returns:
            DataFrame with normalized features maintaining their group relationships
        """
        df = df.copy()
        
        if len(self.numeric_features) > 0:
            df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])
        
        if len(self.temporal_features) > 0:
            df[self.temporal_features] = self.temporal_scaler.fit_transform(df[self.temporal_features])
        
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
        """
        Create a PyTorch DataLoader with sliding window sequences for time series data.

        Generates sequence data using a sliding window approach and validates data integrity
        at each step to ensure robust sequence creation for LSTM training.

        Args:
            X: Feature DataFrame
            y: Target series containing CO2 values
            seq_length: Length of each sequence window
            batch_size: Number of sequences per batch

        Returns:
            DataLoader containing sequence data for LSTM training

        Raises:
            ValueError: If data length is insufficient for sequence length or if no valid sequences can be created
        """
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
            X_tensor = torch.FloatTensor(Xs)  # shape: [num_sequences, seq_length, num_features]
            y_tensor = torch.FloatTensor(ys).reshape(-1, 1)
            
            # Verify tensor shapes
            expected_feature_dim = X.shape[1]
            if X_tensor.shape[2] != expected_feature_dim:
                raise ValueError(f"Feature dimension mismatch. Expected {expected_feature_dim}, got {X_tensor.shape[2]}")
            
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
        """
        Prepare and preprocess data for LSTM training with temporal validation splitting.

        Executes the full preprocessing pipeline including validation, missing value handling,
        feature engineering, normalization, and sequence creation while preserving temporal ordering.

        Args:
            df: Raw DataFrame containing atmospheric measurements
            seq_length: Number of time steps in each sequence. Defaults to 30.
            batch_size: Number of sequences per batch. Defaults to 32.
            validation_split: Fraction of data for validation. Defaults to 0.1.
            test_split: Fraction of data for testing. Defaults to 0.1.

        Returns:
            Dictionary containing:
                - train_loader: DataLoader for training data
                - val_loader: DataLoader for validation data
                - test_loader: DataLoader for test data
                - input_dim: Number of input features
                - target_scaler: Scaler used for the target variable
        """  
        self._validate_data(df, check_missing=True)
        df = df.sort_values('date').copy()
        df = self._handle_missing_values(df)
        df = self._create_cyclic_features(df)
        df['site'] = self.site_encoder.fit_transform(df['site'])
        df = self._normalize_features(df)
        df['ppm'] = self.target_scaler.fit_transform(df[['ppm']])
        
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