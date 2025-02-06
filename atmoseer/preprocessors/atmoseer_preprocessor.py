import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Union

class AtmoSeerPreprocessor:
    """
    A preprocessor class for preparing greenhouse gas emission data for deep learning models.
    
    This class handles all necessary preprocessing steps for time series forecasting of greenhouse gas emissions, including 
    feature scaling, categorical encoding, and creating PyTorch DataLoaders while preserving temporal ordering.
    
    Attributes:
        scaler (StandardScaler): Standardizes numeric features to zero mean and unit variance
        site_encoder (LabelEncoder): Encodes categorical site identifiers
        season_encoder (LabelEncoder): Encodes seasonal information
        numeric_features (list): List of features requiring standardization
    """
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.site_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        self.numeric_features = ['latitude', 'longitude', 'altitude', 'co2_change_rate', 'biomass_density']
        
    def _create_dataloader(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        seq_length: int, 
        batch_size: int
    ) -> DataLoader:
        """
        Creates PyTorch DataLoader objects for time series sequences while preserving temporal ordering.
        
        This method implements a sliding window approach to create sequences of observations for time series forecasting. 
        Each sequence contains seq_length consecutive observations, and the target is the next value after the sequence.
        
        Parameters:
            X (pd.DataFrame): Feature DataFrame containing the input variables
            y (pd.Series): Target series containing the values to predict
            seq_length (int): Length of each sequence (lookback window) for the time series model
            batch_size (int): Number of sequences to include in each batch for training
        
        Returns:
            DataLoader: PyTorch DataLoader containing batched sequences of input features and their
                        corresponding target values, maintaining temporal order
        """
        Xs, ys = [], []
        
        # Create sliding windows of sequences while maintaining temporal order
        for i in range(len(X) - seq_length):
            sequence = X.iloc[i:i+seq_length].values
            target = y.iloc[i+seq_length]
            
            # Verify sequence integrity
            if len(sequence) == seq_length:
                Xs.append(sequence)
                ys.append(target)
        
        # Convert to PyTorch tensors with appropriate shapes for LSTM input
        X_tensor = torch.FloatTensor(Xs)  # shape: [num_sequences, seq_length, num_features]
        y_tensor = torch.FloatTensor(ys).reshape(-1, 1)
        
        # Create a dataset that pairs input sequences with their targets
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        seq_length: int = 30, 
        batch_size: int = 32
    ) -> Dict[str, Union[DataLoader, int]]:
        """
        Prepares greenhouse gas emission data for time series forecasting by applying necessary
        transformations and creating training, validation, and test sets.
        
        This method handles the complete data preparation pipeline including:
        1. Verifying chronological ordering
        2. Feature standardization and encoding
        3. Creating sequences for time series prediction
        4. Splitting the data and maintaining temporal order with an 80-10-10 split for train-val-test 
        
        Parameters:
            df (pd.DataFrame): Raw DataFrame containing greenhouse gas emission data
            seq_length (int, optional): Number of time steps to include in each sequence. Defaults to 30,
                                        representing a month of daily observations
            batch_size (int, optional): Number of sequences per batch. Defaults to 32, balancing
                                        computational efficiency and model stability
        
        Returns:
            Dict[str, Union[DataLoader, int]]: Dictionary containing:
                - train_loader: DataLoader for training data
                - val_loader: DataLoader for validation data
                - test_loader: DataLoader for test data
                - input_dim: Number of input features
        """
        # Maintain temporal integrity by sorting data chronologically, this double checks temporal order
        df = df.sort_values('date')
        
        # Standardize numeric features to zero mean and unit variance to make every feature contribute equally
        df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features]) # cyclical and lag scales will stay as is
        
        # Encode categorical features
        df['site'] = self.site_encoder.fit_transform(df['site'])
        df['season'] = self.season_encoder.fit_transform(df['season'])
        
        # Training data will not need the date and day, as long-short term dependencies capture yearly patterns (month-to-month, not day-to-day)
        X = df.drop(columns=['date', 'day', 'ppm'])
        y = df['ppm']
        
        # Split training data chronologically
        train_size = 0.8
        val_size = 0.1
        
        # Calculate split indices based on percentages
        train_idx = int(len(X) * train_size)
        val_idx = int(len(X) * (train_size + val_size))
        
        # Split data chronologically to maintain temporal relationships, 80-10-10 split
        X_train = X[:train_idx]
        y_train = y[:train_idx]
        X_val = X[train_idx:val_idx]
        y_val = y[train_idx:val_idx]
        X_test = X[val_idx:]
        y_test = y[val_idx:]
        
        # Convert to tensors (for PyTorch) and create dataloaders with temporal ordering preserved
        train_loader = self._create_dataloader(X_train, y_train, seq_length, batch_size)
        val_loader = self._create_dataloader(X_val, y_val, seq_length, batch_size)
        test_loader = self._create_dataloader(X_test, y_test, seq_length, batch_size)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'input_dim': X.shape[1]     # number of input features
        }