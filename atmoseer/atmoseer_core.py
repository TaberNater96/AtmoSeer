import torch
import torch.nn as nn
import random
from configs.atmoseer_config import ModelConfig, TrainConfig, BayesianTunerConfig
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import gc
from typing import Dict, Any, Optional, Tuple
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
        train_config: TrainConfig = TrainConfig()
    ) -> None:
        """
        Initialize AtmoSeer with the specified configuration.
        
        Args:
            model_config (ModelConfig): Configuration dataclass containing model hyperparameters.
                                        If not provided, uses default values from ModelConfig.

            train_config (TrainConfig): Configuration dataclass containing training hyperparameters.
                                        If not provided, uses default values from TrainConfig.
        """
        # Initialize parent nn.Module class to enable PyTorch functionality
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        
        # Verify proper input_dim adjustment based on gas type 
        if hasattr(model_config, 'gas_type') and model_config.gas_type != 'co2':
            print(f"Initializing AtmoSeer for {model_config.gas_type} with input_dim={model_config.input_dim}")
            
        self.input_norm = nn.LayerNorm(model_config.input_dim)
        
        self.lstm = nn.LSTM(
            input_size=model_config.input_dim,
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
        
    def _init_weights(self:"AtmoSeer") -> None:
        """
        Initialize network weights using specialized strategies for different layer types.
        
        LSTM weights use orthogonal initialization to maintain consistent gradient magnitudes, while other weight matrices 
        use Xavier/Glorot uniform initialization for better gradient flow, and all bias terms are initialized to zero.
        The initialization strategy varies by layer type:

        - LSTM weights: Initialized using orthogonal initialization, which helps maintain
          consistent gradient magnitudes across deep networks and is particularly effective
          for recurrent architectures. This helps mitigate vanishing/exploding gradients, 
          which are extremely common in RNNs.
          
        - Other weight matrices: Initialized using Xavier/Glorot uniform initialization,
          which maintains variance of activations and gradients across layers by considering
          the size of input/output dimensions. This promotes better gradient flow in the
          fully connected layers.
          
        - Bias terms: Initialized to zero, as bias terms don't benefit from special
          initialization in this architecture and zero initialization is a stable default.
        """
        # Automatically detect weight and bias parameters by their name to apply specialized initialization
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
        x = self.input_norm(x) # shape: [batch_size, seq_length, input_dim]
        
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
        # Re-initialize the training configuration so that the newly optimized hyperparameters are used, and not reset to default
        self.train_config = train_config
        self.to(train_config.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=train_config.learning_rate)
        criterion = nn.MSELoss() # monitor regression metric for validation loss
        
        # Ff after 5 consecutive epochs, there is no improvement, multiply the learning rate by 0.5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,   
            min_lr=1e-6 # floor the learning rate
        )
        
        # Track best validation loss for model checkpointing
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0
        early_stopping_counter = 0
        min_delta = train_config.min_delta         # minimum change in validation loss to qualify as improvement
        training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
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
            
            # Update the learning rate based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['learning_rates'].append(current_lr)
            
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
    
    def generate_forecast(
        self: "AtmoSeer",
        initial_sequence: torch.Tensor,
        forecast_length: int,
        confidence_interval: float = 0.95,
        noise_scale: float = 0.1,
        device: Optional[torch.device] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate future predictions with uncertainty bounds using an autoregressive approach.
        
        This method implements an iterative forecasting process where each prediction becomes part of the input for the next 
        prediction. It uses Monte Carlo sampling with added Gaussian noise to estimate prediction uncertainty, which naturally 
        grows over time as predictions are chained together. Instead of making a prediction based off of one single point, this 
        forecast method will create a normal distribution around a specific prediction point using 100 normally distributed values, 
        where the mean is the prediction point and the standard deviation is the noise_scale. This will create a range of possible 
        values that the prediction could be, which will be used to create the uncertainty bounds. The further out into the future 
        that the predictions go, the wider the uncertainty bounds become. The Bayesian Tuner will go through many trials to find 
        the optimal sequence length (lookback window in days) and then this forecast method will take that sequence length and use 
        it to generate predictions. For dates that are past this sequence length, the predicted values will be entirely based on 
        other predicted values (not trained data points), which will increase the uncertainty by a larger and larger amount.
        
        Args:
            initial_sequence (torch.Tensor): Starting sequence of shape (1, sequence_length, features)
                                             containing the most recent known observations. 
                
            forecast_length (int): Number of time steps to predict into the future.
            
            confidence_interval (float, optional): Probability mass to include in the uncertainty bounds 
                                                   (default: 0.95).
                
            noise_scale (float, optional): Standard deviation of the Gaussian noise added to predictions 
                                           (default: 0.1).
                
            device (torch.device, optional): Device to run predictions on. If None, uses the device where the 
                                             model parameters are located.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing three arrays of length forecast_length:
                - 'predictions': Point estimates for each future time step
                - 'upper_bound': Upper confidence bound for each prediction
                - 'lower_bound': Lower confidence bound for each prediction
        """
        if device is None:
            device = next(self.parameters()).device
            
        self = self.to(device)
        initial_sequence = initial_sequence.to(device)
        self.eval()
        
        # Create a copy of the initial sequence to avoid modifying the input
        last_sequence = initial_sequence.clone()
        
        forecast_pred = []
        forecast_upper = []
        forecast_lower = []
        
        with torch.no_grad():
            for _ in range(forecast_length):
                pred = self(last_sequence)
                pred = pred.cpu().numpy()
                
                # Creates 100 variations of the prediction by adding random Gaussian noise for uncertainty estimation
                noise = np.random.normal(0, noise_scale, 100)
                predictions_with_noise = pred + noise[:, np.newaxis]
                
                # Calculate confidence interval bounds using percentiles
                lower = np.percentile(predictions_with_noise, (1 - confidence_interval) * 100 / 2)       # 2.5 percentile
                upper = np.percentile(predictions_with_noise, 100 - (1 - confidence_interval) * 100 / 2) # 97.5 percentile
                
                forecast_pred.append(pred[0][0])
                forecast_upper.append(upper)
                forecast_lower.append(lower)
                
                # For the next sequence, shift the window and update all features use the last sequence's features but shift them, 
                last_features = last_sequence[:, -1:, :].clone() 
                last_features[:, :, 0] = torch.tensor(pred[0][0]).to(device)  # update only the target variable
                
                # Shift the sequence window and add the new features
                last_sequence = torch.cat((
                    last_sequence[:, 1:, :],  # remove oldest timestep
                    last_features
                ), dim=1)
                
        return {
            'predictions': np.array(forecast_pred),
            'upper_bound': np.array(forecast_upper),
            'lower_bound': np.array(forecast_lower)
        }

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
            'model_state_dict': self.state_dict(),  # learned parameters
            'model_config': self.model_config,      # architecture configuration
            'train_config': self.train_config
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
        model = cls(model_config=checkpoint['model_config'], train_config=checkpoint['train_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
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
        self: "BayesianTuner",
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
        
    def setup_logging(self: "BayesianTuner") -> None:
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
    
    def _cleanup_memory(self: "BayesianTuner") -> None:
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
        self: "BayesianTuner",
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
    
    def _cleanup_old_trials(self: "BayesianTuner") -> None:
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
    
    def _objective(
        self: "BayesianTuner", 
        **params
    ) -> float:
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
            'sequence_length': int(params['sequence_length']),
            'gas_type': self.config.gas_type
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
            
            # Small regularization term to encourage exploration around known well performant parameters
            regularization = 0.01 * (abs(params['hidden_dim'] - 256) + abs(params['num_layers'] - 2) / 2)
            
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
            
            return -(val_loss + regularization)  # negative because we want to maximize
            
        except Exception as e:
            print(f"Trial {self.current_trial} failed: {str(e)}")
            self._cleanup_memory()
            return float('-inf')  # return worst possible score on failed configurations
    
    def optimize(self: "BayesianTuner") -> Tuple[Dict[str, Any], float]:
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
            init_points=0,     # no random exploration
            n_iter=9           # remaining trials use Bayesian optimization for a total of 10
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