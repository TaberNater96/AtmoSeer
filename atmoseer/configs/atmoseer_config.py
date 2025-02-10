from dataclasses import dataclass, field
from typing import Dict, Any
import torch
from pathlib import Path

@dataclass
class ModelConfig:
    input_dim: int = 12
    hidden_dim: int = 256
    num_layers: int = 2
    batch_first: bool = True
    dropout: float = 0.25
    bidirectional: bool = True
    sequence_length: int = 30  # lookback window (30 days default)
    
@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 50
    early_stopping_patience: int = 10
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradient_clip_val: float = 0.5
    gradient_accumulation_steps: int = 4
    min_delta: float = 1e-5

@dataclass
class BayesianTunerConfig:
    gas_type: str                      # type of gas being modeled (co2, ch4, n2o, sf6)
    n_trials: int = 50
    timeout: int = None
    random_state: int = 10
    
    param_bounds: Dict[str, tuple] = field(default_factory=lambda: {
        'hidden_dim': (256, 512),        # model capacity
        'num_layers': (2, 3),            # model depth
        'dropout': (0.2, 0.4),           # regularization
        'sequence_length': (30, 90),     # temporal context
        'learning_rate': (1e-5, 1e-3),   # optimization rate
        'batch_size': (32, 128)          # training size
    })
    
    models_dir: Path = Path('../atmoseer/models')
    
    # Resource management
    max_memory_gb: float = 0.95  
    cleanup_trials: bool = True 
    
    def __post_init__(self):
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create gas-specific directories
        self.gas_dir = self.models_dir / self.gas_type
        self.trials_dir = self.gas_dir / 'trials'
        self.best_model_dir = self.gas_dir / 'best_model'
        
        for dir_path in [self.gas_dir, self.trials_dir, self.best_model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)