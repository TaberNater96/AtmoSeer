from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    input_dim: int = 14
    hidden_dim: int = 256
    num_layers: int = 2
    batch_first: bool = True
    dropout: float = 0.2
    bidirectional: bool = True
    sequence_length: int = 30  # lookback window
    
@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 50
    early_stopping_patience: int = 10
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 4