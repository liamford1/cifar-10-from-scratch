from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    conv1_filters: int = 32
    conv2_filters: int = 64
    kernel_size: int = 3
    dense1_units: int = 256
    dense2_units: int = 10
    dropout_rate: float = 0.5

@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    batch_size: int = 64
    epochs: int = 100
    validation_split: float = 0.2
    gradient_clip_norm: float = 1.0
    