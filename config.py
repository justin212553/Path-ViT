from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    cnn_backbone:           str   = "resnet50"
    embed_dim:              int   = 512
    num_heads:              int   = 8
    num_transformer_layers: int   = 6
    dropout:                float = 0.1
    max_grid_size:          int   = 128


@dataclass
class DataConfig:
    preprocessed_root: str   = "data/patches"  # data/preprocess.py 출력 디렉토리
    val_centers:       Tuple = (1,)                # center 1을 val로 사용
    num_workers:       int   = 4


@dataclass
class TrainConfig:
    epochs:       int   = 50
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs:int   = 5
    device:       str   = "cuda"
    seed:         int   = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
