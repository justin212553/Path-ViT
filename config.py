from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    embed_dim:              int   = 512
    num_heads:              int   = 8
    num_transformer_layers: int   = 6
    dropout:                float = 0.1
    max_grid_size:          int   = 128


@dataclass
class DataConfig:
    wsi_root:    str   = "data/patches_train"   # preprocess 출력 디렉토리
    test_root:   str   = "data/patches_eval"    # preprocess_eval 출력 디렉토리
    csv_path:    str   = "data/stage_labels.csv"
    val_centers: Tuple = (1,)
    num_workers: int   = 4


@dataclass
class TrainConfig:
    epochs:       int   = 50
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    device:       str   = "cuda"
    seed:         int   = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
