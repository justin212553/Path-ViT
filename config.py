from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    embed_dim:              int   = 256
    num_heads:              int   = 8
    num_transformer_layers: int   = 6
    dropout:                float = 0.1
    max_grid_size:          int   = 1500


@dataclass
class DataConfig:
    patches_root: str   = "data/patches"  # WSI 단위 MIL train/val 공용 노드 루트
    csv_path:     str   = "data/stage_labels.csv"
    num_workers: int   = 4
    max_patches: int   = 4000  # WSI당 최대 패치 수 (None=무제한)


@dataclass
class TrainConfig:
    epochs:                int   = 32
    lr:                    float = 1e-4
    weight_decay:          float = 1e-4
    device:                str   = "cuda"
    seed:                  int   = 42
    # --- GPU 최적화 파라미터 ---
    # gradient accumulation: effective batch = accum_steps WSIs
    # (MIL은 DataLoader batch_size=1 고정이므로 accumulation으로 보완)
    accumulate_grad_steps: int   = 4
    warmup_epochs:         int   = 3       # linear LR warmup → cosine decay
    # AMP dtype: "auto" → A30은 bfloat16, V100은 float16 자동 선택 / "none" 비활성화
    amp_dtype:             str   = "auto"
    # 대형 WSI(수천 패치)에서 CNN OOM 방지용 서브배치
    cnn_chunk_size:        int   = 64


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
