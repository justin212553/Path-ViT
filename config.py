from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    embed_dim:              int   = 512
    num_heads:              int   = 8
    num_transformer_layers: int   = 6
    dropout:                float = 0.1
    max_grid_size:          int   = 1500


@dataclass
class DataConfig:
    wsi_root:    str   = "data/patches_train"
    test_root:   str   = "data/patches_eval"
    csv_path:    str   = "data/stage_labels.csv"
    val_ratio:   float = 0.2   # 각 클래스에서 validation으로 쓸 비율 (stratified)
    num_workers: int   = 8  # SBATCH --cpus-per-task=8 에 맞춤


@dataclass
class TrainConfig:
    epochs:                int   = 15
    lr:                    float = 1e-4
    weight_decay:          float = 1e-4
    device:                str   = "cuda"
    seed:                  int   = 42
    # --- GPU 최적화 파라미터 ---
    # gradient accumulation: effective batch = accum_steps WSIs
    # (MIL은 DataLoader batch_size=1 고정이므로 accumulation으로 보완)
    accumulate_grad_steps: int   = 8
    warmup_epochs:         int   = 3       # linear LR warmup → cosine decay
    # AMP dtype: "auto" → A30은 bfloat16, V100은 float16 자동 선택 / "none" 비활성화
    amp_dtype:             str   = "auto"
    # 대형 WSI(수천 패치)에서 CNN OOM 방지용 서브배치
    cnn_chunk_size:        int   = 256


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
