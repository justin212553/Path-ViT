from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ModelConfig:
    embed_dim:              int   = 64
    num_heads:              int   = 4
    num_transformer_layers: int   = 2
    dropout:                float = 0.25
    num_landmarks:          int   = 128   # Nystrom attention landmark 수 (근사 정밀도/속도 트레이드오프)
    # 대형 WSI(N 수만 패치) backward 메모리 절감용. 끄면 메모리↑ 속도↑
    # (precomputed feature 모드처럼 메모리 여유가 있을 때 끄면 학습 시간 단축 가능)
    grad_checkpoint:        bool  = True


@dataclass
class DataConfig:
    wsi_root_tcga: str          = "data/tcga_paad_wsi"
    wsi_root_cptac: str         = "data/cptac_pda_wsi"
    patches_root_tcga: str      = "data/patches_tcga"
    patches_root_cptac: str     = "data/patches_cptac"
    num_workers: int            = 4
    precomputed: bool           = True
    seed: int                   = 42  # 환자 단위 train/val split 재현성 (data/dataset.py 참조)


@dataclass
class TrainConfig:
    epochs:                int   = 30
    lr:                    float = 1e-4
    weight_decay:          float = 1e-2
    device:                str   = "cuda"
    seed:                  int   = 42
    # --- GPU 최적화 파라미터 ---
    # gradient accumulation 단위 = 환자 1명(보유한 모든 노드 누적 후 1 step, train.py 참조)
    warmup_epochs:         int   = 10      # linear LR warmup → cosine decay
    # 대형 WSI(수천 패치)에서 CNN OOM 방지용 서브배치
    cnn_chunk_size:        int   = 64
    # Cox PH loss는 위험집합(risk set) 비교를 위해 여러 환자를 한 배치로 묶어야 한다.
    # 값이 클수록 risk set 추정이 안정적이지만, 환자별 forward activation을 배치가 찰 때까지
    # 모두 메모리에 들고 있어야 하므로 GPU 메모리 사용량도 함께 늘어난다.
    cox_batch_size:        int   = 16


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
