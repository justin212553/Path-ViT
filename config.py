from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    embed_dim:              int   = 128
    num_heads:              int   = 8
    num_transformer_layers: int   = 2
    dropout:                float = 0.25
    num_landmarks:          int   = 128   # Nystrom attention landmark 수 (근사 정밀도/속도 트레이드오프)
    # 대형 WSI(N 수만 패치) backward 메모리 절감용. 끄면 메모리↑ 속도↑
    # (precomputed feature 모드처럼 메모리 여유가 있을 때 끄면 학습 시간 단축 가능)
    grad_checkpoint:        bool  = True


@dataclass
class DataConfig:
    patches_root: str   = "data/patches"  # WSI 단위 MIL train/val 공용 노드 루트
    csv_path:     str   = "data/stage_labels.csv"
    num_workers: int   = 4
    # True(기본): data/extract_features.py로 미리 뽑아둔 features.pt를 사용
    # False(--image): 패치 jpg/png를 매번 ResNet50으로 디코딩/forward
    precomputed: bool  = True


@dataclass
class TrainConfig:
    epochs:                int   = 50
    lr:                    float = 1e-4
    weight_decay:          float = 1e-2
    device:                str   = "cuda"
    seed:                  int   = 42
    # --- GPU 최적화 파라미터 ---
    # gradient accumulation 단위 = 환자 1명(보유한 모든 노드 누적 후 1 step, train.py 참조)
    warmup_epochs:         int   = 10      # linear LR warmup → cosine decay
    # 대형 WSI(수천 패치)에서 CNN OOM 방지용 서브배치
    cnn_chunk_size:        int   = 64


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
