from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ModelConfig:
    # embed_dim=256, num_heads=4 → head_dim=64 (Transformer/BERT/ViT 전반의 표준 관례).
    # WSI-MIL 레퍼런스(CLAM/TransMIL)는 보통 512를 쓰지만, 이 프로젝트 코호트(TCGA-PAAD/
    # CPTAC-PDA)는 그 논문들보다 표본이 훨씬 적어 과적합 위험이 커 절반 크기로 절충한다.
    #
    # 2026-07-19: "압축이 너무 심해 신호가 노이즈로 뭉개졌을 수도 있다"는 반대 가설을 실제로
    # embed_dim=256/num_heads=4/num_transformer_layers=2로 검증해봤다 — PMA_EX_SS_AUX
    # (tcga->cptac, seed42) train_c_index 0.87까지 과적합되면서 val 0.53대 정체, external
    # C=0.47(기존 작은 모델 0.625~0.632 대비 붕괴)로 명확한 negative result. "표본 대비
    # 과적합"이 병목이라는 기존 진단이 재확인됐다 — 원래 값(절반 크기)으로 되돌린다.
    embed_dim:              int   = 64
    num_heads:              int   = 2
    num_transformer_layers: int   = 1      # TransMIL도 동일하게 2-layer Nystromformer를 사용
    dropout:                float = 0.3    # Transformer/ViT 표준 기본값
    num_landmarks:          int   = 128   # Nystrom attention landmark 수 (근사 정밀도/속도 트레이드오프)
    # 대형 WSI(N 수만 패치) backward 메모리 절감용. 끄면 메모리↑ 속도↑
    # (precomputed feature 모드처럼 메모리 여유가 있을 때 끄면 학습 시간 단축 가능)
    grad_checkpoint:        bool  = False
    # 2026-08-03: --tile-augment(--image, 진짜 real-time augmentation) 경로에서 타일 디코딩+
    # 증강(models/vit_m1.py::_patch_tokens의 ThreadPoolExecutor)에 쓰는 스레드 수. 이 CPU 작업이
    # DataLoader(__getitem__은 patch_paths만 반환하는 경량 작업)가 아니라 forward() 안에서 도는
    # 것이라 DataConfig.num_workers와는 별개다 — SLURM --cpus-per-task로 예약한 CPU 개수에
    # 맞춰야 실제로 CPU 디코딩과 GPU 연산이 겹쳐 돈다(train.py --tile-decode-workers).
    tile_decode_workers:    int   = 4
    # 2026-07-22: 슬라이드당 패치 수 실측(TCGA train, 평균 131/중앙값 67/최대 544)이
    # num_landmarks=128보다도 작은 경우가 절반 이상 — Nystrom이 "N개를 landmark개로 근사"가
    # 아니라 오히려 패딩(zero) 토큰을 landmark에 섞어 넣는 역효과를 냈을 수 있다는 의심으로
    # (findings_backlog.md), 이 규모에서는 O(N^2)이 전혀 부담 없다는 점(N<=544)까지 확인하고
    # 일반 self-attention(nn.MultiheadAttention)으로 교체하는 옵션을 추가한다.
    use_nystrom:             bool  = True
    # 2026-07-22: attention이 이미 uniform으로 붕괴한 상태에서 좌표 임베딩이 실제로 기여하는지
    # 직접 검증(findings_backlog.md, train.py --no-spatial-embed).
    use_spatial_embed:       bool  = True
    # 2026-07-23: 얼린 Stage1 + 잔차 branch(models/spatial_residual.py) 방식은 PAAD(152명)·
    # BRCA(1058명) 양쪽 모두에서 실패(BRCA조차 test_c_index 0.7151->0.6015로 붕괴) — late-fusion
    # 잔차로 붙이는 방식 자체의 문제로 보고, 대신 WSI branch 내부(ViTEncoder)에서 직접 절대좌표
    # 임베딩을 상대offset 기반 attention bias(Swin류, models/vit_encoder.py::RelativeBiasFullAttention)
    # 로 교체해 처음부터 함께 학습시키는 실험용 플래그(train.py --rel-bias-attention). True면
    # use_nystrom/use_spatial_embed는 자동으로 False로 강제된다(ViT_M1.__init__).
    use_rel_bias_attn:       bool  = False
    # 2026-07-23: use_rel_bias_attn(dense, O(N^2))을 BRCA(슬라이드당 패치 수 중앙값 10,309/최대
    # 67,268, data/brca_slide_manifest.csv)에 그대로 돌리자 즉시 CUDA OOM — PAAD(N<=544) 전용
    # 가정이 BRCA엔 안 맞는다. kNN 그래프로 제한한 희소 버전(models/vit_encoder.py::
    # KNNBiasAttention, train.py --knn-bias-attention). use_rel_bias_attn과 배타적(둘 다 True면
    # use_rel_bias_attn 우선).
    use_knn_bias_attn:       bool  = False
    knn_attn_k:               int   = 8
    knn_attn_edge_dropout:    float = 0.2
    # 2026-07-23: kNN-bias-attention 단독은 PAAD(pre-augment)에서도 기존 Nystrom+절대좌표
    # baseline보다 낮았다(internal 0.6309->0.6094, external 0.6289->0.5880). Nystrom(전역)을
    # 완전히 대체하는 대신 같은 레이어에서 kNN(국소)과 병렬로 더하는 hybrid
    # (models/vit_encoder.py::HybridLocalGlobalAttention, train.py --hybrid-attention).
    # use_rel_bias_attn/use_knn_bias_attn과 달리 use_nystrom/use_spatial_embed를 강제로 끄지
    # 않는다 — global 경로(Nystrom)가 계속 절대좌표 임베딩을 활용해야 하므로.
    use_hybrid_attn:          bool  = False
    # 2026-07-23: 학습형 spatial attention(kNN/hybrid, PMA/M4A 둘 다)이 pre-augment PAAD에서
    # 전부 baseline을 못 넘은 뒤, "새 attention 파라미터 자체가 과적합 유인"이라는 가설을
    # 검증하는 저비용 대안 — models/spatial_features.py(학습 파라미터 없음)를 risk_head의
    # 5번째 관점으로 추가한다(train.py --spatial-autocorr/--attn-dispersion, ViT_PMA 전용).
    use_spatial_autocorr:     bool  = False
    use_attn_dispersion:      bool  = False
    # 2026-07-23: idea #3 — KNNBiasAttention의 학습되는 RelativePositionBias(MLP)를 고정(학습
    # 파라미터 없는) 거리감쇠 커널 bias=-dist/tau로 교체(train.py --knn-fixed-bias-attention).
    # q/k/v/out projection은 그대로 학습되므로 "bias MLP만" 없앤 것이 효과가 있는지 확인한다.
    use_knn_fixed_bias_attn:  bool  = False
    knn_bias_tau:              float = 50.0
    # 2026-07-23: PSA-MIL(WACV 2026, arXiv:2503.16284) "learnable distance-decayed prior"의
    # 경량 버전 — tau를 고정 상수 대신 head별로 학습되는 스칼라(head 개수만큼, 보통 2개)로
    # 둔다(train.py --learnable-tau, --knn-fixed-bias-attention과 함께 사용).
    knn_bias_learnable_tau:   bool  = False


@dataclass
class DataConfig:
    wsi_root_tcga: str          = "data/tcga_paad_wsi"
    wsi_root_cptac: str         = "data/cptac_pda_wsi"
    patches_root_tcga: str      = "data/patches_tcga"
    patches_root_cptac: str     = "data/patches_cptac"
    num_workers: int            = 0
    precomputed: bool           = True
    seed: int                   = 42  # case 단위 train/val/test stratified split 재현성 (data/dataset.py 참조)


@dataclass
class TrainConfig:
    epochs:                int   = 30
    lr:                    float = 1e-5
    weight_decay:          float = 1e-1
    device:                str   = "cuda"
    seed:                  int   = 42
    # --- GPU 최적화 파라미터 ---
    # gradient accumulation 단위 = 환자 1명(보유한 모든 노드 누적 후 1 step, train.py 참조)
    warmup_epochs:         int   = 3       # linear LR warmup → cosine decay (epochs의 10%, 표준 warmup 비율)
    # 대형 WSI(수천 패치)에서 CNN OOM 방지용 서브배치
    cnn_chunk_size:        int   = 64
    # Cox PH loss는 위험집합(risk set) 비교를 위해 여러 환자를 한 배치로 묶어야 한다.
    # 값이 클수록 risk set 추정이 안정적이지만, 환자별 forward activation을 배치가 찰 때까지
    # 모두 메모리에 들고 있어야 하므로 GPU 메모리 사용량도 함께 늘어난다.
    cox_batch_size:        int   = 16


@dataclass
class LightTrainConfig:
    """WSI 없이 Clinical/RNA만 쓰는 모델(M5/M6/M6X/M7, train_light.py) 학습 설정.

    TrainConfig와 lr/weight_decay가 다른 이유: TrainConfig.lr(1e-5)은 ViT self-attention +
    ABMIL이 포함된 WSI 스택의 학습 안정성을 위해 낮게 잡은 값이다. 여기 모델들은 그런
    구조가 전혀 없는 작은 MLP(clinical/RNA 인코더 + risk_head)뿐이라 그 정도로 낮출 이유가
    없다 — 실제로 train_clinical_rna_only.py(M7)가 lr=1e-3(Adam 기본값 수준)으로 지금까지
    가장 좋은 external 성능(0.575)을 냈다. M5/M6/M6X를 train.py에 배선하면서 이 값 대신
    TrainConfig를 그대로 물려받은 게 이 프로젝트의 실수였다 — train_light.py는 그 실수를
    바로잡되, 여전히 config.py 한 곳에서 모든 하이퍼파라미터를 관리한다는 원칙은 유지한다.

    embed_dim/dropout(모델 폭)은 여기 두지 않는다 — train.py로 배선된 M5/M6/M6X와 동일한
    ModelConfig(cfg.model)를 그대로 써서, train_light.py와 train.py의 결과 차이가 "아키텍처"가
    아니라 "이 학습 설정(lr/schedule)"때문임을 명확히 분리해 비교할 수 있게 한다.
    """
    epochs:                int   = 30
    lr:                    float = 1e-3
    weight_decay:          float = 1e-2
    device:                str   = "cuda"
    seed:                  int   = 42
    warmup_epochs:         int   = 3
    cox_batch_size:        int   = 16


@dataclass
class Config:
    model: ModelConfig      = field(default_factory=ModelConfig)
    data:  DataConfig       = field(default_factory=DataConfig)
    train: TrainConfig      = field(default_factory=TrainConfig)
    light: LightTrainConfig = field(default_factory=LightTrainConfig)
