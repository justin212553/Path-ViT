"""
M1/M2/M3/PMA 동시 학습 — frozen CNN backbone을 선택한 모델들이 공유해 실시간 augmentation
비용을 줄이는 스크립트 (2026-07-28). 어떤 모델을 같이 돌릴지 --M1/--M2/--M3/--PMA로 그때그때
켜고 끌 수 있다 — 나중에 또 여러 모델을 동시 학습해야 할 때 재사용하기 위한 범용 버전.

배경: models/cnn_encoder.py::CNNEncoder는 backbone(ResNet50)+proj(Linear→embed_dim) 두
단계인데, train.py가 항상 backbone만 requires_grad_(False)로 얼린다 — 즉 backbone은 어떤
모델을 골라도 전부 완전히 동일한(학습 안 되는) 값을 내놓는다. 반면 proj/ViT(Nystromformer)/
pooling/risk_head는 모델마다 다른 loss로 학습되므로 공유 불가. 벤치마크(2026-07-28) 결과
패치 1개당 CNN backbone forward가 ~259ms, 그 이후(ViT+pooling+risk_head)가 ~2.5ms로 backbone이
전체 비용의 99%를 차지한다 — 이 backbone 계산을 여러 모델이 공유하면 이론상 Nx가 아니라
~1.03x 시간에 N개 모델을 다 학습시킬 수 있다.

모델 매핑(전부 --rna-genes literature_1500 "_EX" 기준):
    --M1  WSI만                    (ViT_M1)
    --M2  WSI + Clinical           (ViT_M2)
    --M3  WSI + RNA(clinical 제외) (ViT_PMA, use_clinical=False + dispersion)
    --PMA WSI + RNA + Clinical     (ViT_PMA, PMA_EX_SS_AUX + dispersion — M4 슬롯)
M1/M2는 train.py의 ViT_M1/ViT_M2를 그대로 재사용한다(cnn/vit/attn_pool/forward()를 공유하도록
이미 설계돼 있음 — models/vit_m2.py 참조). M3/PMA는 같은 ViT_PMA 클래스를 use_clinical만 다르게
써서 만든다.

train.py는 건드리지 않는다 — 필요한 헬퍼(스케줄러/로그포맷/평가/seed 등)는 그대로 import해서
재사용하고, "CNN을 한 번만 계산해 선택된 모델들에 나눠준다"는 새 로직만 이 파일에 둔다.

사용법:
    python train_multi.py --dataset tcga --seed 42 --M1 --M2 --M3 --tile-augment --external --group-ts m123_pilot
    python train_multi.py --dataset tcga --seed 42 --M1 --PMA --tile-augment --external  # 필요한 것만 골라서
"""
import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids
from data.patch_utils import PATCH_TRANSFORM_512, PATCH_TRANSFORM_AUGMENTED_CACHED, TileLRUCache
from models import ViT_M1, ViT_M2, ViT_PMA
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils import load_env, send_slack
from utils.losses import cox_ph_loss
from utils.metrics import compute_time_dependent_auc

# train.py를 건드리지 않고 그대로 재사용 — main() 실행 없이 helper만 import(하단 __main__ 가드 확인됨).
from train import set_seed, _build_scheduler, _log_line, _identity_collate, evaluate


def _shared_backbone_features(shared_cnn, patient_slides, tile_cache, transform, device,
                               patch_keep_frac: float = 1.0):
    """슬라이드별로 이미지를 디코딩(+실시간 augmentation)한 뒤, frozen backbone+pool까지만
    "한 번" 계산해 (N, 2048) raw pooled feature를 얻는다 — proj(embed_dim 투영)는 모델마다
    따로 학습되므로 여기서 하지 않는다(각 모델의 forward(features=...)가 자기 proj를 적용).

    2026-07-30: patch_keep_frac<1.0이면(--patch-keep-frac, PatchDropout) 슬라이드별로 패치를
    이 비율만큼 랜덤 서브샘플한다 — train.py::_patient_risk와 동일한 관례(findings_backlog.md
    7번 항목). backbone feature를 모델 4개가 공유하므로, 서브샘플링도 모델별이 아니라 여기서
    슬라이드당 1번만 해야 한다(안 그러면 모델마다 다른 패치를 보게 돼 "backbone 공유"라는
    전제 자체가 깨진다). 이전엔 --patch-keep-frac 인자만 있고 실제로 아무 데도 안 쓰이던
    죽은 코드였다 — val_ds/val_loader와 같은 종류의 버그.

    이 함수는 학습 루프에서만 호출된다(eval은 train.py::evaluate가 모델별로 개별 처리하며
    항상 전체 패치를 씀) — 그래서 train.py처럼 model.training 체크가 따로 필요 없다.

    Returns: [(coords_gpu, pooled_2048_feat), ...] 슬라이드 순서대로.
    """
    out = []
    with torch.no_grad():
        for slide in patient_slides:
            coords = slide["coords"]
            patch_paths = slide["patch_paths"]
            if patch_keep_frac < 1.0:
                n = coords.shape[0]
                k = max(1, round(n * patch_keep_frac))
                idx = torch.randperm(n)[:k]
                coords = coords[idx]
                patch_paths = [patch_paths[i] for i in idx.tolist()]
            coords = coords.to(device, non_blocking=True)
            imgs = torch.stack([transform(tile_cache.get(p)) for p in patch_paths]).to(device, non_blocking=True)
            pooled = shared_cnn.pool(shared_cnn.backbone(imgs))  # (N, 2048) — backbone은 세 모델 다 동일(frozen)
            out.append((coords, pooled))
    return out


def _model_patient_risk(model, patient_slides, shared_feats, device):
    """공유 backbone feature(shared_feats)를 받아 모델 하나의 (risk score, aux_loss)를 계산한다.
    train.py::_patient_risk와 동일한 분기 로직(combine_with_clinical_rna/combine_with_clinical/
    M1 직결)이지만, coords/features를 shared_feats에서 가져온다는 점만 다르다.

    2026-07-30 변경 두 가지:
    (1) M1(ViT_M1)이 patient_spatial_feat를 계산만 하고 실제로는 안 쓰던 버그를 고쳤다 —
        use_attn_dispersion=True인 M1/M2도 이제 risk_head에 반영된다(M2는 combine_with_clinical
        의 spatial_feat 인자로).
    (2) model.rna_aux_head가 있으면(--rna-aux-weight, train.py와 동일한 관례) RNA-free
        meanpool_embed로 RNA 발현을 예측해 aux_loss를 같이 반환한다.

    Returns: (risk, aux_loss) — aux_loss는 model.rna_aux_head가 있을 때만 텐서, 없으면 None.
    """
    z_rna = None
    rna_true = None
    if hasattr(model, "rna_encoder"):
        rna = patient_slides[0]["rna"].to(device, non_blocking=True)
        z_rna = model.encode_rna(rna)
        rna_true = rna

    slide_embeds, slide_spatial_feats, slide_meanpool_embeds = [], [], []
    for (coords, pooled), slide in zip(shared_feats, patient_slides):
        forward_kwargs = {"rna_context": z_rna} if z_rna is not None else {}
        out = model(coords, features=pooled, **forward_kwargs)
        slide_embeds.append(out["embed"])
        if "spatial_feat" in out:
            slide_spatial_feats.append(out["spatial_feat"])
        if "meanpool_embed" in out:
            slide_meanpool_embeds.append(out["meanpool_embed"])

    patient_embed = torch.stack(slide_embeds).mean(dim=0)
    patient_spatial_feat = torch.stack(slide_spatial_feats).mean(dim=0) if slide_spatial_feats else None

    aux_loss = None
    if hasattr(model, "rna_aux_head") and slide_meanpool_embeds:
        patient_meanpool = torch.stack(slide_meanpool_embeds).mean(dim=0)
        rna_pred = model.rna_aux_head(patient_meanpool)
        aux_loss = F.mse_loss(rna_pred, rna_true)

    if hasattr(model, "combine_with_clinical_rna"):
        age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
        sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
        patient_embed = model.combine_with_clinical_rna(
            patient_embed, age_years, sex_idx, z_rna, spatial_feat=patient_spatial_feat,
        )
    elif hasattr(model, "combine_with_clinical"):
        age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
        sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
        patient_embed = model.combine_with_clinical(
            patient_embed, age_years, sex_idx, spatial_feat=patient_spatial_feat,
        )
    elif patient_spatial_feat is not None:
        # M1(ViT_M1)은 combine 메서드가 없어 patient_embed에 직접 이어붙인다.
        patient_embed = torch.cat([patient_embed, patient_spatial_feat], dim=-1)

    risk = model.risk_head(patient_embed.unsqueeze(0)).view(1)
    return risk, aux_loss


class _RiskAccumulator:
    """모델 1개분 risk/time/event를 cox_batch_size만큼 모았다가 flush(loss+backward+step)한다.
    train.py::train_one_epoch::_flush()와 동일한 역할이지만 모델 3개를 각자 독립적으로 관리한다.

    2026-07-30: rna_aux_weight>0(model.rna_aux_head 있는 모델, --rna-aux-weight)이면 cox loss에
    aux_loss 평균을 더한다 — train.py::train_one_epoch::_compute_loss와 동일한 관례.
    """

    def __init__(self, model, optimizer, batch_size: int, rna_aux_weight: float = 0.0):
        self.model, self.optimizer, self.batch_size = model, optimizer, batch_size
        self.rna_aux_weight = rna_aux_weight
        self.risks, self.times, self.events, self.aux_losses = [], [], [], []
        self.total_loss, self.total_batches = 0.0, 0

    def add(self, risk, time_, event_, aux_loss=None):
        self.risks.append(risk)
        self.times.append(time_)
        self.events.append(event_)
        if aux_loss is not None:
            self.aux_losses.append(aux_loss)
        if len(self.risks) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.risks:
            return
        risk_t  = torch.cat(self.risks)
        time_t  = torch.cat(self.times).to(risk_t.device)
        event_t = torch.cat(self.events).to(risk_t.device)
        loss = cox_ph_loss(risk_t, time_t, event_t)
        if self.rna_aux_weight > 0 and self.aux_losses:
            loss = loss + self.rna_aux_weight * torch.stack(self.aux_losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if torch.isfinite(loss) and torch.isfinite(grad_norm):
            self.optimizer.step()
            self.total_loss += loss.item()
            self.total_batches += 1
        self.risks, self.times, self.events, self.aux_losses = [], [], [], []

    def epoch_loss(self) -> float:
        v = self.total_loss / max(self.total_batches, 1)
        self.total_loss, self.total_batches = 0.0, 0
        return v


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--external", action="store_true")
    p.add_argument("--group-ts", type=str, default=None)
    p.add_argument(
        "--tile-augment", action="store_true",
        help="진짜 real-time augmentation(raw image + 매 epoch flip/jitter/blur). 이게 이 스크립트의 "
             "존재 이유(CNN backbone 공유는 raw-image 모드에서만 의미가 있음 — precomputed 모드는 "
             "이미 feature가 캐싱돼 있어 공유할 backbone forward 자체가 없다).",
    )
    p.add_argument("--rna-genes", type=str, default="literature_1500")
    p.add_argument(
        "--patch-keep-frac", type=float, default=0.8,
        help="PatchDropout(train.py --patch-keep-frac와 동일한 관례, findings_backlog.md 7번 "
             "항목). 1.0이면 비활성. 2026-07-30 이전엔 인자만 있고 실제로 안 쓰이던 죽은 "
             "코드였다 — _shared_backbone_features()에서 슬라이드당 1번(모델 4개가 backbone "
             "feature를 공유하므로) 랜덤 서브샘플하도록 고쳤다.",
    )
    p.add_argument(
        "--tile-cache-maxsize", type=int, default=24576,
        help="2026-07-29: --tile-augment 타일 RAM 캐시 상한(항목 개수, data/patch_utils.py::"
             "TileLRUCache). 무제한 전량 캐싱(build_tile_cache, ~22GB 추정)이 이 스크립트(모델 "
             "4개 동시 학습 — 추가 오버헤드가 얹힘)에서 32GB RAM 머신을 스와핑으로 6시간 넘게 "
             "멈추게 한 사고 이후 추가. 기본 24576개(타일당 ~0.75MB 기준 약 18GB)로, 원래 목표치"
             "(~22GB)보다는 낮게, 그리고 사용자 판단(12GB는 과함)에 따라 18GB까지 밀어붙인 값 — "
             "모델 4개분 오버헤드·다른 프로세스가 얹혀도 32GB 안에서 여유를 남긴다.",
    )
    p.add_argument("--M1", action="store_true", help="WSI만(ViT_M1)을 같이 학습할 모델 집합에 포함한다.")
    p.add_argument("--M2", action="store_true", help="WSI+Clinical(ViT_M2)을 포함한다.")
    p.add_argument(
        "--M3", action="store_true",
        help="WSI+RNA(clinical 제외, ViT_PMA use_clinical=False + dispersion)를 포함한다.",
    )
    p.add_argument(
        "--PMA", action="store_true",
        help="WSI+RNA+Clinical(ViT_PMA, PMA_EX_SS_AUX + dispersion — 기존 M4 슬롯)을 포함한다.",
    )
    p.add_argument(
        "--rna-aux-weight", type=float, default=0.0,
        help="2026-07-30: train.py --rna-aux-weight와 동일(models/rna_predictor.py::"
             "RNAPredictionHead) — WSI 표현(RNA-free meanpool_embed)이 RNA 발현도 예측하게 하는 "
             "보조과제. rna_encoder가 있는 모델(--M3/--PMA)에만 적용된다 — M1/M2는 RNA가 없어 "
             "대응되는 게 없으므로 대상에서 제외(WSI-RNA 결합 전략의 일부로 취급, dim 튜닝처럼 "
             "모델 간 균일하게 맞출 항목이 아니라는 판단, 2026-07-30). 0.0(기본)이면 비활성.",
    )
    return p.parse_args()


def main():
    load_env()
    args = _parse_args()
    set_seed(args.seed)
    cfg = Config()
    cfg.data.seed, cfg.train.seed = args.seed, args.seed
    cfg.data.precomputed = False  # 이 스크립트는 raw-image 공유 전용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import contextlib
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset] if args.external else None
    gene_ids = literature_guided_gene_ids(1500) if args.rna_genes == "literature_1500" else None

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])

    # 세 모델이 같은 case 풀/순서를 보게, dataset은 M3(clinical+RNA 둘 다 필요) 기준 하나로
    # 통일해서 만든다 — M1/M2용 patient_slides에서는 rna 필드를 그냥 안 쓰면 된다(재조인 불필요).
    # M1-M7 전체 비교(_EX 기준)가 어차피 같은 case 풀을 전제하므로 이게 공정한 비교이기도 하다.
    ds_kwargs = dict(with_clinical=True, with_rna=True, rna_gene_ids=gene_ids)

    def _make_ds(split, transform):
        return WSISurvivalDataset(cfg.data, dataset=args.dataset, split=split, transform=transform, **ds_kwargs)

    eval_transform = PATCH_TRANSFORM_512
    train_transform = PATCH_TRANSFORM_AUGMENTED_CACHED if args.tile_augment else PATCH_TRANSFORM_512

    train_ds = _make_ds("train", train_transform)
    val_ds   = _make_ds("val",   eval_transform)
    test_ds  = _make_ds("test",  eval_transform)
    external_ds = (
        WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", transform=eval_transform, **ds_kwargs)
        if external_dataset else None
    )

    all_patch_paths = [p for i in range(len(train_ds)) for slide in train_ds[i] for p in slide["patch_paths"]]
    print(f"실시간 augmentation용 타일 프리로드: {len(all_patch_paths):,}개 패치 "
          f"(LRU 캐시 상한 {args.tile_cache_maxsize:,}개)")
    tile_cache = None
    if args.tile_augment:
        tile_cache = TileLRUCache(maxsize=args.tile_cache_maxsize)
        tile_cache.preload(all_patch_paths)

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0, pin_memory=True)
    train_loader    = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader      = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader     = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    external_loader = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds else None

    if not (args.M1 or args.M2 or args.M3 or args.PMA):
        raise ValueError("--M1/--M2/--M3/--PMA 중 최소 하나는 켜야 합니다.")
    if (args.M3 or args.PMA) and gene_ids is None:
        raise ValueError("--M3/--PMA는 RNA 인코더가 필요합니다 — --rna-genes를 확인하세요.")

    # 2026-07-30: dispersion(학습 파라미터 없는 공간 특징, models/spatial_features.py)을 M3/PMA
    # 전용에서 M1/M2까지 확장 — ABMIL(M1/M2)도 attn_weights를 만드므로 원칙적으로 동일하게 적용
    # 가능한데 PMA 계열만 주는 건 "이 모델만 특별 취급"이 되어 공정한 비교를 해친다는 판단
    # (findings_backlog.md 참조). ViT_PMA는 cfg.model.use_attn_dispersion을 읽고, ViT_M1/ViT_M2는
    # 생성자 인자로 명시적으로 받는다(vit_m1.py/vit_m2.py 2026-07-30 변경).
    cfg.model.use_attn_dispersion = True

    models = {}
    if args.M1:
        models["M1"] = ViT_M1(cfg.model, precomputed=False, backbone="resnet50",
                               use_attn_dispersion=True).to(device)
    if args.M2:
        models["M2"] = ViT_M2(cfg.model, age_mean=age_mean, age_std=age_std,
                               precomputed=False, backbone="resnet50",
                               use_attn_dispersion=True).to(device)
    if args.M3:
        models["M3"] = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(gene_ids),
                                precomputed=False, backbone="resnet50", use_clinical=False).to(device)
    if args.PMA:
        models["PMA"] = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(gene_ids),
                                 precomputed=False, backbone="resnet50", use_clinical=True).to(device)

    # 2026-07-30: --rna-aux-weight — rna_encoder가 있는 모델(M3/PMA)에만 붙인다. WSI-RNA 결합
    # 전략의 일부로 취급(M1/M2엔 대응 없음, dim 튜닝과 달리 모델 간 균일화 대상이 아님 — 사용자
    # 판단). optimizer 생성 *이전에* 붙여야 rna_aux_head 파라미터가 옵티마이저에 포함된다.
    if args.rna_aux_weight > 0:
        for name, m in models.items():
            if hasattr(m, "rna_encoder"):
                m.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(gene_ids)).to(device)

    for m in models.values():
        m.cnn.backbone.requires_grad_(False)
    # backbone/pool은 선택된 모델 전부 frozen+동일 아키텍처(같은 사전학습 가중치)라, 그중 아무
    # 인스턴스나 하나를 "공유용"으로 통일해 계산해도 결과가 달라지지 않는다.
    shared_cnn = next(iter(models.values())).cnn

    print("선택된 모델: " + ", ".join(models) + " | " +
          " | ".join(f"{name} params={sum(p.numel() for p in m.parameters()):,}" for name, m in models.items()))
    optimizers = {
        name: torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()),
                                 lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        for name, m in models.items()
    }
    schedulers = {name: _build_scheduler(opt, cfg) for name, opt in optimizers.items()}
    accumulators = {
        name: _RiskAccumulator(m, optimizers[name], cfg.train.cox_batch_size,
                                rna_aux_weight=args.rna_aux_weight if hasattr(m, "rna_aux_head") else 0.0)
        for name, m in models.items()
    }

    model_tag = "".join(models)  # 예: "M1M2M3" 또는 "M1PMA"
    if args.rna_aux_weight > 0:
        model_tag += "_AUX"

    # 2026-07-30: val_ds/val_loader는 만들어져 있었지만 실제로는 어디서도 안 쓰이던 죽은
    # 코드였다 — epoch마다 마지막 가중치를 그대로 최종 평가에 썼는데, train.py(원본)는 매
    # epoch val로 평가해 best-val 체크포인트를 저장/로드하는 방식이라 다른 실험들과 공정한
    # 비교가 안 됐다. 여기서도 같은 패턴(모델별 best-val 체크포인트)으로 맞춘다.
    ckpt_dir = Path(__file__).parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_paths = {name: ckpt_dir / f"survival_{args.dataset}_{model_tag}_{name}_best_multi.pt"
                  for name in models}
    best_scores = {name: -1.0 for name in models}

    if WANDB_AVAILABLE:
        run_ts = datetime.now().strftime("%m%d::%H%M")
        wandb.init(project="Path-ViT", name=f"{args.dataset.upper()}_{model_tag}_seed{args.seed}_{run_ts}",
                   group=f"{model_tag}_{args.group_ts or run_ts}",
                   config={"dataset": args.dataset, "seed": args.seed, "tile_augment": args.tile_augment,
                           "models": list(models)})

    epoch_times = []
    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        for m in models.values():
            m.train()
            if hasattr(m, "cnn") and m.cnn.backbone is not None:
                m.cnn.backbone.eval()

        for patient_slides in train_loader:
            if len(patient_slides) == 0:
                continue
            with amp_ctx:
                shared_feats = _shared_backbone_features(shared_cnn, patient_slides, tile_cache, train_transform,
                                                          device, patch_keep_frac=args.patch_keep_frac)
                time_t  = patient_slides[0]["OS_time"]
                event_t = patient_slides[0]["OS_event"]
                for name, m in models.items():
                    risk, aux_loss = _model_patient_risk(m, patient_slides, shared_feats, device)
                    accumulators[name].add(risk, time_t, event_t, aux_loss)

        for acc in accumulators.values():
            acc.flush()
        for sch in schedulers.values():
            sch.step()

        val_metrics = {}
        for name, m in models.items():
            m.eval()
            val_metrics[name] = evaluate(m, val_loader, cfg, device, amp_ctx, eval_transform)
        # 다음 epoch 맨 위(for epoch 루프 시작 부분)에서 다시 m.train()을 걸어주므로 여기서
        # train 모드로 되돌릴 필요 없음.

        for name, m in models.items():
            c_index = val_metrics[name]["c_index"]
            score = c_index if not math.isnan(c_index) else -1.0
            if score > best_scores[name]:
                best_scores[name] = score
                torch.save({
                    "model_state_dict": m.state_dict(),
                    "epoch":            epoch + 1,
                    "val_c_index":      score,
                    "val_hr":           val_metrics[name]["hr"],
                    "val_log_rank_p":   val_metrics[name]["log_rank_p"],
                }, ckpt_paths[name])

        dt = time.time() - t0
        epoch_times.append(dt)
        losses = {name: acc.epoch_loss() for name, acc in accumulators.items()}
        print(f"Epoch {epoch+1:3d} | {dt:6.1f}s | " +
              " | ".join(f"{name}_loss={losses[name]:.4f}" for name in models) + " | " +
              " | ".join(f"{name}_val_c={val_metrics[name]['c_index']:.4f}" for name in models))
        if WANDB_AVAILABLE:
            log_dict = {f"train/{name}_loss": losses[name] for name in models}
            log_dict |= {f"val/{name}_c_index": val_metrics[name]["c_index"] for name in models}
            log_dict["epoch_seconds"] = dt
            wandb.log(log_dict, step=epoch + 1)

    print(f"\n평균 epoch 시간: {sum(epoch_times)/len(epoch_times):.1f}s "
          f"({len(models)}개 모델({model_tag}) 합계 — 독립 실행 {len(models)}회 대비 예상 절감률은 벤치마크 참조)")

    # 최종 평가 전, epoch 루프에서 저장해둔 모델별 best-val 체크포인트로 되돌린다 — 마지막
    # epoch이 아니라 best-val 시점 가중치로 test/external을 평가해야 다른 실험들과 공정하게
    # 비교된다.
    for name, m in models.items():
        if ckpt_paths[name].exists():
            ckpt = torch.load(ckpt_paths[name], map_location=device)
            m.load_state_dict(ckpt["model_state_dict"])
            print(f"{name}: best checkpoint 로드(epoch {ckpt['epoch']}, "
                  f"val_c_index={ckpt['val_c_index']:.4f})")

    # 최종 평가는 모델당 한 번뿐이라 공유 최적화가 필요 없음 — 기존 evaluate()를 그대로 재사용.
    for name, m in models.items():
        m.eval()
        train_metrics = evaluate(m, DataLoader(train_ds, shuffle=False, **dl_kwargs), cfg, device, amp_ctx, eval_transform)
        test_metrics  = evaluate(m, test_loader, cfg, device, amp_ctx, eval_transform)
        test_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"],
            test_metrics["times"], test_metrics["events"], test_metrics["risks"],
        )
        print(f"\n=== {name} Internal Test ===")
        print(_log_line(f"{name}_test", test_metrics, test_td_auc))
        if external_loader is not None:
            external_metrics = evaluate(m, external_loader, cfg, device, amp_ctx, eval_transform)
            external_td_auc = compute_time_dependent_auc(
                train_metrics["times"], train_metrics["events"],
                external_metrics["times"], external_metrics["events"], external_metrics["risks"],
            )
            print(f"=== {name} External Test ({external_dataset}) ===")
            print(_log_line(f"{name}_external", external_metrics, external_td_auc))
            if WANDB_AVAILABLE:
                wandb.run.summary[f"{name}_external_c_index"] = external_metrics["c_index"]
                wandb.run.summary[f"{name}_test_c_index"] = test_metrics["c_index"]

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        load_env()
        send_slack(f":x: *Path-ViT (M1/M2/M3 동시학습) 에러*\n```{type(e).__name__}: {e}```")
        raise
