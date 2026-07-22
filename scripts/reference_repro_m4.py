"""
레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) M4(PathologyRNASeqClinicalMIL, WSI+RNA+
Clinical) 코드를 그대로 가져와, 우리 데이터를 넣어 학습/평가한다 — reference_repro_m7.py의
WSI 포함 버전. reference_repo/(git clone)의 실제 nn.Module(MorphologyBurdenPooling,
RNASeqGuidedPathologyFusion 포함)과 cox_ph_loss/harrell_c_index를 직접 import해서 쓴다.

단순화한 부분(사용자 방향 확인됨):
  - feature_extractor: nn.Identity() — 우리 backbone(ResNet50 Lunit SwAV, precomputed
    features.pt, 2048dim)을 그대로 "tile_images" 자리에 넣는다(M4ModelConfig.feature_dim=2048).
    레퍼런스는 UNI2-h(1536dim)를 쓰지만, backbone 자체를 바꿔봐도 개선이 없었던 게 이미 확인된
    사실이라(findings_backlog.md) 우리 캐시된 feature를 그대로 재사용해도 무방하다는 전제.
  - 공간 임베딩(coord_dim=6, x/y_norm 등): use_spatial_embedding=False로 끈다 — 우리 좌표
    포맷(row,col 그리드)이 레퍼런스 포맷과 달라, 이 축은 우리 자체 novelty 영역으로 남겨두고
    이번 통제실험에서는 제외한다(레퍼런스 M4의 self-attention 부재도 이미 확인된 차이).
  - 슬라이드: 케이스당 대표 슬라이드 1장만 사용(data/dataset.py::one_slide_per_case=True,
    TCGA는 DX 우선, CPTAC는 GDC로 확인된 tumor 우선) — 레퍼런스도 환자당 WSI 1개만 쓴다.

학습 레시피는 M4_Train.ipynb 그대로: lr=5e-5, weight_decay=1e-3, epochs=50, patience=15,
case_batch_size=16, grad_clip=1.0, ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6).

--protocol external의 val은 early stopping 전용(모델 선택에 쓰임)이라 그 자체로는 held-out test가
아니다 — 2026-07-21(3차): tcga "test" split(학습에 전혀 관여하지 않은 순수 held-out)을 추가로
평가해 internal_test_c_index(tcga)로 함께 보고한다. 우리 자체 모델들이 보고해온 "Internal C"와
동일한 정의라 나란히 비교 가능하다.

사용법:
    python -m scripts.reference_repro_m4 --protocol external --seed 42
    python -m scripts.reference_repro_m4 --protocol pooled --split-seed 42
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
_REF_ROOT = _ROOT / "reference_repo"

sys.path.insert(0, str(_ROOT))
from config import Config
from data.dataset import WSISurvivalDataset, literature_guided_gene_ids
from data.patch_utils import list_patch_paths, PATCH_TRANSFORM_AUGMENTED
from models.cnn_encoder import CNNEncoder

sys.path.insert(0, str(_REF_ROOT))
from scripts.models.discrete_survival import cox_ph_loss, harrell_c_index  # noqa: E402  (reference_repo)
from scripts.models.m4_pathology_rnaseq_clinical_mil import (  # noqa: E402  (reference_repo)
    M4ModelConfig, PathologyRNASeqClinicalMIL,
)

FEATURE_DIM = 2048  # ResNet50 Lunit SwAV (우리 기본 backbone, features.pt)
MAX_TILES_PER_SLIDE = 512
AUG_ENCODE_BATCH = 8  # utils/extract_features_augmented.py와 동일한 배치 크기(1024px 원본 기준 실측)

# 2026-07-21(5차): "레퍼런스는 매 epoch 실시간 augmentation을 쓴다"가 M4(WSI 포함)와 M7(WSI 없음)
# 간 잔여 격차의 마지막 남은 유력 후보로 좁혀졌다(findings_backlog.md — M7은 pooled split_seed=126
# 에서 헤드라인과 거의 정확히 일치(0.702 vs 0.701)했지만 M4는 같은 seed에서도 0.674로 여전히
# 0.048 모자람 — WSI 쪽에만 남은 격차). --tile-augment는 train forward에서만 patch_paths를
# 다시 읽어 PATCH_TRANSFORM_AUGMENTED로 매번 새로 augment한 뒤 frozen encoder(backbone+pool,
# proj 없이 2048dim 그대로)로 인코딩한다 — val/test/internal_test는 항상 기존 features.pt
# (augmentation 없음, 레퍼런스도 eval엔 미적용).


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_patients(cfg, dataset: str, split: str, rna_gene_ids: list[str]) -> list[dict]:
    """WSISurvivalDataset(one_slide_per_case=True)로 환자당 대표 슬라이드 1개의 item dict를 뽑는다."""
    ds = WSISurvivalDataset(
        cfg, dataset=dataset, split=split,
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
        one_slide_per_case=True,
    )
    patients = []
    for i in range(len(ds)):
        slides = ds[i]
        assert len(slides) == 1, "one_slide_per_case=True인데 슬라이드가 여러 장입니다"
        patients.append(slides[0])
    return patients


# 2026-07-21(4차): reference_repro_m7.py와 동일한 이유(M4_Train.ipynb의
# REQUIRE_COMPLETE_24M_HORIZONS=True 발견) — 단일 조건으로 요약: event=0(censored)이면서
# OS_time이 24개월(730.5일) 미만인 case는 후보 pool에서 제외.
HORIZON_24M_DAYS = 24 * 30.4375


def _complete_horizon_mask(os_time, os_event) -> bool:
    return not (int(os_event) == 0 and float(os_time) < HORIZON_24M_DAYS)


def _pooled_patients(cfg, rna_gene_ids: list[str], require_complete_horizon: bool = False) -> list[dict]:
    """both 코호트를 합친 대표-슬라이드 1장/환자 전체 목록(pooled split 대상 모집단)."""
    ds = WSISurvivalDataset(
        cfg, dataset="both", split="all",
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
        one_slide_per_case=True,
    )
    patients = []
    for i in range(len(ds)):
        slides = ds[i]
        patients.append(slides[0])
    if require_complete_horizon:
        before = len(patients)
        patients = [p for p in patients if _complete_horizon_mask(p["OS_time"].item(), p["OS_event"].item())]
        print(f"[require-complete-horizon] {before} -> {len(patients)} case (24개월 미만 censored {before - len(patients)}명 제외)")
    return patients


def _sample_tile_indices(n_tiles: int, max_tiles: int, training: bool) -> torch.Tensor:
    """레퍼런스 scripts/models/m1_pathology_mil.py::sample_tiles와 동일한 규칙 —
    train: 무작위 서브샘플, eval: 등간격(linspace) 결정론적 서브샘플."""
    if n_tiles <= max_tiles:
        return torch.arange(n_tiles)
    if training:
        return torch.randperm(n_tiles)[:max_tiles]
    return torch.linspace(0, n_tiles - 1, steps=max_tiles).long()


def _slide_dir(cfg_data, dataset: str, slide_id: str) -> Path:
    return Path(getattr(cfg_data, f"patches_root_{dataset}")) / "tiles" / slide_id


def _build_aug_encoder(device) -> CNNEncoder:
    encoder = CNNEncoder(embed_dim=1, with_backbone=True).to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


@torch.no_grad()
def _encode_augmented(aug_encoder, patch_paths: list[Path], device) -> torch.Tensor:
    """patch_paths를 매번 새로 읽어 PATCH_TRANSFORM_AUGMENTED로 augment한 뒤 frozen
    backbone+pool(2048dim, proj 없음)로 인코딩한다 — utils/extract_features_augmented.py의
    단일 슬라이드 버전이지만 캐싱 없이 매 호출(=매 epoch)마다 다시 계산한다."""
    chunks = []
    for i in range(0, len(patch_paths), AUG_ENCODE_BATCH):
        batch = torch.stack([
            PATCH_TRANSFORM_AUGMENTED(Image.open(p).convert("RGB"))
            for p in patch_paths[i:i + AUG_ENCODE_BATCH]
        ]).to(device, non_blocking=True)
        raw = aug_encoder.backbone(batch)
        chunks.append(aug_encoder.pool(raw))
    return torch.cat(chunks)


def _forward_patient(
    model, patient: dict, device, training: bool, age_mean: float, age_std: float,
    tile_augment: bool = False, aug_encoder=None, cfg_data=None,
) -> torch.Tensor:
    if training and tile_augment:
        patch_paths = list_patch_paths(_slide_dir(cfg_data, patient["dataset"], patient["slide_id"]))
        idx = _sample_tile_indices(len(patch_paths), MAX_TILES_PER_SLIDE, training)
        sampled_paths = [patch_paths[i] for i in idx.tolist()]
        tile_images = _encode_augmented(aug_encoder, sampled_paths, device)
    else:
        features = patient["features"]  # (N_tiles, FEATURE_DIM) — augmentation 없는 캐시된 feature
        idx = _sample_tile_indices(features.shape[0], MAX_TILES_PER_SLIDE, training)
        tile_images = features[idx].to(device)
    coords = torch.zeros(len(idx), 6, device=device)  # use_spatial_embedding=False라 미사용
    age_z = (patient["age_years"] - age_mean) / age_std
    clinical = torch.stack([
        age_z, patient["sex_idx"].eq(0).float(), patient["sex_idx"].eq(1).float(),
    ]).unsqueeze(0).to(device)
    rnaseq = patient["rna"].unsqueeze(0).to(device).float()
    out = model(tile_images, coords, rnaseq, clinical)
    return out["logits"].reshape(())


@torch.no_grad()
def _evaluate(model, patients: list[dict], device, age_mean: float, age_std: float) -> float:
    model.eval()
    risks, times, events = [], [], []
    for p in patients:
        risks.append(_forward_patient(model, p, device, training=False, age_mean=age_mean, age_std=age_std).item())
        times.append(float(p["OS_time"].item()))
        events.append(int(p["OS_event"].item()))
    return harrell_c_index(np.array(times), np.array(events), np.array(risks))


def _train_and_eval(
    train, val, test, args, device, age_mean: float, age_std: float, internal_test=None,
    tile_augment: bool = False, aug_encoder=None, cfg_data=None,
) -> tuple:
    config = M4ModelConfig(
        feature_dim=FEATURE_DIM, coord_dim=6, clinical_dim=3, rnaseq_dim=len(literature_guided_gene_ids(1500)),
        spatial_dim=32, clinical_embed_dim=16, rnaseq_hidden_dim=256, rnaseq_embed_dim=256,
        fusion_dim=128, burden_hidden_dim=64, n_outputs=1, dropout=0.40, rnaseq_dropout=0.25,
        max_tiles=MAX_TILES_PER_SLIDE, freeze_feature_extractor=True, use_spatial_embedding=False,
    )
    model = PathologyRNASeqClinicalMIL(nn.Identity(), config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    best_val_c, best_state, epochs_since_improve = -1.0, None, 0
    n_train = len(train)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            risks = torch.stack([
                _forward_patient(model, train[i], device, training=True, age_mean=age_mean, age_std=age_std,
                                  tile_augment=tile_augment, aug_encoder=aug_encoder, cfg_data=cfg_data)
                for i in batch_idx
            ])
            times = torch.tensor([train[i]["OS_time"].item() for i in batch_idx], device=device)
            events = torch.tensor([train[i]["OS_event"].item() for i in batch_idx], device=device)
            loss = cox_ph_loss(risks, times, events)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_c = _evaluate(model, val, device, age_mean=age_mean, age_std=age_std)
        score = val_c if not np.isnan(val_c) else -1.0
        scheduler.step(score)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | lr={lr_now:.2e} | loss={total_loss / max(n_batches, 1):.4f} | val_c_index={val_c:.4f}")

        if score > best_val_c:
            best_val_c, best_state, epochs_since_improve = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (best val_c_index={best_val_c:.4f})")
                break

    model.load_state_dict(best_state)
    test_c = _evaluate(model, test, device, age_mean=age_mean, age_std=age_std)
    if internal_test is not None:
        internal_test_c = _evaluate(model, internal_test, device, age_mean=age_mean, age_std=age_std)
        return best_val_c, test_c, internal_test_c
    return best_val_c, test_c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=str, default="external", choices=["external", "pooled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--require-complete-horizon", action="store_true",
                         help="--protocol pooled 전용: M4_Train.ipynb의 REQUIRE_COMPLETE_24M_HORIZONS와 "
                              "동일하게, 24개월 미만 censored case를 후보 pool에서 제외한다.")
    parser.add_argument("--tile-augment", action="store_true",
                         help="train forward에서만 patch_paths를 매 epoch 다시 읽어 실시간으로 "
                              "augment(PATCH_TRANSFORM_AUGMENTED)한 뒤 frozen encoder로 재인코딩한다 "
                              "— val/test/internal_test는 항상 캐시된 features.pt(augmentation 없음).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    rna_gene_ids = literature_guided_gene_ids(1500)

    internal_test = None
    if args.protocol == "external":
        set_seed(args.seed)
        cfg.data.seed = args.seed
        train = _load_patients(cfg.data, "tcga", "train", rna_gene_ids)
        val   = _load_patients(cfg.data, "tcga", "val",   rna_gene_ids)
        internal_test = _load_patients(cfg.data, "tcga", "test", rna_gene_ids)
        test  = _load_patients(cfg.data, "cptac", "all",  rna_gene_ids)
        print(f"[external] train(tcga)={len(train)}  val(tcga)={len(val)}  "
              f"internal_test(tcga)={len(internal_test)}  test(cptac)={len(test)}")
        eval_label = "external_test_c_index(cptac)"

    else:  # pooled
        set_seed(args.seed)
        pool = _pooled_patients(cfg.data, rna_gene_ids, require_complete_horizon=args.require_complete_horizon)
        case_ids = [p["case_id"] for p in pool]
        datasets = [p["dataset"] for p in pool]
        events = [int(p["OS_event"].item()) for p in pool]
        stratify = [f"{d}_event{e}" for d, e in zip(datasets, events)]
        idx_all = list(range(len(pool)))

        idx_train_valid, idx_test = train_test_split(idx_all, test_size=0.2, random_state=args.split_seed, stratify=stratify)
        stratify_tv = [stratify[i] for i in idx_train_valid]
        idx_train, idx_val = train_test_split(idx_train_valid, test_size=0.25, random_state=args.split_seed, stratify=stratify_tv)

        train = [pool[i] for i in idx_train]
        val   = [pool[i] for i in idx_val]
        test  = [pool[i] for i in idx_test]
        print(f"[pooled] candidate pool={len(pool)}  train={len(train)}  valid={len(val)}  test={len(test)}")
        eval_label = "pooled_test_c_index"

    train_ages = np.array([p["age_years"].item() for p in train], dtype="float64")
    age_mean, age_std = float(train_ages.mean()), float(train_ages.std(ddof=0))
    print(f"train age_mean={age_mean:.2f} age_std={age_std:.2f}")

    aug_encoder = _build_aug_encoder(device) if args.tile_augment else None
    result = _train_and_eval(train, val, test, args, device, age_mean, age_std, internal_test=internal_test,
                              tile_augment=args.tile_augment, aug_encoder=aug_encoder, cfg_data=cfg.data)
    print(f"\n=== RESULT (protocol={args.protocol}, seed={args.seed}, split_seed={args.split_seed}) ===")
    print(f"best_val_c_index={result[0]:.4f}")
    print(f"{eval_label}={result[1]:.4f}")
    if len(result) == 3:
        print(f"internal_test_c_index(tcga)={result[2]:.4f}")


if __name__ == "__main__":
    main()
