"""
diagnose_m4_branch_gradients.py에서 RNA gradient가 WSI보다 계속 3배 가까이 컸던 것을 보고
나온 질문(사용자): "WSI도 clinical처럼 cox_add 가산항으로 따로 빼면 얼마나 기여하는지/성능이
얼마나 바뀌는지 보자". risk_head(WSI+RNA concat -> 1개 스칼라)를 WSI 전용/RNA 전용 두 개의
독립 선형항으로 쪼갠 뒤(risk = wsi_linear(z_wsi) + rna_linear(z_rna) + clinical_linear(z_clin)),
한 fold/seed(디폴트 fold=0, seed=84 — 실제 2seed x 5fold 학습에서 이미 나온 baseline과 직접
비교 가능한 조합)만 학습해서:
  1) internal/external c-index가 기존(단일 risk_head, 이미 학습된 baseline: internal 0.503,
     external 0.622, .logs/run_m4_pdac1500cnvmut_kfold_local.log)보다 오르는지
  2) diagnose_m7_branch_contrib.py와 동일한 방식(항별 risk 분산 설명비율)으로 WSI/RNA/clinical
     각각이 최종 risk에 얼마나 기여하는지

를 확인한다. models/vit_m4.py 자체는 건드리지 않고(다른 모델/이미 도는 학습에 영향 없음), risk_head를
학습 전에 여기서만 교체한다 — model.risk_head가 그냥 nn.Module이라 다른 어떤 배선도 안 바꿔도
_patient_risk()가 그대로 동작한다(train.py를 import해 재사용).

사용법:
    python -m scripts.experiment_m4_wsi_cox_add --dataset tcga --seed 84 --fold 0 --n-folds 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_consistency_gene_ids
from models import ViT_M4
from models.clinical_encoder import (
    age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, mutation_stats_from_df, STAGE_FIELDS,
)
from utils.losses import cox_ph_loss
from train import (
    _patient_risk, _build_scheduler, _identity_collate, _make_amp_ctx, evaluate,
    _stage_ord_from_patient, _margin_ord_from_patient, _mutation_ord_from_patient,
    _branch_param_groups,
)


class SplitAdditiveHead(nn.Module):
    """model.risk_head 대체 — 입력 (1, 2*embed_dim)=[z_wsi‖z_rna]을 반으로 갈라 각각 독립
    선형층(risk_head와 동일하게 LayerNorm+Linear)에 통과시킨 뒤 더한다. clinical_linear(이미
    별도 가산항, models/vit_m4.py)와 대칭인 구조 — 세 항 다 순수 가산(additive)이 된다.
    forward 호출마다 last_wsi_term/last_rna_term에 값을 저장해 나중에 분산 분해에 쓴다."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.wsi_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.rna_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.last_wsi_term = None
        self.last_rna_term = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_wsi, z_rna = x[:, : self.embed_dim], x[:, self.embed_dim :]
        wsi_term = self.wsi_head(z_wsi)
        rna_term = self.rna_head(z_rna)
        self.last_wsi_term = wsi_term.detach()
        self.last_rna_term = rna_term.detach()
        return (wsi_term + rna_term).view(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clinical-lr-mult", type=float, default=1.0,
                         help="2026-09-04: train.py --clinical-lr-mult 이식 — 지금까지 M4에서 "
                              "가장 잘 나온 개입(CLR100)을 WSI-split 구조 위에도 얹어보기 위함 "
                              "(사용자 지시). --lr-mult-warmup-epochs와 함께 써야 안전.")
    parser.add_argument("--lr-mult-warmup-epochs", type=int, default=0,
                         help="train.py --lr-mult-warmup-epochs와 동일 — clinical-lr-mult를 "
                              "1.0배에서 목표 배율까지 이 epoch 수에 걸쳐 선형으로 올린다.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()
    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset]

    cfg = Config()
    cfg.data.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs

    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    mutation_stats = mutation_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids,
        fold=args.fold, n_folds=args.n_folds,
    )
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", **ds_kwargs)
    val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   **ds_kwargs)
    test_ds  = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  **ds_kwargs)
    external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} external={len(external_ds)}")

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_loader    = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader      = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader     = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    external_loader = DataLoader(external_ds, shuffle=False, **dl_kwargs)

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_M4(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
        use_mutation=True, mutation_stats=mutation_stats,
    ).to(device)
    # risk_head(단일 2D->1 선형)를 WSI/RNA 독립 가산항으로 교체 — 다른 배선은 전혀 안 바꾼다.
    model.risk_head = SplitAdditiveHead(cfg.model.embed_dim).to(device)

    # 2026-09-04: train.py의 (버그 수정된) --clinical-lr-mult 로직 그대로 이식. risk_head는
    # SplitAdditiveHead로 교체됐지만 _BRANCH_ATTRS 어디에도 안 걸려("other" 취급) — 원래
    # ViT_M4의 risk_head도 마찬가지라 이 스크립트만의 특별 취급은 필요 없다.
    lr_mult_warmup_targets: list[tuple[int, float]] = []
    if args.clinical_lr_mult != 1.0:
        branch_groups = _branch_param_groups(model)
        base_params = list(branch_groups["other"]) + branch_groups["rna"] + branch_groups["wsi"]
        param_groups = [{"params": base_params, "lr": cfg.train.lr}]
        if not branch_groups["clinical"]:
            raise ValueError("--clinical-lr-mult != 1.0인데 clinical 파라미터가 없는 모델입니다.")
        param_groups.append({"params": branch_groups["clinical"], "lr": cfg.train.lr * args.clinical_lr_mult})
        lr_mult_warmup_targets.append((len(param_groups) - 1, args.clinical_lr_mult))
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)
    batch_size = cfg.train.cox_batch_size
    chunk_size = cfg.train.cnn_chunk_size

    best_val_c = -1.0
    best_state = None
    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        if args.lr_mult_warmup_epochs > 0 and lr_mult_warmup_targets:
            progress = min((epoch + 1) / args.lr_mult_warmup_epochs, 1.0)
            for group_idx, target_mult in lr_mult_warmup_targets:
                effective_mult = 1.0 + (target_mult - 1.0) * progress
                optimizer.param_groups[group_idx]["lr"] = lr_now * effective_mult
        model.train()
        if hasattr(model, "cnn") and model.cnn.backbone is not None:
            model.cnn.backbone.eval()
        risks, times, events = [], [], []

        def _flush():
            nonlocal risks, times, events
            if not risks:
                return
            risk_t = torch.cat(risks)
            time_t = torch.cat(times).to(device)
            event_t = torch.cat(events).to(device)
            loss = cox_ph_loss(risk_t, time_t, event_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            risks, times, events = [], [], []

        for patient_slides in train_loader:
            if len(patient_slides) == 0:
                continue
            risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size, 1.0)
            risks.append(risk)
            times.append(patient_slides[0]["OS_time"])
            events.append(patient_slides[0]["OS_event"])
            if len(risks) >= batch_size:
                _flush()
        _flush()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, cfg, device, amp_ctx, None, desc=f"epoch{epoch+1}val")
        print(f"epoch {epoch+1:3d} | val_c_index={val_metrics['c_index']:.4f}")
        if val_metrics["c_index"] > best_val_c:
            best_val_c = val_metrics["c_index"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nbest val_c_index={best_val_c:.4f} — 이 시점 가중치로 test/external 평가")
    model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, None, desc="test")
    external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, None, desc="external")
    print(f"\n=== 결과 (fold={args.fold}, seed={args.seed}) ===")
    print(f"  internal test c_index = {test_metrics['c_index']:.4f}  "
          f"(baseline 단일 risk_head: 0.5033, .logs/kfold_preds 재계산)")
    print(f"  external    c_index = {external_metrics['c_index']:.4f}  "
          f"(baseline 단일 risk_head: 0.6219, .logs/run_m4_pdac1500cnvmut_kfold_local.log)")

    # 2026-09-03: "일단 한번 해보자"(사용자) — 2seed x 5fold 전체로 돌려 pool_multiseed_
    # kfold_preds.py/pool_multiseed_external_preds.py로 baseline(M4_..._COX_ADD)과 직접
    # 비교하기 위해, 같은 파일명 관례(train.py의 kfold_preds/external_preds 저장 형식)로
    # CSV를 남긴다. 태그에 _WSISPLIT을 붙여 baseline과 절대 안 섞이게 한다.
    model_tag = "M4_PDACCONS1500_CNV_STG_R_MUT_COX_ADD_WSISPLIT"
    if args.clinical_lr_mult != 1.0:
        model_tag += f"_CLR{int(args.clinical_lr_mult)}"
    if args.lr_mult_warmup_epochs > 0:
        model_tag += f"_LRMW{args.lr_mult_warmup_epochs}"
    import csv
    kfold_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"
    kfold_dir.mkdir(parents=True, exist_ok=True)
    kfold_path = kfold_dir / f"{args.dataset}_{model_tag}_seed{args.seed}_fold{args.fold}of{args.n_folds}.csv"
    with open(kfold_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
        for cid, risk, t, e in zip(test_metrics["case_ids"], test_metrics["risks"],
                                    test_metrics["times"], test_metrics["events"]):
            writer.writerow([cid, risk, t, e])
    print(f"  -> internal predictions saved: {kfold_path}")

    ext_dir = Path(__file__).parent.parent / ".logs" / "external_preds"
    ext_dir.mkdir(parents=True, exist_ok=True)
    ext_path = ext_dir / f"{external_dataset}_{model_tag}_seed{args.seed}_fold{args.fold}of{args.n_folds}.csv"
    with open(ext_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
        for cid, risk, t, e in zip(external_metrics["case_ids"], external_metrics["risks"],
                                    external_metrics["times"], external_metrics["events"]):
            writer.writerow([cid, risk, t, e])
    print(f"  -> external predictions saved: {ext_path}")

    print("\n=== 항별(WSI/RNA/clinical) risk 기여도 분해 — internal 전체(split=all) ===")
    all_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="all", **{
        k: v for k, v in ds_kwargs.items() if k not in ("fold", "n_folds")
    })
    all_loader = DataLoader(all_ds, batch_size=1, collate_fn=_identity_collate, num_workers=0)
    model.eval()
    terms = {"wsi": [], "rna": [], "clinical": []}
    with torch.no_grad():
        for patient_slides in all_loader:
            if not patient_slides:
                continue
            _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size)
            terms["wsi"].append(model.risk_head.last_wsi_term.item())
            terms["rna"].append(model.risk_head.last_rna_term.item())
            p = patient_slides[0]
            age_years = p["age_years"].to(device)
            sex_idx = p["sex_idx"].to(device)
            stage_ord = _stage_ord_from_patient(patient_slides, device)
            margin_ord = _margin_ord_from_patient(patient_slides, device)
            mutation_ord = _mutation_ord_from_patient(patient_slides, device)
            clin_embed = model._clinical_embed(age_years, sex_idx, margin_ord, stage_ord=stage_ord,
                                                mutation_ord=mutation_ord)
            terms["clinical"].append(model.clinical_linear(clin_embed).view(1).item())

    arrs = {k: np.array(v) for k, v in terms.items()}
    total = sum(arrs.values())
    total_var = total.var()
    print(f"  N={len(total)}명, total risk std={total.std():.4f}\n")
    print(f"  {'항':10s} {'mean':>10s} {'std':>10s} {'설명분산비율':>12s}")
    for name, arr in arrs.items():
        explained = arr.var() / total_var if total_var > 0 else float("nan")
        print(f"  {name:10s} {arr.mean():10.4f} {arr.std():10.4f} {explained:12.2%}")


if __name__ == "__main__":
    main()
