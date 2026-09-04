"""
diagnose_wsi_gradients.py(ViT_PMA용)를 ViT_M4(--M4, cox_add)로 옮긴 버전 — 2026-09-03 M4로
pdac_consistency_1500+CNV+mutation 레시피를 옮긴 뒤(train.py), "RNA가 여전히 리딩 팩터인가"
(사용자 질문)를 branch별 gradient L2 norm으로 직접 확인한다.

M4(cox_add)는 M7과 달리 WSI+RNA가 risk_head에서 concat되어 함께 나오고(clinical만 별도
가산항) — diagnose_m7_branch_contrib.py식 "설명분산비율" 분해가 WSI/RNA를 갈라내지 못한다.
gradient norm은 encoder별 파라미터로 역전파되는 신호 크기를 직접 재므로 이 문제가 없다
(diagnose_wsi_gradients.py와 동일 원리).

CNV는 RNA 벡터에 이미 concat된 채로 rna_encoder에 들어가므로(data/dataset.py::with_cnv),
"rna_encoder" gradient norm에 RNA+CNV 신호가 함께 반영된다(따로 분리 안 됨).

사용법:
    python -m scripts.diagnose_m4_branch_gradients --seed 84 --epochs 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, pdac_consistency_gene_ids
from models import ViT_M4
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, mutation_stats_from_df
from utils.losses import cox_ph_loss
from train import _patient_risk, _build_scheduler, _identity_collate, _make_amp_ctx


def _param_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    wsi_params = list(model.cnn.parameters()) + list(model.attn_pool.parameters())
    if model.vit is not None:
        wsi_params += list(model.vit.parameters())
    groups = {
        "wsi(cnn+vit+attn_pool)": wsi_params,
        "rna_encoder(RNA+CNV concat)": list(model.rna_encoder.parameters()),
        "clinical_linear(age/sex+stg+margin+mut)": list(model.clinical_linear.parameters()),
        "risk_head": list(model.risk_head.parameters()),
    }
    return groups


def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    return sq ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()

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
    )
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_identity_collate)
    print(f"train patients: {len(train_ds)}")

    rna_input_dim = len(rna_gene_ids) + 8  # +8 = CNV(pathway8 8카테고리 평균)
    model = ViT_M4(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
        use_mutation=True, mutation_stats=mutation_stats,
    ).to(device)

    groups = _param_groups(model)
    print("브랜치별 파라미터 수:")
    for name, params in groups.items():
        print(f"  {name}: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    batch_size = cfg.train.cox_batch_size
    chunk_size = cfg.train.cnn_chunk_size

    epoch_norms = {name: [] for name in groups}

    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "cnn") and model.cnn.backbone is not None:
            model.cnn.backbone.eval()
        # _patient_risk가 combine_mode="cox_add"면 clinical_linear 가산항까지 이미 내부에서
        # 더해서 반환한다(train.py, 2026-09-03 mutation_ord 조건부 배선 포함) — 여기서 따로
        # clinical_term을 재계산해 더하면 이중 계산이 되므로, 반환값을 그대로 최종 risk로 쓴다.
        risks, times, events = [], [], []
        batch_norms = {name: [] for name in groups}

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
            for name, params in groups.items():
                batch_norms[name].append(_grad_norm(params))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            risks, times, events = [], [], []

        for patient_slides in train_loader:
            if len(patient_slides) == 0:
                continue
            p = patient_slides[0]
            risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size, 1.0)
            risks.append(risk)
            times.append(p["OS_time"])
            events.append(p["OS_event"])
            if len(risks) >= batch_size:
                _flush()
        _flush()
        scheduler.step()

        for name in groups:
            epoch_norms[name].append(float(np.mean(batch_norms[name])) if batch_norms[name] else 0.0)

        line = " | ".join(f"{name}={epoch_norms[name][-1]:.4f}" for name in groups)
        print(f"epoch {epoch+1:3d} | {line}")

    print("\n=== 브랜치별 gradient L2 norm(배치 평균) 요약 ===")
    for name in groups:
        arr = np.array(epoch_norms[name])
        print(f"  {name:40s}: epoch1={arr[0]:.4f}  마지막5epoch평균={arr[-5:].mean():.4f}  전체평균={arr.mean():.4f}")

    print("\n=== 상대 비율(risk_head 대비) ===")
    risk_head_mean = np.array(epoch_norms["risk_head"]).mean()
    for name in groups:
        ratio = np.array(epoch_norms[name]).mean() / risk_head_mean if risk_head_mean > 0 else float("nan")
        print(f"  {name:40s}: risk_head 대비 {ratio:.3f}배")


if __name__ == "__main__":
    main()
