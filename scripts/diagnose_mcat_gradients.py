"""
ViT_MCAT(--MCAT) 학습을 재현하며(1시드, 진단 목적이라 멀티시드 불필요), 매 Cox 배치
(cox_batch_size명 단위) backward 직후·optimizer.step() 이전에 브랜치별 파라미터
gradient L2 norm + epoch별 평균 Cox loss를 잰다. scripts/diagnose_wsi_gradients.py
(ViT_PMA용)를 ViT_MCAT에 맞게 그대로 적용한 것.

배경: 2026-08-31 --eval-internal-ckpt 진단(train.py)에서, 실제 30-epoch 학습이 끝난
seed84/fold0 MCAT checkpoint를 봤더니 pathway token(GeneGroupEncoder 출력)은 환자별로
잘 갈라지는데(cosine sim 0.0039, collapse 아님) co-attention entropy는 0.9998로 여전히
완전 uniform이었다 — "query 1개(M4A)라서 저용량이라 붕괴한다"는 Phase 1의 핵심 가설이
반박된 상태. "구조(query 개수)가 아니라 loss/gradient가 문제 아니냐"는 다음 질문에 답하기
위해, 이 스크립트는 학습 "도중" attn_pool(MultiQueryCoAttentionPooling)이 애초에
gene_group_encoder/risk_head 대비 얼마나 강한 gradient를 받는지, 그리고 Cox loss 자체가
epoch에 걸쳐 실제로 내려가는지를 직접 관찰한다. attn_pool의 gradient norm이 학습 초반부터
이미 다른 브랜치보다 훨씬 작다면(diagnose_wsi_gradients.py가 PMA에서 확인한 것과 동일
패턴), 8개 query로 늘려도 attention이 붕괴하는 게 "구조"가 아니라 "그 방향으로 학습
신호 자체가 거의 안 온다"는 뜻이 된다 — 이러면 query를 더 늘리는 방향(설계 변경)보다
loss 자체를 손보는 방향(예: attn_pool 전용 보조 loss, 또는 PORPOISE처럼 attention에
의존하지 않는 구조)이 맞다는 근거가 된다.

사용법:
    python -m scripts.diagnose_mcat_gradients --seed 84
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import (
    WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection,
    pathway_category_gene_ids,
)
from models import ViT_MCAT
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils.losses import cox_ph_loss
from train import _patient_risk, _build_scheduler, _identity_collate, _make_amp_ctx


def _param_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        # attn_pool을 cnn+vit와 분리한다 — PMA 진단(diagnose_wsi_gradients.py)에서 "WSI 브랜치
        # 전체"가 아니라 "attn_pool 자체"가 다른 모듈 대비 100~250배 작았던 것이 핵심 발견이라,
        # 여기서도 뭉뚱그리면 그 신호를 놓친다.
        "cnn+vit": list(model.cnn.parameters()) + list(model.vit.parameters()),
        "attn_pool": list(model.attn_pool.parameters()),
        "gene_group_encoder": list(model.gene_group_encoder.parameters()),
        "clinical_encoder": list(model.clinical_encoder.parameters()),
        "risk_head": list(model.risk_head.parameters()),
    }
    if hasattr(model, "rna_aux_head"):
        groups["rna_aux_head"] = list(model.rna_aux_head.parameters())
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
    parser.add_argument("--backbone", type=str, default="resnet50")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()

    cfg = Config()
    cfg.data.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    gene_sets = pathway_category_gene_ids()
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])

    ds_kwargs = dict(with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
                      feature_backbone=args.backbone)
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_identity_collate)
    print(f"train patients: {len(train_ds)}")

    model = ViT_MCAT(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
                      gene_ids=rna_gene_ids, gene_sets=gene_sets,
                      precomputed=cfg.data.precomputed, backbone=args.backbone).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)

    groups = _param_groups(model)
    print("브랜치별 파라미터 수:")
    for name, params in groups.items():
        print(f"  {name}: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    patch_keep_frac = 0.8
    rna_aux_weight = 1.0
    batch_size = cfg.train.cox_batch_size
    chunk_size = cfg.train.cnn_chunk_size

    epoch_norms = {name: [] for name in groups}
    epoch_losses = []

    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "cnn") and model.cnn.backbone is not None:
            model.cnn.backbone.eval()
        risks, times, events, aux_losses = [], [], [], []
        batch_norms = {name: [] for name in groups}
        batch_losses = []

        def _flush():
            nonlocal risks, times, events, aux_losses
            if not risks:
                return
            risk_t = torch.cat(risks)
            time_t = torch.cat(times).to(device)
            event_t = torch.cat(events).to(device)
            cox_loss = cox_ph_loss(risk_t, time_t, event_t)
            loss = cox_loss
            if aux_losses:
                loss = loss + rna_aux_weight * torch.stack(aux_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            for name, params in groups.items():
                batch_norms[name].append(_grad_norm(params))
            batch_losses.append(cox_loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            risks, times, events, aux_losses = [], [], [], []

        for patient_slides in train_loader:
            if len(patient_slides) == 0:
                continue
            risk, aux_loss, _ = _patient_risk(
                model, patient_slides, device, amp_ctx, None, chunk_size, patch_keep_frac
            )
            risks.append(risk)
            times.append(patient_slides[0]["OS_time"])
            events.append(patient_slides[0]["OS_event"])
            if aux_loss is not None:
                aux_losses.append(aux_loss)
            if len(risks) >= batch_size:
                _flush()
        _flush()
        scheduler.step()

        for name in groups:
            epoch_norms[name].append(float(np.mean(batch_norms[name])) if batch_norms[name] else 0.0)
        epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else float("nan"))

        line = " | ".join(f"{name}={epoch_norms[name][-1]:.4f}" for name in groups)
        print(f"epoch {epoch+1:3d} | cox_loss={epoch_losses[-1]:.4f} | {line}")

    print("\n=== Cox loss 추이 ===")
    losses = np.array(epoch_losses)
    print(f"  epoch1={losses[0]:.4f}  마지막5epoch평균={losses[-5:].mean():.4f}  "
          f"최솟값={losses.min():.4f}(epoch{int(losses.argmin())+1})")

    print("\n=== 브랜치별 gradient L2 norm(배치 평균) 요약 ===")
    for name in groups:
        arr = np.array(epoch_norms[name])
        print(f"  {name:28s}: epoch1={arr[0]:.4f}  마지막5epoch평균={arr[-5:].mean():.4f}  전체평균={arr.mean():.4f}")

    print("\n=== 상대 비율(risk_head 대비) ===")
    risk_head_mean = np.array(epoch_norms["risk_head"]).mean()
    for name in groups:
        ratio = np.array(epoch_norms[name]).mean() / risk_head_mean if risk_head_mean > 0 else float("nan")
        print(f"  {name:28s}: risk_head 대비 {ratio:.3f}배")


if __name__ == "__main__":
    main()
