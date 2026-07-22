"""
PMA_EX_SS_AUX 학습 레시피를 그대로 재현하되(1시드, 진단 목적이라 멀티시드 불필요), 매
Cox 배치(cox_batch_size=16명 단위) backward 직후·optimizer.step() 이전에 브랜치별
파라미터 gradient L2 norm을 잰다.

배경: diagnose_wsi_reliance.py(옵션 2/3)가 "학습이 다 끝난 뒤" WSI 관련 attention(co-attention
4-관점, 패치 ABMIL)이 거의 완전히 uniform하게 수렴했고, z_wsi를 아예 지워도 성능이 안
변한다는 걸 보여줬다. 이 스크립트는 그게 "왜" 그렇게 되는지 학습 과정 자체에서 확인한다 —
Cox loss가 WSI 브랜치 파라미터(cnn.proj+ViT/Nystromformer+attn_pool+component_coattn)에
애초에 RNA/Clinical 브랜치만큼 강한 학습 신호를 주고 있는지, epoch별 gradient norm으로
직접 비교한다. WSI 브랜치 norm이 학습 초반부터 이미 RNA/Clinical보다 훨씬 작다면,
"모델이 WSI를 학습할 기회 자체를 별로 못 받았다"는 뜻이고, attention 붕괴(uniform)의
직접적 원인 후보가 된다.

사용법:
    python -m scripts.diagnose_wsi_gradients --seed 42
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils.losses import cox_ph_loss
from train import _patient_risk, _build_scheduler, _identity_collate, _make_amp_ctx


def _param_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        "wsi(cnn+vit+pool+coattn)": (
            list(model.cnn.parameters()) + list(model.vit.parameters())
            + list(model.attn_pool.parameters()) + list(model.component_coattn.parameters())
        ),
        "rna_encoder": list(model.rna_encoder.parameters()),
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
    parser.add_argument("--seed", type=int, default=42)
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
    rna_gene_ids = literature_guided_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])

    ds_kwargs = dict(with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids)
    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_identity_collate)
    print(f"train patients: {len(train_ds)}")

    model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
                     precomputed=cfg.data.precomputed).to(device)
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

    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "cnn") and model.cnn.backbone is not None:
            model.cnn.backbone.eval()
        risks, times, events, aux_losses = [], [], [], []
        batch_norms = {name: [] for name in groups}

        def _flush():
            nonlocal risks, times, events, aux_losses
            if not risks:
                return
            risk_t = torch.cat(risks)
            time_t = torch.cat(times).to(device)
            event_t = torch.cat(events).to(device)
            loss = cox_ph_loss(risk_t, time_t, event_t)
            if aux_losses:
                loss = loss + rna_aux_weight * torch.stack(aux_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            for name, params in groups.items():
                batch_norms[name].append(_grad_norm(params))
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

        line = " | ".join(f"{name}={epoch_norms[name][-1]:.4f}" for name in groups)
        print(f"epoch {epoch+1:3d} | {line}")

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
