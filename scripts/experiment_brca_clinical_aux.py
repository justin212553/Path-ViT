"""
사용자 요청(2026-09-04): "BRCA에도 aux loss 돌려서 떨어지는지 확인해보자" —
scripts/experiment_cptac_clinical_aux.py를 BRCA(N~1058)로 재현. CPTAC에서는 aux loss(staging+
PNI+면역침윤 3-task 평균 cross-entropy)가 30 epoch 내내 랜덤-추측 수준(~1.0)에서 거의 안
줄었다 — WSI 인코더가 이 표본 크기에서는 그 태스크 자체를 못 배운다는 뜻이었다.

BRCA는 PNI/면역침윤 라벨이 아예 없다(cBioPortal PanCanAtlas 확인 완료, 2026-09-04). 대신
staging(ajcc_t, 두 코호트 다 있음)은 그대로 쓰고, 나머지 둘은 BRCA에 실제로 있고 이미 WSI로
잘 학습된다고 확인된 축으로 대체한다:
  - IDC vs ILC 조직형(scripts/diagnose_histology_from_wsi_brca.py에서 AUC 0.925로 이미 검증)
  - PAM50 분자 아형(LumA/LumB/Basal/Her2/Normal, 5클래스)
즉 "PNI/면역침윤과 똑같은 것"이 아니라 "BRCA에서 구할 수 있는, 비슷한 역할(추가 분류 축)을
하는 대체재"다 — 목적은 동일 메커니즘(WSI meanpool_embed -> 보조 분류 헤드)이 표본이 큰
코호트에서는 실제로 학습되는지(aux loss가 떨어지는지) 보는 것.

architecture/학습 루프는 scripts/train_brca_m4.py(ViT_PMA, backbone=uni, gene-selection
consistency)와 scripts/experiment_cptac_clinical_aux.py(clinical_aux_head + branch_risk_out
side-channel, train.py 2026-09-04 확장분)를 그대로 재사용한다.

사용법:
    python -m scripts.experiment_brca_clinical_aux --seed 84 --clinical-aux-weight 1.0
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from models.vit_pma import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, stage_stats_from_csv
from train import _patient_risk, _build_scheduler, _make_amp_ctx, evaluate
from utils.losses import cox_ph_loss
from scripts.brca_common import (
    CLINICAL_PATH, BRCASlideDataset, _identity_collate, load_case_table_kfold, load_rna_matrix,
)

AUX_LABEL_PATH = _ROOT / "data" / "brca_idcilc_pam50_aux.csv"
N_T_CLASSES = 5   # Tis/T1/T2/T3/T4 (models/clinical_encoder.py::_STAGE_ORDINAL_MAPS["ajcc_t"])
N_PAM50_CLASSES = 5


class BRCAClinicalAuxClassifier(nn.Module):
    """experiment_cptac_clinical_aux.py::ClinicalAuxClassifier와 같은 원리(WSI meanpool_embed
    입력, train.py::_patient_risk의 clinical_aux_head 훅과 동일 인터페이스) — BRCA에 실제로
    있는 라벨(staging/IDC-ILC/PAM50)에 맞게 클래스 수만 다르다. train.py 훅은 두 번째/세 번째
    인자 이름을 pni_ord/immune_ord로 부르지만 의미상 여기서는 idcilc_ord/pam50_ord다(훅 자체는
    클래스 수·의미와 무관하게 그냥 3개 ordinal target을 넘길 뿐이라 재사용 가능)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.stage_head = nn.Linear(hidden_dim, N_T_CLASSES)
        self.idcilc_head = nn.Linear(hidden_dim, 2)
        self.pam50_head = nn.Linear(hidden_dim, N_PAM50_CLASSES)

    def forward(self, wsi_meanpool_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(wsi_meanpool_embed)
        return {"stage": self.stage_head(h), "idcilc": self.idcilc_head(h), "pam50": self.pam50_head(h)}

    def loss(self, wsi_meanpool_embed: torch.Tensor, stage_ord: torch.Tensor,
              idcilc_ord: torch.Tensor, pam50_ord: torch.Tensor) -> torch.Tensor | None:
        logits = self.forward(wsi_meanpool_embed)
        losses = []
        if stage_ord.item() >= 0:
            losses.append(F.cross_entropy(logits["stage"].unsqueeze(0), stage_ord.unsqueeze(0)))
        if idcilc_ord.item() >= 0:
            losses.append(F.cross_entropy(logits["idcilc"].unsqueeze(0), idcilc_ord.unsqueeze(0)))
        if pam50_ord.item() >= 0:
            losses.append(F.cross_entropy(logits["pam50"].unsqueeze(0), pam50_ord.unsqueeze(0)))
        if not losses:
            return None
        return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clinical-aux-weight", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()
    cfg = Config()
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs

    aux_df = pd.read_csv(AUX_LABEL_PATH).set_index("case_id")
    aux_map = {cid: (int(r["idcilc_label"]), int(r["pam50_label"])) for cid, r in aux_df.iterrows()}
    print(f"IDC/ILC+PAM50 라벨 로드: {len(aux_map)}명 (data/brca_idcilc_pam50_aux.csv)")

    gene_path = Path("data/brca_rna_gene_selection_consistency/selected_genes.csv")
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)
    stage_stats = stage_stats_from_csv(CLINICAL_PATH)

    cases = load_case_table_kfold(args.seed, args.fold, args.n_folds, external_tss=None)
    rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(Path("data/brca_slide_manifest.csv"))
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    print(f"case 수: train={int((cases['split']=='train').sum())} val={int((cases['split']=='val').sum())} "
          f"test={int((cases['split']=='test').sum())}")

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_ds = BRCASlideDataset(cases[cases["split"] == "train"], rna_df, manifest, with_staging=True)
    val_ds   = BRCASlideDataset(cases[cases["split"] == "val"],   rna_df, manifest, with_staging=True)
    test_ds  = BRCASlideDataset(cases[cases["split"] == "test"],  rna_df, manifest, with_staging=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=True, backbone="uni", use_staging=True, stage_stats=stage_stats,
    ).to(device)
    model.clinical_aux_head = BRCAClinicalAuxClassifier(cfg.model.embed_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)
    batch_size = cfg.train.cox_batch_size
    chunk_size = cfg.train.cnn_chunk_size

    best_val_c = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "cnn") and model.cnn.backbone is not None:
            model.cnn.backbone.eval()
        risks, times, events, aux_losses = [], [], [], []
        epoch_aux_values = []

        def _flush():
            nonlocal risks, times, events, aux_losses
            if not risks:
                return
            risk_t = torch.cat(risks)
            time_t = torch.cat(times).to(device)
            event_t = torch.cat(events).to(device)
            loss = cox_ph_loss(risk_t, time_t, event_t)
            if aux_losses:
                loss = loss + args.clinical_aux_weight * torch.stack(aux_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            risks, times, events, aux_losses = [], [], [], []

        for patient_slides in train_loader:
            if len(patient_slides) == 0:
                continue
            case_id = patient_slides[0]["case_id"]
            idcilc_label, pam50_label = aux_map.get(case_id, (-1, -1))
            branch_risk_out = {
                "pni_ord": torch.tensor(idcilc_label, dtype=torch.long, device=device),
                "immune_ord": torch.tensor(pam50_label, dtype=torch.long, device=device),
            }
            risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, chunk_size, 1.0,
                                        branch_risk_out=branch_risk_out)
            risks.append(risk)
            times.append(patient_slides[0]["OS_time"])
            events.append(patient_slides[0]["OS_event"])
            clinical_aux_loss = branch_risk_out.get("clinical_aux_loss")
            if clinical_aux_loss is not None:
                aux_losses.append(clinical_aux_loss)
                epoch_aux_values.append(clinical_aux_loss.item())
            if len(risks) >= batch_size:
                _flush()
        _flush()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, cfg, device, amp_ctx, None, desc=f"epoch{epoch+1}val")
        aux_mean = sum(epoch_aux_values) / len(epoch_aux_values) if epoch_aux_values else float("nan")
        print(f"epoch {epoch+1:3d} | val_c_index={val_metrics['c_index']:.4f} "
              f"| aux_loss_mean(n={len(epoch_aux_values)}/{len(train_ds)})={aux_mean:.4f}")
        if val_metrics["c_index"] > best_val_c:
            best_val_c = val_metrics["c_index"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nbest val_c_index={best_val_c:.4f}")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, None, desc="test")
    print(f"internal(brca test) c_index = {test_metrics['c_index']:.4f}")


if __name__ == "__main__":
    main()
