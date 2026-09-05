"""
사용자 요청(2026-09-04): "CPTAC엔 이미 staging/신경침윤(PNI)/면역침윤이 환자 단위로 있으니,
이 3가지를 classification하는 aux를 만들어서 CPTAC로 돌리자."

TCGA는 PNI/면역침윤 라벨이 어디에도 없다(GDC/cBioPortal/원논문 세 군데 확인 완료, 2026-09-04
project_wsi_weak_contribution_investigation 메모 참조) — staging만 두 코호트 다 있다. 그래서
이 실험은 **CPTAC를 학습 코호트로, TCGA를 외부검증으로 뒤집는다**(오늘까지의 표준 TCGA->CPTAC
방향과 반대 — 이 실험 전용, 논문 헤드라인 방향을 바꾸는 게 아니다).

models/clinical_aux_classifier.py::ClinicalAuxClassifier(models/stage_predictor.py와 동일 설계
원칙 — RNA/clinical-free WSI meanpool_embed 입력, 예측값은 버리고 그래디언트만 WSI 인코더로
흘림)를 model.clinical_aux_head로 붙이고, train.py::_patient_risk의 branch_risk_out
side-channel(입력 방향으로 확장, 2026-09-04)에 pni_ord/immune_ord를 미리 채워 넣어 매 환자
호출마다 aux loss를 뽑는다. 메인 Cox loss에 --clinical-aux-weight로 가중합.

레시피는 오늘 하루 채택한 M4 baseline과 동일(pdac_consistency_1500+CNV+mutation+staging+
margin+cox_add+uni2native) — 학습 코호트만 CPTAC로 바뀐다.

사용법:
    python -m scripts.experiment_cptac_clinical_aux --fold 0 --n-folds 5 --seed 84 --clinical-aux-weight 1.0
    python -m scripts.experiment_cptac_clinical_aux --clinical-aux-weight 0.0   # aux 없는 baseline 비교용
"""
import argparse
import csv
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
from models.clinical_encoder import (
    age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, mutation_stats_from_df,
)
from models.clinical_aux_classifier import ClinicalAuxClassifier
from utils.losses import cox_ph_loss
from train import (
    _patient_risk, _build_scheduler, _identity_collate, _make_amp_ctx, evaluate,
    _branch_param_groups,
)

PNI_IMMUNE_PATH = _ROOT / "data" / "cptac_pni_immune_aux.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clinical-aux-weight", type=float, default=1.0,
                         help="0.0이면 aux 완전히 꺼짐(비교용 baseline) — clinical_aux_head는 여전히 "
                              "만들어지지만 loss에 안 더해짐(그래디언트도 안 흐름, requires_grad는 살아있으나 "
                              "역전파 경로에 안 들어가므로 사실상 no-op).")
    parser.add_argument("--clinical-lr-mult", type=float, default=1.0)
    parser.add_argument("--lr-mult-warmup-epochs", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()

    cfg = Config()
    cfg.data.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs

    train_dataset, external_dataset = "cptac", "tcga"  # 오늘까지의 표준(tcga->cptac)을 이 실험만 뒤집음

    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[train_dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[train_dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[train_dataset]))
    mutation_stats = mutation_stats_from_df(pd.read_csv(CLINICAL_PATHS[train_dataset]))

    pni_immune_df = pd.read_csv(PNI_IMMUNE_PATH).set_index("case_id")
    pni_immune_map = {
        cid: (int(row["pni_label"]), int(row["immune_label"]))
        for cid, row in pni_immune_df.iterrows()
    }
    print(f"PNI/면역침윤 라벨 로드: {len(pni_immune_map)}명 (data/cptac_pni_immune_aux.csv)")

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone="uni2native",
        fold=args.fold, n_folds=args.n_folds,
    )
    train_ds = WSISurvivalDataset(cfg.data, dataset=train_dataset, split="train", **ds_kwargs)
    val_ds   = WSISurvivalDataset(cfg.data, dataset=train_dataset, split="val",   **ds_kwargs)
    test_ds  = WSISurvivalDataset(cfg.data, dataset=train_dataset, split="test",  **ds_kwargs)
    external_ds_kwargs = {k: v for k, v in ds_kwargs.items() if k not in ("fold", "n_folds")}
    external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **external_ds_kwargs)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} external({external_dataset})={len(external_ds)}")

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_loader    = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader      = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader     = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    external_loader = DataLoader(external_ds, shuffle=False, **dl_kwargs)

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_M4(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, backbone="uni2native", combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
        use_mutation=True, mutation_stats=mutation_stats,
    ).to(device)
    model.clinical_aux_head = ClinicalAuxClassifier(cfg.model.embed_dim).to(device)

    lr_mult_warmup_targets: list[tuple[int, float]] = []
    if args.clinical_lr_mult != 1.0:
        branch_groups = _branch_param_groups(model)
        base_params = list(branch_groups["other"]) + branch_groups["rna"] + branch_groups["wsi"]
        param_groups = [{"params": base_params, "lr": cfg.train.lr}]
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
        risks, times, events, aux_losses = [], [], [], []
        epoch_aux_values = []  # 2026-09-04: 디버그용 — 이 epoch 동안 실제로 aux loss가 몇 번, 어떤 값으로 걸렸는지

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
            pni_label, immune_label = pni_immune_map.get(case_id, (-1, -1))
            branch_risk_out = {
                "pni_ord": torch.tensor(pni_label, dtype=torch.long, device=device),
                "immune_ord": torch.tensor(immune_label, dtype=torch.long, device=device),
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
        aux_mean = np.mean(epoch_aux_values) if epoch_aux_values else float("nan")
        print(f"epoch {epoch+1:3d} | val_c_index={val_metrics['c_index']:.4f} "
              f"| aux_loss_mean(n={len(epoch_aux_values)}/{len(train_ds)})={aux_mean:.4f}")
        if val_metrics["c_index"] > best_val_c:
            best_val_c = val_metrics["c_index"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nbest val_c_index={best_val_c:.4f} — 이 시점 가중치로 test/external 평가")
    model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, None, desc="test")
    external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, None, desc="external")
    print(f"\n=== 결과 (fold={args.fold}, seed={args.seed}, clinical_aux_weight={args.clinical_aux_weight}) ===")
    print(f"  internal(cptac test) c_index = {test_metrics['c_index']:.4f}")
    print(f"  external(tcga)       c_index = {external_metrics['c_index']:.4f}")

    model_tag = f"CPTACTRAIN_PDACCONS1500_CNV_MUT_STG_R_COX_ADD_CLINAUX{args.clinical_aux_weight}"
    if args.clinical_lr_mult != 1.0:
        model_tag += f"_CLR{int(args.clinical_lr_mult)}"
    kfold_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"
    kfold_dir.mkdir(parents=True, exist_ok=True)
    kfold_path = kfold_dir / f"{train_dataset}_{model_tag}_seed{args.seed}_fold{args.fold}of{args.n_folds}.csv"
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


if __name__ == "__main__":
    main()
