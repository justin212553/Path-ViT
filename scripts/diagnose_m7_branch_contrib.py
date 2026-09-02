"""
M7(ClinicalRNAOnly, combine_mode="cox_add") 체크포인트를 열어, clinical/RNA 두 branch가
실제로 risk에 얼마나 기여하는지 뜯어본다. WSI가 없어 scripts/diagnose_hdp_checkpoint_weights.py
보다 훨씬 가볍고, 로컬에서도 바로 돌아간다(uni2native 등 WSI 데이터 불필요).

배경(2026-09-02): HDP_Pretrain_Cluster 진단에서 clinical_linear(margin+staging, 실제 신호가
있다고 검증된 항)조차 설명분산비율 0.00%로 나온 걸 보고 "RNA가 너무 강한 신호라 그런 거
아니냐, 혹시 literature_1500_intersection(Cox test 기반 gene selection)에 leakage가 있는
거 아니냐"는 의심(사용자) — 실제로 확인해보니 paper-spec 5-fold CV의 test 환자 중 약 60%가
그 유전자 선정에 쓰인 "train" 91명에 이미 포함돼 있었다(gene selection이 fold-aware가 아니라
고정 단일 6:2:2 split(seed=42 기본값)으로 한 번만 계산돼 그대로 재사용됨).

이 스크립트로, 같은 fold/seed에서 leakage 있는 버전(--rna-genes literature_1500_intersection)과
없는 버전(--rna-genes pathway8, OS 라벨을 전혀 안 보는 순수 문헌 큐레이션)의 clinical vs RNA
기여도를 직접 비교한다.

사용법:
    python -m scripts.diagnose_m7_branch_contrib --dataset tcga --seed 84 --fold 0 --n-folds 5 --rna-genes pathway8
    python -m scripts.diagnose_m7_branch_contrib --dataset tcga --seed 84 --fold 0 --n-folds 5 --rna-genes literature_1500_intersection
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
from data.dataset import (
    WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection, pathway_category_gene_ids,
)
from models import ClinicalRNAOnly
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, STAGE_FIELDS


def _identity_collate(batch: list) -> list:
    return batch[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--rna-genes", type=str, required=True,
                         choices=["literature_1500_intersection", "pathway8"])
    parser.add_argument("--clinical-lr-mult", type=float, default=1.0)
    parser.add_argument("--rna-lr-mult", type=float, default=1.0)
    parser.add_argument("--ckpt", type=str, default=None,
                         help="명시 안 하면 train_light.py의 기본 저장 경로를 재구성.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_prefix = "M7"
    rna_pathway_categories = None
    if args.rna_genes == "pathway8":
        rna_pathway_categories = pathway_category_gene_ids()
        rna_gene_ids = None
        rna_input_dim = len(rna_pathway_categories)
        model_prefix += "_PW8"
    else:
        rna_gene_ids = literature_guided_gene_ids_intersection(1500)
        rna_input_dim = len(rna_gene_ids)
        model_prefix += "_INT1500"
    model_prefix += "_STG_R_COX_ADD"
    # train_light.py의 정확한 태그 순서(COX_ADD -> CLR -> RLR -> FOLD)와 일치해야 checkpoint
    # 파일명이 맞아떨어진다.
    if args.clinical_lr_mult != 1.0:
        model_prefix += f"_CLR{args.clinical_lr_mult:g}"
    if args.rna_lr_mult != 1.0:
        model_prefix += f"_RLR{args.rna_lr_mult:g}"
    if args.fold is not None:
        model_prefix += f"_FOLD{args.fold}OF{args.n_folds}"

    ckpt_path = Path(args.ckpt) if args.ckpt else (
        _ROOT / "models" / "checkpoint" / f"survival_{args.dataset}_best_{model_prefix.lower()}_seed{args.seed}_light.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 없음 — --ckpt로 직접 경로를 지정하세요.")
    print(f"checkpoint: {ckpt_path}")

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))

    model = ClinicalRNAOnly(
        Config().model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        combine_mode="cox_add", use_margin=True, margin_stats=margin_stats, use_age_sex=True,
        use_staging=True, stage_stats=stage_stats,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"best epoch={ckpt.get('epoch')} val_c_index={ckpt.get('val_c_index'):.4f}\n")

    print("=== 1) zero-init 선형층 weight norm (학습 후) ===")
    w_rna = model.risk_head[1].weight.detach()
    w_clin = model.clinical_linear.weight.detach()
    print(f"  risk_head(RNA, zero-init 아님)   shape={tuple(w_rna.shape)} L2 norm={w_rna.norm().item():.4f}")
    print(f"  clinical_linear                 shape={tuple(w_clin.shape)} L2 norm={w_clin.norm().item():.4f} "
          f"max|w|={w_clin.abs().max().item():.4f}")

    print("\n=== 2) internal 전체(split=all)에서 항별 실제 risk 기여도 ===")
    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True,
                      with_rna=True, rna_gene_ids=rna_gene_ids, rna_pathway_categories=rna_pathway_categories)
    cfg = Config()
    cfg.data.seed = args.seed
    all_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="all", **ds_kwargs)
    loader = DataLoader(all_ds, batch_size=1, collate_fn=_identity_collate, num_workers=0)

    terms = {"rna": [], "clin": []}
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            p = patient_slides[0]
            comp = model(
                p["age_years"].to(device), p["sex_idx"].to(device), p["rna"].to(device),
                margin_ord=p["margin_ord"].to(device), stage_ord={f: p[f].to(device) for f in STAGE_FIELDS},
                return_components=True,
            )
            for k, v in comp.items():
                terms[k].append(v.item())

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
