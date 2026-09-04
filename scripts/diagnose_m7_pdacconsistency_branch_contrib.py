"""
diagnose_m7_branch_contrib.py를 pdac_consistency_1500+CNV+mutation 레시피(train_light.py --M7)로
옮긴 버전 — M4 쪽(diagnose_m4_frozrna_branch_contrib.py)과 나란히, WSI 없는 M7에서도 RNA/clinical
기여도가 얼마나 벌어지는지 확인한다(사용자 질문, 2026-09-04). ClinicalRNAOnly.forward가
return_components=True를 이미 네이티브로 지원해서(risk_rna/risk_clin이 정확히 가산되는 구조 —
LayerNorm 트릭 없이도 정확한 분해) M4보다 훨씬 간단하다.

사용법:
    python -m scripts.diagnose_m7_pdacconsistency_branch_contrib --seed 84 --fold 0 --n-folds 5
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
from models import ClinicalRNAOnly
from models.clinical_encoder import (
    age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, mutation_stats_from_df, STAGE_FIELDS,
)


def _identity_collate(batch: list) -> list:
    return batch[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt) if args.ckpt else (
        _ROOT / "models" / "checkpoint" /
        f"survival_{args.dataset}_best_m7_pdaccons1500_cnv_stg_r_mut_cox_add_"
        f"fold{args.fold}of{args.n_folds}_seed{args.seed}_light.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 없음 — --ckpt로 직접 경로를 지정하세요.")
    print(f"checkpoint: {ckpt_path}")

    rna_gene_ids = pdac_consistency_gene_ids(1500)
    rna_input_dim = len(rna_gene_ids) + 8  # +8 = CNV(pathway8 8카테고리 평균)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    mutation_stats = mutation_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))

    model = ClinicalRNAOnly(
        Config().model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        combine_mode="cox_add", use_margin=True, margin_stats=margin_stats, use_age_sex=True,
        use_staging=True, stage_stats=stage_stats, use_mutation=True, mutation_stats=mutation_stats,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"best epoch={ckpt.get('epoch')} val_c_index={ckpt.get('val_c_index'):.4f}\n")

    from models.clinical_encoder import MUTATION_FIELDS

    ds_kwargs = dict(
        with_clinical=True, with_margin=True, with_staging=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids,
    )
    cfg = Config()
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
                margin_ord=p["margin_ord"].to(device),
                stage_ord={f: p[f].to(device) for f in STAGE_FIELDS},
                mutation_ord={f: p[f].to(device) for f in MUTATION_FIELDS},
                return_components=True,
            )
            for k, v in comp.items():
                terms[k].append(v.item())

    arrs = {k: np.array(v) for k, v in terms.items()}
    total = sum(arrs.values())
    total_var = total.var()
    print(f"N={len(total)}명, total risk std={total.std():.4f}\n")
    print(f"  {'항':10s} {'mean':>10s} {'std':>10s} {'설명분산비율':>12s}")
    for name, arr in arrs.items():
        explained = arr.var() / total_var if total_var > 0 else float("nan")
        print(f"  {name:10s} {arr.mean():10.4f} {arr.std():10.4f} {explained:12.4%}")


if __name__ == "__main__":
    main()
