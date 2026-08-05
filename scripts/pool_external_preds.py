"""
k-fold로 학습한 5개 fold-checkpoint를 전부 로드해 "같은" external 코호트 전체를 평가하고,
환자별 risk score를 5-fold 평균(ensemble mean-pool)한 뒤 c-index를 단 한 번만 계산한다.

기존 pool_kfold_preds.py는 internal(서로 겹치지 않는 test fold를 이어붙임)용이고,
external은 5개 fold 체크포인트가 전부 같은 전체 코호트를 보므로 "이어붙이기"가 아니라
"환자별 risk 평균"이 pooling에 해당한다(summarize_kfold.py의 mean±std of 5 c-index와는
다른, prediction-level ensemble).

사용법 (M7_INT1500_R_COX_ADD, seed84):
    python scripts/pool_external_preds.py --dataset tcga --model-prefix M7_INT1500_R_COX_ADD \
        --seed 84 --n-folds 5 --rna-genes literature_1500_intersection \
        --clinical-margin --combine-mode cox_add
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models import ClinicalRNAOnly
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv
from train_light import evaluate, _identity_collate
from utils.metrics import compute_survival_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=["tcga", "cptac"])
    parser.add_argument("--model-prefix", type=str, required=True, help="예: M7_INT1500_R_COX_ADD")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--rna-genes", type=str, default="literature_1500_intersection")
    parser.add_argument("--clinical-margin", action="store_true")
    parser.add_argument("--no-age-sex", action="store_true")
    parser.add_argument("--combine-mode", type=str, default="concat", choices=["concat", "film", "cox_add"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset]

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset]) if args.clinical_margin else None

    assert args.rna_genes.endswith("_intersection"), "이 스크립트는 현재 _intersection 유전자셋만 지원"
    rna_gene_ids = literature_guided_gene_ids_intersection(int(args.rna_genes.split("_")[1]))
    rna_input_dim = len(rna_gene_ids)

    external_ds = WSISurvivalDataset(
        cfg.data, dataset=external_dataset, split="all",
        with_clinical=True, with_margin=args.clinical_margin, with_rna=True,
        rna_gene_ids=rna_gene_ids, rna_purist=False, restrict_case_ids=None,
    )
    external_loader = DataLoader(external_ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    per_fold_risks = []
    case_ids_ref, times_ref, events_ref = None, None, None

    for fold in range(args.n_folds):
        model = ClinicalRNAOnly(
            cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
            clinical_dim=None, rna_dim=None, clinical_dropout=0.0, sex_onehot=False,
            combine_mode=args.combine_mode, risk_hidden_dim=None, risk_dropout=0.0,
            use_margin=args.clinical_margin, margin_stats=margin_stats,
            use_age_sex=not args.no_age_sex,
        ).to(device)

        tag = f"{args.model_prefix}_FOLD{fold}OF{args.n_folds}".lower()
        ckpt_path = ckpt_dir / f"survival_{args.dataset}_best_{tag}_seed{args.seed}_light.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        result = evaluate(model, external_loader, device)
        if case_ids_ref is None:
            case_ids_ref, times_ref, events_ref = result["case_ids"], result["times"], result["events"]
        else:
            assert result["case_ids"] == case_ids_ref, f"fold {fold}: case_id 순서가 다른 fold와 다릅니다"
        per_fold_risks.append(result["risks"])
        print(f"  fold {fold}: external c_index={result['c_index']:.4f} (개별, 참고용) <- {ckpt_path.name}")

    pooled_risks = np.mean(np.stack(per_fold_risks, axis=0), axis=0)
    metrics = compute_survival_metrics(pooled_risks, times_ref, events_ref)
    print(f"\nPooled (ensemble mean risk) external test — {args.dataset}->{external_dataset} "
          f"{args.model_prefix} seed={args.seed} ({args.n_folds}-fold, N={len(case_ids_ref)}, "
          f"events={int(events_ref.sum())})")
    print(f"  c_index={metrics['c_index']:.4f} | HR={metrics['hr']:.3f} "
          f"[{metrics['hr_ci_lower']:.3f}, {metrics['hr_ci_upper']:.3f}] | log_rank_p={metrics['log_rank_p']:.4f}")


if __name__ == "__main__":
    main()
