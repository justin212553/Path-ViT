"""
HDP_Pretrain_Cluster 학습된 checkpoint를 열어, 각 branch(RNA/clinical/pretrain-hist/growth/
maturity)가 실제로 risk에 얼마나 기여하는지 뜯어본다. HPC 전용(uni2native WSI 데이터 필요).

배경(2026-09-01): scripts/diagnose_hdp_feature_signal.py로 "raw feature 자체가 생존과
무관하다"는 걸 신경망 없이 확인했다. 이번엔 반대쪽 — 실제로 학습된 모델이 이 무의미한 입력을
보고 어떻게 반응했는지(weight를 0 근처로 죽였는지, 아니면 죽이지 못한 채 노이즈를 risk에
계속 더하고 있는지)를 직접 연다. models/hdp.py::HDP.forward/models/hdp_cluster.py::
HDPCluster.forward에 새로 추가한 return_components=True로 항별(rna/clin/hist/growth/maturity)
risk 기여도를 분리해서 받는다.

두 가지를 본다:
  1. weight norm — clinical_linear/hist_linear/growth_linear/maturity_linear는 전부
     zero-init이었다. 학습 후에도 norm이 작으면(risk_head 대비) "모델이 이 branch를 거의
     안 쓰기로 했다"는 뜻.
  2. 실제 항별 risk 기여도의 표준편차(z-score 아닌 raw 입력들이 섞여 있어 weight norm만으론
     비교가 정확하지 않음 — growth_vec은 CNN 출력이라 스케일이 임의적) — 환자마다 이 항이
     얼마나 다른 값을 내는지가 "이 항이 risk 순위에 실제로 영향을 주는지"를 직접 보여준다.
     전체 risk 분산 대비 각 항 분산의 비율(설명 분산 비율)도 같이 낸다.

사용법(HPC):
    python -m scripts.diagnose_hdp_checkpoint_weights --dataset tcga --seed 84 --fold 0 --n-folds 5
    python -m scripts.diagnose_hdp_checkpoint_weights --ckpt models/checkpoint/survival_tcga_best_....pt \
        --dataset tcga --fold 0 --n-folds 5
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
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models.hdp_cluster import HDPCluster
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, STAGE_FIELDS
from train_light import _load_cluster_histograms
from train_hdp_pretrain_cluster import _load_head, PrecomputeCache, _identity_collate


def _patient_components(model, patient_slides, precompute, cluster_hist_lookup, device) -> dict:
    p = patient_slides[0]
    case_id = p["case_id"]
    maps, feat_cat, w_cat = precompute(case_id, patient_slides, device)
    growth_vecs = [model.growth_cnn(m.to(device)) for m in maps]
    growth_vec = torch.stack(growth_vecs).mean(dim=0)
    maturity_scalar = model.maturity_mlp(feat_cat, w_cat)
    cluster_hist = cluster_hist_lookup[case_id].to(device, non_blocking=True)
    kwargs = {}
    if getattr(model, "use_margin", False):
        kwargs["margin_ord"] = p["margin_ord"].to(device, non_blocking=True)
    if getattr(model, "use_staging", False):
        kwargs["stage_ord"] = {f: p[f].to(device, non_blocking=True) for f in STAGE_FIELDS}
    return model(
        p["age_years"].to(device, non_blocking=True), p["sex_idx"].to(device, non_blocking=True),
        p["rna"].to(device, non_blocking=True), cluster_hist, growth_vec, maturity_scalar,
        return_components=True, **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--growth-dim", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=None,
                         help="명시 안 하면 train_hdp_pretrain_cluster.py의 기본 저장 경로를 재구성.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_prefix = f"HDP_PRETRAIN_CLUSTER_INT1500_STG_R_GROWTH{args.growth_dim}"
    if args.fold is not None:
        model_prefix += f"_FOLD{args.fold}OF{args.n_folds}"
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        _ROOT / "models" / "checkpoint" / f"survival_{args.dataset}_best_{model_prefix.lower()}_seed{args.seed}_light.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 없음 — --ckpt로 직접 경로를 지정하세요.")
    print(f"checkpoint: {ckpt_path}")

    head = _load_head(device)
    precompute = PrecomputeCache(head)
    hist_datasets = [args.dataset]
    cluster_hist_lookup, hist_dim = _load_cluster_histograms(hist_datasets, source="pretrain")

    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)

    model = HDPCluster(
        Config().model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
        hist_dim=hist_dim, k=1, feat_dim=1536, growth_dim=args.growth_dim,
        use_margin=True, margin_stats=margin_stats, use_age_sex=True,
        use_staging=True, stage_stats=stage_stats,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"best epoch={ckpt.get('epoch')} val_c_index={ckpt.get('val_c_index'):.4f}\n")

    print("=== 1) zero-init 선형층 weight norm (학습 후) ===")
    named = {
        "risk_head(RNA, zero-init 아님)": model.risk_head[1],
        "clinical_linear": model.clinical_linear,
        "hist_linear(pretrain 5차원)": model.hist_linear,
        "growth_linear": model.growth_linear,
        "maturity_linear": model.maturity_linear,
    }
    for name, layer in named.items():
        w = layer.weight.detach()
        print(f"  {name:32s} shape={tuple(w.shape)} L2 norm={w.norm().item():.4f} "
              f"max|w|={w.abs().max().item():.4f}")

    print("\n=== 2) internal 전체(train+val+test)에서 항별 실제 risk 기여도 ===")
    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True,
                      with_rna=True, rna_gene_ids=rna_gene_ids, feature_backbone="uni2native")
    cfg = Config()
    cfg.data.seed = args.seed
    all_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="all", **ds_kwargs)
    loader = DataLoader(all_ds, batch_size=1, collate_fn=_identity_collate, num_workers=0)

    terms = {"rna": [], "clin": [], "hist": [], "growth": [], "maturity": []}
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            comp = _patient_components(model, patient_slides, precompute, cluster_hist_lookup, device)
            for k, v in comp.items():
                terms[k].append(v.item())

    arrs = {k: np.array(v) for k, v in terms.items()}
    total = sum(arrs.values())
    total_var = total.var()
    print(f"  N={len(total)}명, total risk std={total.std():.4f}\n")
    print(f"  {'항':12s} {'mean':>10s} {'std':>10s} {'설명분산비율':>12s}")
    for name, arr in arrs.items():
        explained = arr.var() / total_var if total_var > 0 else float("nan")
        print(f"  {name:12s} {arr.mean():10.4f} {arr.std():10.4f} {explained:12.2%}")


if __name__ == "__main__":
    main()
