"""
freeze-rna 체크포인트(scripts/experiment... 아님, train.py --freeze-rna로 학습된 실제 모델)의
WSI/RNA/clinical 모달리티 기여도를 뜯어본다(사용자 질문, 2026-09-03).

diagnose_m7_branch_contrib.py/experiment_m4_wsi_cox_add.py와 달리 freeze-rna는 risk_head가
여전히 [z_wsi‖z_rna] concat을 통째로 받는 단일 선형층이라(재학습 없이 그대로 쓰는 게 목적 —
freeze-rna 자체를 다시 학습시키면 이 진단의 의미가 없어짐), risk_head=Sequential(LayerNorm,
Linear) 구조를 이용해 **정확한(근사 아닌) 가산 분해**를 한다: LayerNorm은 concat 전체 기준으로
정규화하지만, 그 뒤에 오는 Linear는 입력 차원마다 독립적으로 가중합하므로

    LN(concat) = [n_wsi ‖ n_rna]  (LayerNorm 출력을 그대로 반으로 자름)
    risk = W_wsi·n_wsi + W_rna·n_rna + bias

가 부동소수점 오차 이내로 정확히 성립한다(0벡터로 치환하는 counterfactual ablation과 달리
LayerNorm의 정규화 기준 자체를 안 건드리므로 왜곡이 없다 — 최초 시도했던 zero-ablation 버전은
rna_only 설명분산비율이 396%로 나와 LayerNorm 왜곡이 크다는 게 확인돼 이 방식으로 교체함).

사용법:
    python -m scripts.diagnose_m4_frozrna_branch_contrib --seed 84 --fold 0 --n-folds 5
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
from models.clinical_encoder import (
    age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df, mutation_stats_from_df,
)
from train import _patient_risk, _make_amp_ctx, _identity_collate, _stage_ord_from_patient, _margin_ord_from_patient, _mutation_ord_from_patient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--rna-genes", type=str, default="pdac_consistency_1500",
                         choices=["pdac_consistency_1500", "literature_1500"],
                         help="이 체크포인트를 학습할 때 실제로 쓴 RNA 패널과 반드시 일치해야 "
                              "한다 — 모델 입력 차원(1508)은 같아도 유전자 정체가 다르면 결과가 "
                              "무의미해진다(사용자 지시로 literature_1500도 지원 추가, 2026-09-04).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()

    ckpt_path = Path(args.ckpt) if args.ckpt else (
        _ROOT / "models" / "checkpoint" /
        f"survival_{args.dataset}_seed{args.seed}_PDACCONS1500_CNV_STG_R_MUT_M4_PDACCONS1500_CNV_STG_R_MUT_"
        f"COX_ADD_WSRNA_FROZRNA_FOLD{args.fold}OF{args.n_folds}_best_clinical_rna.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 없음 — --ckpt로 직접 경로를 지정하세요.")
    print(f"checkpoint: {ckpt_path}")

    if args.rna_genes == "pdac_consistency_1500":
        rna_gene_ids = pdac_consistency_gene_ids(1500)
    else:
        from data.dataset import literature_guided_gene_ids
        rna_gene_ids = literature_guided_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    mutation_stats = mutation_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.dataset]))
    rna_input_dim = len(rna_gene_ids) + 8

    cfg = Config()
    model = ViT_M4(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
        use_mutation=True, mutation_stats=mutation_stats,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"best epoch={ckpt.get('epoch')} val_c_index={ckpt.get('val_c_index'):.4f}\n")

    embed_dim = cfg.model.embed_dim

    # risk_head 입력([z_wsi‖z_rna] concat, models/vit_m4.py::combine_with_clinical_rna)을
    # _patient_risk() 내부에서 그대로 가로채기 위한 캡처용 래퍼 — _patient_risk의 검증된
    # 슬라이드 루프/AMP/청크 로직을 그대로 재사용하고(직접 재구현 시 실수 위험), risk_head
    # 호출 시점의 입력 텐서만 옆에서 훔쳐본다.
    orig_risk_head = model.risk_head
    captured: dict[str, torch.Tensor] = {}

    class _CapturingHead(torch.nn.Module):
        def forward(self, x):
            captured["concat"] = x.detach().clone()
            return orig_risk_head(x)

    model.risk_head = _CapturingHead()

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids,
    )
    all_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="all", **ds_kwargs)
    loader = DataLoader(all_ds, batch_size=1, collate_fn=_identity_collate, num_workers=0)

    terms = {"wsi_only": [], "rna_only": [], "clinical": [], "full": []}
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            p = patient_slides[0]

            full_risk, _, _ = _patient_risk(model, patient_slides, device, amp_ctx, None, cfg.train.cnn_chunk_size)
            concat = captured["concat"]  # (1, 2*embed_dim) = [z_wsi, z_rna], risk_head 입력(LN 전)
            normed = orig_risk_head[0](concat)  # LayerNorm(전체 concat 기준 정규화) — 여기까진 안 쪼갬
            n_wsi, n_rna = normed[:, :embed_dim], normed[:, embed_dim:]
            linear = orig_risk_head[1]
            w_wsi, w_rna = linear.weight[:, :embed_dim], linear.weight[:, embed_dim:]
            risk_wsi_only = (n_wsi * w_wsi).sum(dim=-1) + linear.bias  # 정확한 가산항(근사 아님)
            risk_rna_only = (n_rna * w_rna).sum(dim=-1)                # bias는 wsi_only 쪽에만 더함

            age_years = p["age_years"].to(device)
            sex_idx = p["sex_idx"].to(device)
            stage_ord = _stage_ord_from_patient(patient_slides, device)
            margin_ord = _margin_ord_from_patient(patient_slides, device)
            mutation_ord = _mutation_ord_from_patient(patient_slides, device)
            clin_embed = model._clinical_embed(age_years, sex_idx, margin_ord, stage_ord=stage_ord,
                                                mutation_ord=mutation_ord)
            clinical_term = model.clinical_linear(clin_embed).view(1)

            terms["wsi_only"].append(risk_wsi_only.item())
            terms["rna_only"].append(risk_rna_only.item())
            terms["clinical"].append(clinical_term.item())
            terms["full"].append(full_risk.item())

    arrs = {k: np.array(v) for k, v in terms.items()}
    recon_err = np.abs((arrs["wsi_only"] + arrs["rna_only"] + arrs["clinical"]) - arrs["full"]).max()
    print(f"검산: wsi_only+rna_only+clinical vs full 최대 오차={recon_err:.6f} (0에 가까우면 가산 분해 정확)")
    total_var = arrs["full"].var()
    print(f"N={len(arrs['full'])}명, full risk std={arrs['full'].std():.4f}\n")
    print(f"  {'항':12s} {'mean':>10s} {'std':>10s} {'설명분산비율(대 full)':>18s}")
    for name in ("wsi_only", "rna_only", "clinical"):
        arr = arrs[name]
        explained = arr.var() / total_var if total_var > 0 else float("nan")
        print(f"  {name:12s} {arr.mean():10.6f} {arr.std():10.6f} {explained:18.6%}")


if __name__ == "__main__":
    main()
