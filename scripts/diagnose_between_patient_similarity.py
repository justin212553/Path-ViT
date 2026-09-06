"""
사용자 질문(2026-09-05): "이게(effective rank가) 늘어나면 늘어날수록 성능 향상에 꼭 기여를
할까는 다른 이야기인 거지? 대충 학습 잘 된 벡터라면 어느정도여야 하지? 예컨데면, RNA 벡터는
cosine similarity가 몇이야?" — scripts/diagnose_nystrom_oversmoothing*.py는 "한 환자 안에서
패치들끼리 얼마나 닮았는지"(within-patient)를 쟀다. 이건 Nystrom의 oversmoothing 여부를 보는
데는 맞는 척도지만, "이 표현이 실제로 쓸모있는 정보를 담고 있는가"를 재는 척도는 아니다 —
diversity(rank가 높다/유사도가 낮다)는 필요조건이지 충분조건이 아니다(노이즈 방향으로 다양해도
rank는 높게 나온다).

더 직접적인 질문은 "환자마다 이 표현이 서로 다른가"(between-patient) 다 — 이 프로젝트에서
가장 강한 신호로 알려진 RNA branch(z_rna, RNA co-attention을 실제로 구동하는 벡터)의 환자 간
평균 pairwise cosine similarity를 기준점(calibration reference)으로 삼아, WSI branch의 환자
간 유사도(z_wsi, component_coattn 출력)와 직접 비교한다. RNA가 "잘 작동하는 표현"의 기준이라면,
WSI가 이 기준보다 훨씬 유사도가 높다(=환자를 잘 구분 못 한다)는 것 자체가 문제의 증거가 된다.

사용법:
    python -m scripts.diagnose_between_patient_similarity --ckpt <path> --backbone uni2native
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
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_df
from train import _patient_risk, _identity_collate, _make_amp_ctx


def _mean_pairwise_cosine(tokens: torch.Tensor) -> float:
    n = tokens.shape[0]
    if n < 2:
        return float("nan")
    x = torch.nn.functional.normalize(tokens.float(), dim=-1)
    s = x.sum(dim=0)
    total = (s @ s) - n
    return (total / (n * (n - 1))).item()


def _effective_rank(tokens: torch.Tensor) -> float:
    x = tokens.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eigvals / eigvals.sum().clamp(min=1e-12)
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--num-transformer-layers", type=int, default=1)
    parser.add_argument("--cluster-pool", action="store_true",
                         help="train.py --cluster-pool로 학습한 체크포인트용 — self.vit/ABMIL 대신 "
                              "K=10 군집 대표 토큰을 co-attention에 바로 넣는 모델(models/vit_pma.py).")
    parser.add_argument("--train-dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx()
    cfg = Config()
    cfg.model.num_transformer_layers = args.num_transformer_layers

    rna_gene_ids = pdac_consistency_gene_ids(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS[args.train_dataset])
    stage_stats = stage_stats_from_df(pd.read_csv(CLINICAL_PATHS[args.train_dataset]))

    ds_kwargs = dict(
        with_clinical=True, with_staging=True, with_margin=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    ds = WSISurvivalDataset(cfg.data, dataset=args.train_dataset, split="all", **ds_kwargs)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"{args.train_dataset} N={len(ds)}, backbone={args.backbone}, num_transformer_layers={args.num_transformer_layers}")

    rna_input_dim = len(rna_gene_ids) + 8
    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, backbone=args.backbone, combine_mode="cox_add",
        use_staging=True, stage_stats=stage_stats, use_margin=True, margin_stats=margin_stats,
        cluster_pool=args.cluster_pool,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"체크포인트 로드: {Path(args.ckpt).name} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")

    captured = {}

    def _hook(module, inputs, output):
        patient_embed = inputs[0]         # (4, D) — co-attention 이전, 4개 관점(mean/std/attn/top) 평균
        z_wsi, _ = output                  # (D,)  — co-attention 이후
        captured["z_before"] = patient_embed.mean(dim=0).detach()
        captured["z_wsi"] = z_wsi.detach()

    handle = model.component_coattn.register_forward_hook(_hook)

    z_rna_list, z_wsi_list, z_before_list = [], [], []
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            rna = patient_slides[0]["rna"].to(device, non_blocking=True)
            z_rna = model.encode_rna(rna)
            _ = _patient_risk(model, patient_slides, device, amp_ctx, None, cfg.train.cnn_chunk_size)
            z_rna_list.append(z_rna.float().cpu())
            z_before_list.append(captured["z_before"].float().cpu())
            z_wsi_list.append(captured["z_wsi"].float().cpu())
    handle.remove()

    Zr = torch.stack(z_rna_list)       # (N_patients, D)
    Zb = torch.stack(z_before_list)    # (N_patients, D) — co-attention 이전(4관점 단순평균)
    Zw = torch.stack(z_wsi_list)       # (N_patients, D) — co-attention 이후
    print(f"\n=== 환자 간(between-patient) 유사도, N={Zr.shape[0]}명, D={Zr.shape[1]} ===")
    print(f"z_rna (RNA branch, co-attention을 구동하는 벡터):")
    print(f"  평균 pairwise cosine similarity = {_mean_pairwise_cosine(Zr):.4f}  |  effective rank = {_effective_rank(Zr):.2f}")
    print(f"z_before (WSI branch, Nystrom+ABMIL 다 거쳤지만 co-attention 이전, 4관점 단순평균):")
    print(f"  평균 pairwise cosine similarity = {_mean_pairwise_cosine(Zb):.4f}  |  effective rank = {_effective_rank(Zb):.2f}")
    print(f"z_wsi (WSI branch, component_coattn 출력, co-attention까지 다 거친 최종 벡터):")
    print(f"  평균 pairwise cosine similarity = {_mean_pairwise_cosine(Zw):.4f}  |  effective rank = {_effective_rank(Zw):.2f}")


if __name__ == "__main__":
    main()
