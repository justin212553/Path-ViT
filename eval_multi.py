"""
eval_multi.py — train_multi.py로 학습한 M1/M2/M3/PMA 체크포인트를 internal(test)/external로
평가하는 스크립트 (2026-07-30).

HPC quota 소진 등으로 train_multi.py의 학습(epoch 루프 + best-val 체크포인트 저장)은 끝났지만
마지막 internal/external 평가 단계를 못 돌린 경우를 위한 것 — 재학습 없이, 다운로드해둔
체크포인트(models/checkpoint/survival_{dataset}_{model_tag}_{name}_best_multi.pt)만으로
train_multi.py 끝부분과 동일한 평가를 재현한다.

모델 생성 로직(ViT_M1/ViT_M2/ViT_PMA 생성자 인자, use_attn_dispersion 등)은 train_multi.py와
100% 동일하게 맞춰야 state_dict가 정확히 로드된다 — 그쪽 코드가 바뀌면 이 파일도 같이 맞출 것.

사용법:
    python eval_multi.py --dataset tcga --external --model-tag M1M2M3PMA --M1 --M2 --M3 --PMA
"""
import argparse
import contextlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids
from data.patch_utils import PATCH_TRANSFORM_512
from models import ViT_M1, ViT_M2, ViT_PMA
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from utils.metrics import compute_time_dependent_auc

# train.py를 건드리지 않고 그대로 재사용(train_multi.py와 동일한 관례).
from train import _identity_collate, _log_line, evaluate


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"],
                    help="체크포인트를 학습시킨 train_multi.py --dataset과 동일해야 함.")
    p.add_argument("--external", action="store_true",
                    help="internal(test)과 별도로 반대 코호트 전체(split=all)도 평가.")
    p.add_argument("--rna-genes", type=str, default="literature_1500",
                    help="train_multi.py --rna-genes와 동일해야 함(기본 literature_1500).")
    p.add_argument("--model-tag", type=str, required=True,
                    help="체크포인트 파일명의 <model_tag> 부분(예: M1M2M3PMA) — train_multi.py "
                         "실행 시 켰던 --M1/--M2/--M3/--PMA 조합을 이어붙인 문자열 그대로.")
    p.add_argument("--M1", action="store_true", help="WSI만(ViT_M1) 평가 대상에 포함.")
    p.add_argument("--M2", action="store_true", help="WSI+Clinical(ViT_M2) 포함.")
    p.add_argument("--M3", action="store_true", help="WSI+RNA(clinical 제외, ViT_PMA) 포함.")
    p.add_argument("--PMA", action="store_true", help="WSI+RNA+Clinical(ViT_PMA) 포함.")
    p.add_argument(
        "--rna-aux-weight", type=float, default=0.0,
        help="train_multi.py --rna-aux-weight와 동일한 값을 줘야 함(0보다 크면 M3/PMA에 "
             "rna_aux_head를 붙인다) — 체크포인트가 그 값으로 학습됐다면 반드시 맞춰야 "
             "state_dict가 로드된다(rna_aux_head.* 키 유무).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    if not (args.M1 or args.M2 or args.M3 or args.PMA):
        raise ValueError("--M1/--M2/--M3/--PMA 중 최소 하나는 켜야 합니다.")

    cfg = Config()
    cfg.data.precomputed = False  # train_multi.py와 동일(raw-image 파이프라인으로 학습된 체크포인트)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else contextlib.nullcontext())

    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset] if args.external else None
    gene_ids = literature_guided_gene_ids(1500) if args.rna_genes == "literature_1500" else None
    if (args.M3 or args.PMA) and gene_ids is None:
        raise ValueError("--M3/--PMA는 RNA 인코더가 필요합니다 — --rna-genes를 확인하세요.")
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])

    # 2026-07-30: train_multi.py가 dispersion을 M1/M2까지 확장하면서 무조건 True로 바뀌었다 —
    # 여기도 맞춘다(안 맞으면 risk_head 차원/dispersion_scale 키가 체크포인트와 어긋나
    # load_state_dict가 실패한다).
    cfg.model.use_attn_dispersion = True

    ds_kwargs = dict(with_clinical=True, with_rna=True, rna_gene_ids=gene_ids)
    eval_transform = PATCH_TRANSFORM_512

    def _make_ds(dataset: str, split: str) -> WSISurvivalDataset:
        return WSISurvivalDataset(cfg.data, dataset=dataset, split=split, transform=eval_transform, **ds_kwargs)

    train_ds = _make_ds(args.dataset, "train")  # time-dependent AUC censoring 분포 기준값용
    test_ds  = _make_ds(args.dataset, "test")
    external_ds = _make_ds(external_dataset, "all") if external_dataset else None

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0, pin_memory=True)
    train_loader    = DataLoader(train_ds, shuffle=False, **dl_kwargs)
    test_loader     = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    external_loader = DataLoader(external_ds, shuffle=False, **dl_kwargs) if external_ds else None

    models = {}
    if args.M1:
        models["M1"] = ViT_M1(cfg.model, precomputed=False, backbone="resnet50",
                               use_attn_dispersion=True).to(device)
    if args.M2:
        models["M2"] = ViT_M2(cfg.model, age_mean=age_mean, age_std=age_std,
                               precomputed=False, backbone="resnet50",
                               use_attn_dispersion=True).to(device)
    if args.M3:
        models["M3"] = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(gene_ids),
                                precomputed=False, backbone="resnet50", use_clinical=False).to(device)
        if args.rna_aux_weight > 0:
            models["M3"].rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(gene_ids)).to(device)
    if args.PMA:
        models["PMA"] = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(gene_ids),
                                 precomputed=False, backbone="resnet50", use_clinical=True).to(device)
        if args.rna_aux_weight > 0:
            models["PMA"].rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(gene_ids)).to(device)

    ckpt_dir = Path(__file__).parent / "models" / "checkpoint"
    for name, m in models.items():
        ckpt_path = ckpt_dir / f"survival_{args.dataset}_{args.model_tag}_{name}_AUX_best_multi.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} 없음 — --model-tag/--dataset이 체크포인트 파일명과 맞는지 확인.")
        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        print(f"{name}: 체크포인트 로드(epoch {ckpt['epoch']}, val_c_index={ckpt['val_c_index']:.4f})")

    for name, m in models.items():
        train_metrics = evaluate(m, train_loader, cfg, device, amp_ctx, eval_transform)
        test_metrics  = evaluate(m, test_loader, cfg, device, amp_ctx, eval_transform)
        test_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"],
            test_metrics["times"], test_metrics["events"], test_metrics["risks"],
        )
        print(f"\n=== {name} Internal Test ===")
        print(_log_line(f"{name}_test", test_metrics, test_td_auc))

        if external_loader is not None:
            external_metrics = evaluate(m, external_loader, cfg, device, amp_ctx, eval_transform)
            external_td_auc = compute_time_dependent_auc(
                train_metrics["times"], train_metrics["events"],
                external_metrics["times"], external_metrics["events"], external_metrics["risks"],
            )
            print(f"=== {name} External Test ({external_dataset}) ===")
            print(_log_line(f"{name}_external", external_metrics, external_td_auc))


if __name__ == "__main__":
    main()
