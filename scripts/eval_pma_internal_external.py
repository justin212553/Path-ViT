"""
학습이 끝난 PMA(WSI+Clinical+RNA multi-modal, --PMA) checkpoint를, train.py가 학습 종료 직후
best checkpoint로 수행하는 것과 똑같은 프로토콜로 internal(같은 코호트 held-out test)/
external(반대 코호트 전체) 평가한다 — 재학습 없이 기존 checkpoint만으로 결과를 재확인할 때
쓴다(train.py::main()의 "best checkpoint" 평가 블록을 그대로 재현, evaluate()/
compute_time_dependent_auc() 등 동일 함수를 import해서 씀).

기본값은 survival_tcga_seed42_EXTfdr0.1_SS_AUX_PMA_EXTfdr0.1_SS_AUX_AUG_DISP_best_pma.pt에
맞춰져 있다 — 이 checkpoint는 다음 커맨드로 학습됐다(체크포인트 파일명의 태그 구성 규칙
train.py:1608-1642/1183-1263 역산, model_state_dict에 cnn.backbone.* 존재로 --image 확인):
    python train.py --dataset tcga --seed 42 --PMA --external \
        --rna-genes literature_fdr0.1_tcga_only --patch-keep-frac 0.8 --rna-aux-weight 1.0 \
        --image --tile-augment --attn-dispersion

다른 PMA checkpoint를 평가하려면 --dataset/--seed/--rna-genes/--rna-aux-weight/--image/
--tile-augment/--attn-dispersion을 그 checkpoint를 학습할 때 쓴 값과 정확히 맞춰야 한다
(architecture kwarg가 하나라도 다르면 model.load_state_dict()에서 바로 에러가 난다 —
이 스크립트에서 그 mismatch를 조용히 넘기지 않고 fail-fast하는 게 의도된 안전장치다).

사용법(PathViT-ray conda env):
    python -m scripts.eval_pma_internal_external
    python -m scripts.eval_pma_internal_external --checkpoint models/checkpoint/other_pma.pt \
        --dataset cptac --seed 84 --rna-genes literature_fdr0.1_cptac_only
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 로컬(비-HPC) 환경에서 SSL_CERT_FILE이 잘못 세팅돼 있으면 timm이 backbone(ImageNet 사전학습
# ResNet50) 가중치를 HF Hub에서 받아올 때 SSL 컨텍스트 생성이 실패한다 — 다른 로컬 PowerShell
# 스크립트들(scripts/_pma_*_local_ext.ps1)과 동일한 대응.
os.environ.pop("SSL_CERT_FILE", None)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, resolve_tcga_only_rna_genes
from data.patch_utils import PATCH_TRANSFORM, PATCH_TRANSFORM_512
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from train import set_seed, evaluate, _log_line, _identity_collate
from utils.metrics import compute_time_dependent_auc

DEFAULT_CKPT = (
    _ROOT / "models" / "checkpoint"
    / "survival_tcga_seed42_EXTfdr0.1_SS_AUX_PMA_EXTfdr0.1_SS_AUX_AUG_DISP_best_pma.pt"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"],
                         help="checkpoint 학습에 쓴 --dataset(internal held-out test 코호트). "
                              "external test는 반대 코호트 전체로 자동 결정된다.")
    parser.add_argument("--seed", type=int, default=42,
                         help="checkpoint 학습에 쓴 --seed — train/val/test split 재현에 필요.")
    parser.add_argument("--rna-genes", type=str, default="literature_fdr0.1_tcga_only",
                         help="checkpoint 학습에 쓴 --rna-genes(입력 유전자셋 = rna_input_dim 결정).")
    parser.add_argument("--rna-aux-weight", type=float, default=1.0,
                         help="0보다 크면 rna_aux_head를 붙인다(state_dict에 있으면 반드시 필요).")
    parser.add_argument("--image", action=argparse.BooleanOptionalAction, default=True,
                         help="raw 이미지 실시간 인코딩 모드(--no-image면 precomputed features.pt). "
                              "이 checkpoint는 cnn.backbone 가중치가 저장돼 있어 --image로 학습됐다.")
    parser.add_argument("--tile-augment", action=argparse.BooleanOptionalAction, default=True,
                         help="원 학습이 --tile-augment였는지 — eval에는 증강이 안 걸리지만 "
                              "eval_transform 선택(PATCH_TRANSFORM_512 vs PATCH_TRANSFORM, "
                              "train.py:1383)에 --image와 함께 영향을 준다.")
    parser.add_argument("--attn-dispersion", action=argparse.BooleanOptionalAction, default=True,
                         help="ViT_PMA의 dispersion_scale 파라미터 존재 여부 — 원 학습과 다르면 "
                              "load_state_dict()가 실패한다.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tile-decode-workers", type=int, default=None,
                         help="cfg.model.tile_decode_workers 덮어쓰기(기본: config.py 값, 4) — "
                              "--image 모드에서 forward() 안 타일 디코딩+인코딩 스레드풀 크기. "
                              "HPC --cpus-per-task에 맞춰 늘리면 순수 속도만 개선(결과엔 무관).")
    parser.add_argument("--auc-days", type=str, default="365,730,1095")
    args = parser.parse_args()

    auc_days = tuple(int(x.strip()) for x in args.auc_days.split(",") if x.strip())

    cfg = Config()
    cfg.data.precomputed = not args.image
    cfg.data.seed = args.seed
    cfg.train.seed = args.seed
    cfg.data.num_workers = args.num_workers
    if args.tile_decode_workers is not None:
        cfg.model.tile_decode_workers = args.tile_decode_workers
    if args.attn_dispersion:
        cfg.model.use_attn_dispersion = True

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )

    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset]

    rna_gene_ids = resolve_tcga_only_rna_genes(args.rna_genes)
    rna_input_dim = len(rna_gene_ids)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])

    ds_kwargs = dict(
        with_clinical=True, with_staging=False, with_rna=True,
        feature_backbone="resnet50", rna_gene_ids=rna_gene_ids,
    )
    # train.py:1383과 동일한 조건 — eval(train/test/external 전부)은 항상 증강 없이, --image
    # --tile-augment 조합일 때만 512 리사이즈(train과 동일 유효 배율)를 쓴다.
    eval_transform = PATCH_TRANSFORM_512 if (args.tile_augment and args.image) else PATCH_TRANSFORM

    train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", transform=eval_transform, **ds_kwargs)
    test_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test", transform=eval_transform, **ds_kwargs)
    external_ds = WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", transform=eval_transform, **ds_kwargs)

    dl_kwargs = dict(batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=cfg.data.num_workers)
    train_loader = DataLoader(train_ds, **dl_kwargs)
    test_loader = DataLoader(test_ds, **dl_kwargs)
    external_loader = DataLoader(external_ds, **dl_kwargs)

    model = ViT_PMA(
        cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
        precomputed=cfg.data.precomputed, backbone="resnet50",
    ).to(device)
    if args.rna_aux_weight > 0:
        model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index'):.4f})")

    # censoring 분포(time-dependent AUC)는 internal/external 둘 다 학습 코호트(--dataset)의
    # train split 기준으로 추정한다 — train.py:1863-1869/1908-1912와 동일한 관례.
    train_metrics = evaluate(model, train_loader, cfg, device, amp_ctx, train_ds.transform, desc="train_eval")

    test_metrics = evaluate(model, test_loader, cfg, device, amp_ctx, test_ds.transform, desc="internal test")
    test_td_auc = compute_time_dependent_auc(
        train_metrics["times"], train_metrics["events"],
        test_metrics["times"], test_metrics["events"], test_metrics["risks"],
        eval_days=auc_days,
    )
    print(f"\n=== Internal Test ({args.dataset}/test, n={len(test_ds)}) ===")
    print(_log_line("test", test_metrics, test_td_auc))

    external_metrics = evaluate(model, external_loader, cfg, device, amp_ctx, external_ds.transform, desc="external")
    external_td_auc = compute_time_dependent_auc(
        train_metrics["times"], train_metrics["events"],
        external_metrics["times"], external_metrics["events"], external_metrics["risks"],
        eval_days=auc_days,
    )
    print(f"\n=== External Test ({external_dataset} 전체 코호트, n={len(external_ds)}) ===")
    print(_log_line("external", external_metrics, external_td_auc))


if __name__ == "__main__":
    main()
