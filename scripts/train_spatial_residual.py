"""
Stage2 — 공간정보 잔차(residual) branch 학습 (models/spatial_residual.py::SpatialResidualBranch).

Stage1(오늘 밤 PMA_EX_SS_AUX_AUG real-time augment 체크포인트, TCGA external에서 처음으로
M7을 넘긴 레시피)을 통째로 얼리고, 그 위에 kNN 그래프+relative position bias 기반 공간
branch를 잔차로만 학습시킨다("Spatial Blindness in Whole-Slide MIL", arXiv:2605.17449의
ResTopoMIL 설계 이식) — 기존 MultiComponentPooling이 gradient를 다 흡수해 공간 신호가
학습되지 않는 문제(findings_backlog.md, WSI gradient가 RNA의 1/4)를 "따로 학습"으로 피한다.

Stage1이 완전히 얼어있으므로 base risk와 (Nystrom 이전) patch_tokens을 학습 시작 전에 딱
한 번만 계산해 캐싱한다(_precompute_patient) — 이후 epoch 루프는 CNN/RNA/Clinical을 전혀
건드리지 않고 작은 그래프 branch만 학습하므로 초 단위로 빠르다.

사용 (PathViT-ray 환경):
  python -m scripts.train_spatial_residual --external --epochs 200
"""
import argparse
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids
from data.patch_utils import PATCH_TRANSFORM_512
from models.vit_pma import ViT_PMA
from models.rna_predictor import RNAPredictionHead
from models.clinical_encoder import age_stats_from_csv
from models.spatial_residual import SpatialResidualBranch, shuffle_margin_loss
from utils import load_env
from utils.losses import cox_ph_loss
from utils.metrics import compute_survival_metrics

DEFAULT_STAGE1_CKPT = "models/checkpoint/survival_tcga_seed42_EX_SS_AUX_best_pma.pt"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str, default=DEFAULT_STAGE1_CKPT)
    p.add_argument("--dataset", type=str, default="tcga", choices=["tcga", "cptac"])
    p.add_argument("--external", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=8, help="kNN 그래프 이웃 수")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1, help="attention/feature dropout")
    p.add_argument("--edge-dropout", type=float, default=0.2,
                    help="학습 중 kNN 엣지를 무작위로 이 비율만큼 제거 — 고정 캐시(patch_tokens)를 "
                         "매 epoch 반복해서 보며 과적합하는 걸 막는다(findings_backlog.md 참조)")
    p.add_argument("--layer-type", type=str, default="gcn", choices=["gcn", "attention"],
                    help="gcn(기본) = attention 없는 단순 이웃평균(논문의 simple GNN 재현). "
                         "attention = q/k/v+relative bias(첫 시도, dropout 유무 둘 다 Stage1 "
                         "단독보다 낮게 나옴 — 비교용)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20,
                    help="val_c_index가 이 epoch 수만큼 갱신 안 되면 조기 종료(BRCA M7과 동일 관례)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--shuffle-margin", type=float, default=0.3)
    p.add_argument("--shuffle-margin-weight", type=float, default=0.3)
    p.add_argument("--group-ts", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def _precompute_patient(model, patient_slides, device, transform, chunk_size):
    """PMA 전용 _patient_risk 재현 — base_risk 계산과 CNN forward를 공유해 patch_tokens도
    같이 뽑는다(중복 연산 방지). model은 이미 .eval() + 전부 freeze된 상태여야 한다.

    [해상도 일치, 2026-07-23] Stage1이 --image 없이(precomputed 모드, 기존 features.pt) 학습된
    경우 slide["features"]를 그대로 forward_pooled로 태운다 — features.pt는 Stage1의 proj가
    실제로 학습에 쓴 그 해상도(원본, 리사이즈 없음)로 이미 뽑혀 있으므로, 여기서 raw 이미지를
    다른 해상도(예: 512)로 다시 디코딩하면 train.py에서 겪었던 것과 같은 종류의 train/eval
    해상도 불일치가 재발한다(findings_backlog.md). --image로 학습된 Stage1만 raw 이미지+
    transform 경로를 쓴다(train.py::_patient_risk와 동일한 분기).
    """
    rna = patient_slides[0]["rna"].to(device, non_blocking=True)
    z_rna = model.encode_rna(rna)

    slide_embeds, slides_cache = [], []
    for slide in patient_slides:
        coords = slide["coords"].to(device, non_blocking=True)
        features = slide.get("features")
        if features is not None:
            out = model(coords, features=features, rna_context=z_rna)
        else:
            out = model(coords, patch_paths=slide["patch_paths"], transform=transform,
                         chunk_size=chunk_size, rna_context=z_rna)
        slide_embeds.append(out["embed"])
        slides_cache.append((out["patch_tokens"].float().cpu(), coords.cpu()))

    patient_embed = torch.stack(slide_embeds).mean(dim=0)
    age_years = patient_slides[0]["age_years"].to(device, non_blocking=True)
    sex_idx   = patient_slides[0]["sex_idx"].to(device, non_blocking=True)
    combined  = model.combine_with_clinical_rna(patient_embed, age_years, sex_idx, z_rna)
    base_risk = model.risk_head(combined.unsqueeze(0)).view(()).float().cpu()

    return {
        "base_risk": base_risk,
        "slides":    slides_cache,
        "OS_time":   patient_slides[0]["OS_time"],
        "OS_event":  patient_slides[0]["OS_event"],
    }


def _precompute_split(model, ds, device, transform, chunk_size, desc: str) -> list[dict]:
    cache = []
    for i in range(len(ds)):
        cache.append(_precompute_patient(model, ds[i], device, transform, chunk_size))
    print(f"  [{desc}] {len(cache)}명 캐싱 완료")
    return cache


def _branch_risk(branch: SpatialResidualBranch, patient: dict, device) -> torch.Tensor:
    slide_reprs = [
        branch.encode(pt.to(device, non_blocking=True), c.to(device, non_blocking=True))
        for pt, c in patient["slides"]
    ]
    pooled = torch.stack(slide_reprs).mean(dim=0)
    return branch.head(branch.head_drop(pooled)).view(())


@torch.no_grad()
def _evaluate(branch: SpatialResidualBranch, cache: list[dict], device) -> dict:
    branch.eval()
    risks, times, events = [], [], []
    for patient in cache:
        final_risk = patient["base_risk"].to(device) + _branch_risk(branch, patient, device)
        risks.append(final_risk.item())
        times.append(float(patient["OS_time"].item()))
        events.append(int(patient["OS_event"].item()))
    import numpy as np
    return compute_survival_metrics(np.array(risks), np.array(times), np.array(events))


def main():
    load_env()
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    cfg.data.seed  = args.seed
    cfg.train.seed = args.seed

    external_dataset = {"tcga": "cptac", "cptac": "tcga"}[args.dataset] if args.external else None

    # [캐싱] Stage1은 완전히 얼어있어 결정론적이라, base_risk/patch_tokens는 (Stage1 체크포인트,
    # dataset, seed) 조합이 같으면 항상 동일하다 — k/hidden_dim 등 Stage2 하이퍼파라미터를 바꿔가며
    # 여러 번 돌릴 걸 감안해, 한 번 계산한 뒤엔 디스크에 남겨 CNN forward(수십 분)를 건너뛴다.
    cache_dir = Path("data/spatial_residual_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage1_tag = Path(args.stage1_ckpt).stem
    cache_path = cache_dir / f"{args.dataset}_seed{args.seed}_{stage1_tag}_ext{args.external}.pt"

    if cache_path.exists():
        print(f"Stage1 precompute 캐시 로드: {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_cache, val_cache, test_cache, external_cache = (
            cached["train"], cached["val"], cached["test"], cached["external"]
        )
        ckpt = {"epoch": cached["stage1_epoch"], "val_c_index": cached["stage1_val_c_index"]}
    else:
        age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS[args.dataset])
        rna_gene_ids  = literature_guided_gene_ids(1500)
        rna_input_dim = len(rna_gene_ids)

        ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        # [해상도 일치] --image로(raw 이미지) 학습된 체크포인트만 cnn.backbone.*를 저장한다 —
        # 이걸로 Stage1이 어느 모드였는지 자동 판별해서, precomputed 모드였다면 기존
        # features.pt(Stage1이 실제로 학습에 쓴 해상도 그대로)를 재사용하고, raw 이미지
        # 모드였다면 그때와 같은 512 리사이즈로 다시 디코딩한다.
        stage1_used_raw_image = any(
            k.startswith("cnn.backbone.") for k in ckpt["model_state_dict"]
        )
        cfg.data.precomputed = not stage1_used_raw_image
        precompute_transform = PATCH_TRANSFORM_512 if stage1_used_raw_image else None

        ds_kwargs = dict(
            with_clinical=True, with_rna=True, feature_backbone="resnet50",
            rna_gene_ids=rna_gene_ids,
        )
        if stage1_used_raw_image:
            ds_kwargs["transform"] = PATCH_TRANSFORM_512
        train_ds = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="train", **ds_kwargs)
        val_ds   = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="val",   **ds_kwargs)
        test_ds  = WSISurvivalDataset(cfg.data, dataset=args.dataset, split="test",  **ds_kwargs)
        external_ds = (
            WSISurvivalDataset(cfg.data, dataset=external_dataset, split="all", **ds_kwargs)
            if external_dataset else None
        )

        # --- Stage1: 체크포인트와 정확히 같은 구성으로 모델을 만들고 통째로 얼린다 ---
        stage1 = ViT_PMA(
            cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
            precomputed=not stage1_used_raw_image, backbone="resnet50",
        ).to(device)
        # Stage1이 --rna-aux-weight 1.0으로 학습돼 rna_aux_head가 state_dict에 포함돼 있다 —
        # load_state_dict 전에 똑같이 붙여야 키가 일치한다.
        stage1.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
        stage1.load_state_dict(ckpt["model_state_dict"])
        stage1.eval()
        for p in stage1.parameters():
            p.requires_grad_(False)
        print(f"Stage1 로드 완료: {args.stage1_ckpt} (epoch={ckpt.get('epoch')}, "
              f"val_c_index={ckpt.get('val_c_index')}, raw_image={stage1_used_raw_image})")

        chunk_size = cfg.train.cnn_chunk_size
        print("Stage1 forward로 base_risk + patch_tokens 사전계산 중(1회성)...")
        train_cache = _precompute_split(stage1, train_ds, device, precompute_transform, chunk_size, "train")
        val_cache   = _precompute_split(stage1, val_ds,   device, precompute_transform, chunk_size, "val")
        test_cache  = _precompute_split(stage1, test_ds,  device, precompute_transform, chunk_size, "test")
        external_cache = (
            _precompute_split(stage1, external_ds, device, precompute_transform, chunk_size, "external")
            if external_ds else None
        )
        del stage1  # Stage2부터는 전혀 불필요 — GPU 메모리 반납
        torch.cuda.empty_cache()

        torch.save({
            "train": train_cache, "val": val_cache, "test": test_cache, "external": external_cache,
            "stage1_epoch": ckpt.get("epoch"), "stage1_val_c_index": ckpt.get("val_c_index"),
        }, cache_path)
        print(f"Stage1 precompute 캐시 저장: {cache_path}")

    # --- Stage2: 공간 잔차 branch만 학습 ---
    branch = SpatialResidualBranch(
        patch_dim=cfg.model.embed_dim, hidden_dim=args.hidden_dim, k=args.k,
        num_layers=args.num_layers, dropout=args.dropout, edge_dropout=args.edge_dropout,
        layer_type=args.layer_type,
    ).to(device)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    global WANDB_AVAILABLE
    if WANDB_AVAILABLE:
        # wandb 네트워크 타임아웃(2026-07-23, 90초 CommError로 precompute 결과를 통째로 날린
        # 전례)으로 학습 자체가 죽지 않게 — 실패하면 이 run만 wandb 없이 진행한다.
        try:
            wandb.init(
                project="Path-ViT", name=f"{args.dataset.upper()}_SPATIALRESID_seed{args.seed}_{run_ts}",
                group=f"SPATIALRESID_{group_ts}",
                config={**vars(args), "stage1_val_c_index": ckpt.get("val_c_index")},
                settings=wandb.Settings(init_timeout=120),
            )
        except Exception as e:
            print(f"wandb.init 실패({e!r}) — 이 run은 wandb 없이 진행합니다.")
            WANDB_AVAILABLE = False

    ckpt_dir = Path("models/checkpoint")
    ckpt_dir.mkdir(exist_ok=True)
    branch_ckpt_path = ckpt_dir / f"spatial_residual_{args.layer_type}_{args.dataset}_seed{args.seed}_best.pt"

    best_val_c = -1.0
    epochs_since_improve = 0
    for epoch in range(args.epochs):
        branch.train()
        random.shuffle(train_cache)
        risks, times, events, margin_losses = [], [], [], []
        total_loss, total_steps = 0.0, 0
        batch_size = 16

        def _flush():
            nonlocal risks, times, events, margin_losses, total_loss, total_steps
            if not risks:
                return
            risk_t  = torch.stack(risks)
            time_t  = torch.cat(times).to(device)
            event_t = torch.cat(events).to(device)
            loss = cox_ph_loss(risk_t, time_t, event_t)
            if margin_losses:
                loss = loss + args.shuffle_margin_weight * torch.stack(margin_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1
            risks, times, events, margin_losses = [], [], [], []

        for patient in train_cache:
            final_risk = patient["base_risk"].to(device) + _branch_risk(branch, patient, device)
            risks.append(final_risk.view(1))
            times.append(patient["OS_time"])
            events.append(patient["OS_event"])
            pt, c = patient["slides"][0]
            margin_losses.append(
                shuffle_margin_loss(branch, pt.to(device), c.to(device), margin=args.shuffle_margin)
            )
            if len(risks) >= batch_size:
                _flush()
        _flush()

        val_metrics = _evaluate(branch, val_cache, device)
        loss_avg = total_loss / max(total_steps, 1)
        print(
            f"Epoch {epoch+1:3d} | loss={loss_avg:.4f} | "
            f"val_c_index={val_metrics['c_index']:.4f} | val_HR={val_metrics['hr']:.3f} | "
            f"val_logrank_p={val_metrics['log_rank_p']:.4f}"
        )
        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss": loss_avg,
                "val_performance/c_index": val_metrics["c_index"],
                "val_performance/hr": val_metrics["hr"],
                "val_performance/log_rank_p": val_metrics["log_rank_p"],
            }, step=epoch + 1)

        if val_metrics["c_index"] > best_val_c:
            best_val_c = val_metrics["c_index"]
            epochs_since_improve = 0
            torch.save({"branch_state_dict": branch.state_dict(), "epoch": epoch + 1,
                        "val_c_index": best_val_c}, branch_ckpt_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"  -> val_c_index가 {args.patience} epoch 동안 갱신되지 않아 조기 종료 "
                      f"(epoch {epoch+1}, best={best_val_c:.4f} @ epoch {epoch+1-epochs_since_improve})")
                break

    best = torch.load(branch_ckpt_path, map_location=device, weights_only=False)
    branch.load_state_dict(best["branch_state_dict"])
    test_metrics = _evaluate(branch, test_cache, device)
    print(f"\n=== Internal Test 성능 (best checkpoint, epoch {best['epoch']}) ===")
    print(f"test_c_index={test_metrics['c_index']:.4f} | test_HR={test_metrics['hr']:.3f} "
          f"| test_logrank_p={test_metrics['log_rank_p']:.4f}")
    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"] = test_metrics["c_index"]
        wandb.run.summary["test_hr"] = test_metrics["hr"]
        wandb.run.summary["test_log_rank_p"] = test_metrics["log_rank_p"]
        wandb.finish()

    if external_cache:
        external_metrics = _evaluate(branch, external_cache, device)
        print(f"\n=== External Test 성능 ({external_dataset}) ===")
        print(f"external_c_index={external_metrics['c_index']:.4f} | "
              f"external_HR={external_metrics['hr']:.3f} | "
              f"external_logrank_p={external_metrics['log_rank_p']:.4f}")
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="Path-ViT",
                    name=f"{args.dataset.upper()}_XSPATIALRESID_seed{args.seed}_{run_ts}",
                    group=f"SPATIALRESID_{group_ts}",
                    config={**vars(args), "external_dataset": external_dataset},
                    settings=wandb.Settings(init_timeout=120),
                )
            except Exception as e:
                print(f"wandb.init 실패({e!r}) — external 결과는 위 print 로그로만 남습니다.")
                WANDB_AVAILABLE = False
            if WANDB_AVAILABLE:
                wandb.run.summary["external_c_index"] = external_metrics["c_index"]
                wandb.run.summary["external_hr"] = external_metrics["hr"]
                wandb.run.summary["external_log_rank_p"] = external_metrics["log_rank_p"]
                wandb.finish()


if __name__ == "__main__":
    main()
