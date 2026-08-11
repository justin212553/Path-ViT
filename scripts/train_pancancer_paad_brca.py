"""
PAAD(TCGA-PAAD, N=152)와 BRCA(TCGA-BRCA, N~1058)를 공동학습해, "WSI trunk가 표본을 늘리면
더 잘 배우는가"(scripts/train_brca_m4.py/train_brca_m7.py 결과: BRCA에서 M4=0.7155가
M7=0.6621을 넘음 — findings_backlog.md, PAAD에서는 반대로 PMA<M6/M7)를 PAAD 쪽에도 실제로
옮겨올 수 있는지 검증한다. self-supervised warmup(라벨 불필요)과 달리, 이건 두 코호트의
"진짜 생존 라벨"을 같이 써서 WSI trunk를 학습시키는 방식 — PORPOISE(Chen et al. 2022)의
pan-cancer 설계(14개 암종을 한 모델로 공동학습)와 같은 원리를 PAAD+BRCA 2개 코호트로
축소 적용한 것.

[공유 설계 — weight tying]
공유(model_paad/model_brca가 같은 nn.Module 객체를 참조):
    cnn(backbone feature projection) + vit(Nystromformer 공간 컨텍스트) + attn_pool
    (MultiComponentPooling: mean/std/attn/top-k) — 순수 WSI 인코더. "표본을 늘리면
    이득을 보는지" 검증 대상이 바로 이 부분이라 여기만 공유한다.
비공유(코호트별 독립 가중치):
    rna_encoder(유전자 패널이 다름 — PAAD literature_1500_intersection vs BRCA 자체
    top1500 고분산 유전자), clinical_linear/buffers(PAAD는 age/sex+margin+staging,
    BRCA는 age/sex만 있음 — data/brca_clinical.csv에 staging/margin 컬럼 자체가 없음),
    component_coattn(RNA-guided WSI 성분 co-attention), risk_head. risk_head를 공유하지
    않는 이유: cox_ph_loss는 코호트별로 따로 계산(stratified)한다 — 두 암종의 baseline
    hazard가 완전히 달라 risk score를 직접 섞어 비교(같은 risk set에 넣음)하면 안 되기
    때문에, 애초에 "PAAD 환자와 BRCA 환자를 직접 비교하는 학습 신호"는 존재하지 않는다.

[학습 루프] 매 결합 스텝마다 PAAD 배치 1개 + BRCA 배치 1개를 순서대로 forward해 loss를
더한 뒤 한 번만 optimizer.step() — 공유 모듈(cnn/vit/attn_pool)의 gradient는 두 코호트
loss가 합쳐진 걸 받고, 코호트 전용 모듈은 자기 코호트 loss의 gradient만 받는다. PAAD는
91명(train)뿐이라 한 epoch 안에 다 순회하고, BRCA는 훨씬 크므로 매 epoch 셔플된 상태를
필요한 만큼만 순환(cycling) 소비한다(굳이 BRCA 전체를 한 epoch 안에 다 쓸 필요는 없음).

[평가] PAAD internal(held-out test, baseline과 동일 seed 6:2:2 split)이 진짜 관심사다
("internal 밖에 안 될 테지만" — BRCA는 검증 목적의 보조 코호트). PAAD external(cptac)도
참고용으로 같이 낸다. BRCA 쪽 성능은 로그만 남기고 비교 대상으로 삼지 않는다.

[backbone=uni인 이유] BRCA는 raw WSI가 없고(HF 데이터셋 Dearcat/CPathPatchFeature에
uni/chief/gigap/r50만 있고 uni2 없음, 확인함) UNI2로 새로 뽑으려면 원본 슬라이드를 다시
받아야 해서(예전에 "너무 느리다"고 접었던 작업), 이미 양쪽 다 로컬에 있는 UNI(v1) feature로
진행한다 — 이 프로젝트에서 이미 "UNI vs UNI2, 통제된 조건에서 거의 동일" 결과가 있어
공동학습 메커니즘 자체를 검증하는 데는 지장 없다고 판단.

사용법: python -m scripts.train_pancancer_paad_brca --seed 84
"""
import argparse
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, literature_guided_gene_ids_intersection
from models.vit_pma import ViT_PMA
from models.rna_predictor import RNAPredictionHead
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv
from train import set_seed, _build_scheduler, _log_line, _patient_risk, evaluate, WANDB_AVAILABLE
from utils.losses import cox_ph_loss
from utils.metrics import compute_time_dependent_auc
from scripts.brca_common import (
    CLINICAL_PATH as BRCA_CLINICAL_PATH, BRCASlideDataset, _identity_collate,
    load_case_table as brca_load_case_table, load_rna_matrix as brca_load_rna_matrix,
    MANIFEST_PATH as BRCA_MANIFEST_PATH,
)

if WANDB_AVAILABLE:
    import wandb

BRCA_GENE_PATH = Path("data/brca_rna_gene_selection/selected_genes_top_1500.csv")


def _make_amp_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _run_cohort_batch(model, patient_batch, device, amp_ctx, chunk_size, patch_keep_frac, rna_aux_weight):
    """한 코호트의 배치(환자 리스트) 하나에 대한 loss(backward는 호출부가 함)."""
    risks, times, events, aux_losses = [], [], [], []
    for patient_slides in patient_batch:
        risk, aux_loss, _ = _patient_risk(
            model, patient_slides, device, amp_ctx, None, chunk_size, patch_keep_frac=patch_keep_frac,
        )
        risks.append(risk)
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if aux_loss is not None:
            aux_losses.append(aux_loss)
    time_t = torch.cat(times).to(device)
    event_t = torch.cat(events).to(device)
    loss = cox_ph_loss(torch.cat(risks), time_t, event_t)
    if rna_aux_weight > 0 and aux_losses:
        loss = loss + rna_aux_weight * torch.stack(aux_losses).mean()
    return loss


def _infinite_shuffled_patients(dataset, seed_offset: int = 0):
    """BRCA train set을 무한 순환(매 pass마다 재셔플)하는 제너레이터 — DataLoader 없이
    dataset[i] 직접 인덱싱(각 아이템이 이미 _identity_collate 형식과 동일한 list)."""
    rng = random.Random(1000 + seed_offset)
    n = len(dataset)
    while True:
        idxs = list(range(n))
        rng.shuffle(idxs)
        for i in idxs:
            yield dataset[i]


def train_joint_epoch(model_paad, model_brca, paad_loader, brca_cycle, optimizer, all_params, device, amp_ctx,
                       chunk_size, patch_keep_frac, rna_aux_weight, batch_size, brca_loss_weight):
    model_paad.train()
    model_brca.train()
    total_loss, n_steps = 0.0, 0
    paad_batch = []
    for patient_slides in paad_loader:
        paad_batch.append(patient_slides)
        if len(paad_batch) < batch_size:
            continue
        brca_batch = [next(brca_cycle) for _ in range(batch_size)]

        optimizer.zero_grad()
        loss_paad = _run_cohort_batch(model_paad, paad_batch, device, amp_ctx, chunk_size, patch_keep_frac, rna_aux_weight)
        loss_brca = _run_cohort_batch(model_brca, brca_batch, device, amp_ctx, chunk_size, patch_keep_frac, rna_aux_weight)
        total = loss_paad + brca_loss_weight * loss_brca
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        if not (torch.isfinite(total) and torch.isfinite(grad_norm)):
            print(f"  [경고] non-finite total(={total.item()})/grad_norm(={grad_norm.item()}) 스텝 스킵")
            optimizer.zero_grad()
        else:
            optimizer.step()
            total_loss += total.item()
            n_steps += 1
        paad_batch = []
    return total_loss / max(n_steps, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--fold", type=int, default=None,
                         help="주어지면(0-based) PAAD 쪽에 단일 6:2:2 대신 k-fold를 쓴다 "
                              "(data/dataset.py::WSISurvivalDataset fold와 동일 관례). BRCA는 "
                              "이 실험에서 held-out 평가 대상이 아니라 보조 학습 코호트일 "
                              "뿐이라 fold 개념을 적용하지 않고 --seed 기준 6:2:2의 train "
                              "부분만 계속 쓴다.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patch-keep-frac", type=float, default=0.8)
    parser.add_argument("--rna-aux-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--brca-loss-weight", type=float, default=1.0,
                         help="BRCA loss에 곱하는 가중치. BRCA가 PAAD보다 표본이 훨씬 커서 "
                              "필요하면 낮춰 균형을 맞출 수 있다(기본 1.0=동등).")
    parser.add_argument("--group-ts", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.data.seed = cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.model.use_attn_dispersion = True
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = _make_amp_ctx(device)
    chunk_size = cfg.train.cnn_chunk_size
    start_time = datetime.now()

    # ---- PAAD 모델(baseline PMA 레시피, backbone=uni만 다름) ----
    paad_rna_genes = literature_guided_gene_ids_intersection(1500)
    paad_age_mean, paad_age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    paad_margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    paad_stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])
    model_paad = ViT_PMA(
        cfg.model, age_mean=paad_age_mean, age_std=paad_age_std, rna_input_dim=len(paad_rna_genes),
        backbone="uni", combine_mode="cox_add", use_margin=True, margin_stats=paad_margin_stats,
        use_age_sex=True, use_staging=True, stage_stats=paad_stage_stats,
    ).to(device)
    if args.rna_aux_weight > 0:
        model_paad.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(paad_rna_genes)).to(device)

    # ---- BRCA 모델(age/sex만, margin/staging 없음 — data/brca_clinical.csv에 컬럼 자체가 없음) ----
    if not BRCA_GENE_PATH.exists():
        raise FileNotFoundError(f"{BRCA_GENE_PATH} 없음 — 먼저 scripts/select_brca_rna_genes.py 실행 필요")
    brca_gene_ids = pd.read_csv(BRCA_GENE_PATH)["gene_id"].tolist()
    brca_age_mean, brca_age_std = age_stats_from_csv(BRCA_CLINICAL_PATH)
    model_brca = ViT_PMA(
        cfg.model, age_mean=brca_age_mean, age_std=brca_age_std, rna_input_dim=len(brca_gene_ids),
        backbone="uni", combine_mode="cox_add", use_margin=False, use_age_sex=True, use_staging=False,
    ).to(device)
    if args.rna_aux_weight > 0:
        model_brca.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(brca_gene_ids)).to(device)

    # ---- weight tying: WSI trunk(cnn/vit/attn_pool)만 공유 ----
    model_brca.cnn = model_paad.cnn
    model_brca.vit = model_paad.vit
    model_brca.attn_pool = model_paad.attn_pool

    n_shared = sum(p.numel() for p in model_paad.cnn.parameters()) + \
               sum(p.numel() for p in model_paad.vit.parameters()) + \
               sum(p.numel() for p in model_paad.attn_pool.parameters())
    print(f"공유 WSI trunk 파라미터 수: {n_shared:,}")
    print(f"PAAD 모델 전체 파라미터 수: {sum(p.numel() for p in model_paad.parameters()):,}")
    print(f"BRCA 모델 전체 파라미터 수: {sum(p.numel() for p in model_brca.parameters()):,}")

    # ---- 데이터셋 ----
    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True, with_rna=True,
                      rna_gene_ids=paad_rna_genes, feature_backbone="uni")
    fold_kwargs = dict(fold=args.fold, n_folds=args.n_folds) if args.fold is not None else {}
    paad_train_ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="train", **fold_kwargs, **ds_kwargs)
    paad_val_ds   = WSISurvivalDataset(cfg.data, dataset="tcga", split="val",   **fold_kwargs, **ds_kwargs)
    paad_test_ds  = WSISurvivalDataset(cfg.data, dataset="tcga", split="test",  **fold_kwargs, **ds_kwargs)
    paad_ext_ds   = WSISurvivalDataset(cfg.data, dataset="cptac", split="all",  **ds_kwargs)

    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    paad_train_loader = DataLoader(paad_train_ds, shuffle=True,  **dl_kwargs)
    paad_eval_loader  = DataLoader(paad_train_ds, shuffle=False, **dl_kwargs)
    paad_val_loader   = DataLoader(paad_val_ds,   shuffle=False, **dl_kwargs)
    paad_test_loader  = DataLoader(paad_test_ds,  shuffle=False, **dl_kwargs)
    paad_ext_loader   = DataLoader(paad_ext_ds,   shuffle=False, **dl_kwargs)

    brca_cases = brca_load_case_table(args.seed)
    brca_rna_df = brca_load_rna_matrix(brca_gene_ids)
    brca_manifest = pd.read_csv(BRCA_MANIFEST_PATH)
    brca_train_ds = BRCASlideDataset(brca_cases[brca_cases["split"] == "train"], brca_rna_df, brca_manifest)
    brca_cycle = _infinite_shuffled_patients(brca_train_ds, seed_offset=args.seed)

    print(f"PAAD: train={len(paad_train_ds)} val={len(paad_val_ds)} test={len(paad_test_ds)} external={len(paad_ext_ds)}")
    print(f"BRCA: train={len(brca_train_ds)} (val/test는 이번 실험에서 미평가 — PAAD internal이 관심사)")

    # ---- optimizer: 공유 파라미터가 중복 등록되지 않도록 id 기준으로 dedupe ----
    seen = {}
    for p in list(model_paad.parameters()) + list(model_brca.parameters()):
        seen[id(p)] = p
    all_params = list(seen.values())
    print(f"optimizer 파라미터 수(중복 제거 후): {sum(p.numel() for p in all_params):,}")

    optimizer = torch.optim.AdamW(
        [p for p in all_params if p.requires_grad], lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg)

    base_model_prefix = "PANCANCER_PAAD_BRCA_uni_STG_R_DISP_COX_ADD"
    model_prefix = base_model_prefix
    if args.fold is not None:
        model_prefix += f"_FOLD{args.fold}OF{args.n_folds}"
    run_ts = datetime.now().strftime("%m%d::%H%M")
    group_ts = args.group_ts or run_ts
    if WANDB_AVAILABLE:
        wandb.init(project="Path-ViT", name=f"{model_prefix}_seed{args.seed}_{run_ts}",
                   group=f"{base_model_prefix}_{group_ts}",
                   config={"epochs": cfg.train.epochs, "lr": cfg.train.lr, "seed": args.seed,
                           "fold": args.fold, "n_folds": args.n_folds,
                           "brca_loss_weight": args.brca_loss_weight, "model": model_prefix})

    ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"survival_tcga_uni_seed{args.seed}_{model_prefix}_best_pancancer.pt"

    best_score, best_metrics = -1.0, {}
    for epoch in range(cfg.train.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_joint_epoch(model_paad, model_brca, paad_train_loader, brca_cycle, optimizer, all_params,
                                  device, amp_ctx, chunk_size, args.patch_keep_frac, args.rna_aux_weight,
                                  args.batch_size, args.brca_loss_weight)
        train_metrics = evaluate(model_paad, paad_eval_loader, cfg, device, amp_ctx, None, desc="paad_train_eval")
        val_metrics = evaluate(model_paad, paad_val_loader, cfg, device, amp_ctx, None, desc="paad_val")
        val_td_auc = compute_time_dependent_auc(
            train_metrics["times"], train_metrics["events"], val_metrics["times"], val_metrics["events"], val_metrics["risks"],
        )
        scheduler.step()

        c_index = val_metrics.get("c_index", float("nan"))
        score = c_index if c_index == c_index else -1.0  # NaN-safe
        print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | loss={loss:.4f} | "
              f"paad_train_c_index={train_metrics['c_index']:.4f} | " + _log_line("paad_val", val_metrics, val_td_auc))

        if WANDB_AVAILABLE:
            wandb.log({"train/loss": loss, "train/lr": lr_now, "train/c_index": train_metrics["c_index"],
                       "val_performance/c_index": val_metrics["c_index"], "val_performance/hr": val_metrics["hr"],
                       "val_performance/log_rank_p": val_metrics["log_rank_p"],
                       "val_performance/auc_mean": val_td_auc["auc_mean"]}, step=epoch + 1)

        if score > best_score:
            best_score = score
            best_metrics = {**val_metrics, "epoch": epoch + 1}
            torch.save({"model_state_dict": model_paad.state_dict(), "epoch": epoch + 1, "val_c_index": best_score}, ckpt_path)
            print(f"  -> checkpoint saved (c_index={best_score:.4f}, HR={val_metrics['hr']:.3f}, "
                  f"log-rank p={val_metrics['log_rank_p']:.4f})")
            if WANDB_AVAILABLE:
                wandb.run.summary["best_val_c_index"] = best_score
                wandb.run.summary["best_epoch"] = epoch + 1

    # ---- 최종(마지막 epoch) PAAD internal/external ----
    train_metrics_final = evaluate(model_paad, paad_eval_loader, cfg, device, amp_ctx, None, desc="final_paad_train_eval")
    test_metrics_final = evaluate(model_paad, paad_test_loader, cfg, device, amp_ctx, None, desc="final_paad_test")
    test_td_auc_final = compute_time_dependent_auc(
        train_metrics_final["times"], train_metrics_final["events"],
        test_metrics_final["times"], test_metrics_final["events"], test_metrics_final["risks"],
    )
    print(f"\n=== PAAD Internal Test (마지막 epoch {cfg.train.epochs} 모델, best-val 선택 없음) ===")
    print(_log_line("final_paad_test", test_metrics_final, test_td_auc_final))
    ext_metrics_final = evaluate(model_paad, paad_ext_loader, cfg, device, amp_ctx, None, desc="final_paad_external")
    ext_td_auc_final = compute_time_dependent_auc(
        train_metrics_final["times"], train_metrics_final["events"],
        ext_metrics_final["times"], ext_metrics_final["events"], ext_metrics_final["risks"],
    )
    print(f"=== PAAD External Test (cptac, 마지막 epoch 모델) ===")
    print(_log_line("final_paad_external", ext_metrics_final, ext_td_auc_final))

    # ---- best checkpoint PAAD internal/external ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_paad.load_state_dict(ckpt["model_state_dict"])
    train_metrics_best = evaluate(model_paad, paad_eval_loader, cfg, device, amp_ctx, None, desc="best_paad_train_eval")
    test_metrics_best = evaluate(model_paad, paad_test_loader, cfg, device, amp_ctx, None, desc="best_paad_test")
    test_td_auc_best = compute_time_dependent_auc(
        train_metrics_best["times"], train_metrics_best["events"],
        test_metrics_best["times"], test_metrics_best["events"], test_metrics_best["risks"],
    )
    print(f"\n=== PAAD Internal Test (best checkpoint epoch {ckpt['epoch']}) ===")
    print(_log_line("paad_test", test_metrics_best, test_td_auc_best))
    ext_metrics_best = evaluate(model_paad, paad_ext_loader, cfg, device, amp_ctx, None, desc="best_paad_external")
    ext_td_auc_best = compute_time_dependent_auc(
        train_metrics_best["times"], train_metrics_best["events"],
        ext_metrics_best["times"], ext_metrics_best["events"], ext_metrics_best["risks"],
    )
    print(f"=== PAAD External Test (cptac, best checkpoint) ===")
    print(_log_line("paad_external", ext_metrics_best, ext_td_auc_best))

    # ---- 멀티시드 pooling 스크립트(scripts/pool_multiseed_kfold_preds.py/pool_multiseed_
    # external_preds.py) 입력용 CSV 저장 — train.py의 --fold/--eval-external-ckpt 저장 관례와
    # 동일 경로/파일명(base_model_prefix 기준, FOLD/seed는 pooling 스크립트가 자체적으로
    # 채워 넣으므로 여기서는 base_model_prefix — FOLD 접미사 없는 버전 — 를 그대로 쓴다).
    if args.fold is not None:
        import csv
        kfold_dir = Path(__file__).parent.parent / ".logs" / "kfold_preds"
        kfold_dir.mkdir(parents=True, exist_ok=True)
        kfold_path = kfold_dir / f"tcga_{model_prefix}_seed{args.seed}_fold{args.fold}of{args.n_folds}.csv"
        with open(kfold_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, risk, t, e in zip(test_metrics_best["case_ids"], test_metrics_best["risks"],
                                        test_metrics_best["times"], test_metrics_best["events"]):
                writer.writerow([cid, risk, t, e])
        print(f"  -> internal fold predictions saved: {kfold_path}")

        ext_dir = Path(__file__).parent.parent / ".logs" / "external_preds"
        ext_dir.mkdir(parents=True, exist_ok=True)
        ext_path = ext_dir / f"cptac_{model_prefix}_seed{args.seed}_fold{args.fold}of{args.n_folds}.csv"
        with open(ext_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "risk", "OS_time", "OS_event"])
            for cid, risk, t, e in zip(ext_metrics_best["case_ids"], ext_metrics_best["risks"],
                                        ext_metrics_best["times"], ext_metrics_best["events"]):
                writer.writerow([cid, risk, t, e])
        print(f"  -> external predictions saved: {ext_path}")

    if WANDB_AVAILABLE:
        wandb.run.summary["test_c_index"] = test_metrics_best["c_index"]
        wandb.run.summary["external_c_index"] = ext_metrics_best["c_index"]
        wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    print(f"\n소요 시간: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
