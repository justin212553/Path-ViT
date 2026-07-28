"""
BRCA로 WSI trunk(proj + Nystromformer)만 RNA-aux로 pretrain — PAAD로 가중치 이전용.

배경(findings_backlog.md 2026-07-22 최상위 발견 참조): TCGA-BRCA(1058명)에서 PMA_EX_SS_AUX가
M7을 유의미하게 이겼다(WSI가 표본이 커지면 순증분 기여) — 근데 TCGA-PAAD는 152명뿐이라 같은
아키텍처를 처음부터 거기서 학습시키면 여전히 신호가 부족하다. 그래서 WSI trunk(형태학적 표현을
만드는 부분)만 BRCA에서 먼저 학습시켜두고, PAAD로 가중치를 이전해 fine-tune하자는 계획.

[전이 대상 = trunk만] `models/vit_pma.py::ViT_PMA`(및 base `ViT_M1`)에서 RNA는 두 곳에서만
개입한다: (1) `RNAEncoder`(원본 유전자 벡터 -> z_rna, 유전자 목록/순서에 직접 의존), (2)
`component_coattn`이 z_rna를 query로 씀. `rna_aux_head`(models/rna_predictor.py, HE2RNA류
보조과제)가 실제로 붙는 지점은 `meanpool_embed = ctx_tokens.mean(dim=0)` — **RNA가 개입하기
전, `self.cnn.proj`(UNI feature projection)와 `self.vit`(ViTEncoder/Nystromformer)만 지난
단계**다. 즉 BRCA/PAAD의 RNA 유전자 목록이 완전히 달라도(top1500 vs literature_1500, 겹침
12.3%뿐 — 실측 확인함) 이 pretrain이 실제로 학습시키는 `proj`+`vit`는 유전자 정체성을 전혀
보지 않으므로 문제가 되지 않는다 — PAAD로 옮길 때는 `RNAEncoder`/`rna_aux_head`/
`component_coattn`은 버리고 PAAD의 literature_1500으로 새로 초기화한다.

[뭉뚱그리지 않는다] `--rna-genes pathway8`(8개 카테고리 평균)이 부호 없는 단순 평균이 반대
방향 유전자를 상쇄시켜 명확히 실패했던 전적이 있다(findings_backlog.md 10번 항목, external
C 0.52 이하로 붕괴) — 그 교훈을 따라 이번에도 BRCA top1500 개별 유전자를 그대로(가공 없이)
예측 타깃으로 쓴다. BRCA/PAAD 두 유전자셋의 겹치는 185개가 증식/면역/기질/상피 같은 범용
암 생물학 프로그램 위주였다는 사전 확인(가이드 대화 참조)이 이 raw 방식이 아주 무모하지는
않을 것이라는 근거다.

[case 제한 없음] survival(OS) 라벨을 전혀 안 쓰는 순수 회귀(MSE) 보조과제라 train/val/test
split을 지킬 이유가 없다 — BRCA 자체를 다시 평가하지 않고 PAAD로 넘어갈 것이므로, BRCA
전체 1058 case를 다 pretrain에 쓴다(2번 항목 "case 수를 최대한 확보" 원칙과 일치).

출력:
    models/checkpoint/brca_pretrain_wsi_trunk.pt   {"cnn": state_dict, "vit": state_dict}
    (train.py --pretrained-wsi-trunk로 로드해 PAAD fine-tune에 사용)

사용법:
    python -m scripts.pretrain_brca_wsi_trunk --epochs 30
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from models.vit_m1 import ViT_M1
from models.rna_predictor import RNAPredictionHead
from train import set_seed, _build_scheduler, WANDB_AVAILABLE
from scripts.brca_common import (
    BRCASlideDataset, _identity_collate, common_case_ids, load_rna_matrix, CLINICAL_PATH,
    MANIFEST_PATH,
)

if WANDB_AVAILABLE:
    import wandb

OUT_DIR = Path("data/brca_rna_gene_selection")
CKPT_DIR = Path(__file__).parent.parent / "models" / "checkpoint"


def _patient_meanpool(model, patient_slides, device, patch_keep_frac: float) -> torch.Tensor:
    """patient_slides(BRCASlideDataset 산출물)를 순회해 슬라이드 평균 meanpool_embed를 낸다.

    train.py::_patient_risk의 slide 루프/patch_keep_frac 서브샘플 로직과 동일 — clinical/rna
    결합이 없다는 점만 다르다(이 pretrain은 RNA를 예측 *타깃*으로만 쓰고 입력엔 안 씀).
    """
    slide_embeds = []
    for slide in patient_slides:
        coords = slide["coords"]
        features = slide["features"]
        if model.training and patch_keep_frac < 1.0:
            n = coords.shape[0]
            k = max(1, round(n * patch_keep_frac))
            idx = torch.randperm(n)[:k]
            coords, features = coords[idx], features[idx]
        coords = coords.to(device, non_blocking=True)
        out = model(coords, features=features)
        slide_embeds.append(out["meanpool_embed"])
    return torch.stack(slide_embeds).mean(dim=0)


def _run_epoch(model, loader, device, patch_keep_frac: float, batch_size: int,
               optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss, n_patients = 0.0, 0
    preds, targets = [], []

    def _flush():
        nonlocal preds, targets, total_loss, n_patients
        if not preds:
            return
        loss = F.mse_loss(torch.stack(preds), torch.stack(targets))
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], max_norm=1.0
            )
            optimizer.step()
        total_loss += loss.item() * len(preds)
        n_patients += len(preds)
        preds, targets = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for patient_slides in loader:
            if len(patient_slides) == 0:
                continue
            meanpool = _patient_meanpool(model, patient_slides, device, patch_keep_frac if training else 1.0)
            rna_pred = model.rna_aux_head(meanpool)
            preds.append(rna_pred)
            targets.append(patient_slides[0]["rna"].to(device, non_blocking=True))
            if training and len(preds) >= batch_size:
                _flush()
        _flush()
    return total_loss / max(n_patients, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-genes", type=int, default=1500)
    parser.add_argument("--gene-list-path", type=str, default=None,
                         help="--n-genes 기반 기본 경로(selected_genes_top_{n}.csv) 대신 임의의 "
                              "gene_id 목록 csv를 aux 타깃으로 쓴다. findings_backlog.md 2026-07-22 "
                              "'다른 task' 재시도 — BRCA/PAAD 겹치는 185개(증식/면역/기질 등 범용"
                              "프로그램 위주)만으로 pretrain해 negative transfer를 줄이는 실험용.")
    parser.add_argument("--ckpt-tag", type=str, default=None,
                         help="출력 체크포인트 파일명 접미사(예: overlap185) — 지정 안 하면 "
                              "brca_pretrain_wsi_trunk.pt(기존 top1500 버전)를 덮어쓴다.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None,
                         help="cfg.train.lr(기본 1e-5, ViT_PMA 생존예측용으로 튜닝된 값을 그대로 "
                              "물려받은 것) 덮어쓰기 — 이 pretrain은 손실 함수(MSE)도 표본 규모"
                              "(952명)도 다르니 더 큰 값(예: 1e-4)이 30 epoch 안에 더 잘 수렴할 "
                              "수 있다(findings_backlog.md 2026-07-22, val_mse가 30 epoch 끝까지 "
                              "plateau 없이 계속 하강 중이었던 관찰).")
    parser.add_argument("--patch-keep-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1,
                         help="순수 모니터링용 held-out(재사용/전이 대상 아님, PAAD 평가와 무관).")
    args = parser.parse_args()

    cfg = Config()
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = datetime.now()

    gene_path = Path(args.gene_list_path) if args.gene_list_path else OUT_DIR / f"selected_genes_top_{args.n_genes}.csv"
    gene_ids = pd.read_csv(gene_path)["gene_id"].tolist()
    rna_input_dim = len(gene_ids)
    rna_df = load_rna_matrix(gene_ids)

    case_ids = common_case_ids()
    clinical = pd.read_csv(CLINICAL_PATH).set_index("case_id")
    manifest = pd.read_csv(MANIFEST_PATH)
    rng = np.random.RandomState(args.seed)
    shuffled = np.array(case_ids)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_frac))
    val_ids, train_ids = shuffled[:n_val], shuffled[n_val:]
    print(f"BRCA pretrain case 수: {len(case_ids)}  (train={len(train_ids)}, val(모니터링용)={len(val_ids)})"
          f"  — survival 라벨 미사용, 전체 case를 trunk pretrain에 사용")
    print(f"RNA 타깃 유전자 수: {rna_input_dim} (top{args.n_genes}, 가공 없이 개별 유전자 그대로)")

    train_table = clinical.loc[train_ids].reset_index()
    val_table = clinical.loc[val_ids].reset_index()
    train_ds = BRCASlideDataset(train_table, rna_df, manifest)
    val_ds = BRCASlideDataset(val_table, rna_df, manifest)
    dl_kwargs = dict(batch_size=1, collate_fn=_identity_collate, num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    model = ViT_M1(cfg.model, precomputed=True, backbone="uni").to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
    trainable = list(model.cnn.parameters()) + list(model.vit.parameters()) + list(model.rna_aux_head.parameters())
    print(f"trunk(cnn+vit)+rna_aux_head params: {sum(p.numel() for p in trainable):,}  "
          f"(attn_pool/risk_head는 이 pretrain에서 미사용, 저장 안 함)")

    run_ts = datetime.now().strftime("%m%d::%H%M")
    if WANDB_AVAILABLE:
        wandb.init(
            project="Path-ViT",
            name=f"BRCA_PRETRAIN_WSI_TRUNK_seed{args.seed}_{run_ts}",
            group=f"BRCA_PRETRAIN_WSI_TRUNK_{run_ts}",
            config={
                "epochs": cfg.train.epochs, "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay,
                "seed": args.seed, "n_genes": args.n_genes, "patch_keep_frac": args.patch_keep_frac,
                "embed_dim": cfg.model.embed_dim, "dataset": "brca_pretrain",
            },
        )

    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg)

    ckpt_name = f"brca_pretrain_wsi_trunk_{args.ckpt_tag}.pt" if args.ckpt_tag else "brca_pretrain_wsi_trunk.pt"
    CKPT_PATH = CKPT_DIR / ckpt_name
    best_val_loss = float("inf")
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg.train.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = _run_epoch(model, train_loader, device, args.patch_keep_frac,
                                 cfg.train.cox_batch_size, optimizer=optimizer)
        val_loss = _run_epoch(model, val_loader, device, args.patch_keep_frac, cfg.train.cox_batch_size)
        scheduler.step()
        print(f"Epoch {epoch+1:3d} | lr={lr_now:.2e} | train_mse={train_loss:.4f} | val_mse={val_loss:.4f}")
        if WANDB_AVAILABLE:
            wandb.log({"train/mse": train_loss, "train/lr": lr_now, "val/mse": val_loss}, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"cnn": model.cnn.state_dict(), "vit": model.vit.state_dict(),
                 "epoch": epoch + 1, "val_mse": val_loss},
                CKPT_PATH,
            )
            print(f"  -> trunk checkpoint saved (val_mse={val_loss:.4f})")

    if WANDB_AVAILABLE:
        wandb.run.summary["best_val_mse"] = best_val_loss
        wandb.finish()

    elapsed = datetime.now() - start_time
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    print(f"\n저장: {CKPT_PATH} (best val_mse={best_val_loss:.4f})")
    print(f"소요 시간: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
