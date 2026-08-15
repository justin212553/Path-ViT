"""
PMA(baseline, PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD) WSI 브랜치의 구조적 결함 여부를
현재 레시피(uni2/cox_add/margin/staging/attn-dispersion)에서 재확인한다 — 2026-07-21의
diagnose_wsi_reliance.py/diagnose_wsi_gradients.py(구식 레시피: uni 백본, cox_add/staging/
margin/dispersion 없음)가 발견한 두 가지(patch attention entropy 거의 uniform, RNA gradient가
WSI의 ~4배)가 지금 레시피에서도 여전한지 확인하는 게 목적. 재학습 없이, 이미 학습 끝난
체크포인트 하나로 forward(+ 진단용 backward 1회)만 돌린다 — 한 폴드(seed42 fold0)로 빠르게
확인한다.

측정 항목:
  (A) patch-level ABMIL attention entropy — models/multi_component_pooling.py 내부
      AttentionPooling이 만드는 patch attn_weights가 uniform(entropy→1)인지 peaked(entropy→0)인지.
  (B) 4-component co-attention entropy — component_coattn(RNA query)이 mean/std/attn/top-k
      4개 관점 중 하나로 쏠리는지(entropy→0) 균등하게 섞는지(entropy→1).
  (C) risk_head 입력 스케일 비교 — z_wsi(component_coattn 출력) vs z_rna(RNA 인코더 출력)의
      raw L2 norm. 두 값이 크게 다르면 risk_head의 LayerNorm이 암묵적으로 큰 쪽에 쏠릴 수 있다
      (attn_dispersion 좌표스케일 버그와 같은 버그 클래스 — 2026-07-30 발견/수정 전례).
  (D) 학습 중 gradient norm 비교 — model.cnn/vit(WSI 인코더+ViT), model.attn_pool/
      component_coattn(WSI pooling), model.rna_encoder(RNA), model.risk_head/clinical_linear
      (fusion+clinical) 각 서브모듈의 파라미터 gradient L2 norm을 한 배치(cox_batch_size=16)에
      대해 실제로 backward 한 번 돌려서 측정 — 2026-07-21 발견(RNA가 WSI의 ~4배)이 지금도
      재현되는지 직접 확인.

사용법: python -m scripts.diagnose_pma_wsi_structure --seed 42 --fold 0
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
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv, STAGE_FIELDS
from models.rna_predictor import RNAPredictionHead
from train import _patient_risk, _make_amp_ctx
from utils.losses import cox_ph_loss

MODEL_TAG = "PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD"


def _identity_collate(batch):
    return batch[0]


def _entropy(w: torch.Tensor) -> float:
    """정규화된 Shannon entropy(0~1). 1이면 완전 uniform, 0이면 완전 peaked(한 곳에 몰림)."""
    n = w.numel()
    if n <= 1:
        return float("nan")
    p = w.clamp(min=1e-12)
    h = -(p * p.log()).sum().item()
    return h / np.log(n)


def _load_fold_case_ids(seed: int, fold: int) -> set[str]:
    import csv
    path = (_ROOT / ".logs" / "kfold_preds" /
            f"tcga_{MODEL_TAG}_FOLD{fold}OF5_seed{seed}_fold{fold}of5.csv")
    with open(path, newline="") as f:
        return {row["case_id"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])

    cfg = Config()
    cfg.model.use_attn_dispersion = True
    model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=len(rna_gene_ids),
                     backbone="uni2", combine_mode="cox_add", use_margin=True, margin_stats=margin_stats,
                     use_age_sex=True, use_staging=True, stage_stats=stage_stats).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, len(rna_gene_ids)).to(device)

    ckpt_glob = list((_ROOT / "models" / "checkpoint").glob(
        f"survival_tcga_uni2_seed{args.seed}_*{MODEL_TAG}_FOLD{args.fold}OF5_best_pma.pt"))
    if len(ckpt_glob) != 1:
        raise RuntimeError(f"checkpoint {len(ckpt_glob)}개 매칭 (1개여야 함): {ckpt_glob}")
    ckpt = torch.load(ckpt_glob[0], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"체크포인트: {ckpt_glob[0].name} (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")

    ds_kwargs = dict(with_clinical=True, with_margin=True, with_staging=True, with_rna=True,
                      rna_gene_ids=rna_gene_ids, feature_backbone="uni2")
    fold_case_ids = _load_fold_case_ids(args.seed, args.fold)
    ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="all", restrict_case_ids=fold_case_ids, **ds_kwargs)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"환자 수: {len(ds.cases)} (seed={args.seed} fold={args.fold} test set)")

    # ---------- (A)(B)(C): forward-only, eval mode ----------
    model.eval()
    patch_entropies, coattn_entropies = [], []
    wsi_norms, rna_norms = [], []
    with torch.no_grad():
        for patient_slides in loader:
            p = patient_slides[0]
            rna = p["rna"].to(device)
            z_rna = model.encode_rna(rna)
            comps, patch_attn_ents = [], []
            for slide in patient_slides:
                out = model(slide["coords"].to(device), features=slide["features"])
                comps.append(out["embed"])
                patch_attn_ents.append(_entropy(out["attn_weights"]))
            patient_embed = torch.stack(comps).mean(dim=0)  # (4, D)
            z_wsi, coattn_w = model.component_coattn(patient_embed, z_rna)

            patch_entropies.append(np.mean(patch_attn_ents))
            coattn_entropies.append(_entropy(coattn_w))
            wsi_norms.append(z_wsi.norm().item())
            rna_norms.append(z_rna.norm().item())

    print("\n=== (A) patch-level ABMIL attention entropy (1=완전 uniform, 0=완전 peaked) ===")
    print(f"  mean={np.mean(patch_entropies):.4f}  std={np.std(patch_entropies):.4f}  "
          f"min={np.min(patch_entropies):.4f}  max={np.max(patch_entropies):.4f}")

    print("\n=== (B) 4-component co-attention entropy (RNA query, mean/std/attn/top-k 중 선택) ===")
    print(f"  mean={np.mean(coattn_entropies):.4f}  std={np.std(coattn_entropies):.4f}  "
          f"min={np.min(coattn_entropies):.4f}  max={np.max(coattn_entropies):.4f}")

    print("\n=== (C) risk_head 입력 스케일 — z_wsi vs z_rna raw L2 norm ===")
    print(f"  z_wsi: mean={np.mean(wsi_norms):.4f}  std={np.std(wsi_norms):.4f}")
    print(f"  z_rna: mean={np.mean(rna_norms):.4f}  std={np.std(rna_norms):.4f}")
    print(f"  비율(z_rna/z_wsi) = {np.mean(rna_norms) / np.mean(wsi_norms):.3f}")

    # ---------- (D): gradient norm, 실제 training-step 재현(같은 환자셋에서 첫 배치) ----------
    model.train()
    amp_ctx = _make_amp_ctx() if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    batch_size = cfg.train.cox_batch_size
    risks, times, events, aux_losses = [], [], [], []
    for i, patient_slides in enumerate(loader):
        if i >= batch_size:
            break
        risk, aux_loss, _ = _patient_risk(
            model, patient_slides, device, amp_ctx, transform=None,
            chunk_size=cfg.train.cnn_chunk_size, patch_keep_frac=1.0,
        )
        risks.append(risk)
        times.append(patient_slides[0]["OS_time"])
        events.append(patient_slides[0]["OS_event"])
        if aux_loss is not None:
            aux_losses.append(aux_loss)

    time_t = torch.cat(times).to(device)
    event_t = torch.cat(events).to(device)
    # baseline 레시피(--rna-aux-weight 1.0)와 동일하게 RNA 예측 보조loss를 포함한다 —
    # 이 보조loss가 WSI 브랜치(model.vit/cnn)에도 gradient를 흘려보내는 실제 경로이므로 빼면
    # 지금 레시피의 WSI 브랜치가 받는 gradient를 과소평가하게 된다.
    loss = cox_ph_loss(torch.cat(risks), time_t, event_t)
    if aux_losses:
        loss = loss + 1.0 * torch.stack(aux_losses).mean()
    model.zero_grad()
    loss.backward()

    def _grad_norm(module) -> float:
        sq = sum(p.grad.norm().item() ** 2 for p in module.parameters() if p.grad is not None)
        return sq ** 0.5

    print(f"\n=== (D) gradient norm 비교 (배치={len(risks)}명, loss={loss.item():.4f}) ===")
    print(f"  model.cnn(WSI 인코더 투영)         : {_grad_norm(model.cnn):.4f}")
    if model.vit is not None:
        print(f"  model.vit(patch-mixing transformer) : {_grad_norm(model.vit):.4f}")
    print(f"  model.attn_pool(4관점 pooling)      : {_grad_norm(model.attn_pool):.4f}")
    print(f"  model.component_coattn(RNA co-attn) : {_grad_norm(model.component_coattn):.4f}")
    print(f"  model.rna_encoder(RNA)              : {_grad_norm(model.rna_encoder):.4f}")
    print(f"  model.risk_head                     : {_grad_norm(model.risk_head):.4f}")
    print(f"  model.clinical_linear(cox_add)      : {_grad_norm(model.clinical_linear):.4f}")


if __name__ == "__main__":
    main()
