"""
[조사 3] 이전 이상치 감사(scripts/audit_leverage_patients.py)에서 나온 "빼면 c-index가 오히려
오르는" 오분류 환자들의 patch attention을 실제 WSI 위에 시각화한다.

models/multi_component_pooling.py::MultiComponentPooling이 "attn" 성분을 계산할 때 쓰는
patch별 attention weight(N,)를 그 환자를 held-out으로 학습한 checkpoint에서 그대로 뽑아,
실제 패치 이미지(data/patches_{tcga,cptac}/tiles/<slide_id>/r####_c####.jpg)로 만든 저해상도
모자이크 위에 반투명 오버레이한다. 조사 1(probe_wsi_only_signal.py)에서 WSI 단독 신호가
내부/외부 모두 c-index~0.50 수준(사실상 무신호)으로 나온 뒤라, 이 attention이 실제 조직
구조(종양/기질/괴사 등)에 그럴듯하게 몰리는지, 아니면 사실상 무작위/균일한지 눈으로 확인하는
목적이다.

환자별로 어느 (seed, fold) checkpoint가 held-out이었는지는 .logs/kfold_preds/의 CSV 파일명에서
찾는다(그 case_id가 들어있는 fold 파일 = 그 checkpoint가 그 환자를 한 번도 학습에 안 쓴 것).

사용법: python scripts/generate_attention_heatmaps.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, PATCHES_ROOT_ATTRS, literature_guided_gene_ids_intersection
from data.patch_utils import list_patch_paths
from models import ViT_PMA
from models.clinical_encoder import age_stats_from_csv, margin_stats_from_csv, stage_stats_from_csv
from models.rna_predictor import RNAPredictionHead

# scripts/audit_leverage_patients.py "빼면 c-index가 오히려 오르는" 상위 목록에서, 두 실패
# 유형(사망인데 risk 낮게 잡음 / 생존인데 risk 높게 잡음)을 각각 대표하도록 6명 선정.
PATIENTS = [
    # case_id, 실패유형 설명
    ("TCGA-FB-A4P5", "event=1(179일 사망) risk 과소평가 — 최악 1위"),
    ("TCGA-HZ-A49I", "event=1(308일 사망) risk 과소평가"),
    ("TCGA-US-A77G", "event=1(12일 조기사망) risk 과소평가"),
    ("TCGA-IB-A5SO", "event=1(365일 사망) risk 과소평가"),
    ("TCGA-IB-7885", "censored(1257일 생존) risk 과대평가"),
    ("TCGA-IB-A6UF", "censored(666일 생존) risk 과대평가"),
]

N_FOLDS = 5
CKPT_DIR = _ROOT / "models" / "checkpoint"
PRED_DIR = _ROOT / ".logs" / "kfold_preds"
OUT_DIR = _ROOT / ".logs" / "attention_heatmaps"
CELL_PX = 28  # 모자이크에서 패치 1개를 이만큼의 정사각형으로 축소


def _find_holdout_seed_fold(case_id: str) -> tuple[int, int]:
    """이 환자가 test split에 들어간 (seed, fold)를 kfold_preds CSV에서 찾는다(seed 오름차순 첫 매칭)."""
    for seed in (42, 84, 126):
        for fold in range(N_FOLDS):
            path = PRED_DIR / f"tcga_PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_FOLD{fold}OF{N_FOLDS}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row["case_id"] == case_id:
                        return seed, fold
    raise ValueError(f"{case_id}: 어느 fold에서도 못 찾음")


def _ckpt_path(seed: int, fold: int) -> Path:
    matches = list(CKPT_DIR.glob(
        f"survival_tcga_uni2_seed{seed}_*STG_R_DISP_COX_ADD_FOLD{fold}OF{N_FOLDS}_best_pma.pt"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(f"seed={seed} fold={fold}: checkpoint {len(matches)}개 매칭됨")
    return matches[0]


def _build_model(device, rna_input_dim, age_mean, age_std, margin_stats, stage_stats, ckpt_path):
    cfg = Config()
    cfg.model.use_attn_dispersion = True
    model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                     backbone="uni2", combine_mode="cox_add", use_margin=True, margin_stats=margin_stats,
                     use_age_sex=True, use_staging=True, stage_stats=stage_stats).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _identity_collate(batch):
    return batch[0]


def _build_mosaic_and_heatmap(slide_dir: Path, coords: torch.Tensor, attn_weights: np.ndarray):
    """실제 패치 jpg로 저해상도 모자이크를, attn_weights(percentile 정규화)로 히트맵 배열을 만든다."""
    patch_paths = list_patch_paths(slide_dir)
    max_row = int(coords[:, 0].max().item()) + 1
    max_col = int(coords[:, 1].max().item()) + 1

    mosaic = np.zeros((max_row * CELL_PX, max_col * CELL_PX, 3), dtype=np.uint8)
    heat = np.full((max_row, max_col), np.nan)

    # percentile rank로 정규화 — attention이 소수 패치에 극단적으로 몰려도 시각적으로 잘 보이게
    order = np.argsort(np.argsort(attn_weights))
    pct = order / max(1, len(attn_weights) - 1)

    for i, (r, c) in enumerate(coords.tolist()):
        img = Image.open(patch_paths[i]).convert("RGB").resize((CELL_PX, CELL_PX))
        mosaic[r * CELL_PX:(r + 1) * CELL_PX, c * CELL_PX:(c + 1) * CELL_PX] = np.array(img)
        heat[r, c] = pct[i]

    return mosaic, heat


@torch.no_grad()
def _slide_attention(model, device, slide):
    fwd = model(slide["coords"].to(device), features=slide["features"].to(device))
    return fwd["attn_weights"].cpu().numpy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rna_gene_ids = literature_guided_gene_ids_intersection(1500)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATHS["tcga"])
    margin_stats = margin_stats_from_csv(CLINICAL_PATHS["tcga"])
    stage_stats = stage_stats_from_csv(CLINICAL_PATHS["tcga"])

    for case_id, note in PATIENTS:
        seed, fold = _find_holdout_seed_fold(case_id)
        print(f"{case_id} ({note}) -> held-out at seed={seed} fold={fold}")
        ckpt_path = _ckpt_path(seed, fold)
        model = _build_model(device, len(rna_gene_ids), age_mean, age_std, margin_stats, stage_stats, ckpt_path)

        cfg = Config()
        cfg.data.seed = seed
        ds = WSISurvivalDataset(cfg.data, dataset="tcga", split="all", feature_backbone="uni2",
                                 restrict_case_ids={case_id})
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate)
        patient_slides = next(iter(loader))

        n_slides = len(patient_slides)
        fig, axes = plt.subplots(n_slides, 2, figsize=(9, 4.5 * n_slides), squeeze=False)
        for si, slide in enumerate(patient_slides):
            attn = _slide_attention(model, device, slide)
            slide_dir = _ROOT / getattr(cfg.data, PATCHES_ROOT_ATTRS["tcga"]) / "tiles" / slide["slide_id"]
            mosaic, heat = _build_mosaic_and_heatmap(slide_dir, slide["coords"], attn)

            axes[si, 0].imshow(mosaic)
            axes[si, 0].set_title(f"{slide['slide_id'][:24]}... (조직)")
            axes[si, 0].axis("off")

            axes[si, 1].imshow(mosaic)
            axes[si, 1].imshow(heat, cmap="inferno", alpha=0.6, vmin=0, vmax=1,
                                extent=[0, mosaic.shape[1], mosaic.shape[0], 0])
            axes[si, 1].set_title("attention (percentile, 밝을수록 높음)")
            axes[si, 1].axis("off")

        fig.suptitle(f"{case_id} — {note}", fontsize=11)
        fig.tight_layout()
        out_path = OUT_DIR / f"{case_id}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  저장: {out_path}")


if __name__ == "__main__":
    main()
