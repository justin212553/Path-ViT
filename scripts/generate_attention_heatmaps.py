"""
M4/PMA의 attention이 실제로 뭘 보고 있는지 시각적으로 확인한다 — patch-level ABMIL
attention(MultiComponentPooling의 "attn" 성분)과, RNA-query co-attention이 4개 pooling
관점(mean/std/attn/top-k) 중 어디에 가중치를 주는지 둘 다 뽑아서 보여준다.

2026-08-31(2차): PAAD/uni2native로 만들었던 첫 버전은 HPC에도 원본 WSI 재타일링 JPG가 없어서
(features_uni2native.pt/coords_uni2native.pt만 있고 조직 이미지 자체가 없음 — 확인됨) 실행
불가였다. TCGA-BRCA(scripts/brca_common.py, HuggingFace Dearcat/CPathPatchFeature에서 받은
사전추출 UNI(v1) feature)로 대상을 바꾼다 — 단, BRCA도 원본 패치 이미지가 없다(coords.pt+
features_uni.pt뿐, 확인됨: data/patches_tcga_brca/tiles/<slide_id>/에 jpg 없음). 그래서 조직
이미지 위 오버레이가 아니라 좌표 산점도(scatter) 형태의 순수 attention heatmap이다 —
BRCA 슬라이드당 패치 수가 최대 67,268개라 이미지 모자이크(PAAD 버전 방식)는 메모리상으로도
비현실적이었을 것.

BRCA M4 실제 학습 레시피(scripts/train_brca_m4.py 기본값, sbatch/train_brca_m4_internal_hpc.sh)는
PAAD M4와 달리 combine_mode="concat"(cox_add 아님), clinical margin/staging 없음(age/sex만),
backbone="uni"(UNI v1) — 체크포인트와 정확히 같은 구조로 재구성해야 state_dict가 로드된다.

패치별 attention은 model.forward()가 반환하는 attn_weights(N,) — exclude="attn"이어도
항상 계산됨(models/multi_component_pooling.py). co-attention 가중치는
model.combine_with_clinical_rna() 내부에서 버려지는 값이라(z_wsi, _ = self.component_coattn(...)),
동일한 patient_embed/z_rna로 component_coattn을 한 번 더(부작용 없음, eval 모드라 dropout
없어 결정적) 직접 호출해서 뽑는다.

사용법:
    python -m scripts.generate_attention_heatmaps
    python -m scripts.generate_attention_heatmaps --case-ids TCGA-E2-A15M,TCGA-B6-A0RO
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from models.vit_pma import ViT_PMA
from models.clinical_encoder import age_stats_from_csv
from models.rna_predictor import RNAPredictionHead
from scripts.brca_common import CLINICAL_PATH, MANIFEST_PATH, load_case_table, load_rna_matrix, BRCASlideDataset

# seed84 internal(institution split 없음, 전체 1058명 6:2:2) test set(212명, paper/brca/
# kfold_preds/brca_BRCA_PMA_TOP1500_SS_AUX_seed84.csv)에서 risk 오분류가 가장 심한 3+3명.
PATIENTS = [
    # matplotlib 기본 폰트(DejaVu Sans, HPC도 동일)에 한글 glyph가 없어 title에 그대로 쓰면
    # 깨진다(2026-08-31 스모크테스트로 확인) — 영어로 표기.
    ("TCGA-E2-A15M", "died day336, risk underestimated — worst"),
    ("TCGA-D8-A1XC", "died day377, risk underestimated"),
    ("TCGA-E2-A14Z", "died day563, risk underestimated"),
    ("TCGA-B6-A0RO", "censored day4929, risk overestimated — worst"),
    ("TCGA-A2-A0CW", "censored day3283, risk overestimated (highest abs. risk)"),
    ("TCGA-E2-A1LH", "censored day3247, risk overestimated"),
]

DEFAULT_SEED = 84
N_GENES = 1500
CKPT_DIR = _ROOT / "models" / "checkpoint"
OUT_DIR = _ROOT / ".logs" / "attention_heatmaps"
GENE_PATH = _ROOT / "data" / "brca_rna_gene_selection" / f"selected_genes_top_{N_GENES}.csv"

# variant -> (ViT_PMA kwargs 오버라이드, 체크포인트 파일명에 반드시 포함될 substring).
# 지금은 BRCA에서 학습된 게 baseline(train_brca_m4_internal_hpc.sh, combine_mode 기본값
# "concat", 나머지 인자도 전부 train_brca_m4.py 기본값)뿐이다 — no_coattn/no_abmil/no_nystrom을
# BRCA로도 돌리면 여기에 항목만 추가하면 된다(sbatch/train_brca_m4_internal_hpc.sh에
# --no-coattn/--drop-component attn/--skip-patch-vit를 추가한 변형 스크립트 필요).
VARIANTS = {
    "baseline": (dict(), "brca_pma_top1500_ss_aux"),
}
_ALL_COMPONENT_NAMES = ("mean", "std", "attn(ABMIL)", "top-k")
_ALL_COMPONENT_KEYS = ("mean", "std", "attn", "top")  # models/multi_component_pooling.py 순서와 동일


def _component_names(drop_component: str | None) -> tuple[str, ...]:
    return tuple(n for k, n in zip(_ALL_COMPONENT_KEYS, _ALL_COMPONENT_NAMES) if k != drop_component)


def _find_ckpt(tag: str, seed: int) -> Path:
    candidates = list(CKPT_DIR.glob(f"survival_brca_best_{tag}_seed{seed}.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"seed={seed} tag={tag!r}: 체크포인트 {len(candidates)}개 매칭됨 "
            f"({[p.name for p in candidates]}) — sbatch/train_brca_m4_internal_hpc.sh를 먼저 "
            f"돌렸는지 확인할 것."
        )
    return candidates[0]


def _build_model(device, variant: str, seed: int, rna_input_dim, age_mean, age_std):
    kwargs, tag = VARIANTS[variant]
    ckpt_path = _find_ckpt(tag, seed)
    cfg = Config()
    # scripts/train_brca_m4.py 기본 레시피 그대로 — combine_mode="concat"(기본값),
    # use_margin/use_staging 없음(age/sex만), backbone="uni"(UNI v1, PAAD의 uni2native가 아님).
    model = ViT_PMA(cfg.model, age_mean=age_mean, age_std=age_std, rna_input_dim=rna_input_dim,
                     precomputed=True, backbone="uni", **kwargs).to(device)
    model.rna_aux_head = RNAPredictionHead(cfg.model.embed_dim, rna_input_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  [{variant}] {ckpt_path.name} 로드 완료 (epoch={ckpt.get('epoch')}, val_c_index={ckpt.get('val_c_index')})")
    return model


@torch.no_grad()
def _patient_attention_and_coattn(model, device, patient_slides, z_rna):
    """환자의 슬라이드별 (coords(N,2), attn_weights(N,)) 리스트와, train.py::_patient_risk와
    동일하게 슬라이드 평균 patient_embed로 계산한 4-component co-attention 가중치(4,)
    (model.use_coattn=False면 None) 하나를 반환한다."""
    per_slide = []
    components_list = []
    for slide in patient_slides:
        coords = slide["coords"].to(device)
        features = slide["features"].to(device)
        fwd = model(coords, features=features)
        per_slide.append((slide["coords"].numpy(), fwd["attn_weights"].cpu().numpy()))
        components_list.append(fwd["embed"])
    patient_embed = torch.stack(components_list).mean(dim=0)  # (4 또는 3, D) — train.py와 동일 관례
    coattn_weights = None
    if getattr(model, "use_coattn", True):
        _, cw = model.component_coattn(patient_embed, z_rna)
        coattn_weights = cw.cpu().numpy()
    return per_slide, coattn_weights


def _scatter_heatmap(ax, coords: np.ndarray, attn_weights: np.ndarray, title: str):
    """조직 이미지가 없어(BRCA는 사전추출 feature+좌표뿐, jpg 없음) 좌표 산점도로만 attention을
    표시한다 — 패치 수가 최대 6만개대라 이미지 모자이크(PAAD 버전)는 메모리상 비현실적."""
    order = np.argsort(np.argsort(attn_weights))
    pct = order / max(1, len(attn_weights) - 1)
    marker_size = max(1, min(12, 3000 / max(1, len(attn_weights)) * 20))
    sc = ax.scatter(coords[:, 1], -coords[:, 0], c=pct, cmap="inferno", s=marker_size,
                     vmin=0, vmax=1, rasterized=True)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", choices=list(VARIANTS), default=None,
                         help="반복 지정 가능. 생략하면 등록된 전부(지금은 baseline뿐).")
    parser.add_argument("--case-ids", type=str, default=None,
                         help="콤마로 구분한 case_id 목록. 생략하면 PATIENTS(기본 6명) 사용.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help=f"체크포인트/split seed(기본 {DEFAULT_SEED} — train_brca_m4_internal_hpc.sh와 동일).")
    args = parser.parse_args()

    variants = args.variant or list(VARIANTS)
    patients = [(cid, "") for cid in args.case_ids.split(",")] if args.case_ids else PATIENTS
    seed = args.seed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not GENE_PATH.exists():
        raise FileNotFoundError(f"{GENE_PATH} 없음 — scripts/select_brca_rna_genes.py 산출물이 있어야 함.")
    import pandas as pd
    gene_ids = pd.read_csv(GENE_PATH)["gene_id"].tolist()
    rna_df = load_rna_matrix(gene_ids)
    manifest = pd.read_csv(MANIFEST_PATH)
    age_mean, age_std = age_stats_from_csv(CLINICAL_PATH)
    cases = load_case_table(seed, external_tss=None)  # train_brca_m4_internal_hpc.sh와 동일(institution split 없음)

    print("=== 체크포인트 로드 ===")
    models = {v: _build_model(device, v, seed, len(gene_ids), age_mean, age_std) for v in variants}

    for case_id, note in patients:
        print(f"\n{case_id} ({note})")
        row = cases[cases["case_id"] == case_id]
        if len(row) == 0:
            print(f"  [SKIP] {case_id}: case 테이블에 없음")
            continue
        ds = BRCASlideDataset(row, rna_df, manifest)
        patient_slides = ds[0]
        rna = patient_slides[0]["rna"].to(device, non_blocking=True)

        n_slides = len(patient_slides)
        fig, axes = plt.subplots(n_slides, len(variants), figsize=(4.5 * len(variants), 4.5 * n_slides),
                                  squeeze=False)
        for vi, variant in enumerate(variants):
            model = models[variant]
            with torch.no_grad():
                z_rna = model.encode_rna(rna)
            per_slide, coattn = _patient_attention_and_coattn(model, device, patient_slides, z_rna)
            drop_component = VARIANTS[variant][0].get("drop_component")
            coattn_str = (
                " | ".join(f"{n}={w:.2f}" for n, w in zip(_component_names(drop_component), coattn))
                if coattn is not None else "N/A(--no-coattn)"
            )
            for si, (coords, attn) in enumerate(per_slide):
                # BRCASlideDataset이 반환하는 슬라이드 dict엔 slide_id가 없다(공용 case 필드만
                # 있음, scripts/brca_common.py 참조) — 슬라이드 순번으로만 구분.
                title = f"{variant} — slide {si + 1}/{n_slides}\nco-attn: {coattn_str}"
                _scatter_heatmap(axes[si, vi], coords, attn, title)

        fig.suptitle(f"{case_id} — {note}" if note else case_id, fontsize=11)
        fig.tight_layout()
        out_path = OUT_DIR / f"{case_id}_{'_'.join(variants)}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  저장: {out_path}")


if __name__ == "__main__":
    main()
