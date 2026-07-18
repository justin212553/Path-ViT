"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 데이터셋 — 환자(case) 단위 MIL.

환자당 다중 슬라이드를 리스트로 묶는 구조다. 슬라이드→환자 매핑은 파일명 파싱이 아니라
data/preprocess_cptac.py 산출물인 slide_index_task*.csv의 case_id 컬럼을 그대로 쓴다.

각 아이템 = 환자(case) 1명이 보유한 모든 슬라이드 리스트(dict).
DataLoader는 batch_size=1 + collate_fn=lambda batch: batch[0] 로 사용해야 한다.

반환 형식 (환자 1명의 슬라이드 수만큼의 리스트, 각 원소는 dict):
    patch_paths / features: precomputed 여부에 따라 둘 중 하나만 존재
    coords:      (N, 2) int64   [row, col]  (파일명 r####_c#### 파싱)
    case_id:     str
    slide_id:    str
    dataset:     "tcga" | "cptac"
    OS_time:     (1,) float32
    OS_event:    (1,) int64   (1=사망, 0=생존/censored)
    age_years / sex_idx: with_clinical=True일 때만 존재 (float32 스칼라 / long 스칼라, 0=male 1=female)
    rna:         with_rna=True일 때만 존재 ((G,) float32 — 코호트 내부 z-score 정규화된
                 유전자 발현. data/rna_{tcga,cptac}.csv(extract_rna_clinical.py 산출물, 전체
                 protein-coding 유전자) 중 Bailey 2016 + Moffitt 2015 PDAC subtype 분류
                 유전자만 골라 쓴다 — pdac_subtype_gene_ids() 참조, G ≈ 340)

data/extract_os_labels.py 산출물(data/os_labels_{tcga,cptac}.csv)에 없는 case(=raw clinical.tsv에
없거나 vital_status 미상이라 OS를 알 수 없는 환자)의 슬라이드는 라벨이 없으므로 제외한다.
with_clinical=True인 경우 data/clinical_{tcga,cptac}.csv(=RNA/clinical 모두 있는 case만 남긴
data/extract_rna_clinical.py 산출물)에 없는 case도 추가로 제외된다. with_rna=True인 경우
data/rna_{tcga,cptac}.csv에 없는 case도 마찬가지로 제외된다(실제로는 두 파일이 같은
extract_rna_clinical.py 실행에서 함께 나온 산출물이라 case 집합이 이미 동일하다).

train/val/test는 case 단위 6:2:2 stratified split이다 — (dataset, OS_event) 조합별로 case를
seed로 섞은 뒤 순서대로 잘라 배정하므로, 코호트 비율과 사망/생존 비율이 세 split에 고르게
유지된다. dataset="both"면 tcga+cptac 전체를 하나의 풀로 합친 뒤 이 방식으로 나눈다
(dataset="tcga"|"cptac" 하나만 주면 그 코호트 하나만 대상으로 같은 방식으로 나눈다).
학습에 쓰지 않은 반대 코호트 전체(split="all")를 추가 external test로 쓸 수도 있다
(train.py --external 참조, 기본은 미사용 옵션).

사용법 예:
    from config import DataConfig
    from data.dataset import WSISurvivalDataset
    train_ds = WSISurvivalDataset(DataConfig(), dataset="both", split="train")
    val_ds   = WSISurvivalDataset(DataConfig(), dataset="both", split="val")
    test_ds  = WSISurvivalDataset(DataConfig(), dataset="both", split="test")
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DataConfig
from data.patch_utils import (
    FEATURES_FILENAME, FEATURES_NORM_FILENAME, FEATURES_UNI_FILENAME,
    PATCH_TRANSFORM, list_patch_paths, _parse_coord,
)

FEATURES_FILENAME_BY_BACKBONE = {
    "resnet50":      FEATURES_FILENAME,
    "uni":           FEATURES_UNI_FILENAME,
    "resnet50_norm": FEATURES_NORM_FILENAME,  # Macenko stain-normalized (utils/extract_features_stain_norm.py)
}
from models.clinical_encoder import SEX_TO_IDX, STAGE_FIELDS, encode_stage_value

OS_LABEL_PATHS = {
    "tcga":  Path("data/os_labels_tcga.csv"),
    "cptac": Path("data/os_labels_cptac.csv"),
}
CLINICAL_PATHS = {
    "tcga":  Path("data/clinical_tcga.csv"),
    "cptac": Path("data/clinical_cptac.csv"),
}
RNA_PATHS = {
    "tcga":  Path("data/rna_tcga.csv"),
    "cptac": Path("data/rna_cptac.csv"),
}
COMMON_GENES_PATH         = Path("data/common_genes.csv")
BAILEY_SUBTYPE_GENES_PATH  = Path("data/bailey_subtype_genes.tsv")
MOFFITT_SUBTYPE_GENES_PATH = Path("data/moffitt_subtype_genes.tsv")


@lru_cache(maxsize=1)
def pdac_subtype_gene_ids() -> list[str]:
    """
    두 PDAC 분자 subtype 분류 체계의 유전자만 추려 RNA 벡터 차원을 줄인다 — ~2만 개
    protein-coding 유전자를 그대로 MLP에 넣으면 코호트당 case 수(~150)에 비해 과적합
    위험이 너무 크다.

      - Bailey et al. 2016(Nature): 4-subtype(Squamous/Progenitor/Immunogenic/ADEX) 분류
        유전자. data/bailey_subtype_genes.tsv(rmoffitt/pdacR의 Bailey_readable_list.tsv
        재배포) 중 subtype이 한 곳에만 유일하게 배정된 유전자("not unique" 제외).
      - Moffitt et al. 2015(Nat Genet): tumor-intrinsic(Basal-like/Classical, 25개씩) +
        stroma(Normal/Activated, 25개씩) 분류 유전자. data/moffitt_subtype_genes.tsv
        (rmoffitt/pdacR의 data/gene_lists.rds에서 추출) — Bailey 목록과 겹치는 유전자는
        2개(KRT6A, S100A2)뿐이라 대부분 상호보완적인 신호를 더한다.

    두 목록을 합쳐 data/common_genes.csv(gene_id, gene_name — extract_rna_clinical.py가
    만든 TCGA∩CPTAC protein-coding 교집합)로 ENSG id에 매핑한다. 비-protein_coding
    (면역글로불린 V/C 유전자 등)이거나 구식/별칭 심볼이라 common_genes.csv에 없는 유전자는
    매핑에서 자연히 빠진다.
    """
    bailey  = pd.read_csv(BAILEY_SUBTYPE_GENES_PATH, sep="\t")
    bailey  = bailey.loc[bailey["subtype"] != "not unique", "gene_symbol"]
    moffitt = pd.read_csv(MOFFITT_SUBTYPE_GENES_PATH, sep="\t")["gene_symbol"]
    symbols = pd.concat([bailey, moffitt]).unique()

    # gene_name은 PAR(pseudoautosomal) 유전자 등 극소수가 서로 다른 gene_id에 중복 배정돼
    # 있어(예: CD99, IL3RA) reindex 전에 첫 항목만 남긴다 — 위 두 목록 유전자에는 해당 없음.
    common_genes = pd.read_csv(COMMON_GENES_PATH).drop_duplicates(subset="gene_name", keep="first")
    name_to_id   = common_genes.set_index("gene_name")["gene_id"]
    gene_ids     = name_to_id.reindex(symbols).dropna().unique()
    return sorted(gene_ids.tolist())


@lru_cache(maxsize=None)
def literature_guided_gene_ids(top_n: int = 1500) -> list[str]:
    """
    pdac_subtype_gene_ids()의 대안 — data/select_rnaseq_genes.py 산출물을 로드한다.

    subtype 분류(Bailey/Moffitt)가 아니라 **생존 예측**에 직접 최적화된 기준으로 고른
    유전자셋이다: 문헌 큐레이션 PDAC 유전자(8개 카테고리, PDAC_LITERATURE_GENE_SETS)를
    train split(--dataset both 기준, val/test 라벨 미사용) 내부 TCGA/CPTAC 각각의
    univariate Cox score test 순위로 우선 배치하고, 남는 자리는 나머지 유전자의 Cox
    순위(Stouffer meta-analysis로 두 코호트 결합)로 채운다. 레퍼런스
    (Leeyoungsup/pancreatic_cancer_pathology) scripts/select_rnaseq_gene_features.py
    방법론을 그대로 재구현한 것 — 원 논문은 1000/1500/2000개를 ablation으로 비교한다.
    """
    path = Path(f"data/rna_gene_selection/selected_genes_top_{top_n}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — 먼저 실행: python -m data.select_rnaseq_genes --n-genes {top_n}"
        )
    return sorted(pd.read_csv(path)["gene_id"].tolist())


def pathway_category_gene_ids() -> dict[str, list[str]]:
    """
    literature_guided_gene_ids()의 대안 — 개별 유전자 1500개 대신, 문헌 큐레이션 PDAC 유전자
    8개 카테고리(PDAC_LITERATURE_GENE_SETS, data/select_rnaseq_genes.py)를 카테고리 -> ENSG id
    목록으로 반환한다. --rna-genes pathway8에서 사용 — WSISurvivalDataset이 카테고리별 유전자
    z-score의 평균(카테고리당 1개, 총 8차원)을 RNA 입력으로 구성한다(SurvPath의 pathway token
    방식과 같은 방향). literature_1500이 "순수 통계적 순위(Cox test)"로 차원을 줄였다면, 이건
    "생물학적 도메인 지식(레퍼런스가 미리 정의한 8개 범주)"으로 줄이는 대안 축이다.
    """
    path = Path("data/rna_gene_selection/literature_curated_genes.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 실행: python -m data.select_rnaseq_genes")
    df = pd.read_csv(path)
    df = df[df["available"]]
    return {cat: sorted(g["gene_id"].tolist()) for cat, g in df.groupby("category")}
PATCHES_ROOT_ATTRS = {
    "tcga":  "patches_root_tcga",
    "cptac": "patches_root_cptac",
}
DATASET_CHOICES = ("tcga", "cptac", "both")
SPLIT_CHOICES   = ("train", "val", "test", "all")
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2  # 나머지 0.2는 test


def _load_slide_index(patches_root: Path) -> pd.DataFrame:
    """data/preprocess_cptac.py가 --num-tasks 샤드별로 나눠 쓴 slide_index_task*.csv를 모두 합친다."""
    paths = sorted(patches_root.glob("slide_index_task*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"{patches_root}에 slide_index_task*.csv가 없습니다 — "
            "먼저 python -m data.preprocess.py 을 실행하세요."
        )
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _stratified_case_split(case_df: pd.DataFrame, seed: int) -> dict:
    """
    (dataset, OS_event) 조합별로 case를 6:2:2(train/val/test)로 나눈다.

    그룹 내에서 seed로 섞은 뒤 순서대로 잘라 배정하므로, 코호트 구성비와 사망/생존 비율이
    세 split 전체에 고르게 유지된다(dataset="both"일 때도 tcga/cptac 비율이 유지됨).

    Args:
        case_df: index=case_id, columns=["dataset", "OS_event"] (case당 1행)
        seed:    셔플 재현성
    Returns:
        {case_id: "train"|"val"|"test"}
    """
    rng = np.random.RandomState(seed)
    split_of_case = {}
    for _, group in case_df.groupby(["dataset", "OS_event"]):
        case_ids = group.index.to_numpy().copy()
        rng.shuffle(case_ids)
        n       = len(case_ids)
        n_train = min(round(n * TRAIN_FRAC), n)
        n_val   = min(round(n * VAL_FRAC), n - n_train)
        for i, case_id in enumerate(case_ids):
            if i < n_train:
                split_of_case[case_id] = "train"
            elif i < n_train + n_val:
                split_of_case[case_id] = "val"
            else:
                split_of_case[case_id] = "test"
    return split_of_case


class WSISurvivalDataset(Dataset):
    """
    Args:
        cfg:           DataConfig (patches_root_tcga/cptac, precomputed, seed 참조)
        dataset:       "tcga" | "cptac" | "both" ("both"면 두 코호트를 하나의 풀로 합친다)
        split:         "train" | "val" | "test" — case 단위 6:2:2 stratified split
                       ((dataset, OS_event) 조합 기준, _stratified_case_split 참조)
                       "all"이면 split을 나누지 않고 dataset의 case 전체를 반환한다 —
                       학습에 전혀 쓰이지 않은 별도 코호트를 통째로 external test로 평가할 때
                       쓴다(예: --dataset cptac으로 학습한 모델을 dataset="tcga", split="all"로
                       평가).
        transform:     패치에 적용할 transform (precomputed=False일 때만 사용)
        with_clinical: True면 data/clinical_{tcga,cptac}.csv(age_years, sex)를 case_id로
                       inner-join한다 — clinical 정보가 없는 case의 슬라이드는 제외되고,
                       각 아이템 dict에 age_years/sex_idx가 추가된다(models/vit_m2.py::ViT_M2,
                       train.py --M2 용).
        with_staging:  with_clinical=True와 함께만 쓸 수 있다. True면 같은 clinical CSV에서
                       AJCC 병기(T/N/M)+grade도 함께 join해, 각 아이템 dict에 STAGE_FIELDS
                       (ajcc_t/ajcc_n/ajcc_m/tumor_grade) 각각을 순서형 정수 텐서로 추가한다
                       (encode_stage_value() 규약 - "미상"은 -1). train.py --clinical-staging
                       (ClinicalEncoder 입력에 병기 추가)과 --stage-aux-weight
                       (models/stage_predictor.py, WSI 인코더 보조과제) 둘 다 이 플래그가 필요.
        with_rna:      True면 data/rna_{tcga,cptac}.csv(유전자 발현)를 case_id로 inner-join한다 —
                       RNA 정보가 없는 case의 슬라이드는 제외되고, 각 아이템 dict에 rna가
                       추가된다. 컬럼은 전체 protein-coding 유전자(~2만 개)가 아니라
                       pdac_subtype_gene_ids()로 추린 Bailey 2016 + Moffitt 2015 PDAC subtype
                       분류 유전자(~340개)만 쓴다 — case 수(코호트당 ~150) 대비 과적합을
                       줄이기 위함. 유전자 벡터는 case당 1번만 lookup에 저장하고(슬라이드
                       수만큼 중복 저장 방지) merged 테이블에는 case_id만 inner-join한다.
                       dataset="both"면 두 코호트의 유전자 컬럼 순서가 같아야 하며
                       (extract_rna_clinical.py가 보장), 다르면 에러.
                       (models/vit_m4.py::ViT_M4, train.py --M4 용)
        feature_backbone: precomputed=True일 때 어느 backbone의 캐싱된 feature 파일을 읽을지
                       선택한다 — "resnet50"(기본, features.pt) 또는 "uni"(features_uni.pt).
                       data/extract_features.py --backbone으로 미리 추출해둔 파일이 있어야
                       한다. 모델(ViT_M1 등) 생성 시 backbone 인자와 반드시 일치시켜야 한다.
        rna_gene_ids:  with_rna=True일 때 사용할 유전자 ENSG id 목록. None(기본)이면
                       pdac_subtype_gene_ids()(Bailey/Moffitt subtype 분류용, ~340개)를
                       쓴다. literature_guided_gene_ids(top_n)(data/select_rnaseq_genes.py
                       산출물, 생존 예측에 직접 최적화된 유전자셋)를 넘기면 그걸 대신 쓴다.

    아이템 단위 = 환자 1명. __getitem__은 그 환자가 가진 모든 슬라이드의 dict 리스트를 반환한다.
    """

    def __init__(
        self,
        cfg: DataConfig,
        dataset: str = "both",
        split: str = "train",
        transform=None,
        with_clinical: bool = False,
        with_staging: bool = False,
        with_rna: bool = False,
        feature_backbone: str = "resnet50",
        rna_gene_ids: list[str] | None = None,
        rna_pathway_categories: dict[str, list[str]] | None = None,
    ):
        if dataset not in DATASET_CHOICES:
            raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got {dataset!r}")
        if split not in SPLIT_CHOICES:
            raise ValueError(f"split must be one of {SPLIT_CHOICES}, got {split!r}")
        if feature_backbone not in FEATURES_FILENAME_BY_BACKBONE:
            raise ValueError(
                f"feature_backbone must be one of {list(FEATURES_FILENAME_BY_BACKBONE)}, "
                f"got {feature_backbone!r}"
            )
        if with_staging and not with_clinical:
            raise ValueError("with_staging=True는 with_clinical=True와 함께만 쓸 수 있습니다.")

        self.transform        = transform or PATCH_TRANSFORM
        self.precomputed      = cfg.precomputed
        self.with_clinical    = with_clinical
        self.with_staging     = with_staging
        self.with_rna         = with_rna
        self.features_filename = FEATURES_FILENAME_BY_BACKBONE[feature_backbone]
        self.rna_gene_ids     = rna_gene_ids
        self.rna_pathway_categories = rna_pathway_categories

        dataset_names = ["tcga", "cptac"] if dataset == "both" else [dataset]
        self.roots = {name: Path(getattr(cfg, PATCHES_ROOT_ATTRS[name])) for name in dataset_names}

        self.rna_gene_cols = None
        self.rna_lookup    = {}

        parts = []
        for name in dataset_names:
            root = self.roots[name]
            slide_df = _load_slide_index(root)
            slide_df = slide_df[(slide_df["status"] == "ok") & (slide_df["n_tiles_kept"] > 0)].copy()
            slide_df["dataset"] = name

            os_df  = pd.read_csv(OS_LABEL_PATHS[name])
            merged = slide_df.merge(os_df[["case_id", "OS_time", "OS_event"]], on="case_id", how="inner")

            if with_clinical:
                clinical_cols = ["case_id", "age_years", "sex"]
                if with_staging:
                    clinical_cols += list(STAGE_FIELDS)
                clinical_df = pd.read_csv(CLINICAL_PATHS[name])[clinical_cols]
                merged = merged.merge(clinical_df, on="case_id", how="inner")

            if with_rna:
                rna_df = pd.read_csv(RNA_PATHS[name])
                if self.rna_pathway_categories is not None:
                    # --rna-genes pathway8: 개별 유전자가 아니라 카테고리 평균 z-score를 쓴다 —
                    # target_ids는 8개 카테고리에 속한 전체 유전자의 합집합.
                    target_ids = set(g for genes in self.rna_pathway_categories.values() for g in genes)
                else:
                    target_ids = set(self.rna_gene_ids) if self.rna_gene_ids is not None else set(pdac_subtype_gene_ids())
                gene_cols    = [c for c in rna_df.columns if c in target_ids]
                if self.rna_gene_cols is None:
                    self.rna_gene_cols = gene_cols
                elif gene_cols != self.rna_gene_cols:
                    raise ValueError(
                        f"[{name}] {RNA_PATHS[name]}의 유전자 컬럼이 다른 코호트와 다릅니다 — "
                        "data.extract_rna_clinical을 다시 실행해 공통 유전자셋을 맞추세요."
                    )
                rna_matrix = rna_df[gene_cols].to_numpy(dtype="float32")  # (num_cases, G)

                if self.rna_pathway_categories is not None:
                    # (num_cases, G) -> (num_cases, 8) : 카테고리별 유전자 z-score 평균으로 집계
                    # (SurvPath의 pathway token 방식과 같은 방향 — 개별 유전자 대신 생물학적으로
                    # 함께 작동하는 유전자 그룹의 평균 신호를 입력으로 써서 표본 대비 차원을 줄인다).
                    col_index = {c: i for i, c in enumerate(gene_cols)}
                    cat_names = sorted(self.rna_pathway_categories.keys())
                    agg = np.zeros((rna_matrix.shape[0], len(cat_names)), dtype="float32")
                    for ci, cat in enumerate(cat_names):
                        idxs = [col_index[g] for g in self.rna_pathway_categories[cat] if g in col_index]
                        agg[:, ci] = rna_matrix[:, idxs].mean(axis=1)
                    rna_matrix = agg
                    self.rna_category_names = cat_names

                # 유전자(또는 카테고리) 벡터는 case당 1번만 lookup에 저장하고(슬라이드 수만큼
                # 중복 저장 방지), merged 테이블에는 필터링용 case_id만 inner-join한다.
                self.rna_lookup.update(zip(rna_df["case_id"], rna_matrix))
                merged = merged.merge(rna_df[["case_id"]], on="case_id", how="inner")

            def _has_patches(slide_id: str, root=root) -> bool:
                d = root / "tiles" / slide_id
                if self.precomputed:
                    return (d / self.features_filename).exists()
                return (next(d.glob("*.jpg"), None) or next(d.glob("*.png"), None)) is not None

            has_patches = merged["slide_id"].apply(_has_patches)
            parts.append(merged[has_patches].reset_index(drop=True))

        all_items = pd.concat(parts, ignore_index=True)
        if all_items.empty:
            joined = ["os_labels"]
            if with_clinical:
                joined.append("clinical")
            if with_rna:
                joined.append("rna")
            reason = "/".join(joined) + " 병합 결과"
            raise RuntimeError(
                f"[{dataset}] 사용 가능한 슬라이드가 없습니다 — preprocess 산출물과 {reason}를 확인하세요."
            )

        if split == "all":
            # external test용 — 코호트 전체를 split 없이 그대로 쓴다.
            self.items = all_items.reset_index(drop=True)
        else:
            # case 단위 6:2:2 stratified split — (dataset, OS_event) 조합별로 seed 고정 셔플 후 배정
            case_df = all_items.groupby("case_id").agg(dataset=("dataset", "first"), OS_event=("OS_event", "first"))
            split_of_case = _stratified_case_split(case_df, seed=cfg.seed)
            all_items["_split"] = all_items["case_id"].map(split_of_case)
            self.items = all_items[all_items["_split"] == split].reset_index(drop=True)

        if self.items.empty:
            raise RuntimeError(
                f"[{dataset}/{split}] 해당 split에 남은 case가 없습니다 — 코호트 규모가 너무 "
                f"작아 6:2:2 split이 비어버렸을 수 있습니다."
            )

        self.cases = sorted(self.items["case_id"].unique())

    def __len__(self) -> int:
        return len(self.cases)

    def _load_slide(self, row) -> dict:
        slide_dir   = self.roots[row["dataset"]] / "tiles" / row["slide_id"]
        patch_paths = list_patch_paths(slide_dir)

        coords = torch.tensor(
            [_parse_coord(p.name) for p in patch_paths],
            dtype=torch.long,
        )
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()

        item = {
            "coords":   coords,
            "case_id":  row["case_id"],
            "slide_id": row["slide_id"],
            "dataset":  row["dataset"],
            "OS_time":  torch.tensor([row["OS_time"]], dtype=torch.float32),
            "OS_event": torch.tensor([row["OS_event"]], dtype=torch.long),
        }

        if self.with_clinical:
            item["age_years"] = torch.tensor(row["age_years"], dtype=torch.float32)
            item["sex_idx"]   = torch.tensor(SEX_TO_IDX[row["sex"]], dtype=torch.long)
            if self.with_staging:
                for field in STAGE_FIELDS:
                    ord_val = encode_stage_value(field, row[field])
                    item[field] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)

        if self.with_rna:
            item["rna"] = torch.from_numpy(self.rna_lookup[row["case_id"]])

        if self.precomputed:
            features = torch.load(slide_dir / self.features_filename, weights_only=True)
            if len(features) != len(patch_paths):
                raise RuntimeError(
                    f"{slide_dir}: {self.features_filename} 행 수({len(features)})가 패치 수"
                    f"({len(patch_paths)})와 다릅니다 — utils.extract_features를 다시 실행하세요."
                )
            item["features"] = features
        else:
            item["patch_paths"] = patch_paths

        return item

    def __getitem__(self, idx: int) -> list:
        case_id   = self.cases[idx]
        case_rows = self.items[self.items["case_id"] == case_id]
        return [self._load_slide(row) for _, row in case_rows.iterrows()]
