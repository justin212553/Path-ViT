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

train/val/test는 case 단위 6:2:2 stratified split이다 — (dataset, OS_event) 조합별로(use_stage_
stratify=True면 stage도 함께) case를 seed로 섞은 뒤 순서대로 잘라 배정하므로, 코호트 비율과
사망/생존 비율(+옵션 시 병기 구성)이 세 split에 고르게
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
    FEATURES_FILENAME, FEATURES_NORM_FILENAME, FEATURES_UNI_FILENAME, FEATURES_UNI2_FILENAME,
    FEATURES_UNI2OFFICIAL_FILENAME, COORDS_UNI2OFFICIAL_FILENAME,
    FEATURES_UNI2NATIVE_FILENAME, COORDS_UNI2NATIVE_FILENAME,
    PATCH_TRANSFORM, list_patch_paths, _parse_coord,
)

FEATURES_FILENAME_BY_BACKBONE = {
    "resnet50":      FEATURES_FILENAME,
    "uni":           FEATURES_UNI_FILENAME,
    "uni2":          FEATURES_UNI2_FILENAME,  # UNI2-h(ViT-H/14, models/uni2_encoder.py), utils/extract_features.py --backbone uni2
    "resnet50_norm": FEATURES_NORM_FILENAME,  # Macenko stain-normalized (utils/extract_features_stain_norm.py)
    # 2026-08-12: MahmoodLab 공식 UNI2-h feature(256px@20x) — patch grid가 우리 자체 추출본과
    # 전혀 달라 coords도 별도 파일(COORDS_UNI2OFFICIAL_FILENAME)에서 읽는다(_load_slide 참조,
    # list_patch_paths/파일명-파싱 coords 경로를 타지 않음).
    "uni2official":  FEATURES_UNI2OFFICIAL_FILENAME,
    # 2026-08-12: 우리 raw WSI를 우리 파이프라인으로 256px@0.5MPP 재타일링한 UNI2-h feature
    # (scripts/reconcile_uni2native_features.py) — uni2official과 같은 이유로 coords도 별도 파일.
    "uni2native":    FEATURES_UNI2NATIVE_FILENAME,
}

# feature 파일과 짝을 이루는 별도 coords 파일에서 읽어야 하는 backbone들(patch grid가 우리 자체
# JPG 추출본과 달라 list_patch_paths/파일명-파싱 coords를 쓸 수 없음) — _load_slide 참조.
_PAIRED_COORDS_FILENAME_BY_BACKBONE = {
    "uni2official": COORDS_UNI2OFFICIAL_FILENAME,
    "uni2native":   COORDS_UNI2NATIVE_FILENAME,
}
from models.clinical_encoder import (
    SEX_TO_IDX, STAGE_FIELDS, encode_stage_value, encode_margin_value,
    MUTATION_FIELDS, encode_mutation_value,
)

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
# data/extract_cnv.py 산출물 — pathway8(163유전자) 범위 카피수 변이, raw 정수(정상=2). RNA와
# 달리 미리 z-score된 버전을 따로 저장해두지 않아서(2026-09-03 신규 추가) 여기서 즉석으로
# log2-ratio + z-score를 계산한다(cnv_pathway_category_features 참조).
CNV_RAW_PATHS = {
    "tcga":  Path("data/cnv_tcga.csv"),
    "cptac": Path("data/cnv_cptac.csv"),
}
# data/compute_purist_subtype.py 산출물 — PurIST(Rashid et al. 2020) basal-like 확률 1차원.
# 어떤 코호트에도 fit하지 않는 고정 계수 분류기라 --rna-genes purist에서만 사용.
RNA_PURIST_PATHS = {
    "tcga":  Path("data/rna_purist_tcga.csv"),
    "cptac": Path("data/rna_purist_cptac.csv"),
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


@lru_cache(maxsize=None)
def literature_guided_gene_ids_single_cohort(cohort: str, top_n: int = 1500) -> list[str]:
    """
    literature_guided_gene_ids()의 external 프로토콜 전용 버전 — data/select_rnaseq_genes.py
    --single-cohort {cohort} 산출물(data/rna_gene_selection_{cohort}only/)만 무조건 로드한다.
    both-결합 산출물(data/rna_gene_selection/)로는 절대 폴백하지 않는다.

    [왜 필요한가] literature_guided_gene_ids()는 TCGA+CPTAC 두 코호트의 train split을 Stouffer로
    결합해 유전자를 뽑는데, 이 결합 과정에 쓰인 각 코호트의 train case가 그 코호트를 학습에 전혀
    안 쓴 반대쪽 external 프로토콜(--dataset {cohort} --external)의 external test case와 겹친다
    (실측 약 60%, findings_backlog.md). "{cohort}로 학습 -> 반대 코호트 전체를 external test"인
    실행에서는 이 함수로 뽑은, {cohort} train split만 사용한(반대 코호트 데이터 자체를 로드하지
    않는) 유전자셋을 써야 external test의 완전 미노출 전제가 실제로 성립한다.
    """
    path = Path(f"data/rna_gene_selection_{cohort}only/selected_genes_top_{top_n}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — 먼저 실행: "
            f"python -m data.select_rnaseq_genes --single-cohort {cohort} --n-genes {top_n}"
        )
    return sorted(pd.read_csv(path)["gene_id"].tolist())


@lru_cache(maxsize=None)
def literature_guided_gene_ids_fdr_threshold(cohort: str, q_threshold: float = 0.1) -> list[str]:
    """
    literature_guided_gene_ids_single_cohort()의 threshold 기반 버전 — 임의의 고정 개수(top-N)
    대신 data/select_rnaseq_genes.py --fdr-threshold 산출물(문헌 curated 163개 + BH-FDR q <
    q_threshold를 만족하는 유전자)을 로드한다. 최종 유전자 수는 고정되지 않고 그 코호트의 실제
    신호 강도로 결정된다. single-cohort 전용(반대 코호트 미참조)이라 leakage 없음.
    """
    path = Path(f"data/rna_gene_selection_{cohort}only/selected_genes_fdr{q_threshold:g}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — 먼저 실행: "
            f"python -m data.select_rnaseq_genes --single-cohort {cohort} --fdr-threshold {q_threshold:g}"
        )
    return sorted(pd.read_csv(path)["gene_id"].tolist())


@lru_cache(maxsize=None)
def literature_guided_gene_ids_intersection(top_n: int = 1500) -> list[str]:
    """
    2026-08-05: data/select_rnaseq_genes.py --intersection 산출물 로더 — TCGA-only 순위와
    CPTAC-only 순위(각자 자기 코호트 라벨만으로 독립 계산)가 겹치는 유전자만 쓴다. 방향
    상관없이(TCGA->CPTAC, CPTAC->TCGA 둘 다) leakage 없이 쓸 수 있다 — 두 순위 모두 반대
    코호트 라벨을 전혀 참조하지 않았기 때문이다(literature_guided_gene_ids_single_cohort와
    달리 --dataset tcga/cptac 어느 쪽으로 학습해도 동일 리스트 사용 가능).
    """
    path = Path(f"data/rna_gene_selection_intersection/selected_genes_top_{top_n}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — 먼저 실행: python -m data.select_rnaseq_genes --intersection --n-genes {top_n}"
        )
    return sorted(pd.read_csv(path)["gene_id"].tolist())


@lru_cache(maxsize=None)
def variance_gene_ids_single_cohort(cohort: str, top_n: int) -> list[str]:
    """data/select_rnaseq_genes_variance.py --single-cohort {cohort} 산출물 로더 — 생존 라벨을
    전혀 안 보는 고분산(variance) 기준 single-cohort 유전자 패널(반대 코호트 미참조라
    external 프로토콜에 leak-free). 2026-09-03: BRCA에서 확인된 것 — Cox 기반 선택(라벨
    직접 사용)은 fold 경계를 넘는 구조적 leak에 취약했지만 variance 기반 선택은 라벨을 아예
    안 봐서 훨씬 덜 취약했다(findings_backlog.md) — 그 결과를 PAAD에도 적용해보는 실험용."""
    path = Path(f"data/rna_gene_selection_variance_{cohort}only/selected_genes_top_{top_n}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — 먼저 실행: "
            f"python -m data.select_rnaseq_genes_variance --single-cohort {cohort} --n-genes {top_n}"
        )
    return sorted(pd.read_csv(path)["gene_id"].tolist())


def resolve_tcga_only_rna_genes(rna_genes_arg: str) -> list[str]:
    """train.py/train_light.py --rna-genes "{prefix}_{spec}_{cohort}_only" 문자열을 파싱해
    single-cohort 로더 중 맞는 쪽으로 dispatch한다.

    prefix가 "literature"면(기존 동작 그대로): spec이 정수면 top-N
    (literature_guided_gene_ids_single_cohort), "fdr{q}"면 FDR threshold
    (literature_guided_gene_ids_fdr_threshold). prefix가 "variance"면(2026-09-03 추가)
    spec은 항상 정수 top-N(variance_gene_ids_single_cohort) — FDR 개념이 없다(라벨을 안 써서
    p-value 자체가 없음). 이 파싱을 train.py/train_light.py 양쪽에 각각 두면 하나만 고치고
    다른 쪽을 놓치는 사고가 나기 쉬워 여기 한 곳에만 둔다.

    2026-08-04: cohort를 "tcga"로 하드코딩했던 버그를 고쳤다 — "literature_fdr0.1_cptac_only"처럼
    반대 코호트를 단일 코호트로 쓰는 문자열도 받아야 하므로, 문자열 끝에서 두 번째 토큰
    ("_only" 바로 앞)을 실제 cohort로 파싱한다.
    """
    parts = rna_genes_arg.split("_")
    prefix, spec, cohort = parts[0], parts[1], parts[-2]
    if prefix == "variance":
        return variance_gene_ids_single_cohort(cohort, int(spec))
    if spec.startswith("fdr"):
        return literature_guided_gene_ids_fdr_threshold(cohort, float(spec[3:]))
    return literature_guided_gene_ids_single_cohort(cohort, int(spec))


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


def pathway_flat_gene_ids() -> list[str]:
    """pathway_category_gene_ids()와 정확히 같은 163개 문헌 큐레이션 유전자를, 카테고리 평균
    (8차원)이 아니라 개별 유전자 z-score 그대로(163차원) 반환한다 — --rna-genes pathway8_flat.

    2026-09-03: "카테고리로 뭉뚱그리는 게 정말 필요한가, 아니면 그냥 개별 유전자를 그대로 써도
    되는가"를 직접 검증하기 위해 추가(사용자 지시). pathway8의 평균 방식은 표본 대비 차원을
    줄이려는 설계 선택(SurvPath의 pathway token 방식 참조)이었지만, 실제로 과적합을 줄이는
    효과가 있는지는 이 비교 전까지 실측된 적이 없었다 — 라벨을 전혀 안 본다는 leak-free 성질은
    이 함수도 pathway_category_gene_ids()와 동일하게 유지한다(평균을 내느냐 마느냐의 차이일 뿐).
    """
    path = Path("data/rna_gene_selection/literature_curated_genes.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 실행: python -m data.select_rnaseq_genes")
    df = pd.read_csv(path)
    return sorted(df.loc[df["available"], "gene_id"].tolist())


@lru_cache(maxsize=None)
def cnv_pathway_category_features(name: str) -> pd.DataFrame:
    """data/extract_cnv.py 산출물(raw 정수 copy_number, pathway8 163유전자 범위)을 RNA
    pathway8과 정확히 같은 8개 카테고리(pathway_category_gene_ids() 재사용 — CNV 추출을
    애초에 이 유전자 범위로 한정했으므로 카테고리/유전자가 100% 동일)로 평균 낸 (case_id x
    8) DataFrame을 반환한다. 컬럼명은 RNA와 겹치지 않게 "cnv_" 접두어를 붙인다.

    [정규화] raw copy_number(정수, 정상=2)는 그대로 평균 내면 안 된다 — 표준 CNV 분석 관례대로
    log2(copy_number/2 + eps) 로그비(log-ratio)로 바꾼 뒤(0=정상, 음수=결실, 양수=증폭),
    RNA와 동일한 관례(data/extract_rna_clinical.py::main())로 그 코호트 전체 기준
    z-score한다. 2026-09-03 신규 — WSI+유전체 융합이 유의했던 원 PORPOISE 논문 결과를
    재현하기 위해 이 프로젝트가 한 번도 안 써본 CNV 모달리티를 처음 추가.
    """
    raw = pd.read_csv(CNV_RAW_PATHS[name]).set_index("case_id")
    log_ratio = np.log2(raw / 2.0 + 1e-3)
    z = (log_ratio - log_ratio.mean()) / log_ratio.std(ddof=0).replace(0, 1.0)

    categories = pathway_category_gene_ids()
    cat_names = sorted(categories.keys())
    out = pd.DataFrame(index=z.index)
    for cat in cat_names:
        cols = [g for g in categories[cat] if g in z.columns]
        out[f"cnv_{cat}"] = z[cols].mean(axis=1)
    return out
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


def _tcga_barcode_field(slide_id: str, idx: int) -> str:
    """TCGA slide_id(예: "TCGA-2J-AAB1-01Z-00-DX1.<UUID>")의 바코드 필드 앞 2글자.
    idx=3 -> sample type("01"=Primary Tumor, "11"=Solid Tissue Normal 등),
    idx=5 -> portion/analyte("DX"=diagnostic/영구절편, "TS"/"BS"=냉동절편 등)."""
    parts = slide_id.split(".")[0].split("-")
    return parts[idx][:2] if len(parts) > idx else ""


CPTAC_GDC_SLIDE_TYPES_PATH = Path("data/cptac_gdc_slide_sample_type.csv")


def _load_cptac_gdc_slide_types() -> dict:
    """GDC(Genomic Data Commons) API의 CPTAC-3 프로젝트 biospecimen 계층(cases -> samples ->
    portions -> slides)에서 직접 조회한 slide_id -> sample_type("Primary Tumor"/"Solid Tissue
    Normal") 매핑(2026-07-21, api.gdc.cancer.gov/cases?expand=samples.portions.slides로 CPTAC-3/
    Pancreas 170개 case 전체 조회, data/cptac_gdc_slide_sample_type.csv에 저장). 우리가 다운로드한
    CPTAC svs 567장 중 295장(52%)이 이 GDC biospecimen 기록과 매칭되고, 그중 80장(27%)이
    "Solid Tissue Normal"(정상 조직) — tumor/normal 구분 없이 case당 슬라이드를 전부 써온 기존
    방식이 정상 조직 슬라이드까지 섞어 썼다는 걸 실측으로 확인한 근거(findings_backlog.md 14번
    항목). 나머지 272장은 GDC biospecimen에 slide 단위로 등록되지 않은 슬라이드(TCIA CPTAC
    Pathology Portal에는 있지만 GDC에는 formal biospecimen entity로 안 올라간 경우로 추정) —
    tumor 여부를 알 수 없는 미상으로 남긴다."""
    if not CPTAC_GDC_SLIDE_TYPES_PATH.exists():
        return {}
    df = pd.read_csv(CPTAC_GDC_SLIDE_TYPES_PATH)
    return dict(zip(df["slide_id"], df["sample_type"]))


def _slide_tumor_status(slide_id: str, dataset: str, cptac_slide_types: dict) -> int:
    """2026-08-13: --tumor-type-embed용 — 슬라이드의 종양/정상 여부를 0(tumor)/1(normal)/
    2(unknown)으로 인코딩한다. _exclude_normal_slides()가 이미 같은 출처(TCGA 바코드 sample
    type, CPTAC GDC biospecimen)로 정상 조직을 걸러내는 데 쓰던 것과 동일한 정보를, 여기서는
    필터링이 아니라 모델 입력 태그로 재사용한다.

    TCGA: sample type 코드(idx=3)가 "0"으로 시작하면 종양 계열(01=Primary Tumor 등),
    "1"로 시작하면 정상/대조 계열(11=Solid Tissue Normal 등) — TCGA 코드 체계 관례.
    CPTAC: GDC biospecimen 매칭 결과(52%만 커버, 나머지는 unknown)를 그대로 쓴다.
    """
    if dataset == "tcga":
        code = _tcga_barcode_field(slide_id, 3)
        if code.startswith("0"):
            return 0
        if code.startswith("1"):
            return 1
        return 2
    sample_type = cptac_slide_types.get(slide_id)
    if sample_type == "Primary Tumor":
        return 0
    if sample_type == "Solid Tissue Normal":
        return 1
    return 2


def _load_case_stage(dataset_name: str) -> dict:
    """2026-08-14: case_id -> ajcc_stage 문자열 매핑, split stratification 전용(모델 입력과
    무관 — with_clinical=False인 모델도 이 정보로 fold를 나눌 수 있다). data/clinical_{name}.csv에
    ajcc_stage 컬럼이 없거나 파일 자체가 없으면 빈 dict를 반환하고, 호출부에서 "unknown"으로
    처리한다.

    도입 배경: fold별 internal log-rank p가 fold마다 크게 요동쳤는데(2026-08-14 조사), event
    비율/표본 크기는 fold 간 거의 동일한 반면 stage 구성만 뚜렷이 달랐다(나쁜 fold는 Stage IIB가
    77%까지 쏠림, 좋은 fold는 58~62%로 더 다양) — 단일 병기에 쏠린 fold는 애초에 그 안에 위험도
    스펙트럼이 좁아 log-rank 검정력 자체가 약해진다. (dataset, OS_event) 기준 stratification에
    stage를 추가해 이 쏠림을 fold 간에 고르게 분산시킨다.
    """
    path = CLINICAL_PATHS.get(dataset_name)
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if "ajcc_stage" not in df.columns:
        return {}
    return dict(zip(df["case_id"], df["ajcc_stage"].fillna("unknown")))


def _select_representative_slide(all_items: pd.DataFrame) -> pd.DataFrame:
    """케이스당 슬라이드를 1장으로 줄인다(findings_backlog.md 14번 항목 — 레퍼런스(Leeyoungsup/
    pancreatic_cancer_pathology)는 TCGA는 diagnostic(DX) WSI 1개/환자, CPTAC는 SeriesDescription에
    "tumor"가 포함된 series 중 용량이 가장 큰 것 1개/case만 쓰는데, 우리는 지금까지 case당 존재하는
    슬라이드를 전부 써왔다(TCGA 평균 2.52장/case, CPTAC 평균 3.22장/case) — 그 격차를 좁히는 실험용
    옵션).

    TCGA: 바코드에서 sample type(idx=3)이 "01"(Primary Tumor)인 슬라이드만 후보로 삼는다(01이
    하나도 없는 소수 케이스는 통째로 잃지 않기 위해 전체 슬라이드로 폴백). 그중 portion(idx=5)이
    "DX"(진단용/영구절편)인 슬라이드를 TS/BS(냉동절편)보다 우선한다. 동률/유일 후보 안에서는
    n_tiles_kept(조직량)가 가장 큰 슬라이드를 고른다.
    CPTAC: SeriesDescription 같은 tumor/normal 태그는 우리가 가진 데이터(TCIA Aperio SVS)에
    직접 없지만(레퍼런스가 쓴 IDC DICOM 버전에만 있는 필드), GDC biospecimen API에서 같은 정보를
    별도로 확보했다(_load_cptac_gdc_slide_types() 참조) — "Solid Tissue Normal"로 확인된 슬라이드는
    후보에서 제외하고, "Primary Tumor"로 확인된 슬라이드가 있으면 그중에서만 고른다. GDC에 없는
    (tumor 여부 미상인) 슬라이드는 "정상이 아님"으로 간주해 후보에 남긴다. 최종적으로 남은 후보
    중 n_tiles_kept(조직량)가 가장 큰 슬라이드를 대표로 쓴다.
    """
    cptac_slide_types = _load_cptac_gdc_slide_types()
    rows = []
    for _, group in all_items.groupby("case_id"):
        if group["dataset"].iloc[0] == "tcga":
            sample_type = group["slide_id"].map(lambda s: _tcga_barcode_field(s, 3))
            pool = group[sample_type == "01"]
            if pool.empty:
                pool = group
            portion = pool["slide_id"].map(lambda s: _tcga_barcode_field(s, 5))
            dx_pool = pool[portion == "DX"]
            if not dx_pool.empty:
                pool = dx_pool
        else:
            slide_type = group["slide_id"].map(cptac_slide_types.get)
            not_normal = group[slide_type != "Solid Tissue Normal"]
            pool = not_normal if not not_normal.empty else group
            pool_slide_type = pool["slide_id"].map(cptac_slide_types.get)
            tumor_pool = pool[pool_slide_type == "Primary Tumor"]
            if not tumor_pool.empty:
                pool = tumor_pool
        rows.append(pool.loc[pool["n_tiles_kept"].idxmax()])
    return pd.DataFrame(rows).reset_index(drop=True)


def _exclude_normal_slides(all_items: pd.DataFrame) -> pd.DataFrame:
    """확인된 정상 조직 슬라이드만 제외하고, 케이스당 나머지 슬라이드는 전부 그대로 둔다
    (findings_backlog.md 14번 항목 절충안 — _select_representative_slide()의 "대표 1장으로 축소"
    보다 훨씬 덜 급진적인 개입: TCGA 평균 슬라이드/case 2.52→2.28, CPTAC 3.22→2.76, 두 코호트 다
    슬라이드가 0장이 되는 케이스는 없음).

    TCGA: 바코드 sample type(idx=3)이 "01"(Primary Tumor)이 아닌 슬라이드(예: "11"=Solid Tissue
    Normal) 제외.
    CPTAC: GDC biospecimen에서 "Solid Tissue Normal"로 확인된 슬라이드 제외(_load_cptac_gdc_
    slide_types() 참조) — GDC에 없는(tumor 여부 미상인) 슬라이드는 유지한다.
    """
    cptac_slide_types = _load_cptac_gdc_slide_types()
    is_tcga = (all_items["dataset"] == "tcga").to_numpy()

    tcga_keep = all_items["slide_id"].map(lambda s: _tcga_barcode_field(s, 3) == "01").to_numpy()
    cptac_keep = all_items["slide_id"].map(lambda s: cptac_slide_types.get(s) != "Solid Tissue Normal").to_numpy()
    keep = np.where(is_tcga, tcga_keep, cptac_keep)
    return all_items[keep].reset_index(drop=True)


def _dx_only_slides(all_items: pd.DataFrame) -> pd.DataFrame:
    """TCGA에서 portion(idx=5)이 "DX"(진단용/영구절편)가 아닌 슬라이드(TS/BS 등 냉동절편)를
    제외한다. 케이스당 남은 DX 슬라이드는 전부 그대로 둔다(_select_representative_slide()처럼
    대표 1장으로 줄이지 않음) — _exclude_normal_slides()와 같은 절충안 패턴을, "정상 조직
    제외" 대신 "냉동절편 제외"에 적용한 버전.

    도입 배경: MahmoodLab UNI2-h-features(uni2official, 2026-08-14 조사)가 DX 슬라이드만
    포함해(TS/BS 0% 커버) 환자당 평균 슬라이드 수가 확 줄었던 것과, risk_head에 들어가는
    attn_dispersion 좌표 스케일 버그가 동시에 섞여 있어 "DX만 쓰면 성능이 오르는지"를 그
    자체로 분리해서 볼 수 없었다 — 이번엔 좌표 스케일 버그 없이(자체 추출 좌표 그대로) DX만
    남기는 효과 하나만 검증한다.

    DX 슬라이드가 하나도 없는 케이스는 통째로 잃지 않기 위해 전체 슬라이드로 폴백한다
    (_select_representative_slide()와 동일한 관례).
    CPTAC: DX/TS 구분에 해당하는 필드가 우리가 가진 데이터에 없어(TCGA 전용 바코드 관례),
    필터링 없이 그대로 둔다 — external(CPTAC) 평가는 이 옵션과 무관하게 항상 전체 코호트다.
    """
    rows = []
    for _, group in all_items.groupby("case_id"):
        if group["dataset"].iloc[0] == "tcga":
            portion = group["slide_id"].map(lambda s: _tcga_barcode_field(s, 5))
            dx_pool = group[portion == "DX"]
            rows.append(dx_pool if not dx_pool.empty else group)
        else:
            rows.append(group)
    return pd.concat(rows, ignore_index=True)


# 2026-08-14: (dataset, OS_event) 기준 stratification에 stage(_load_case_stage())를 선택적으로
#추가할 수 있게 한다(--stage-stratify, 기본 False로 기존 동작 그대로 유지). fold별 internal
# log-rank p가 요동친 원인 조사에서, event 비율/표본 크기는 fold 간 거의 동일한데 stage 구성만
# 뚜렷이 달랐다(나쁜 fold는 Stage IIB가 77%까지 쏠림) — 단일 병기에 쏠린 fold는 그 안의 위험도
# 스펙트럼이 좁아 log-rank 검정력 자체가 약해진다. 다만 이 stratify key를 바꾸면 같은 seed라도
# fold 배정(어떤 환자가 어느 fold에 들어가는지) 자체가 달라져 기존 결과와 split이 안 맞게
# 되므로, 기본은 꺼둔 채 옵션으로만 제공한다 — 켜서 새로 실험할 땐 baseline도 같은 옵션으로
# 다시 돌려야 공정한 비교가 된다.
#
# 2026-08-14(2차): stage-stratify로도 fold별 log-rank p 변동이 완전히 안 풀려서(fold-평균 std
# 0.181->0.117, 부분 개선에 그침) 추가 요인을 조사한 결과, baseline에서 "fold 안에 고레버리지
# (leave-one-out c-index 델타 최하위 20명, audit_leverage_patients.py류 계산) 환자가 몇 명
# 몰려있는가"가 log-rank p와 rho=0.89로 가장 강하게 상관됐다(연령 rho=0.90과 함께 최상위) —
# 우연히 "예측하기 어려운" 환자들이 한 fold에 몰리면 그 fold의 검정력 자체가 흔들린다는 가설.
# _HIGH_LEVERAGE_CASE_IDS는 baseline(PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD) 3seed pooled
# OOF에서 한 번 계산해 고정한 20명이다 — 특정 모델의 특정 학습 결과에서 역산한 값이라 순환
# 논리적 성격이 있고(이 실험 자체가 "이 20명을 고르게 펴면 그 모델의 log-rank p 변동이
# 줄어드는가"를 보는 진단용 실험이지, 범용적으로 검증된 "어려운 환자 목록"이 아님) 일반화를
# 주장하지 않는다 — --leverage-stratify로만 옵트인.
_HIGH_LEVERAGE_CASE_IDS = frozenset({
    "TCGA-2J-AAB1", "TCGA-FB-A4P5", "TCGA-FB-A545", "TCGA-FB-A78T", "TCGA-FB-AAPQ",
    "TCGA-H6-8124", "TCGA-H6-A45N", "TCGA-HV-A5A3", "TCGA-HZ-8002", "TCGA-HZ-A49I",
    "TCGA-IB-7885", "TCGA-IB-7897", "TCGA-IB-A5SO", "TCGA-IB-A5SQ", "TCGA-IB-A6UF",
    "TCGA-IB-A6UG", "TCGA-IB-A7M4", "TCGA-OE-A75W", "TCGA-US-A77G", "TCGA-XD-AAUI",
})


def _stratify_keys(use_stage_stratify: bool, use_leverage_stratify: bool = False) -> list[str]:
    keys = ["dataset", "OS_event"]
    if use_stage_stratify:
        keys.append("stage")
    if use_leverage_stratify:
        keys.append("high_leverage")
    return keys


def _stratified_case_split(case_df: pd.DataFrame, seed: int, use_stage_stratify: bool = False,
                            use_leverage_stratify: bool = False) -> dict:
    """
    (dataset, OS_event[, stage][, high_leverage]) 조합별로 case를 6:2:2(train/val/test)로 나눈다.

    그룹 내에서 seed로 섞은 뒤 순서대로 잘라 배정하므로, 코호트 구성비·사망/생존 비율(및 옵션
    시 병기 구성/고레버리지 환자 분포)이 세 split 전체에 고르게 유지된다(dataset="both"일
    때도 tcga/cptac 비율이 유지됨).

    Args:
        case_df: index=case_id, columns=["dataset", "OS_event", "stage", "high_leverage"]
                 (case당 1행, 옵션이 꺼져 있으면 해당 컬럼 값이 있어도 무시됨)
        seed:    셔플 재현성
    Returns:
        {case_id: "train"|"val"|"test"}
    """
    rng = np.random.RandomState(seed)
    split_of_case = {}
    for _, group in case_df.groupby(_stratify_keys(use_stage_stratify, use_leverage_stratify)):
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


def _stratified_kfold_assignment(case_df: pd.DataFrame, seed: int, n_folds: int,
                                  use_stage_stratify: bool = False,
                                  use_leverage_stratify: bool = False) -> dict:
    """
    (dataset, OS_event[, stage]) 조합별로 case를 n_folds개 fold에 라운드로빈으로 균등 배정한다.
    K-fold 교차검증의 fold 번호 할당 전용 — train/val 배정은 _kfold_case_split()이 이어서 한다.

    2026-08-14: use_stage_stratify=True면 stage도 stratify key에 들어가 그룹 수가 확 늘어난다
    (dataset x OS_event x stage, 최대 ~14개) — 상당수 그룹이 n_folds(5)보다 작아진다. 그룹마다
    항상 i=0부터 시작해 i%n_folds로 배정하면, 작은 그룹은 항상 fold 0/1/2...번에만 몰리는
    체계적 편향이 생긴다(실측: fold0 N=37 vs fold4 N=25로 뒤틀림). 그룹마다 시작 fold를 무작위
    offset으로 옮겨서, 어느 그룹의 "나머지"가 어느 fold에 쏠릴지 자체를 무작위화한다 — 그룹이
    여럿이면 이 쏠림이 fold 사이에서 평균적으로 상쇄된다. use_stage_stratify=False(그룹 수가
    최대 4개라 그룹이 항상 n_folds보다 훨씬 큼)일 때도 안전하게 동작한다.
    """
    rng = np.random.RandomState(seed)
    fold_of_case = {}
    for _, group in case_df.groupby(_stratify_keys(use_stage_stratify, use_leverage_stratify)):
        case_ids = group.index.to_numpy().copy()
        rng.shuffle(case_ids)
        offset = rng.randint(0, n_folds)
        for i, case_id in enumerate(case_ids):
            fold_of_case[case_id] = (i + offset) % n_folds
    return fold_of_case


def _stratified_binary_split(case_df: pd.DataFrame, seed: int, frac: float,
                              use_stage_stratify: bool = False,
                              use_leverage_stratify: bool = False) -> dict:
    """(dataset, OS_event[, stage][, high_leverage]) 조합별로 case를 frac:(1-frac) 비율로 "train"/"val" 둘로 나눈다."""
    rng = np.random.RandomState(seed)
    split_of_case = {}
    for _, group in case_df.groupby(_stratify_keys(use_stage_stratify, use_leverage_stratify)):
        case_ids = group.index.to_numpy().copy()
        rng.shuffle(case_ids)
        n       = len(case_ids)
        n_train = min(round(n * frac), n)
        for i, case_id in enumerate(case_ids):
            split_of_case[case_id] = "train" if i < n_train else "val"
    return split_of_case


def _kfold_case_split(case_df: pd.DataFrame, seed: int, n_folds: int, fold_idx: int,
                       use_stage_stratify: bool = False,
                       use_leverage_stratify: bool = False) -> dict:
    """
    K-fold 교차검증 split. case_df를 n_folds개로 나눠 fold_idx번째를 test로 쓰고, 나머지
    (n_folds-1)/n_folds 풀은 TRAIN_FRAC:VAL_FRAC 비율 그대로(60:20 관례)로 다시 train/val로
    나눈다 — 즉 fold_idx=0..n_folds-1 전부 돌리면 코호트의 모든 case가 정확히 한 번씩
    test로 쓰이고(pooled out-of-fold 평가로 c-index를 다시 계산하면 internal 표본이 코호트
    전체 크기로 늘어난다), 각 fold의 train 크기는 기존 단일 6:2:2 split과 거의 같다
    (fold 1개를 test로 빼고 남은 (n_folds-1)/n_folds 풀 안에서 다시 75:25로 나누므로,
    n_folds=5면 train=80%*0.75=60%, val=80%*0.25=20% — 기존과 동일).

    Args:
        case_df:  index=case_id, columns=["dataset", "OS_event", "stage", "high_leverage"]
        seed:     fold 배정과 train/val 재분할에 공통으로 쓰는 셔플 시드
        n_folds:  fold 개수
        fold_idx: 이번 호출에서 test로 쓸 fold 번호(0-based)
        use_stage_stratify: True면 stage도 stratify key에 포함(기본 False — 기존 동작 유지,
                            켜면 fold 배정 자체가 기존과 달라짐).
        use_leverage_stratify: True면 high_leverage(_HIGH_LEVERAGE_CASE_IDS 소속 여부)도
                            stratify key에 포함(기본 False, 켜면 fold 배정이 달라짐).
    Returns:
        {case_id: "train"|"val"|"test"}
    """
    fold_of_case = _stratified_kfold_assignment(case_df, seed, n_folds, use_stage_stratify=use_stage_stratify,
                                                 use_leverage_stratify=use_leverage_stratify)
    is_test = case_df.index.map(lambda cid: fold_of_case[cid] == fold_idx)
    test_df, remaining_df = case_df[is_test], case_df[~is_test]

    split_of_case = {case_id: "test" for case_id in test_df.index}
    train_val_frac = TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC)
    split_of_case.update(_stratified_binary_split(remaining_df, seed, frac=train_val_frac,
                                                    use_stage_stratify=use_stage_stratify,
                                                    use_leverage_stratify=use_leverage_stratify))
    return split_of_case


class WSISurvivalDataset(Dataset):
    """
    Args:
        cfg:           DataConfig (patches_root_tcga/cptac, precomputed, seed 참조)
        dataset:       "tcga" | "cptac" | "both" ("both"면 두 코호트를 하나의 풀로 합친다)
        split:         "train" | "val" | "test" — case 단위 6:2:2 stratified split
                       ((dataset, OS_event) 조합 기준, use_stage_stratify=True면 stage도 추가,
                       _stratified_case_split 참조)
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
                       선택한다 — "resnet50"(기본, features.pt), "uni"(features_uni.pt) 또는
                       "uni2"(features_uni2.pt, UNI2-h). data/extract_features.py --backbone으로
                       미리 추출해둔 파일이 있어야 한다. 모델(ViT_M1 등) 생성 시 backbone 인자와
                       반드시 일치시켜야 한다.
        rna_gene_ids:  with_rna=True일 때 사용할 유전자 ENSG id 목록. None(기본)이면
                       pdac_subtype_gene_ids()(Bailey/Moffitt subtype 분류용, ~340개)를
                       쓴다. literature_guided_gene_ids(top_n)(data/select_rnaseq_genes.py
                       산출물, 생존 예측에 직접 최적화된 유전자셋)를 넘기면 그걸 대신 쓴다.
        restrict_case_ids: 주어지면 이 case_id 집합에 없는 환자는 전부 제외한다(다른 필터를
                       전부 통과한 뒤 마지막에 적용). 레퍼런스(Leeyoungsup/pancreatic_cancer_
                       pathology) 코호트 포함 기준(24개월 시점 생존 여부 확정 + WSI 보유)에
                       맞춰 재검증할 때 사용 — reference_cohort.py::reference_eligible_case_ids()
                       참조.
        one_slide_per_case: True면 케이스당 슬라이드를 대표 1장으로 줄인다(기본 False — 케이스가
                       가진 슬라이드를 전부 사용하는 기존 동작 유지). _select_representative_slide()
                       참조 — 레퍼런스의 "환자당 diagnostic WSI 1개(TCGA)/tumor series 중 최대
                       용량 1개(CPTAC)" 큐레이션에 맞춘 실험용 옵션(findings_backlog.md 14번 항목).
                       exclude_normal_slides와 동시에 켜면 exclude_normal_slides가 먼저 적용된
                       뒤(정상 조직 제외) 대표 1장을 고른다.
        exclude_normal_slides: True면 확인된 정상 조직 슬라이드만 제외하고 케이스당 나머지는
                       전부 그대로 둔다(기본 False). _exclude_normal_slides() 참조 —
                       one_slide_per_case보다 훨씬 덜 급진적인 절충안(findings_backlog.md 14번
                       항목, TCGA 평균 슬라이드/case 2.52→2.28, CPTAC 3.22→2.76).
        dx_only_slides: True면 TCGA에서 DX(진단용/영구절편)가 아닌 슬라이드(TS/BS 등 냉동절편)만
                       제외하고 케이스당 남은 DX 슬라이드는 전부 그대로 둔다(기본 False).
                       _dx_only_slides() 참조 — uni2official 조사(2026-08-14)에서 발견한 두 confound
                       (DX-only 슬라이드 감소 + 좌표스케일 버그) 중 좌표스케일 버그 없이 DX-only
                       효과만 분리해서 보기 위한 옵션. CPTAC은 DX/TS 구분 정보가 없어 영향받지 않음.
        feature_filename_override: 주어지면 feature_backbone 대신 이 파일명을 읽는다(예:
                       "features_aug.pt", utils/extract_features_augmented.py 산출물). 해당
                       슬라이드에 이 파일이 없으면 원래 feature_backbone 파일명으로 조용히
                       폴백한다(train.py --tile-augment가 train split에서만 이걸 쓴다 — val/
                       test/external은 항상 기본 feature_backbone).
        fold:          주어지면(0-based) split in {"train","val","test"} 배정에 단일 6:2:2 대신
                       K-fold(_kfold_case_split 참조)를 쓴다 — case_df를 n_folds개로 나눠 이
                       fold를 test로, 나머지를 다시 60:20으로 train/val 배정. fold=0..n_folds-1을
                       전부 돌려 나온 test 예측을 이어붙이면(pooled out-of-fold) internal 표본이
                       단일 split의 20%가 아니라 코호트 전체 크기가 된다. None(기본)이면 기존
                       단일 _stratified_case_split 그대로 동작(하위 호환).
        n_folds:       fold 개수(기본 5). fold=None이면 무시.
        use_stage_stratify: 2026-08-14, 기본 False(기존 동작 그대로) — True면 (dataset,
                       OS_event) split stratification에 ajcc_stage(_load_case_stage())도
                       추가한다. fold별 log-rank p 변동 원인 조사에서 나쁜 fold일수록 단일
                       병기(Stage IIB)에 쏠려 있었던 것에 대한 대응. 켜면 같은 seed라도 fold
                       배정 자체가 기존과 달라지므로, 기존 결과와 직접 비교하려면 baseline도
                       같은 옵션으로 다시 돌려야 한다(train.py --stage-stratify).
        use_leverage_stratify: 2026-08-14, 기본 False(기존 동작 그대로) — True면 split
                       stratification에 high_leverage(_HIGH_LEVERAGE_CASE_IDS 소속 여부)도
                       추가한다. stage-stratify로도 fold별 log-rank p 변동이 fold 단위에서는
                       깔끔히 설명되지 않아, 다변량 상관 분석에서 가장 강했던 두 후보(고레버리지
                       환자 집중도 rho=0.894, 평균 연령 rho=0.900) 중 하나를 직접 통제해보는
                       탐색적 실험 — _HIGH_LEVERAGE_CASE_IDS는 baseline 3seed pooled OOF의
                       leave-one-out c-index delta로 역산한 20명으로, 모델에 종속적인(순환적인)
                       정의임을 유의할 것(일반적인 "어려운 환자" 정답이 아님). 켜면 같은 seed라도
                       fold 배정 자체가 기존과 달라진다(train.py --leverage-stratify).

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
        with_margin: bool = False,
        with_mutation: bool = False,
        with_rna: bool = False,
        feature_backbone: str = "resnet50",
        rna_gene_ids: list[str] | None = None,
        rna_pathway_categories: dict[str, list[str]] | None = None,
        rna_purist: bool = False,
        with_cnv: bool = False,
        restrict_case_ids: set[str] | None = None,
        one_slide_per_case: bool = False,
        exclude_normal_slides: bool = False,
        dx_only_slides: bool = False,
        feature_filename_override: str | None = None,
        fold: int | None = None,
        n_folds: int = 5,
        use_stage_stratify: bool = False,
        use_leverage_stratify: bool = False,
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
        if with_margin and not with_clinical:
            raise ValueError("with_margin=True는 with_clinical=True와 함께만 쓸 수 있습니다.")
        if with_mutation and not with_clinical:
            raise ValueError("with_mutation=True는 with_clinical=True와 함께만 쓸 수 있습니다.")

        self.transform        = transform or PATCH_TRANSFORM
        self.precomputed      = cfg.precomputed
        self.with_clinical    = with_clinical
        self.with_staging     = with_staging
        self.with_margin      = with_margin
        self.with_mutation    = with_mutation
        self.with_rna         = with_rna
        self.feature_backbone   = feature_backbone
        self.features_filename = FEATURES_FILENAME_BY_BACKBONE[feature_backbone]
        self.feature_filename_override = feature_filename_override
        # 2026-08-13: --tumor-type-embed(models/vit_encoder.py) — 슬라이드가 종양/정상 조직인지를
        # ViT 입력 단계에서 조건 신호로 주입하기 위해 매 __getitem__마다 다시 계산하지 않도록
        # 한 번만 로드해 캐싱한다(_load_cptac_gdc_slide_types()는 파일 I/O가 있음).
        self._cptac_slide_types = _load_cptac_gdc_slide_types()
        self.rna_gene_ids     = rna_gene_ids
        self.rna_pathway_categories = rna_pathway_categories
        self.rna_purist       = rna_purist
        self.with_cnv         = with_cnv
        self.use_stage_stratify = use_stage_stratify
        self.use_leverage_stratify = use_leverage_stratify

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
                if with_margin:
                    clinical_cols += ["residual_disease"]
                if with_mutation:
                    clinical_cols += list(MUTATION_FIELDS)
                clinical_df = pd.read_csv(CLINICAL_PATHS[name])[clinical_cols]
                merged = merged.merge(clinical_df, on="case_id", how="inner")

            if with_rna and self.rna_purist and self.rna_pathway_categories is not None:
                # 2026-09-03 추가 — 하이브리드: PurIST 확률(1차원, 고정 계수) + pathway8 카테고리
                # 평균(8차원) concat, 총 9차원. 위 purist+개별유전자 하이브리드(바로 아래 분기)와
                # 원리는 같지만 유전자 대신 카테고리 평균을 붙인다 — 둘 다 생존 라벨을 전혀 안 봐서
                # (PurIST=고정 계수, pathway8=문헌 카테고리 소속 여부만) 완전히 leak-free인 채로
                # PDAC RNA 표현력을 키우는 조합(BRCA의 PAM50+Oncotype DX+pan-cancer 카테고리
                # 확장과 같은 발상, findings_backlog.md 2026-09-03 참조).
                purist_df = pd.read_csv(RNA_PURIST_PATHS[name]).set_index("case_id")
                rna_df    = pd.read_csv(RNA_PATHS[name]).set_index("case_id")
                target_ids = set(g for genes in self.rna_pathway_categories.values() for g in genes)
                gene_cols  = [c for c in rna_df.columns if c in target_ids]
                col_index  = {c: i for i, c in enumerate(gene_cols)}
                cat_names  = sorted(self.rna_pathway_categories.keys())
                gene_matrix = rna_df[gene_cols].to_numpy(dtype="float32")
                agg = np.zeros((gene_matrix.shape[0], len(cat_names)), dtype="float32")
                for ci, cat in enumerate(cat_names):
                    idxs = [col_index[g] for g in self.rna_pathway_categories[cat] if g in col_index]
                    agg[:, ci] = gene_matrix[:, idxs].mean(axis=1)
                cat_df = pd.DataFrame(agg, index=rna_df.index, columns=cat_names)
                self.rna_category_names = cat_names
                combined = purist_df[["purist_basal_prob"]].join(cat_df, how="inner")
                if self.rna_gene_cols is None:
                    self.rna_gene_cols = list(combined.columns)
                elif list(combined.columns) != self.rna_gene_cols:
                    raise ValueError(f"[{name}] PurIST+pathway8 하이브리드 컬럼이 다른 코호트와 다릅니다.")
                rna_matrix = combined.to_numpy(dtype="float32")
                self.rna_lookup.update(zip(combined.index, rna_matrix))
                merged = merged.merge(combined.reset_index()[["case_id"]], on="case_id", how="inner")
            elif with_rna and self.rna_purist and self.rna_gene_ids is not None:
                # 하이브리드: PurIST 확률(1차원, 고정 계수) + 개별 유전자(z-score, 보통 TCGA/CPTAC
                # single-cohort top-N) concat. purist 단독(chance 수준이었음)과 개별 유전자 다수
                # (과적합 경향)의 절충 — case_id 기준 inner join으로 두 소스를 이어붙인다.
                purist_df = pd.read_csv(RNA_PURIST_PATHS[name]).set_index("case_id")
                rna_df    = pd.read_csv(RNA_PATHS[name]).set_index("case_id")
                gene_cols = [c for c in rna_df.columns if c in set(self.rna_gene_ids)]
                combined  = purist_df[["purist_basal_prob"]].join(rna_df[gene_cols], how="inner")
                if self.rna_gene_cols is None:
                    self.rna_gene_cols = list(combined.columns)
                elif list(combined.columns) != self.rna_gene_cols:
                    raise ValueError(
                        f"[{name}] PurIST+유전자 하이브리드 컬럼이 다른 코호트와 다릅니다."
                    )
                rna_matrix = combined.to_numpy(dtype="float32")
                self.rna_lookup.update(zip(combined.index, rna_matrix))
                merged = merged.merge(combined.reset_index()[["case_id"]], on="case_id", how="inner")
            elif with_rna and self.rna_purist:
                # PurIST(data/compute_purist_subtype.py) basal-like 확률 1차원 — 개별 유전자
                # 컬럼이 아니라 이미 계산된 단일 스코어라 아래 gene_cols/rna_matrix 분기와
                # 완전히 별개 경로로 처리한다. rna_tcga.csv/rna_cptac.csv(z-scored literature
                # 유전자 행렬)는 아예 읽지 않는다.
                purist_df = pd.read_csv(RNA_PURIST_PATHS[name])
                rna_matrix = purist_df[["purist_basal_prob"]].to_numpy(dtype="float32")
                self.rna_gene_cols = ["purist_basal_prob"]
                self.rna_lookup.update(zip(purist_df["case_id"], rna_matrix))
                merged = merged.merge(purist_df[["case_id"]], on="case_id", how="inner")
            elif with_rna:
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

                rna_case_ids = rna_df["case_id"]
                if self.with_cnv:
                    # 2026-09-03 추가 — pathway8 카테고리 평균(8차원)에 같은 8개 카테고리의 CNV
                    # 평균(cnv_pathway_category_features, log2-ratio+z-score)을 이어붙여 16차원
                    # "genomic" 벡터로 만든다. CNV 추출(data/extract_cnv.py)이 RNA 전체 코호트를
                    # 커버하지 못하므로(TCGA 135/152, CPTAC 139/144) inner join으로 자연히
                    # CNV 있는 case만 남는다 — 다른 modality(clinical/staging 등)와 동일한 관례.
                    if self.rna_pathway_categories is None:
                        raise ValueError("with_cnv=True는 rna_pathway_categories(즉 --rna-genes pathway8)와 함께만 지원합니다.")
                    cnv_df = cnv_pathway_category_features(name)
                    rna_indexed = pd.DataFrame(rna_matrix, index=rna_case_ids.values, columns=cat_names)
                    combined = rna_indexed.join(cnv_df, how="inner")
                    rna_matrix = combined.to_numpy(dtype="float32")
                    rna_case_ids = pd.Series(combined.index, name="case_id")
                    self.rna_category_names = list(combined.columns)

                # 유전자(또는 카테고리) 벡터는 case당 1번만 lookup에 저장하고(슬라이드 수만큼
                # 중복 저장 방지), merged 테이블에는 필터링용 case_id만 inner-join한다.
                self.rna_lookup.update(zip(rna_case_ids, rna_matrix))
                merged = merged.merge(rna_case_ids.to_frame(name="case_id"), on="case_id", how="inner")

            def _has_patches(slide_id: str, root=root) -> bool:
                d = root / "tiles" / slide_id
                if self.precomputed:
                    return (d / self.features_filename).exists()
                return (next(d.glob("*.jpg"), None) or next(d.glob("*.png"), None)) is not None

            has_patches = merged["slide_id"].apply(_has_patches)
            parts.append(merged[has_patches].reset_index(drop=True))

        all_items = pd.concat(parts, ignore_index=True)
        if exclude_normal_slides:
            all_items = _exclude_normal_slides(all_items)
        if dx_only_slides:
            all_items = _dx_only_slides(all_items)
        if one_slide_per_case:
            all_items = _select_representative_slide(all_items)
        if restrict_case_ids is not None:
            all_items = all_items[all_items["case_id"].isin(restrict_case_ids)].reset_index(drop=True)
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
            case_df = all_items.groupby("case_id").agg(dataset=("dataset", "first"), OS_event=("OS_event", "first"))
            if self.use_stage_stratify:
                # split stratification 전용 stage 컬럼(_load_case_stage 참조) — 모델이
                # with_clinical=False라 all_items에 ajcc_stage가 아예 없어도 여기서 독립적으로
                # 채운다. use_stage_stratify=False면 아예 안 만들고(불필요한 파일 I/O 생략)
                # _stratify_keys()도 "stage"를 안 봐서 이 컬럼 부재가 문제되지 않는다.
                stage_lookup = {}
                for name in case_df["dataset"].unique():
                    stage_lookup.update(_load_case_stage(name))
                case_df["stage"] = case_df.index.map(lambda cid: stage_lookup.get(cid, "unknown"))
            if self.use_leverage_stratify:
                # split stratification 전용 high_leverage 컬럼 — baseline 3seed pooled OOF의
                # leave-one-out c-index delta로 역산한 20명(_HIGH_LEVERAGE_CASE_IDS) 소속 여부.
                # 모델 종속적(순환적)인 정의라 일반적 "어려운 환자" 정답은 아니고, fold별
                # log-rank p 변동의 원인 탐색용 실험적 컬럼이다.
                case_df["high_leverage"] = case_df.index.map(lambda cid: cid in _HIGH_LEVERAGE_CASE_IDS)
            if fold is not None:
                # K-fold — fold 번째를 test로, 나머지를 다시 60:20 비율로 train/val 배정
                split_of_case = _kfold_case_split(case_df, seed=cfg.seed, n_folds=n_folds, fold_idx=fold,
                                                   use_stage_stratify=self.use_stage_stratify,
                                                   use_leverage_stratify=self.use_leverage_stratify)
            else:
                # case 단위 6:2:2 stratified split — (dataset, OS_event[, stage][, high_leverage])
                # 조합별로 seed 고정 셔플 후 배정
                split_of_case = _stratified_case_split(case_df, seed=cfg.seed,
                                                         use_stage_stratify=self.use_stage_stratify,
                                                         use_leverage_stratify=self.use_leverage_stratify)
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
        slide_dir = self.roots[row["dataset"]] / "tiles" / row["slide_id"]

        if self.feature_backbone in _PAIRED_COORDS_FILENAME_BY_BACKBONE:
            # patch grid가 우리 자체 JPG 추출본과 달라(uni2official: MahmoodLab 공식 추출,
            # uni2native: 우리 파이프라인이지만 별도 트리로 재타일링) list_patch_paths/파일명-파싱
            # coords를 쓸 수 없다. 짝을 이루는 coords 파일에서 직접 읽는다(변환 스크립트가
            # features/coords를 같은 행 순서로 저장해뒀으므로 길이 불일치 걱정이 없음).
            patch_paths = None
            coords = torch.load(slide_dir / _PAIRED_COORDS_FILENAME_BY_BACKBONE[self.feature_backbone],
                                 weights_only=True)
        else:
            patch_paths = list_patch_paths(slide_dir)
            coords = torch.tensor(
                [_parse_coord(p.name) for p in patch_paths],
                dtype=torch.long,
            )
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()

        tumor_status = _slide_tumor_status(row["slide_id"], row["dataset"], self._cptac_slide_types)
        item = {
            "coords":       coords,
            "case_id":      row["case_id"],
            "slide_id":     row["slide_id"],
            "dataset":      row["dataset"],
            "OS_time":      torch.tensor([row["OS_time"]], dtype=torch.float32),
            "OS_event":     torch.tensor([row["OS_event"]], dtype=torch.long),
            "tumor_status": torch.tensor(tumor_status, dtype=torch.long),  # 0=tumor,1=normal,2=unknown
        }

        if self.with_clinical:
            item["age_years"] = torch.tensor(row["age_years"], dtype=torch.float32)
            item["sex_idx"]   = torch.tensor(SEX_TO_IDX[row["sex"]], dtype=torch.long)
            if self.with_staging:
                for field in STAGE_FIELDS:
                    ord_val = encode_stage_value(field, row[field])
                    item[field] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)
            if self.with_margin:
                ord_val = encode_margin_value(row["residual_disease"])
                item["margin_ord"] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)
            if self.with_mutation:
                for field in MUTATION_FIELDS:
                    ord_val = encode_mutation_value(row[field])
                    item[field] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)

        if self.with_rna:
            item["rna"] = torch.from_numpy(self.rna_lookup[row["case_id"]])

        if self.precomputed:
            features_filename = self.features_filename
            if self.feature_filename_override is not None and (slide_dir / self.feature_filename_override).exists():
                features_filename = self.feature_filename_override
            features = torch.load(slide_dir / features_filename, weights_only=True)
            if patch_paths is not None and len(features) != len(patch_paths):
                raise RuntimeError(
                    f"{slide_dir}: {features_filename} 행 수({len(features)})가 패치 수"
                    f"({len(patch_paths)})와 다릅니다 — utils.extract_features를 다시 실행하세요."
                )
            if patch_paths is None and len(features) != len(coords):
                raise RuntimeError(
                    f"{slide_dir}: {features_filename} 행 수({len(features)})가 coords 행 수"
                    f"({len(coords)})와 다릅니다 — scripts/convert_uni2h_official_features.py를 다시 실행하세요."
                )
            item["features"] = features
        else:
            item["patch_paths"] = patch_paths

        return item

    def __getitem__(self, idx: int) -> list:
        case_id   = self.cases[idx]
        case_rows = self.items[self.items["case_id"] == case_id]
        return [self._load_slide(row) for _, row in case_rows.iterrows()]
