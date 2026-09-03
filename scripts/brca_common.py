"""
TCGA-BRCA M4/M7 스크립트(train_brca_m4.py, train_brca_m7.py, select_brca_rna_genes.py)가
공유하는 case 목록 + train/val/test/external split 로직.

data/dataset.py::_stratified_case_split(TCGA-PAAD/CPTAC용, (dataset, OS_event) 조합 기준)와
동일한 방법론을 BRCA(단일 코호트라 OS_event만으로 그룹핑)에 맞게 재현한다 — M4와 M7이
반드시 "같은 case, 같은 split"으로 비교돼야 하므로(사용자 지시: "같은 환경일 때 M7을
넘냐 안 넘냐가 문제") 이 모듈 하나로 두 스크립트의 split이 어긋나지 않게 고정한다.

2026-08-30: institution(TCGA barcode 2번째 세그먼트 = Tissue Source Site) 기준 internal/
external split 추가 — PAAD(학습)→CPTAC(외부 코호트) 구도를 BRCA는 단일 코호트라 그대로
재현할 수 없어서, 가장 큰 단일 기관(BH, 142명, 공통 case 1058명 중 13.4%)을 통째로 external
holdout으로 뺀다(사용자 결정 — 여러 기관을 섞기보다 "정말 한 번도 학습에 안 쓴 기관" 하나를
깔끔하게 분리). 같은 기관 슬라이드는 염색/스캐너 특성이 비슷해 site-based split이 병리 ML에서
흔히 쓰는 leakage 방지 표준 방법이기도 하다. RNA 유전자 선택(select_brca_rna_genes.py 산출물,
data/brca_rna_gene_selection/)은 이 재분할 이전 기준으로 이미 고정돼 있고 이번 실험에서는
다시 안 만든다(사용자 지시) — 재선택 없이 그대로 재사용.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.clinical_encoder import SEX_TO_IDX, STAGE_FIELDS, encode_stage_value

CLINICAL_PATH = Path("data/brca_clinical.csv")
MANIFEST_PATH = Path("data/brca_slide_manifest.csv")
RNA_RAW_LOG2_PATH = Path("data/rna_brca_raw_log2.csv")
RNA_ZSCORED_PATH = Path("data/rna_brca.csv")
TILES_ROOT = Path("data/patches_tcga_brca/tiles")

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2  # 나머지 0.2는 test

EXTERNAL_TSS = "BH"  # 2026-08-30: institution-level external holdout 기본값(사용자 결정)


def _tss(case_id: str) -> str:
    """TCGA barcode 2번째 세그먼트(Tissue Source Site = 제출 기관 코드). 예: "TCGA-3C-AALI" -> "3C"."""
    return case_id.split("-")[1]


def common_case_ids() -> list[str]:
    """Clinical(OS) ∩ RNA ∩ WSI(manifest) 세 데이터 모두 존재하는 case_id 목록."""
    clinical = pd.read_csv(CLINICAL_PATH)
    rna = pd.read_csv(RNA_ZSCORED_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    common = set(clinical["case_id"]) & set(rna["case_id"]) & set(manifest["case_id"])
    return sorted(common)


def split_by_institution(case_ids: list[str], external_tss: str | None) -> tuple[list[str], list[str]]:
    """case_ids를 기관(TSS) 기준으로 (internal_case_ids, external_case_ids)로 나눈다.

    external_tss=None이면 전부 internal(기존 동작 그대로, external 없음).
    """
    if external_tss is None:
        return case_ids, []
    external = [c for c in case_ids if _tss(c) == external_tss]
    internal = [c for c in case_ids if _tss(c) != external_tss]
    return internal, external


def stratified_case_split(case_ids: list[str], os_event_by_case: dict, seed: int) -> dict:
    """(OS_event) 그룹별로 case를 6:2:2(train/val/test)로 나눈다.

    data/dataset.py::_stratified_case_split과 동일 로직 — BRCA는 단일 코호트라
    "dataset" 축이 없어 OS_event만으로 그룹핑한다. external로 뺀 case는 이 함수에 아예
    넘기지 않으므로(load_case_table 참조) train/val/test 비율은 항상 internal 인구 기준.
    """
    rng = np.random.RandomState(seed)
    split_of_case = {}
    events = np.array([os_event_by_case[c] for c in case_ids])
    for event_value in sorted(set(events.tolist())):
        group = [c for c, e in zip(case_ids, events) if e == event_value]
        group = np.array(group)
        rng.shuffle(group)
        n = len(group)
        n_train = min(round(n * TRAIN_FRAC), n)
        n_val = min(round(n * VAL_FRAC), n - n_train)
        for i, case_id in enumerate(group):
            if i < n_train:
                split_of_case[case_id] = "train"
            elif i < n_train + n_val:
                split_of_case[case_id] = "val"
            else:
                split_of_case[case_id] = "test"
    return split_of_case


def _stratified_kfold_assignment(case_ids: list[str], os_event_by_case: dict, seed: int, n_folds: int) -> dict:
    """data/dataset.py::_stratified_kfold_assignment과 동일 알고리즘(OS_event 그룹별 라운드로빈
    + 그룹마다 무작위 시작 offset)을 BRCA(단일 코호트, OS_event만으로 그룹핑)에 맞게 재현한다."""
    rng = np.random.RandomState(seed)
    events = np.array([os_event_by_case[c] for c in case_ids])
    fold_of_case = {}
    for event_value in sorted(set(events.tolist())):
        group = np.array([c for c, e in zip(case_ids, events) if e == event_value])
        rng.shuffle(group)
        offset = rng.randint(0, n_folds)
        for i, case_id in enumerate(group):
            fold_of_case[case_id] = (i + offset) % n_folds
    return fold_of_case


def _stratified_binary_split(case_ids: list[str], os_event_by_case: dict, seed: int, frac: float) -> dict:
    """data/dataset.py::_stratified_binary_split과 동일 — OS_event 그룹별로 frac:(1-frac)."""
    rng = np.random.RandomState(seed)
    events = np.array([os_event_by_case[c] for c in case_ids])
    split_of_case = {}
    for event_value in sorted(set(events.tolist())):
        group = np.array([c for c, e in zip(case_ids, events) if e == event_value])
        rng.shuffle(group)
        n = len(group)
        n_train = min(round(n * frac), n)
        for i, case_id in enumerate(group):
            split_of_case[case_id] = "train" if i < n_train else "val"
    return split_of_case


def kfold_case_split(case_ids: list[str], os_event_by_case: dict, seed: int, n_folds: int, fold_idx: int) -> dict:
    """data/dataset.py::_kfold_case_split과 동일 방법론(PAAD paper-spec 프로토콜과 동일 재현성
    보장) — fold_idx번째를 test로, 나머지 (n_folds-1)/n_folds 풀을 다시 TRAIN_FRAC:VAL_FRAC
    비율로 train/val 배정. fold_idx=0..n_folds-1을 전부 돌리면 코호트 전체(internal 인구)가
    정확히 한 번씩 test로 쓰인다(pooled out-of-fold).

    2026-09-01: 기존 load_case_table()의 단일 6:2:2(seed가 fold 배정과 model init을 동시에
    통제)만으로는 M4 vs M7 비교가 "이 데이터 분할 하나에서" 이상의 의미를 못 가진다는 사용자
    지적(BRCA에서 다시드 검증을 split_seed로 착각했던 일) 이후, PAAD와 동일한 진짜 k-fold
    프로토콜을 BRCA에도 적용하기 위해 추가."""
    fold_of_case = _stratified_kfold_assignment(case_ids, os_event_by_case, seed, n_folds)
    test_ids = [c for c in case_ids if fold_of_case[c] == fold_idx]
    remaining_ids = [c for c in case_ids if fold_of_case[c] != fold_idx]
    split_of_case = {c: "test" for c in test_ids}
    train_val_frac = TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC)
    split_of_case.update(_stratified_binary_split(remaining_ids, os_event_by_case, seed, frac=train_val_frac))
    return split_of_case


def load_case_table_kfold(seed: int, fold: int, n_folds: int, external_tss: str | None = EXTERNAL_TSS) -> pd.DataFrame:
    """load_case_table()의 k-fold 버전 — internal 인구(institution external holdout 제외)를
    kfold_case_split()으로 train/val/test(fold) 배정, external_tss 기관은 그대로 "external".

    columns: case_id, OS_time, OS_event, age_years, sex, split
    """
    case_ids = common_case_ids()
    internal_ids, external_ids = split_by_institution(case_ids, external_tss)
    clinical = pd.read_csv(CLINICAL_PATH).set_index("case_id")
    table = clinical.loc[case_ids].reset_index()
    os_event_by_case = dict(zip(table["case_id"], table["OS_event"]))
    split_of_case = kfold_case_split(internal_ids, os_event_by_case, seed, n_folds, fold)
    for cid in external_ids:
        split_of_case[cid] = "external"
    table["split"] = table["case_id"].map(split_of_case)
    return table


def load_rna_matrix(gene_ids: list[str]) -> pd.DataFrame:
    """data/rna_brca.csv(코호트 내부 z-score) 중 지정된 gene_id 컬럼만 골라 반환.

    z-score는 유전자별 독립 계산이라(scripts/select_brca_rna_genes.py 참조), 이미 z-score된
    전체 유전자 테이블에서 컬럼만 서브셋해도 값이 바뀌지 않는다.
    """
    rna = pd.read_csv(RNA_ZSCORED_PATH).set_index("case_id")
    return rna[gene_ids]


def load_rna_matrix_categorized(categories: dict[str, list[str]]) -> pd.DataFrame:
    """PAAD pathway8과 동일한 방식(data/dataset.py::WSISurvivalDataset의 rna_pathway_categories
    처리)을 BRCA에 재현 — 카테고리별 유전자 z-score의 평균을 카테고리당 1개 컬럼으로 반환한다
    (개별 유전자 그대로 쓰는 load_rna_matrix()의 대안, 2026-09-03 추가).

    [왜 필요한가] PAAD에서 "카테고리 평균(8차원)이 개별 유전자 그대로(163차원, pathway8_flat)보다
    작은 코호트(TCGA train~91명)에서 더 안정적"이라는 게 실측 확인됨(findings_backlog.md
    2026-09-03) — 같은 원칙을 BRCA(문헌 패널 165개)에도 적용해 pathway8과 구조적으로 동일한
    비교를 만든다. BRCA는 코호트가 훨씬 커서(train~635명) 이 압축이 실제로 필요한지는 별개
    문제지만, "PDAC은 표본이 작아서 신호가 안 났다"는 가설을 검증하려면 같은 방식으로 맞춰야
    공정한 비교가 된다(사용자 지시).
    """
    rna = pd.read_csv(RNA_ZSCORED_PATH).set_index("case_id")
    cat_names = sorted(categories.keys())
    out = pd.DataFrame(index=rna.index)
    for cat in cat_names:
        cols = [g for g in categories[cat] if g in rna.columns]
        out[cat] = rna[cols].mean(axis=1)
    return out


def load_literature_categories(merge_oncotype: bool = True) -> dict[str, list[str]]:
    """scripts/select_brca_rna_genes_literature.py 산출물(선택된 유전자 + category 라벨)을
    카테고리 -> gene_id 목록 dict로 되돌린다 — load_rna_matrix_categorized() 입력용.

    merge_oncotype=True(기본)면 oncotype_* 6개 서브카테고리(중복 제거 후 일부는 유전자 1~2개
    뿐이라 평균의 의미가 약함)를 "oncotype_dx" 하나로 합친다 — 그 결과 PAM50(1) + Oncotype
    DX(1) + pan-cancer 6개 = 총 8개 카테고리로, PAAD pathway8과 카테고리 개수가 정확히 일치한다
    (2026-09-03, 사용자 지시 — "PADC에서 뭉치기로 했으면 BRCA에서도 뭉치자").
    """
    df = pd.read_csv(Path("data/brca_rna_gene_selection_literature/selected_genes.csv"))
    if merge_oncotype:
        df = df.copy()
        df["category"] = df["category"].where(~df["category"].str.startswith("oncotype_"), "oncotype_dx")
    return {cat: g["gene_id"].tolist() for cat, g in df.groupby("category")}


def load_case_table(seed: int, external_tss: str | None = EXTERNAL_TSS) -> pd.DataFrame:
    """공통 case에 대해 clinical + split 정보를 합친 테이블.

    columns: case_id, OS_time, OS_event, age_years, sex, split
    split: "train"/"val"/"test"(internal, external_tss 기관 제외 인구에서 6:2:2)
           또는 "external"(external_tss 기관 전체, 학습에 전혀 안 쓰임).
    external_tss=None이면 기존 동작(institution split 없음, 전부 internal 6:2:2) 그대로.
    """
    case_ids = common_case_ids()
    internal_ids, external_ids = split_by_institution(case_ids, external_tss)
    clinical = pd.read_csv(CLINICAL_PATH).set_index("case_id")
    table = clinical.loc[case_ids].reset_index()
    os_event_by_case = dict(zip(table["case_id"], table["OS_event"]))
    split_of_case = stratified_case_split(internal_ids, os_event_by_case, seed)
    for cid in external_ids:
        split_of_case[cid] = "external"
    table["split"] = table["case_id"].map(split_of_case)
    return table


def _identity_collate(batch: list) -> list:
    return batch[0]


class BRCACaseDataset(Dataset):
    """M7(ClinicalRNAOnly)용 — WSI 없이 case 단위 clinical+RNA만 반환.

    train_light.py::_patient_risk가 기대하는 형식(환자 1명 = dict 1개짜리 list, dict에
    age_years/sex_idx/rna/OS_time/OS_event)을 그대로 맞춘다.
    """

    def __init__(self, case_table: pd.DataFrame, rna_df: pd.DataFrame, with_staging: bool = False):
        self.rows = case_table.reset_index(drop=True)
        self.rna_df = rna_df
        self.with_staging = with_staging

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> list:
        row = self.rows.iloc[idx]
        item = {
            "case_id":   row["case_id"],  # train_light.py::evaluate()가 요구(2026-08-30 확인,
                                           # BRCASlideDataset은 이미 있었는데 이쪽만 빠져있었음)
            "OS_time": torch.tensor([row["OS_time"]], dtype=torch.float32),
            "OS_event": torch.tensor([row["OS_event"]], dtype=torch.long),
            "age_years": torch.tensor(row["age_years"], dtype=torch.float32),
            "sex_idx": torch.tensor(SEX_TO_IDX[row["sex"]], dtype=torch.long),
            "rna": torch.from_numpy(self.rna_df.loc[row["case_id"]].to_numpy(dtype="float32")),
        }
        if self.with_staging:
            for field in STAGE_FIELDS:
                ord_val = encode_stage_value(field, row[field])
                item[field] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)
        return [item]


class BRCASlideDataset(Dataset):
    """M4(ViT_PMA)용 — case당 슬라이드(보통 1장) 전부를 train.py::_patient_risk가 기대하는
    형식(슬라이드 dict의 list, 각 dict에 coords/features(+age_years/sex_idx/rna/OS_time/
    OS_event 전부 중복 포함 — _patient_risk는 patient_slides[0]에서만 clinical/rna를 읽지만
    data/dataset.py::WSISurvivalDataset도 매 슬라이드에 중복해서 넣는 관례를 그대로 따른다))
    으로 반환한다.

    [좌표 정규화] HF에서 받은 coords.pt는 WSI 픽셀 좌표(예: 0,512,1024,...이지만 슬라이드마다
    타일링 격자가 정확히 512 배수로 정렬돼 있지 않다 — 실측 결과 512/416/256이 섞여 나옴,
    scripts/prepare_brca_data.py 산출물 확인 과정에서 발견)라 그대로 SpatialPositionEmbedding에
    넣으면(row/col 값이 수만대) sin/cos가 여러 바퀴 감겨 위치 정보가 사실상 노이즈가 된다.
    torch.unique(..., return_inverse=True)로 각 슬라이드 내부에서 고유 좌표값의 정렬 순위로
    변환하면 data/dataset.py의 (r, c) 그리드 인덱스 관례와 동등한 작은 정수 좌표를 얻는다 —
    격자가 살짝 불균일해도 상대적 순서는 보존된다.
    """

    def __init__(self, case_table: pd.DataFrame, rna_df: pd.DataFrame, manifest: pd.DataFrame,
                 with_staging: bool = False):
        self.rows = case_table.reset_index(drop=True)
        self.rna_df = rna_df
        self.slides_by_case = manifest.groupby("case_id")
        self.with_staging = with_staging

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _grid_coords(coords: torch.Tensor) -> torch.Tensor:
        _, row_idx = torch.unique(coords[:, 0], sorted=True, return_inverse=True)
        _, col_idx = torch.unique(coords[:, 1], sorted=True, return_inverse=True)
        return torch.stack([row_idx, col_idx], dim=1).long()

    def __getitem__(self, idx: int) -> list:
        row = self.rows.iloc[idx]
        case_id = row["case_id"]
        slide_rows = self.slides_by_case.get_group(case_id)

        common = {
            "case_id":   case_id,  # 2026-08-11: train.py::evaluate()가 case_id를 요구하도록 바뀌어 추가
            "OS_time":   torch.tensor([row["OS_time"]], dtype=torch.float32),
            "OS_event":  torch.tensor([row["OS_event"]], dtype=torch.long),
            "age_years": torch.tensor(row["age_years"], dtype=torch.float32),
            "sex_idx":   torch.tensor(SEX_TO_IDX[row["sex"]], dtype=torch.long),
            "rna":       torch.from_numpy(self.rna_df.loc[case_id].to_numpy(dtype="float32")),
        }
        if self.with_staging:
            for field in STAGE_FIELDS:
                ord_val = encode_stage_value(field, row[field])
                common[field] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)

        slides = []
        for _, srow in slide_rows.iterrows():
            slide_dir = TILES_ROOT / srow["slide_id"]
            coords = self._grid_coords(torch.load(slide_dir / "coords.pt", weights_only=True))
            features = torch.load(slide_dir / "features_uni.pt", weights_only=True)
            slides.append({"coords": coords, "features": features, **common})
        return slides
