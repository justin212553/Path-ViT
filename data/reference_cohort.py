"""
reference_eligible_case_ids() — 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)
`M4_Train.ipynb`의 케이스 포함 기준을 그대로 재현한 case_id 집합을 계산한다
(GitHub 원문 직접 확인, 2026-07-19).

레퍼런스가 실제로 적용하는 eligibility mask(M4_Train.ipynb):
    eligible = os_time.notna() & os_event.notna()
             & required_horizon_mask       # REQUIRE_COMPLETE_24M_HORIZONS=True
             & (n_tiles > 0)               # WSI 보유 (M7도 M4의 split을 그대로 재사용해 상속)
             & has_rnaseq_selected & age.notna() & sex.isin(["male","female"])

`required_horizon_mask`(REQUIRE_COMPLETE_24M_HORIZONS=True일 때)는 6/12/18/24개월
horizon 전부가 "known"이어야 한다 — 한 horizon이 known인 조건은
    (event==1 and time<=horizon) or (time>=horizon)
가장 늦은 horizon(24개월)만 확인하면 그보다 이른 horizon은 자동으로 만족되므로
(24개월 이전에 죽었거나, 24개월까지 살아있었던 게 확인되면 6/12/18개월도 자동으로 known),
아래 구현은 24개월 조건 하나만 검사한다 — 레퍼런스와 동일한 최종 포함/제외 결과를 낸다.

우리 age/sex/RNA 유효성은 이미 WSISurvivalDataset의 with_clinical/with_rna inner-join이
걸러주므로 여기서는 재검사하지 않는다 — 이 함수는 "24개월 horizon 확정 + WSI 보유"라는
레퍼런스 고유의 추가 필터만 담당하고, WSISurvivalDataset(restrict_case_ids=...)로 다른
필터와 함께(AND) 적용된다.
"""
from pathlib import Path

import pandas as pd

from config import DataConfig

HORIZON_24M_DAYS = 24 * 30.4375  # 레퍼런스 MONTH_DAYS=30.4375 그대로

OS_LABEL_PATHS = {
    "tcga":  Path("data/os_labels_tcga.csv"),
    "cptac": Path("data/os_labels_cptac.csv"),
}


def _load_slide_index(patches_root: Path) -> pd.DataFrame:
    paths = sorted(patches_root.glob("slide_index_task*.csv"))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def reference_eligible_case_ids(
    datasets: list[str] = ("tcga", "cptac"),
    cfg: DataConfig | None = None,
) -> set[str]:
    """레퍼런스의 "24개월 시점 생존 여부 확정 + WSI 보유" 기준을 통과하는 case_id 집합.

    Args:
        datasets: 대상 코호트("tcga"/"cptac" 중 일부 또는 전부).
        cfg: WSI 타일 경로(patches_root_tcga/cptac)를 가져올 DataConfig. None이면 기본값.
    Returns:
        case_id 집합(두 코호트를 합친 것 — case_id 자체가 코호트별로 명명 규칙이 달라
        충돌하지 않는다, 예: "TCGA-XX-XXXX" vs "C3L-00017").
    """
    cfg = cfg or DataConfig()
    patches_roots = {"tcga": cfg.patches_root_tcga, "cptac": cfg.patches_root_cptac}

    eligible: set[str] = set()
    for name in datasets:
        os_df = pd.read_csv(OS_LABEL_PATHS[name])
        known_24m = (
            ((os_df["OS_event"] == 1) & (os_df["OS_time"] <= HORIZON_24M_DAYS))
            | (os_df["OS_time"] >= HORIZON_24M_DAYS)
        )

        slide_df = _load_slide_index(Path(patches_roots[name]))
        slide_df = slide_df[(slide_df["status"] == "ok") & (slide_df["n_tiles_kept"] > 0)]
        has_wsi_case_ids = set(slide_df["case_id"])

        eligible_cases = set(os_df.loc[known_24m, "case_id"]) & has_wsi_case_ids
        eligible |= eligible_cases

    return eligible
