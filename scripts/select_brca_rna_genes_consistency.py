"""
data/select_rnaseq_genes_pdac_consistency.py를 BRCA로 옮긴 버전 — "완전히 객관적이면서 도메인
특화"인 RNA 패널이 필요하다는 동일한 요구(2026-09-04, 사용자 지시)를 BRCA에 적용한다.

PDAC 쪽은 JCI Insight(2025)의 5개 독립 PDAC 마이크로어레이 데이터셋 교차분석을 썼는데, BRCA는
그 대신 **Győrffy(2021, Computational and Structural Biotechnology Journal)**의 55개 독립 GEO
유방암 데이터셋(7830명) 통합 생존분석을 쓴다 — PDAC 논문(단순 발현 일관성, survival 라벨 미사용)
보다 오히려 우리 목적에 더 가깝다: 유전자별로 직접 Cox 회귀 + BH-FDR을 relapse-free survival
기준으로 계산해뒀다(우리 TCGA-BRCA 코호트는 전혀 사용되지 않음 — 순수 외부 문헌 소스, leak-free).

사용자가 직접 다운로드한 두 supplementary table(data/brca_rna_gene_selection_consistency_source/):
  - mmc1.xlsx: ER+/HER2- 화학요법 치료군, FDR<0.05 유의 유전자 692개, absolute HR로 정렬됨
  - mmc4.xlsx: basal(삼중음성 유사) 화학요법 치료군, 246개, 동일 형식
(논문에 나온 두 chemo-treated 서브그룹 전부 — untreated 버전(mmc6/mmc7)은 우리 코호트가 전부
수술+항암 표준치료를 받은 집단이 아니라는 보장이 없어 일단 제외.)

여기에 기존 BRCA 도메인 문헌 패널(scripts/select_brca_rna_genes_literature.py 산출물 —
PAM50+Oncotype DX+pan-cancer 6카테고리, 165유전자)을 합집합한다. PDAC 쪽 pathway8과 달리
BRCA는 카테고리 평균 메커니즘이 없어(scripts/brca_common.py::load_rna_matrix가 개별 유전자
컬럼을 그대로 씀) 그냥 flat 유전자 id 리스트 하나로 합친다.

[2026-09-04] "core_driver_tumor_suppressor"(KRAS/TP53/CDKN2A/SMAD4/ARID1A/PIK3CA/PTEN/BRAF/MYC
등 16개) — PDAC_LITERATURE_GENE_SETS 8개 카테고리 중 유일하게 BRCA 문헌 패널에 안 들어갔던
카테고리(select_brca_rna_genes_literature.py는 6개만 재사용, "core_driver"는 이름만 보면
PDAC 전용 같지만 실제로는 TP53/PIK3CA/PTEN/MYC/BRAF처럼 범암종 드라이버라 유방암에도 그대로
말이 됨, 특히 PIK3CA는 BRCA 최빈 돌연변이 중 하나 — 사용자 지적으로 발견) 도 여기서 추가한다.

최종: Győrffy(733) + 기존 BRCA 문헌(165) + core_driver(16, 대부분 겹침) 합집합 = 868개
(1500 목표엔 못 미치지만, PDAC의 "all5"(1680)처럼 억지로 개수를 맞추지 않고 자연스러운 결과를
그대로 받아들인다 — 사용자 결정, 2026-09-04. 필요하면 나중에 basal 코호트의 adjuvant-only
서브셋(mmc5)이나 untreated 코호트(mmc6/mmc7)를 추가해 확장 가능).

출력: data/brca_rna_gene_selection_consistency/selected_genes.csv (gene_id, gene_symbol, source)
사용법: python -m scripts.select_brca_rna_genes_consistency
"""
from pathlib import Path

import pandas as pd

from data.select_rnaseq_genes import PDAC_LITERATURE_GENE_SETS

COMMON_GENES_PATH = Path("data/common_genes.csv")
RNA_BRCA_PATH = Path("data/rna_brca.csv")
GYORFFY_ER_PATH = Path("data/brca_rna_gene_selection_consistency_source/1-s2.0-S2001037021003044-mmc1.xlsx")
GYORFFY_BASAL_PATH = Path("data/brca_rna_gene_selection_consistency_source/1-s2.0-S2001037021003044-mmc4.xlsx")
EXISTING_LIT_PATH = Path("data/brca_rna_gene_selection_literature/selected_genes.csv")
OUT_DIR = Path("data/brca_rna_gene_selection_consistency")

CORE_DRIVER_GENES = PDAC_LITERATURE_GENE_SETS["core_driver_tumor_suppressor"]


def _name_to_id() -> pd.Series:
    common = pd.read_csv(COMMON_GENES_PATH).drop_duplicates("gene_name", keep="first")
    return common.set_index("gene_name")["gene_id"]


def main() -> None:
    name_to_id = _name_to_id()
    rna_cols = set(pd.read_csv(RNA_BRCA_PATH, nrows=1).columns) - {"case_id"}

    er = pd.read_excel(GYORFFY_ER_PATH, header=0)
    basal = pd.read_excel(GYORFFY_BASAL_PATH, header=0)
    gyorffy = pd.concat([er[["Genesymbol"]], basal[["Genesymbol"]]], ignore_index=True).drop_duplicates()
    gyorffy["gene_id"] = gyorffy["Genesymbol"].map(name_to_id)
    gyorffy = gyorffy.dropna(subset=["gene_id"])
    gyorffy = gyorffy[gyorffy["gene_id"].isin(rna_cols)]
    gyorffy_records = [
        {"gene_id": row.gene_id, "gene_symbol": row.Genesymbol, "source": "gyorffy_survival"}
        for row in gyorffy.itertuples()
    ]
    print(f"Győrffy(ER+/HER2- {er['Genesymbol'].nunique()} + basal {basal['Genesymbol'].nunique()}, "
          f"유니크 심볼 {gyorffy['Genesymbol'].nunique()}) -> ENSG 매핑 {gyorffy['gene_id'].nunique()}개")

    existing_lit = pd.read_csv(EXISTING_LIT_PATH)
    existing_lit = existing_lit[existing_lit["gene_id"].isin(rna_cols)]
    lit_records = [
        {"gene_id": row.gene_id, "gene_symbol": row.gene_symbol, "source": f"literature_{row.panel}"}
        for row in existing_lit.itertuples()
    ]
    print(f"기존 BRCA 도메인 문헌(PAM50+Oncotype+pan-cancer): {existing_lit['gene_id'].nunique()}개")

    core_driver_ids = name_to_id.reindex(CORE_DRIVER_GENES).dropna()
    core_driver_ids = core_driver_ids[core_driver_ids.isin(rna_cols)]
    core_records = [
        {"gene_id": gid, "gene_symbol": sym, "source": "core_driver_pan_cancer"}
        for sym, gid in core_driver_ids.items()
    ]
    print(f"core_driver_tumor_suppressor(범암종 드라이버, 추가): {core_driver_ids.nunique()}개 "
          f"(원본 {len(CORE_DRIVER_GENES)}개 중)")

    table = pd.DataFrame(gyorffy_records + lit_records + core_records)
    dedup = table.drop_duplicates(subset="gene_id", keep="first").sort_values("gene_symbol")
    print(f"\n최종 합집합: {len(dedup)}개 (source별: "
          f"{dedup['source'].str.replace(r'_.*', '', regex=True).value_counts().to_dict()})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "selected_genes.csv"
    dedup.to_csv(out_path, index=False)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
