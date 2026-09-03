"""
TCGA-BRCA RNA-seq 유전자 선정 — 문헌 기반(임상 표준 예후 패널) 버전.

scripts/select_brca_rna_genes.py(고분산, 생존 라벨 미사용)와
scripts/select_brca_rna_genes_cox.py(생존 라벨 기반 Cox+FDR, PAAD에서 fold-경계 leak을
못 고친다고 확인됨 — findings_backlog.md 2026-09-03)에 이은 세 번째 대안. PDAC은
`data/select_rnaseq_genes.py::PDAC_LITERATURE_GENE_SETS`처럼 직접 문헌을 뒤져 카테고리를
큐레이션해야 했지만, 유방암은 이미 임상에서 검증된 표준 다유전자 예후 패널이 있어 그걸
그대로 가져다 쓴다 — 우리가 직접 통계 검정을 하지 않으므로 생존 라벨을 전혀 참조하지 않고
(leak 원천 차단), 동시에 개별 코호트의 작은 표본에서 나온 우연한 신호(findings_backlog.md
2026-09-03 항목 — Cox+FDR로 뽑은 유전자가 TCGA/CPTAC 코호트 간 거의 재현이 안 됐던 문제)에서도
자유롭다.

  - PAM50(Parker et al. 2009, NanoString Prosigna) — intrinsic molecular subtype(Luminal A/B,
    HER2-enriched, Basal-like, Normal-like) 분류용 50유전자.
  - Oncotype DX 21-gene Recurrence Score(Paik et al. 2004, NEJM) — 증식(proliferation)/
    에스트로겐 신호(estrogen)/HER2/침습(invasion)/기타 16개 암 관련 유전자 + 5개 기준
    유전자(ACTB/GAPDH/GUSB/RPLP0/TFRC, 발현량 정규화용으로 원 검사에선 쓰이지만 여기서는
    그대로 카테고리 하나로 포함 — 정규화 목적이 아니라 그냥 "문헌에 포함된 유전자"로 취급).

MammaPrint(70-gene, van 't Veer et al. 2002)는 제외했다 — 원 논문이 유전자 기호가 아니라
Agilent probe/EST accession 기준이라 신뢰할 수 있는 공식 gene symbol 목록을 웹 검색만으로
확인하지 못했다(70 probe가 unique gene 56개에 매핑, 상당수가 기호 없는 EST). 잘못된 기호를
넣느니 아예 빼는 게 안전하다고 판단 — 필요하면 genefu R 패키지(Bioconductor)의
`sig.gene70` 데이터로 나중에 정확히 채울 수 있다.

두 패널 합쳐 유니크 유전자 60개(11개 중복 — BAG1/BCL2/BIRC5/CCNB1/ERBB2/ESR1/GRB7/MKI67/
MMP11/MYBL2/PGR, 둘 다 증식/호르몬 신호를 핵심으로 삼아 겹치는 게 자연스럽다). BRCA는
scripts/brca_common.py::load_rna_matrix가 카테고리 평균이 아니라 개별 유전자 컬럼을 그대로
쓰므로(PAAD pathway8과 달리 카테고리 평균 메커니즘이 없음), 여기서도 variance/cox 패널과
동일하게 "유전자 id 리스트" 하나로 저장한다.

ORC6L(PAM50 원 논문 표기, 옛 HGNC 기호) -> ORC6(현재 기호)로 교체, CTSL2(Oncotype 원 논문
표기) -> CTSV(현재 기호)로 교체했다 — 둘 다 data/common_genes.csv에서 확인.

출력:
    data/brca_rna_gene_selection_literature/selected_genes.csv   gene_id, gene_symbol, panel

사용법:
    python -m scripts.select_brca_rna_genes_literature
"""
from pathlib import Path

import pandas as pd

COMMON_GENES_PATH = Path("data/common_genes.csv")
RNA_ZSCORED_PATH = Path("data/rna_brca.csv")
OUT_DIR = Path("data/brca_rna_gene_selection_literature")

# PAM50(Parker et al. 2009) — ORC6L는 현재 HGNC 기호 ORC6로 교체.
PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1",
    "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
]

# Oncotype DX 21-gene Recurrence Score(Paik et al. 2004) — CTSL2는 현재 HGNC 기호 CTSV로 교체.
# STK15는 현재 HGNC 기호 AURKA로 교체. GUS/RPLPO는 각각 GUSB/RPLP0로 표기 통일.
ONCOTYPE_DX_GENES = {
    "proliferation": ["MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2"],
    "estrogen":      ["ESR1", "PGR", "BCL2", "SCUBE2"],
    "her2":          ["ERBB2", "GRB7"],
    "invasion":      ["MMP11", "CTSV"],
    "other":         ["GSTM1", "CD68", "BAG1"],
    "reference":      ["ACTB", "GAPDH", "GUSB", "RPLP0", "TFRC"],
}


def main():
    common = pd.read_csv(COMMON_GENES_PATH).drop_duplicates("gene_name", keep="first")
    name_to_id = common.set_index("gene_name")["gene_id"]
    rna_cols = set(pd.read_csv(RNA_ZSCORED_PATH, nrows=1).columns) - {"case_id"}

    records = []
    for symbol in PAM50_GENES:
        records.append({"gene_symbol": symbol, "panel": "pam50", "category": "pam50_intrinsic"})
    for category, symbols in ONCOTYPE_DX_GENES.items():
        for symbol in symbols:
            records.append({"gene_symbol": symbol, "panel": "oncotype_dx", "category": f"oncotype_{category}"})

    table = pd.DataFrame(records)
    table["gene_id"] = table["gene_symbol"].map(name_to_id)
    missing_id = table[table["gene_id"].isna()]
    if len(missing_id):
        raise ValueError(f"common_genes.csv에 없는 기호(별칭 확인 필요): {missing_id['gene_symbol'].tolist()}")
    missing_rna = table[~table["gene_id"].isin(rna_cols)]
    if len(missing_rna):
        raise ValueError(f"rna_brca.csv 컬럼에 없음: {missing_rna['gene_symbol'].tolist()}")

    print(f"PAM50: {len(PAM50_GENES)}개, Oncotype DX: {sum(len(v) for v in ONCOTYPE_DX_GENES.values())}개 "
          f"(카테고리 {len(ONCOTYPE_DX_GENES)}개)")
    n_dup = table["gene_id"].duplicated().sum()
    print(f"두 패널 합쳐 총 {len(table)}행, 유니크 유전자 {table['gene_id'].nunique()}개 "
          f"(중복 {n_dup}개 — 두 패널 모두에 포함된 유전자)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "selected_genes.csv"
    dedup = table.drop_duplicates(subset="gene_id", keep="first")[["gene_id", "gene_symbol", "panel", "category"]]
    dedup.to_csv(out_path, index=False)
    print(f"저장: {out_path}  ({len(dedup)} unique genes)")


if __name__ == "__main__":
    main()
