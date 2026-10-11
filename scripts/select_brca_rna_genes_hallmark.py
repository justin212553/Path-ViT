"""
2026-10-11: MMP 공식 BRCA RNA 패널(mmp/src/data_csvs/rna/hallmarks/BRCA/rna_clean.csv의 유전자 컬럼,
MSigDB Hallmark 50 pathway 유전자, 4천여 개 gene symbol)을 우리 RNA 파이프라인의 Ensembl gene_id로
옮긴 유전자 목록을 만든다 — train_brca_m4.py --gene-selection hallmark 입력.

목적: BRCA 아키텍처 수준 비교. MMP는 RNA를 pathway token으로 받아 우리 882유전자 패널을 넣을 수 없으므로
반대로 우리 모델(M3/M4)에 MMP의 유전자 세트를 넣는다. 그러면
  - M3(HALL) vs MMP: 입력 유전자 세트가 같아 아키텍처 차이만 남는다
  - M3(HALL) vs M3(CONS882): 아키텍처가 같아 패널 차이만 남는다(우리 패널의 기여 검증)
값은 MMP의 rna_clean.csv가 아니라 우리 raw log2(data/rna_brca_raw_log2.csv)를 쓰고 --fold-safe로 정규화한다
(정규화 규칙까지 우리 모델들과 동일하게 — 바뀌는 것은 유전자 목록 하나).

symbol -> gene_id 매핑은 data/common_genes.csv(gene_id, gene_name). 같은 symbol이 여러 gene_id에 걸리면
raw RNA에 있는 것만, 그래도 여럿이면 전부 제외하고 개수를 보고한다. raw RNA에 없는 symbol도 보고한다.

출력: data/brca_rna_gene_selection_hallmark/selected_genes.csv (gene_id, gene_name)
사용법(HPC, repo 루트): python -m scripts.select_brca_rna_genes_hallmark
"""
from pathlib import Path

import pandas as pd

MMP_RNA = Path("mmp/src/data_csvs/rna/hallmarks/BRCA/rna_clean.csv")
RAW = Path("data/rna_brca_raw_log2.csv")
SYMBOLS = Path("data/common_genes.csv")
OUT = Path("data/brca_rna_gene_selection_hallmark/selected_genes.csv")


def main():
    cols = pd.read_csv(MMP_RNA, nrows=0, index_col=0).columns
    symbols = sorted({c for c in cols if c and not c.lower().startswith("unnamed")})
    raw_ids = set(pd.read_csv(RAW, nrows=0).columns) - {"case_id"}
    m = pd.read_csv(SYMBOLS)
    m = m[m["gene_name"].isin(symbols) & m["gene_id"].isin(raw_ids)]
    multi = m.groupby("gene_name")["gene_id"].nunique()
    ambiguous = sorted(multi[multi > 1].index)
    m = m[~m["gene_name"].isin(ambiguous)].drop_duplicates("gene_name")
    unmapped = sorted(set(symbols) - set(m["gene_name"]) - set(ambiguous))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m[["gene_id", "gene_name"]].sort_values("gene_name").to_csv(OUT, index=False)
    print(f"MMP Hallmark 패널 symbol {len(symbols)}개 -> 우리 raw RNA gene_id {len(m)}개 매핑 "
          f"(매핑 실패 {len(unmapped)}개, symbol 중복으로 제외 {len(ambiguous)}개)")
    if unmapped:
        print(f"  매핑 실패 예: {unmapped[:15]}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
