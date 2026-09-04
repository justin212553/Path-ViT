"""
PORPOISE의 6-gene-family(porpoise_sig, 암종 무관 범용 카테고리)에 대한 대안 — "완전히 객관적이면서도
PDAC 특화"인 큰 유전자 패널이 필요하다는 사용자 요구를 반영(findings_backlog.md 2026-09-03 참조).

출처: Danielle Chang et al., "Multidimensional analyses identify genes of high priority for
pancreatic cancer research", JCI Insight (2025), DOI 10.1172/jci.insight.174264 — PDAC
마이크로어레이 데이터셋 5개를 각각 독립적으로 미분화 발현 분석한 뒤 결합, "5개 중 4개 이상에서
일관되게 상향/하향된 유전자"를 CUG(consistently upregulated)/CDG(consistently downregulated)로
정의했다(우리 TCGA/CPTAC 코호트는 전혀 사용되지 않음 — 순수 문헌 소스). Supplemental Table S1a
(사용자가 직접 다운로드, data/rna_gene_selection_pdac_consistency_source/)에 유전자별 연속값
Rank(cross-dataset 일관성 점수, CUG는 양수/CDG는 음수, 절댓값이 클수록 5개 데이터셋에서 일관됨)가
이미 계산되어 있어 |Rank| 기준으로 바로 top-N을 뽑을 수 있다 — 우리가 따로 통계를 계산하지 않는다.

data/select_rnaseq_genes_variance.py(우리 코호트 분산 기준)나 porpoise_signature_gene_ids()
(PORPOISE의 범용 6카테고리)와 달리, 이 패널은 유전자 선정 전 과정에서 우리 코호트도 PORPOISE의
카테고리 목록도 전혀 참조하지 않는다 — PDAC이라는 암종 자체에 대한 외부 문헌 통계만 쓴다.

[2026-09-03 추가 — all5/all5_pluslit] "|Rank| top-N"은 CUG/CDG 판정 기준 자체(최소 5개 중
4개 일관)를 그대로 물려받는다. 사용자 요청으로 더 엄격한 기준도 추가: Table_S1a의 5개 데이터셋별
원본 블록(GSE71729/GSE62452/GSE28735/GSE16515/GSE15471, 각각 logFC+adj.P.Val)에서
adj.P.Val<0.05 **그리고 5개 전부 방향(logFC 부호) 일관** 유전자만 뽑으면 1730개(up 993+down 737,
ENSG 매핑 1680개) — "5개 중 4개"보다 엄격한 "5개 전부"판. pluslit는 여기에 우리 기존 PDAC 특화
문헌 세트(Bailey 2016 612 + Moffitt 2015 100 + pathway8 8카테고리 163, 합집합 815 심볼)를 더한
버전(합집합 2387 심볼, ENSG 매핑 2167개) — 두 세트가 겹치는 게 158개뿐이라(서로 다른 근거) 합쳐도
중복이 크지 않다.

사용법: python -m data.select_rnaseq_genes_pdac_consistency
산출물:
  data/rna_gene_selection_pdac_consistency/selected_genes_top_{500,1000,1500,2000}.csv
  data/rna_gene_selection_pdac_consistency/selected_genes_all5consistent.csv        (1680개)
  data/rna_gene_selection_pdac_consistency/selected_genes_all5consistent_pluslit.csv (2167개)
"""
import argparse
from pathlib import Path

import pandas as pd

from data.dataset import COMMON_GENES_PATH
from data.select_rnaseq_genes import PDAC_LITERATURE_GENE_SETS

SOURCE_XLSX = Path("data/rna_gene_selection_pdac_consistency_source/jciinsight-10-174264-s248.xlsx")
OUT_DIR = Path("data/rna_gene_selection_pdac_consistency")
BAILEY_PATH = Path("data/bailey_subtype_genes.tsv")
MOFFITT_PATH = Path("data/moffitt_subtype_genes.tsv")

DATASET_BLOCKS = [
    ("GSE71729", "logFC", "adj.P.Val"),
    ("GSE62452", "logFC.1", "adj.P.Val.1"),
    ("GSE28735", "logFC.2", "adj.P.Val.2"),
    ("GSE16515", "logFC.3", "adj.P.Val.3"),
    ("GSE15471", "logFC.4", "adj.P.Val.4"),
]


def _map_to_ensg(symbols) -> pd.Series:
    common_genes = pd.read_csv(COMMON_GENES_PATH).drop_duplicates(subset="gene_name", keep="first")
    name_to_id = common_genes.set_index("gene_name")["gene_id"]
    return name_to_id.reindex(list(symbols)).dropna()


def load_ranked_table() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_XLSX, sheet_name="Table_S1a", header=7, usecols=[0, 1, 2, 3])
    df.columns = ["gene_symbol", "alias", "rank", "pattern"]
    df = df.dropna(subset=["gene_symbol"]).copy()
    df["abs_rank"] = df["rank"].abs()

    mapped = _map_to_ensg(df["gene_symbol"])
    df["gene_id"] = mapped.reindex(df["gene_symbol"]).values
    df = df.dropna(subset=["gene_id"]).drop_duplicates(subset="gene_id", keep="first")
    return df.sort_values("abs_rank", ascending=False).reset_index(drop=True)


def load_all5_consistent_symbols() -> set[str]:
    """5개 데이터셋 블록 전부에서 adj.P.Val<0.05 그리고 logFC 부호가 5개 전부 같은 유전자만."""
    full = pd.read_excel(SOURCE_XLSX, sheet_name="Table_S1a", header=7)
    sig_up, sig_down = {}, {}
    for name, fc_col, p_col in DATASET_BLOCKS:
        sub = full[[name, fc_col, p_col]].dropna(subset=[name])
        sig = sub[sub[p_col] < 0.05]
        sig_up[name] = set(sig.loc[sig[fc_col] > 0, name])
        sig_down[name] = set(sig.loc[sig[fc_col] < 0, name])
    return set.intersection(*sig_up.values()) | set.intersection(*sig_down.values())


def load_existing_pdac_literature_symbols() -> set[str]:
    bailey = pd.read_csv(BAILEY_PATH, sep="\t")
    moffitt = pd.read_csv(MOFFITT_PATH, sep="\t")
    all_pdac = set()
    for genes in PDAC_LITERATURE_GENE_SETS.values():
        all_pdac |= set(genes)
    return set(bailey["gene_symbol"]) | set(moffitt["gene_symbol"]) | all_pdac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-genes", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ranked = load_ranked_table()
    print(f"ENSG 매핑 성공: {len(ranked)} / 3938 (CUG {(ranked['pattern']=='CUG').sum()}, "
          f"CDG {(ranked['pattern']=='CDG').sum()})")
    for n in args.n_genes:
        top_n = ranked.head(n)
        out_path = OUT_DIR / f"selected_genes_top_{n}.csv"
        top_n[["gene_id", "gene_symbol", "rank", "pattern"]].to_csv(out_path, index=False)
        print(f"{out_path}: {len(top_n)}개 (CUG {(top_n['pattern']=='CUG').sum()}, "
              f"CDG {(top_n['pattern']=='CDG').sum()})")

    all5_symbols = load_all5_consistent_symbols()
    all5_mapped = _map_to_ensg(all5_symbols)
    out_path = OUT_DIR / "selected_genes_all5consistent.csv"
    pd.DataFrame({"gene_id": all5_mapped.values, "gene_symbol": all5_mapped.index}).drop_duplicates(
        subset="gene_id"
    ).to_csv(out_path, index=False)
    print(f"{out_path}: {all5_mapped.nunique()}개 (5개 데이터셋 전부 유의+방향일관, symbol {len(all5_symbols)}개 중)")

    lit_symbols = load_existing_pdac_literature_symbols()
    union_symbols = all5_symbols | lit_symbols
    union_mapped = _map_to_ensg(union_symbols)
    out_path = OUT_DIR / "selected_genes_all5consistent_pluslit.csv"
    pd.DataFrame({"gene_id": union_mapped.values, "gene_symbol": union_mapped.index}).drop_duplicates(
        subset="gene_id"
    ).to_csv(out_path, index=False)
    print(f"{out_path}: {union_mapped.nunique()}개 "
          f"(all5consistent {len(all5_symbols)} + 기존 PDAC 문헌 {len(lit_symbols)} 합집합 {len(union_symbols)}개 중)")


if __name__ == "__main__":
    main()
