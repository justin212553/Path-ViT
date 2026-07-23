"""
TCGA-BRCA RNA-seq 유전자 선정 — data/select_rnaseq_genes.py(TCGA-PAAD/CPTAC)의 대안.

PDAC 문헌 큐레이션(Bailey/Moffitt subtype, literature_1500)은 췌장암 전용이라 유방암에는
그대로 못 쓴다(scripts/extract_brca_rna.py 모듈 docstring 참조) — 대신 순수 통계적 기준인
"고분산 유전자 상위 N개"로 차원을 줄인다. literature_1500과 동일한 원칙(val/test 라벨/분포를
유전자 선정에 전혀 쓰지 않는다)을 지키기 위해, 분산은 scripts/brca_common.py가 정한
train split의 case만으로 계산한다(scripts/train_brca_m4.py, train_brca_m7.py와 반드시
동일한 split/seed를 써야 함 — brca_common.stratified_case_split 참조).

분산은 z-score 이전 원본(log2 FPKM-UQ+1, data/rna_brca_raw_log2.csv)으로 계산한다 — 이미
코호트 전체로 z-score된 값(data/rna_brca.csv)은 유전자별 분산이 전부 1이라 순위를 매길 수
없다. 최종적으로 학습에 쓸 값 자체는 그대로 data/rna_brca.csv(z-scored)에서 선택된 유전자
컬럼만 골라 쓰면 된다(유전자별 독립 z-score라 열을 나중에 서브셋해도 값이 바뀌지 않음).

출력:
    data/brca_rna_gene_selection/gene_variance_ranking.csv   전체 유전자 분산 순위(train만)
    data/brca_rna_gene_selection/selected_genes_top_{n}.csv  상위 n개 gene_id

사용법:
    python -m scripts.select_brca_rna_genes                 # 기본: seed 42, top 1500
    python -m scripts.select_brca_rna_genes --n-genes 1000 1500 2000
"""
import argparse
from pathlib import Path

import pandas as pd

from scripts.brca_common import RNA_RAW_LOG2_PATH, load_case_table

OUT_DIR = Path("data/brca_rna_gene_selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                         help="brca_common.stratified_case_split과 동일 seed를 써야 train "
                              "split이 학습 스크립트와 어긋나지 않는다.")
    parser.add_argument("--n-genes", nargs="+", type=int, default=[1500])
    args = parser.parse_args()

    cases = load_case_table(args.seed)
    train_cases = cases.loc[cases["split"] == "train", "case_id"].tolist()
    print(f"전체 case 수: {len(cases)}  (train={len(train_cases)}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())})")

    raw = pd.read_csv(RNA_RAW_LOG2_PATH).set_index("case_id")
    train_raw = raw.loc[raw.index.intersection(train_cases)]
    print(f"분산 계산에 쓰인 train case 수: {len(train_raw)}  (유전자 수: {train_raw.shape[1]})")

    variance = train_raw.var(axis=0, ddof=0).sort_values(ascending=False)
    ranking = variance.reset_index()
    ranking.columns = ["gene_id", "train_variance"]
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUT_DIR / "gene_variance_ranking.csv", index=False)
    print(f"저장: {OUT_DIR / 'gene_variance_ranking.csv'}  ({len(ranking)} genes)")

    for n in args.n_genes:
        selected = ranking.head(n)[["rank", "gene_id"]]
        out_path = OUT_DIR / f"selected_genes_top_{n}.csv"
        selected.to_csv(out_path, index=False)
        print(f"저장: {out_path}  (top {n})")


if __name__ == "__main__":
    main()
