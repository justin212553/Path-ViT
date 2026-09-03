"""
TCGA-PAAD RNA-seq 유전자 선정 — 고분산(variance) 버전, single-cohort(TCGA-only) 전용.

scripts/select_brca_rna_genes.py의 방법론(생존 라벨을 전혀 안 보고, train split 내부에서
발현량 분산이 큰 유전자 상위 N개)을 PAAD에 재현한다. data/select_rnaseq_genes.py(Cox 기반)와
동일하게 --single-cohort 방식만 지원한다 — external 프로토콜(TCGA로 학습 -> CPTAC 전체
external test)에서 CPTAC 라벨/데이터를 전혀 참조하지 않아야 하기 때문(반대 코호트 파일 자체를
안 읽는다).

[2026-09-03, 배경] BRCA에서 확인된 것: Cox 기반 선택(생존 라벨을 직접 봄)은 fold 경계를 넘는
구조적 leak(고정 단일 split을 모든 k-fold에 재사용)에 크게 취약했지만(findings_backlog.md
2026-09-03), variance 기반 선택(생존 라벨을 아예 안 봄)은 같은 구조적 문제를 안고도 leak에
훨씬 덜 취약했다(k-fold 재검증에서 오히려 제일 좋은 성능) — 라벨이 feature selection에
새어들어가는 직접적인 경로 자체가 없기 때문이라는 가설. PAAD(TCGA 152명, CPTAC 144명)는
BRCA(1057명)보다 코호트가 훨씬 작아 1500개를 그대로 가져가는 게 무리일 수 있다는 판단 하에
100/250/500/1000/1500 단계별로 비교한다(사용자 지시).

data/select_rnaseq_genes.py::_train_case_ids_single()을 그대로 재사용해 Cox 버전과 정확히
동일한 train case 집합으로 분산을 계산한다 — 두 방법론을 공정하게 비교하기 위함.

분산은 raw(z-score 이전) log2(FPKM-UQ+1) 값으로 계산해야 한다(z-scored 값은 전체 코호트
기준으로 이미 분산=1로 정규화돼 있어 순위를 매길 수 없음, scripts/select_brca_rna_genes.py와
동일한 원칙) — data/rna_{tcga,cptac}.csv는 이미 z-scored라 못 쓰고,
data/extract_rna_clinical.py::extract_dataset()을 재호출해 raw log2 행렬을 다시 얻는다(로컬
캐시된 GDC raw RNA tsv를 다시 읽음 — 네트워크 호출 없음, file_case_map.csv 캐시 재사용).
최종 학습에 쓸 값 자체는 그대로 data/rna_{tcga,cptac}.csv(z-scored)에서 선택된 유전자 컬럼만
골라 쓰면 된다(유전자별 독립 z-score라 열을 나중에 서브셋해도 값이 바뀌지 않음).

출력:
    data/rna_gene_selection_variance_{cohort}only/gene_variance_ranking.csv
    data/rna_gene_selection_variance_{cohort}only/selected_genes_top_{n}.csv

사용법:
    python -m data.select_rnaseq_genes_variance --single-cohort tcga --n-genes 100 250 500 1000 1500
"""
import argparse
from pathlib import Path

from config import DataConfig
from data.extract_rna_clinical import extract_dataset
from data.select_rnaseq_genes import _train_case_ids_single


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-cohort", type=str, required=True, choices=["tcga", "cptac"],
                         help="분산을 계산할 코호트 — 반대 코호트 파일은 아예 안 읽는다("
                              "external 프로토콜에서 leak-free 보장).")
    parser.add_argument("--n-genes", nargs="+", type=int, default=[100, 250, 500, 1000, 1500])
    args = parser.parse_args()

    cfg = DataConfig()
    train_cases = _train_case_ids_single(cfg, args.single_cohort)
    print(f"train case 수({args.single_cohort} 단일, 반대 코호트 미참조): {len(train_cases)}")

    raw_log2, _clinical, _name_map = extract_dataset(args.single_cohort)
    ids = [c for c in train_cases if c in raw_log2.index]
    print(f"분산 계산에 실제로 쓰인 train case 수: {len(ids)}  (유전자 수: {raw_log2.shape[1]})")

    train_raw = raw_log2.loc[ids]
    variance = train_raw.var(axis=0, ddof=0).sort_values(ascending=False)
    ranking = variance.reset_index()
    ranking.columns = ["gene_id", "train_variance"]
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    out_dir = Path(f"data/rna_gene_selection_variance_{args.single_cohort}only")
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_dir / "gene_variance_ranking.csv", index=False)
    print(f"저장: {out_dir / 'gene_variance_ranking.csv'}  ({len(ranking)} genes)")

    for n in args.n_genes:
        selected = ranking.head(n)[["rank", "gene_id"]]
        out_path = out_dir / f"selected_genes_top_{n}.csv"
        selected.to_csv(out_path, index=False)
        print(f"저장: {out_path}  (top {n})")


if __name__ == "__main__":
    main()
