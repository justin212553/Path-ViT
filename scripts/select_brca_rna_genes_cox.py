"""
TCGA-BRCA RNA-seq 유전자 선정 — 생존 라벨 기반(Cox score test) 버전.

scripts/select_brca_rna_genes.py(고분산 상위 N개, 생존 라벨을 전혀 안 봄)의 대안으로,
data/select_rnaseq_genes.py(TCGA-PAAD/CPTAC)와 동일한 univariate Cox partial-likelihood
score test 방법론을 BRCA에 재현한다(사용자 지시: "RNA를 생존 라벨을 보고 1500개 선정" —
문헌 큐레이션은 PDAC 전용이라 지금 당장은 스킵, 순수 통계 기준만 생존 라벨 기반으로 바꾼다).

data/select_rnaseq_genes.py처럼 두 코호트(TCGA+CPTAC)를 Stouffer로 결합할 필요가 없다 —
BRCA는 단일 코호트라 그 코호트의 train split 안에서 Cox z-score만 계산하면 된다(레퍼런스
데이터셋의 "생존 라벨을 쓰되 val/test는 전혀 보지 않는다" 원칙은 동일하게 지킨다).

scripts/select_brca_rna_genes.py와 동일한 관례로, 분산 계산이 아니라 Cox score test를
raw(z-score 이전) log2 FPKM-UQ+1 값(data/rna_brca_raw_log2.csv)에 대해 수행한다 — Cox test
자체가 스케일에 크게 민감하지 않지만(score test는 순위 기반 위험집합 평균/분산을 쓰므로),
같은 원본 파일을 기준으로 삼아 select_brca_rna_genes.py와 비교 가능하게 유지한다. 최종
학습에 쓰는 값은 그대로 data/rna_brca.csv(z-scored)에서 선택된 컬럼만 골라 쓴다.

[leakage 참고] 여기서 계산하는 train split은 scripts/brca_common.py::load_case_table(seed)
(k-fold 이전의 단일 6:2:2)을 그대로 쓴다 — select_brca_rna_genes.py(고분산 버전)도 동일
관례라 두 유전자 패널이 "같은 split 정의"로 공정하게 비교된다. k-fold 평가 시 fold별로
다시 선정하지 않고 이 고정 패널을 그대로 재사용하는 점도 기존 고분산 버전과 동일 — PAAD의
literature_1500_intersection에서 확인된 것과 같은 종류의 "fold 경계를 넘는 라벨 노출"
caveat이 그대로 적용된다(사용자도 이 점을 인지하고 승인, findings_backlog.md::
RNA 유전자 선정 leakage 참조).

출력:
    data/brca_rna_gene_selection_cox/gene_cox_ranking.csv       전체 유전자 Cox 순위(train만)
    data/brca_rna_gene_selection_cox/selected_genes_top_{n}.csv 상위 n개 gene_id

사용법:
    python -m scripts.select_brca_rna_genes_cox                 # 기본: seed 42, top 1500
    python -m scripts.select_brca_rna_genes_cox --n-genes 1000 1500 2000
"""
import argparse
from pathlib import Path

import pandas as pd

from data.select_rnaseq_genes import cox_score_test_matrix, benjamini_hochberg_qvalues
from scripts.brca_common import RNA_RAW_LOG2_PATH, load_case_table

OUT_DIR = Path("data/brca_rna_gene_selection_cox")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                         help="scripts.select_brca_rna_genes와 동일 관례 — brca_common.load_case_table과 "
                              "동일 seed를 써야 train split이 학습 스크립트와 어긋나지 않는다.")
    parser.add_argument("--n-genes", nargs="+", type=int, default=[1500])
    parser.add_argument(
        "--fdr-threshold", type=float, default=None,
        help="2026-09-02: 주어지면 --n-genes(고정 개수) 대신 BH-FDR q < 이 값을 만족하는 유전자만 "
             "고른다(data/select_rnaseq_genes.py::build_fdr_threshold_ranking과 동일 원리). "
             "top-N 방식은 549 train case/event 61개로 ~2만 유전자를 score test하면 다중검정 "
             "보정 없이 우연만으로도 상위권이 Y염색체 유전자(RBMY1E 등)/후각수용체(OR4K2 등) 같은 "
             "생물학적으로 말이 안 되는 잡음으로 채워지는 게 실측 확인됨 — FDR 보정은 그 잡음을 "
             "걸러내고 실제 신호 강도에 따라 유전자 수가 결정되게 한다(문헌 curated 세트가 없어 "
             "PDAC 버전과 달리 '항상 포함' 로직은 없음). 출력 파일명 selected_genes_fdr{threshold}.csv.",
    )
    args = parser.parse_args()

    cases = load_case_table(args.seed)
    train_cases = cases.loc[cases["split"] == "train"]
    print(f"전체 case 수: {len(cases)}  (train={len(train_cases)}, "
          f"val={int((cases['split']=='val').sum())}, test={int((cases['split']=='test').sum())}, "
          f"external={int((cases['split']=='external').sum())})")

    raw = pd.read_csv(RNA_RAW_LOG2_PATH).set_index("case_id")
    ids = [c for c in train_cases["case_id"] if c in raw.index]
    x = raw.loc[ids].to_numpy(dtype="float64")
    time = train_cases.set_index("case_id").loc[ids, "OS_time"].to_numpy(dtype="float64")
    event = train_cases.set_index("case_id").loc[ids, "OS_event"].to_numpy(dtype="int64")
    print(f"Cox score test에 쓰인 train case 수: {len(ids)}  (event={int(event.sum())})  "
          f"유전자 수: {x.shape[1]}")

    z, chi2_stat, p_value = cox_score_test_matrix(x, time, event)
    ranking = pd.DataFrame({
        "gene_id": raw.columns,
        "cox_z": z,
        "cox_p": p_value,
    })
    ranking["_abs_z"] = ranking["cox_z"].abs()
    ranking = ranking.sort_values(["cox_p", "_abs_z"], ascending=[True, False]).drop(columns="_abs_z").reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["fdr_q"] = benjamini_hochberg_qvalues(ranking["cox_p"].to_numpy())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUT_DIR / "gene_cox_ranking.csv", index=False)
    print(f"저장: {OUT_DIR / 'gene_cox_ranking.csv'}  ({len(ranking)} genes)")

    if args.fdr_threshold is not None:
        selected = ranking[ranking["fdr_q"] < args.fdr_threshold][["rank", "gene_id", "fdr_q"]]
        out_path = OUT_DIR / f"selected_genes_fdr{args.fdr_threshold:g}.csv"
        selected.to_csv(out_path, index=False)
        print(f"저장: {out_path}  (q<{args.fdr_threshold:g}: {len(selected)} genes)")
    else:
        for n in args.n_genes:
            selected = ranking.head(n)[["rank", "gene_id"]]
            out_path = OUT_DIR / f"selected_genes_top_{n}.csv"
            selected.to_csv(out_path, index=False)
            print(f"저장: {out_path}  (top {n})")


if __name__ == "__main__":
    main()
