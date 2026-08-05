"""
PurIST(Purity Independent Subtyping of Tumors, Rashid et al. 2020 Clin Cancer Res 26(1):82-92,
공식 구현 github.com/naimurashid/PurIST) basal-like/classical 단일-샘플 분류기를 TCGA/CPTAC
각 case에 결정론적으로 적용한다.

[왜 필요한가] 지금까지 --rna-genes literature_*_tcga_only는 TCGA train split(n=91)만으로
Cox-score 유전자 순위를 새로 매기는 데이터 기반 절차라, 유전자 개수(500/1000/1500/fdr0.1)를
바꿔도 external(CPTAC) c-index가 0.57~0.59대에서 벗어나지 못했다(findings_backlog.md 15번
항목 후속). PurIST는 반대로 어떤 데이터에도 fit하지 않는 고정 계수 분류기다 — 8개 유전자쌍
(TSP, tumor-specific pair)의 within-sample log2(FPKM-UQ+1) 대소 비교만으로 basal-like 확률을
계산하므로 (a) TCGA n=91에 전혀 노출되지 않고(leakage 원천 차단), (b) 원논문이 여러 플랫폼/
tumor purity에서 안정성을 검증한 rank-based 비교라 between-sample 정규화 없이도 코호트 간
batch effect에 상대적으로 robust하다.

[계수 출처] naimurashid/PurIST 패키지 classifier 객체 — TSP 8쌍, intercept=-6.815,
logit = intercept + sum(coef_i * 1[gene1_i > gene2_i]), prob_basal = sigmoid(logit).

출력: data/rna_purist_{tcga,cptac}.csv (case_id, purist_basal_prob)
      purist_basal_prob가 높을수록 basal-like(예후 나쁨), 낮을수록 classical(예후 좋음)
      — Moffitt et al. 2015 Nat Genet.

사용법:
    python -m data.compute_purist_subtype
    python -m data.compute_purist_subtype --dataset cptac
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.extract_rna_clinical import (
    CLINIC_ROOTS,
    NA_VALUES,
    RNA_ROOTS,
    _gene_id_to_name,
    _list_rna_files,
    _load_or_query_file_case_map,
    _load_qc_flagged_cases,
    _read_tpm,
)

TSP_PAIRS = [
    ("GPR87", "REG4"),
    ("KRT6A", "ANXA10"),
    ("BCAR3", "GATA6"),
    ("PTGES", "CLDN18"),
    ("ITGA3", "LGALS4"),
    ("C16orf74", "DDC"),
    ("S100A2", "SLC40A1"),
    ("KRT5", "CLRN3"),
]
TSP_COEF = [1.994, 2.031, 1.618, 0.922, 1.059, 0.929, 2.505, 0.485]
INTERCEPT = -6.815

OUT_PATHS = {
    "tcga": Path("data/rna_purist_tcga.csv"),
    "cptac": Path("data/rna_purist_cptac.csv"),
}


def _symbol_to_gene_id(tsv_path: Path) -> dict:
    name_map = _gene_id_to_name(tsv_path)  # gene_id(ENSG) -> gene_name(HUGO), protein_coding만
    needed = {g for pair in TSP_PAIRS for g in pair}
    symbol_to_id = {}
    for gid, name in name_map.items():
        if name in needed and name not in symbol_to_id:
            symbol_to_id[name] = gid
    missing = needed - set(symbol_to_id)
    if missing:
        raise ValueError(f"PurIST 유전자를 protein_coding 목록에서 찾지 못함: {sorted(missing)}")
    return symbol_to_id


def compute_for_dataset(dataset: str, keep_qc_flagged: bool = False) -> pd.DataFrame:
    root = RNA_ROOTS[dataset]
    files = _list_rna_files(root)
    file_map = _load_or_query_file_case_map(dataset, root, files["file_id"].tolist())
    merged = files.merge(file_map, on="file_id", how="inner")
    merged = merged[merged["sample_type"] == "Primary Tumor"]

    if not keep_qc_flagged:
        clinical = pd.read_csv(CLINIC_ROOTS[dataset], sep="\t", na_values=NA_VALUES)
        flagged = _load_qc_flagged_cases(root, clinical)
        merged = merged[~merged["case_id"].isin(flagged)]

    symbol_to_id = _symbol_to_gene_id(merged["tsv_path"].iloc[0])

    rows = []
    for case_id, group in merged.groupby("case_id"):
        series_list = [_read_tpm(p) for p in group["tsv_path"]]
        expr = pd.concat(series_list, axis=1).mean(axis=1)  # gene_id -> log2(FPKM-UQ+1)

        logit = INTERCEPT
        for (g1, g2), coef in zip(TSP_PAIRS, TSP_COEF):
            indicator = 1.0 if expr[symbol_to_id[g1]] > expr[symbol_to_id[g2]] else 0.0
            logit += coef * indicator
        prob_basal = 1.0 / (1.0 + np.exp(-logit))
        rows.append({"case_id": case_id, "purist_basal_prob": prob_basal})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="PurIST basal-like/classical 단일-샘플 분류기 적용")
    parser.add_argument("--dataset", type=str, default="both", choices=["tcga", "cptac", "both"])
    parser.add_argument("--keep-qc-flagged", action="store_true")
    args = parser.parse_args()

    datasets = ["tcga", "cptac"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        df = compute_for_dataset(ds, keep_qc_flagged=args.keep_qc_flagged)
        df.to_csv(OUT_PATHS[ds], index=False)
        n_basal = int(df["purist_basal_prob"].gt(0.5).sum())
        print(
            f"[{ds}] {len(df)} case -> {OUT_PATHS[ds]} "
            f"(basal_prob mean={df['purist_basal_prob'].mean():.3f}, "
            f"basal-like(>0.5) {n_basal}/{len(df)} = {n_basal / len(df):.1%})"
        )


if __name__ == "__main__":
    main()
