"""
RNA-seq 유전자 재선정 — 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)의
scripts/select_rnaseq_gene_features.py 방법론을 우리 데이터로 재구현.

findings_backlog.md 2번 항목. 현재 RNAEncoder는 Bailey 2016 + Moffitt 2015 PDAC subtype
"분류"용 유전자(339개, data/dataset.py::pdac_subtype_gene_ids())를 쓰는데, 이건 애초에
"생존 예측"이 아니라 "subtype 분류"에 최적화된 목록이다. 레퍼런스는 생존 예측에 직접
최적화된 별도 기준으로 유전자를 고른다:

  1. 문헌 기반 curated seed gene(PDAC driver/subtype/EMT/stromal/immune/proliferation/
     hypoxia/DNA damage repair 8개 카테고리, PDAC_LITERATURE_GENE_SETS)
  2. train split(우리는 --dataset both의 train split과 동일한 case 집합 사용,
     data/dataset.py::WSISurvivalDataset로 가져와 실제 학습에 쓰이는 split과 어긋나지
     않게 함 — val/test 라벨은 선정에 전혀 쓰지 않는다) 내부에서 TCGA/CPTAC 각각
     독립적으로 gene별 univariate Cox score test 수행
  3. 두 코호트의 Cox z-score를 Stouffer 방식으로 결합(meta_z = sum(z) / sqrt(n_datasets),
     레퍼런스와 동일한 단순 결합 — 코호트별 가중치 없음)
  4. 최종 순위: 공통 RNA-seq에 존재하는 curated gene을 먼저 배치(그 안에서는 Cox 순위로
     정렬), 남는 자리는 나머지 유전자의 Cox 순위로 채움
  5. 상위 1000/1500/2000개를 저장 — data/dataset.py::literature_guided_gene_ids()로 로드

Cox score test는 lifelines로 유전자 18,879개를 하나씩 fitting하면 느리므로, 레퍼런스와
동일하게 벡터화된 score test(효율적 점수 U, Fisher 정보 I, z = U/sqrt(I))로 전체 유전자를
한 번에 계산한다 — 결과는 표준 Cox partial likelihood의 score test와 동일하다.

사용법:
    python -m data.select_rnaseq_genes                      # 기본: 1000/1500/2000 저장
    python -m data.select_rnaseq_genes --n-genes 1500
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from config import DataConfig
from data.dataset import RNA_PATHS, OS_LABEL_PATHS, COMMON_GENES_PATH, WSISurvivalDataset

OUT_DIR = Path("data/rna_gene_selection")
DATASETS = ("tcga", "cptac")

# 레퍼런스 scripts/select_rnaseq_gene_features.py::PDAC_LITERATURE_GENE_SETS 그대로 이식.
# PDAC driver/subtype/EMT/stromal/immune/proliferation/hypoxia/DNA damage repair 8개
# 카테고리, 문헌 근거는 Method.md 참조(Collisson 2011, Moffitt 2015, Bailey 2016,
# Waddell 2015, TCGA 2017, Puleo 2018).
PDAC_LITERATURE_GENE_SETS = {
    "core_driver_tumor_suppressor": [
        "KRAS", "TP53", "CDKN2A", "CDKN2B", "SMAD4", "ARID1A", "KDM6A", "RNF43",
        "GNAS", "TGFBR2", "STK11", "SMARCA4", "PIK3CA", "PTEN", "BRAF", "MYC",
    ],
    "dna_damage_repair_therapy": [
        "BRCA1", "BRCA2", "PALB2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51",
        "MLH1", "MSH2", "MSH6", "PMS2", "ERCC1",
    ],
    "classical_pancreatic_progenitor": [
        "GATA6", "HNF1A", "HNF4A", "HNF4G", "FOXA2", "FOXA3", "PDX1", "MNX1",
        "ONECUT1", "ONECUT2", "KRT19", "EPCAM", "CDH1", "MUC1", "MUC5AC",
        "CEACAM5", "CEACAM6", "CLDN4", "CLDN18", "TFF1", "TFF2", "AGR2",
    ],
    "basal_squamous_mesenchymal": [
        "KRT5", "KRT6A", "KRT6B", "KRT14", "KRT17", "KRT81", "TP63", "KLF5",
        "S100A2", "S100A4", "SERPINB3", "SERPINB4", "VIM", "CDH2", "ZEB1",
        "ZEB2", "SNAI1", "SNAI2", "TWIST1", "ITGA6", "LAMC2",
    ],
    "stroma_ecm_invasion": [
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2",
        "COL6A3", "FN1", "SPARC", "POSTN", "THBS1", "ACTA2", "TAGLN", "FAP",
        "ITGA2", "ITGA3", "ITGB1", "ITGB4", "MMP2", "MMP7", "MMP9", "MMP11",
        "MMP14", "PLAU", "PLAUR", "LOX", "LUM", "DCN", "BGN", "MET",
    ],
    "immune_inflammation_tgf_beta": [
        "CD274", "PDCD1", "CTLA4", "CD8A", "CD8B", "CD3D", "CD3E", "FOXP3",
        "CD68", "CD163", "LYZ", "CXCL12", "CXCR4", "CXCL8", "IL6", "IL6R",
        "STAT3", "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "CCL2", "CCR2", "CSF1", "CSF1R",
    ],
    "proliferation_cell_cycle_apoptosis": [
        "MKI67", "TOP2A", "CCNB1", "CCND1", "CCNE1", "CDK1", "CDK2", "BIRC5",
        "AURKA", "AURKB", "PLK1", "MCM2", "MCM4", "MCM6", "PCNA", "BCL2", "BAX", "CASP3",
    ],
    "hypoxia_metabolism_acinar_program": [
        "HIF1A", "VEGFA", "CA9", "SLC2A1", "LDHA", "HK2", "ENO1", "ALDOA",
        "PNLIP", "CPA1", "CPA2", "CPB1", "CTRB1", "CTRB2", "CLPS", "PRSS1", "REG1A", "REG1B",
    ],
}


def cox_score_test_matrix(
    x: np.ndarray, time: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    전체 유전자(열)에 대해 벡터화된 Cox partial-likelihood score test.

    Args:
        x:     (N, G) — 환자 x 유전자 z-score 행렬
        time:  (N,)   — OS_time
        event: (N,)   — OS_event (1=사망)
    Returns:
        z, chi2_stat, p_value: 각 (G,) — 유전자별 score test 통계량
    """
    order = np.argsort(-time)
    x = x[order].astype(np.float64, copy=False)
    event = event[order].astype(bool, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # risk set(현재 시점에 아직 생존/미관측인 환자 집합)의 누적 평균/분산 — time 내림차순
    # 정렬 후 앞에서부터 누적하면 매 시점의 risk set이 "지금까지의 전체"가 된다.
    risk_count = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    risk_mean = np.cumsum(x, axis=0) / risk_count[:, None]
    risk_var = np.cumsum(x * x, axis=0) / risk_count[:, None] - risk_mean**2
    risk_var = np.clip(risk_var, 1e-12, None)

    event_x, event_mean, event_var = x[event], risk_mean[event], risk_var[event]
    u = (event_x - event_mean).sum(axis=0)       # 효율적 점수(efficient score)
    info = event_var.sum(axis=0)                  # Fisher 정보
    z = u / np.sqrt(np.clip(info, 1e-12, None))
    chi2_stat = z * z
    p_value = chi2.sf(chi2_stat, df=1)
    return z, chi2_stat, p_value


def _train_case_ids_by_dataset(cfg: DataConfig) -> dict[str, list[str]]:
    """--dataset both의 train split과 동일한 case 집합을 코호트별로 분리해 반환.

    실제 학습에 쓰이는 split과 완전히 동일한 기준(WSISurvivalDataset)을 재사용해,
    유전자 선정이 val/test case의 생존 라벨을 보지 않게 한다(레퍼런스와 동일 원칙).
    """
    ds = WSISurvivalDataset(cfg, dataset="both", split="train", with_rna=True)
    cases = ds.items[["case_id", "dataset"]].drop_duplicates()
    return {name: cases.loc[cases["dataset"] == name, "case_id"].tolist() for name in DATASETS}


def rank_genes_by_train_cox(cfg: DataConfig) -> pd.DataFrame:
    rna = {name: pd.read_csv(RNA_PATHS[name]).set_index("case_id") for name in DATASETS}
    os_labels = {name: pd.read_csv(OS_LABEL_PATHS[name]).set_index("case_id") for name in DATASETS}
    train_cases = _train_case_ids_by_dataset(cfg)

    common_genes = sorted(set.intersection(*(set(df.columns) for df in rna.values())))
    rows = pd.DataFrame({"gene_id": common_genes})
    z_cols = []

    for name in DATASETS:
        cases = [c for c in train_cases[name] if c in rna[name].index and c in os_labels[name].index]
        x = rna[name].loc[cases, common_genes].to_numpy(dtype=np.float64)
        time = os_labels[name].loc[cases, "OS_time"].to_numpy(dtype=np.float64)
        event = os_labels[name].loc[cases, "OS_event"].to_numpy(dtype=np.int64)

        rows[f"{name}_train_n"] = len(cases)
        rows[f"{name}_train_events"] = int(event.sum())
        z, chi2_stat, p_value = cox_score_test_matrix(x, time, event)
        rows[f"{name}_cox_z"] = z
        rows[f"{name}_cox_p"] = p_value
        z_cols.append(f"{name}_cox_z")
        print(f"  {name}: train n={len(cases)}, events={int(event.sum())}")

    # Stouffer meta-analysis: meta_z = sum(z) / sqrt(코호트 수) — 레퍼런스와 동일한 단순 결합
    z_matrix = rows[z_cols].to_numpy(dtype=np.float64)
    rows["meta_cox_z"] = z_matrix.sum(axis=1) / np.sqrt(z_matrix.shape[1])
    rows["meta_cox_p"] = 2.0 * norm.sf(np.abs(rows["meta_cox_z"]))
    rows["direction"] = np.where(rows["meta_cox_z"] >= 0, "higher_expr_higher_risk", "higher_expr_lower_risk")
    rows["_abs_z"] = rows["meta_cox_z"].abs()

    rows = rows.sort_values(["meta_cox_p", "_abs_z"], ascending=[True, False]).drop(columns="_abs_z").reset_index(drop=True)
    rows.insert(0, "rank", np.arange(1, len(rows) + 1))
    return rows


def build_literature_table(ranked: pd.DataFrame) -> pd.DataFrame:
    common_genes = pd.read_csv(COMMON_GENES_PATH).drop_duplicates(subset="gene_name", keep="first")
    name_to_id = common_genes.set_index("gene_name")["gene_id"]
    rank_lookup = ranked.set_index("gene_id").to_dict(orient="index")

    records, seen = [], set()
    for category, symbols in PDAC_LITERATURE_GENE_SETS.items():
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            gene_id = name_to_id.get(symbol)
            info = rank_lookup.get(gene_id, {}) if gene_id is not None else {}
            records.append({
                "gene_symbol": symbol, "category": category, "gene_id": gene_id,
                "available": gene_id is not None and gene_id in rank_lookup,
                "cox_rank": info.get("rank"), "meta_cox_z": info.get("meta_cox_z"),
            })
    table = pd.DataFrame(records)
    return table.sort_values(["available", "cox_rank", "gene_symbol"], ascending=[False, True, True], na_position="last").reset_index(drop=True)


def build_literature_guided_ranking(ranked: pd.DataFrame, literature_table: pd.DataFrame) -> pd.DataFrame:
    curated_ids = (
        literature_table[literature_table["available"]]
        .sort_values(["cox_rank", "gene_symbol"])["gene_id"].drop_duplicates().tolist()
    )
    curated_set = set(curated_ids)
    cox_ids = [g for g in ranked["gene_id"].tolist() if g not in curated_set]
    ordered = curated_ids + cox_ids

    guided = ranked.set_index("gene_id").loc[ordered].reset_index().rename(columns={"rank": "cox_only_rank"})
    guided["is_literature_curated"] = guided["gene_id"].isin(curated_set)
    guided.insert(0, "rank", np.arange(1, len(guided) + 1))
    return guided


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-genes", nargs="+", type=int, default=[1000, 1500, 2000])
    args = parser.parse_args()

    cfg = DataConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Ranking genes by train-split univariate Cox score test (TCGA/CPTAC separately)...")
    ranked = rank_genes_by_train_cox(cfg)
    ranked.to_csv(OUT_DIR / "gene_cox_ranking.csv", index=False)
    print(f"  -> {OUT_DIR / 'gene_cox_ranking.csv'} ({len(ranked)} genes)")

    literature_table = build_literature_table(ranked)
    literature_table.to_csv(OUT_DIR / "literature_curated_genes.csv", index=False)
    n_available = int(literature_table["available"].sum())
    print(f"  -> literature curated genes: {len(literature_table)} total, {n_available} found in common RNA-seq")

    guided = build_literature_guided_ranking(ranked, literature_table)
    guided.to_csv(OUT_DIR / "literature_guided_ranking.csv", index=False)
    print(f"  -> {OUT_DIR / 'literature_guided_ranking.csv'}")

    for n in args.n_genes:
        selected = guided.head(n)[["rank", "gene_id", "is_literature_curated"]]
        out_path = OUT_DIR / f"selected_genes_top_{n}.csv"
        selected.to_csv(out_path, index=False)
        n_curated_in_top = int(selected["is_literature_curated"].sum())
        print(f"  -> {out_path} (top {n}, {n_curated_in_top} literature-curated)")


if __name__ == "__main__":
    main()
