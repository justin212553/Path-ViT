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

[2026-08-04, --single-cohort 추가 — leakage 수정] 위 기본 경로(Stouffer로 TCGA+CPTAC 둘 다
결합)는 "--dataset both"(내부 비교) 전용으로만 써야 한다. external 프로토콜(예: --dataset tcga
--external, 학습에 전혀 안 쓴 CPTAC 전체를 external test로 평가)에 이 both-결합 리스트를
그대로 재사용하면, external test로 쓰이는 반대 코호트 환자 중 상당수(실측 약 60%)가 이미 유전자
선정 단계에서 생존 라벨이 쓰인 상태가 된다 — external "완전 미노출" 전제가 깨지는 명백한 data
leakage(findings_backlog.md 참조). --single-cohort {tcga,cptac}를 주면 그 코호트의 train
split(--dataset {tcga,cptac} --external이 실제로 쓰는 것과 동일한 split)만으로, 반대 코호트
데이터는 아예 로드하지 않고 유전자를 선정한다(Stouffer 결합 없음 — 단일 코호트 z-score 그대로).
출력은 기존 both-결합 산출물과 겹치지 않게 별도 디렉토리(data/rna_gene_selection_{cohort}only/)에
저장된다.

사용법:
    python -m data.select_rnaseq_genes                      # 기본: both 결합, 1000/1500/2000 저장
    python -m data.select_rnaseq_genes --n-genes 1500
    python -m data.select_rnaseq_genes --single-cohort tcga --n-genes 1500   # TCGA->CPTAC external용
    python -m data.select_rnaseq_genes --single-cohort cptac --n-genes 1500  # CPTAC->TCGA external용
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
    both 프로토콜 전용 — external 프로토콜에는 쓰지 말 것(모듈 docstring 참조).
    """
    ds = WSISurvivalDataset(cfg, dataset="both", split="train", with_rna=True)
    cases = ds.items[["case_id", "dataset"]].drop_duplicates()
    return {name: cases.loc[cases["dataset"] == name, "case_id"].tolist() for name in DATASETS}


def _train_case_ids_single(cfg: DataConfig, cohort: str) -> list[str]:
    """--dataset {cohort} --external이 실제로 쓰는 train split(그 코호트 단독 6:2:2)과
    동일한 case 집합을 반환한다 — 반대 코호트는 이 함수 호출 전체에서 아예 참조되지 않는다."""
    ds = WSISurvivalDataset(cfg, dataset=cohort, split="train", with_rna=True)
    return ds.items["case_id"].drop_duplicates().tolist()


def rank_genes_by_train_cox(cfg: DataConfig, single_cohort: str | None = None) -> pd.DataFrame:
    """
    Args:
        single_cohort: None(기본)이면 기존 both-결합(Stouffer) 동작. "tcga" 또는 "cptac"을
            주면 그 코호트의 train split만 사용하고(반대 코호트 파일은 로드조차 하지 않음),
            Stouffer 결합 없이 그 코호트 자신의 Cox z-score를 그대로 순위 기준으로 쓴다 —
            external 프로토콜(그 코호트로 학습 -> 반대 코호트 전체를 external test)에서 쓸
            유전자 리스트 전용.
    """
    datasets = (single_cohort,) if single_cohort else DATASETS
    rna = {name: pd.read_csv(RNA_PATHS[name]).set_index("case_id") for name in datasets}
    os_labels = {name: pd.read_csv(OS_LABEL_PATHS[name]).set_index("case_id") for name in datasets}
    train_cases = (
        {single_cohort: _train_case_ids_single(cfg, single_cohort)}
        if single_cohort else _train_case_ids_by_dataset(cfg)
    )

    common_genes = sorted(set.intersection(*(set(df.columns) for df in rna.values())))
    rows = pd.DataFrame({"gene_id": common_genes})
    z_cols = []

    for name in datasets:
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

    # single_cohort=None: Stouffer meta-analysis(meta_z = sum(z)/sqrt(코호트 수), 레퍼런스와
    # 동일한 단순 결합). single_cohort 지정 시 z_cols에 컬럼이 1개뿐이라 그 코호트 자신의
    # z-score가 그대로 meta_cox_z가 된다(sqrt(1)로 나누는 것과 동일 — 결합이 아니라 통과).
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


def benjamini_hochberg_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 보정. raw p-value를 그대로 유전자 ~2만 개에 threshold로 쓰면
    순수 우연만으로도 (2만 * alpha)개가 "유의"하게 걸린다(예: p<0.05 -> ~1000개) — 다중검정
    보정 없이는 hard threshold가 top-N보다 오히려 더 노이즈에 취약하다. statsmodels 의존성을
    피하기 위해 표준 BH 절차를 직접 구현(정렬 -> q_(i) = min_{j>=i}(p_(j)*m/j), 뒤에서부터
    누적 최소로 단조성 강제).
    """
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate((sorted_p * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(m)
    q[order] = q_sorted
    return q


def build_fdr_threshold_ranking(
    ranked: pd.DataFrame, literature_table: pd.DataFrame, q_threshold: float
) -> pd.DataFrame:
    """top-N(임의의 고정 개수) 대신, 통계적으로 정당화되는 컷오프로 유전자를 고른다.

    문헌 curated 유전자(163개)는 top-N 방식과 동일하게 개별 유의성과 무관하게 항상 포함한다
    (문헌 근거 자체가 포함 사유). 나머지는 BH-FDR q < q_threshold를 만족하는 유전자만 추가한다
    — raw p 대신 FDR로 다중검정을 보정한다(위 benjamini_hochberg_qvalues 참조). 최종 개수는
    고정되지 않고 그 코호트의 실제 신호 강도에 따라 결정된다(top-N과의 핵심 차이).
    """
    guided = build_literature_guided_ranking(ranked, literature_table)
    guided = guided.merge(ranked[["gene_id", "meta_cox_p"]], on="gene_id", how="left", suffixes=("", "_dup"))
    guided["fdr_q"] = benjamini_hochberg_qvalues(guided["meta_cox_p"].to_numpy())
    selected = guided[guided["is_literature_curated"] | (guided["fdr_q"] < q_threshold)].copy()
    return selected.sort_values("rank").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-genes", nargs="+", type=int, default=[1000, 1500, 2000])
    parser.add_argument(
        "--single-cohort", type=str, default=None, choices=["tcga", "cptac"],
        help="지정하면 그 코호트의 train split만으로(반대 코호트 미참조, Stouffer 결합 없음) "
             "유전자를 선정한다 — external 프로토콜(그 코호트로 학습 -> 반대 코호트 전체 "
             "external test) 전용. 출력은 data/rna_gene_selection_{cohort}only/에 저장되어 "
             "기존 both-결합 산출물(data/rna_gene_selection/)과 분리된다.",
    )
    parser.add_argument(
        "--fdr-threshold", type=float, default=None,
        help="주어지면 --n-genes(임의의 고정 개수) 대신, BH-FDR q < 이 값을 만족하는 유전자만 "
             "고른다(문헌 curated 163개는 항상 포함). raw p-value를 그대로 threshold로 쓰면 "
             "유전자 ~2만 개 중 다중검정 보정 없이 우연만으로도 p<0.05가 ~1000개 나오므로 "
             "FDR 보정을 강제한다. 최종 유전자 수는 고정되지 않고 실제 신호 강도에 따라 "
             "달라진다 — 출력 파일명 selected_genes_fdr{threshold}.csv.",
    )
    args = parser.parse_args()

    cfg = DataConfig()
    out_dir = Path(f"data/rna_gene_selection_{args.single_cohort}only") if args.single_cohort else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single_cohort:
        opposite = "cptac" if args.single_cohort == "tcga" else "tcga"
        print(f"Ranking genes by {args.single_cohort}-only train-split Cox score test "
              f"({opposite} not loaded at all — external-protocol gene list)...")
    else:
        print("Ranking genes by train-split univariate Cox score test (TCGA/CPTAC separately, "
              "Stouffer-combined; both-protocol only)...")
    ranked = rank_genes_by_train_cox(cfg, single_cohort=args.single_cohort)
    ranked.to_csv(out_dir / "gene_cox_ranking.csv", index=False)
    print(f"  -> {out_dir / 'gene_cox_ranking.csv'} ({len(ranked)} genes)")

    literature_table = build_literature_table(ranked)
    literature_table.to_csv(out_dir / "literature_curated_genes.csv", index=False)
    n_available = int(literature_table["available"].sum())
    print(f"  -> literature curated genes: {len(literature_table)} total, {n_available} found in common RNA-seq")

    guided = build_literature_guided_ranking(ranked, literature_table)
    guided.to_csv(out_dir / "literature_guided_ranking.csv", index=False)
    print(f"  -> {out_dir / 'literature_guided_ranking.csv'}")

    if args.fdr_threshold is not None:
        fdr_selected = build_fdr_threshold_ranking(ranked, literature_table, args.fdr_threshold)
        out_path = out_dir / f"selected_genes_fdr{args.fdr_threshold:g}.csv"
        fdr_selected[["rank", "gene_id", "is_literature_curated", "fdr_q"]].to_csv(out_path, index=False)
        n_curated_in_sel = int(fdr_selected["is_literature_curated"].sum())
        print(f"  -> {out_path} (q<{args.fdr_threshold:g}: {len(fdr_selected)} genes total, "
              f"{n_curated_in_sel} literature-curated + {len(fdr_selected) - n_curated_in_sel} FDR-passing)")
    else:
        for n in args.n_genes:
            selected = guided.head(n)[["rank", "gene_id", "is_literature_curated"]]
            out_path = out_dir / f"selected_genes_top_{n}.csv"
            selected.to_csv(out_path, index=False)
            n_curated_in_top = int(selected["is_literature_curated"].sum())
            print(f"  -> {out_path} (top {n}, {n_curated_in_top} literature-curated)")


if __name__ == "__main__":
    main()
