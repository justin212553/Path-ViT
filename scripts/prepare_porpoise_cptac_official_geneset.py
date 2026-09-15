"""
PORPOISE 공식 유전자 세트(porpoise/datasets_csv_mutsig/tcga_paad_all_clean.csv.zip, mahmoodlab
공식 repo에서 그대로 가져온 파일)로 학습한 모델(sbatch/run_porpoise_official_geneset_paad_mmf_
2seed_array_hpc.sh)의 external validation cohort로 쓸 CPTAC-PDAC genomic CSV를 만든다.

배경: 기존 "official" 재현(results_true_resnet50_mmf)이 사실은 공식 유전자 세트가 아니라
19,962개 전체 protein-coding 유전자를 쓰고 있었다는 게 밝혀져(2026-09-14), datasets_csv_mutsig의
진짜 공식 파일(166환자, RNA-seq 1553유전자 + CNV 82유전자 + mutation 14유전자, 전부 gene symbol
기준)로 다시 학습하기로 했다. 이 스크립트는 그 학습된 모델을 CPTAC-PDAC에 평가하기 위한 동일
포맷의 CSV를 만든다.

**커버리지 갭과 근사 처리 (전부 사용자 승인, 2026-09-14):**
- 공식 데이터를 요청 시점에 새로 받아오는 것(raw CPTAC mutation call을 추가로 확보하는 것 등)은
  기각되었다 — "우리가 RNA 세트를 제3의 소스로 채운 것을 노벨티로 미는 이상, RNA 세트가 어떻게
  구성되어 있는지 또한 모델의 일부"라는 원칙에 따라, PORPOISE 공식 유전자 세트도 있는 그대로
  받아들이고 없는 건 그냥 근사한다.
- RNA-seq 1553개 중 1543개는 data/rna_cptac.csv(이미 cohort-내 z-score됨)에서 symbol->Ensembl
  매핑(data/common_genes.csv)을 거쳐 그대로 가져온다. 나머지 10개(CCL3L1, FCGR2C, HMGN2P46,
  KIR2DS4, LILRA3, MALAT1, MDS2, PPBPP2, TCL6, TDGF1P3 — 대부분 유사유전자/pseudogene)는 심볼
  자체가 우리 GDC harmonized 데이터의 protein-coding 유전자 목록에 없어 z-score 평균값인 0.0으로
  채운다.
- CNV 82개는 전부 0(GISTIC 관례상 copy-neutral)으로 채운다. 처음엔 data/cnv_cptac.csv(163유전자)와
  교집합 5개만 쓰려 했으나, 그 5개조차 단위가 다르다(공식 파일은 GISTIC 상대 점수 -1~2, 우리
  데이터는 raw absolute copy number 1~8) — 변환 없이 섞어 쓰면 근사가 아니라 오염이라 판단해
  82개 전부를 동일하게 neutral 처리한다.
- Mutation 14개 중 우리가 gene-level로 실제 갖고 있는 4개(KRAS, TP53, SMAD4, CDKN2A,
  data/mutations_cptac.csv)는 그대로 쓰고, 나머지 10개(ARID1A, CSMD2, GNAS, MUC16, OBSCN,
  PCDH15, RAS, RNF43, RYR1, TTN)는 0(미돌연변이)으로 근사한다.

이 세 가지 근사는 논문 Methods/Limitations에 반드시 명시해야 한다.

컬럼 순서는 학습에 쓴 공식 CSV와 정확히 동일해야 한다(PorpoiseMMF의 SNN은 순서 무관하게 크기만
맞으면 되지만, 값의 의미가 훈련 시점과 같은 컬럼 위치에 있어야 하므로).

산출물: porpoise/datasets_csv_mutsig/cptac_paad_external_clean.csv.zip

사용법:
    python -m scripts.prepare_porpoise_cptac_official_geneset
"""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from data.dataset import _load_slide_index

PORPOISE_ROOT = Path("porpoise")
OFFICIAL_TRAIN_CSV = PORPOISE_ROOT / "datasets_csv_mutsig" / "tcga_paad_all_clean.csv.zip"
TRUE_RESNET50_PT_DIR = Path("data/porpoise_style_features/cptac/pt_files")
CPTAC_PATCHES_ROOT = Path("data/patches_cptac_uni2native")
OUT_NAME = "cptac_paad_external_clean.csv.zip"

COMMON_GENES_PATH = Path("data/common_genes.csv")
RNA_CPTAC_PATH = Path("data/rna_cptac.csv")
MUT_CPTAC_PATH = Path("data/mutations_cptac.csv")
CLINICAL_CSV_PATH = Path("data/clinical_cptac.csv")


def main():
    print("1) 공식 학습 CSV에서 정확한 컬럼 순서(메타데이터 + 1553 RNA-seq + 82 CNV + 14 mutation) 로드...")
    with zipfile.ZipFile(OFFICIAL_TRAIN_CSV) as zf:
        with zf.open(zf.namelist()[0]) as f:
            official = pd.read_csv(f, nrows=1)
    all_cols = official.columns.tolist()
    meta_cols = ['case_id', 'slide_id', 'site', 'is_female', 'oncotree_code', 'age',
                 'survival_months', 'censorship', 'train']
    rnaseq_genes = [c[:-len('_rnaseq')] for c in all_cols if c.endswith('_rnaseq')]
    cnv_genes = [c[:-len('_cnv')] for c in all_cols if c.endswith('_cnv')]
    mut_genes = [c[:-len('_mut')] for c in all_cols if c.endswith('_mut')]
    gene_cols_ordered = [c for c in all_cols if c not in meta_cols]
    print(f"   RNA-seq {len(rnaseq_genes)}, CNV {len(cnv_genes)}, mutation {len(mut_genes)}, 총 유전자 컬럼 {len(gene_cols_ordered)}")

    print("2) symbol -> Ensembl 매핑, 우리 CPTAC RNA-seq에서 값 채우기...")
    common = pd.read_csv(COMMON_GENES_PATH).drop_duplicates(subset="gene_name", keep="first")
    name_to_id = common.set_index("gene_name")["gene_id"]
    rna_cptac = pd.read_csv(RNA_CPTAC_PATH).set_index("case_id")

    rna_values = {}
    n_rna_missing = 0
    for symbol in rnaseq_genes:
        ens_id = name_to_id.get(symbol)
        if ens_id is not None and ens_id in rna_cptac.columns:
            rna_values[symbol] = rna_cptac[ens_id]
        else:
            n_rna_missing += 1
            rna_values[symbol] = pd.Series(0.0, index=rna_cptac.index)
    print(f"   RNA-seq: {len(rnaseq_genes) - n_rna_missing}/{len(rnaseq_genes)}개 실측값, {n_rna_missing}개 0.0(z-score 평균)으로 근사")

    print("3) CNV 82개는 전부 0(copy-neutral)으로 근사 (단위 불일치로 실측 데이터 사용 안 함)...")
    cnv_values = {symbol: pd.Series(0.0, index=rna_cptac.index) for symbol in cnv_genes}

    print("4) Mutation 14개 중 실측 4개(KRAS/TP53/SMAD4/CDKN2A) 사용, 나머지 10개는 0(미돌연변이)으로 근사...")
    mut_cptac = pd.read_csv(MUT_CPTAC_PATH).set_index("case_id")
    mut_values = {}
    n_mut_real = 0
    for symbol in mut_genes:
        col = f"{symbol}_mut"
        if col in mut_cptac.columns:
            mut_values[symbol] = mut_cptac[col].reindex(rna_cptac.index).fillna(0.0)
            n_mut_real += 1
        else:
            mut_values[symbol] = pd.Series(0.0, index=rna_cptac.index)
    print(f"   Mutation: {n_mut_real}/{len(mut_genes)}개 실측값, {len(mut_genes) - n_mut_real}개 0(미돌연변이)으로 근사")

    print("5) true-ResNet50(PORPOISE 스펙) pt_files 및 슬라이드->환자 매핑 스캔...")
    if not TRUE_RESNET50_PT_DIR.is_dir():
        raise FileNotFoundError(f"{TRUE_RESNET50_PT_DIR} 없음 — sbatch/extract_porpoise_style_features_cptac_hpc.sh 먼저 완료 필요")
    available_slide_ids = {p.stem for p in TRUE_RESNET50_PT_DIR.glob("*.pt")}
    slide_index = _load_slide_index(CPTAC_PATCHES_ROOT)
    slide_index = slide_index[slide_index["slide_id"].isin(available_slide_ids)]
    slide_df = slide_index[["case_id", "slide_id"]].drop_duplicates()
    print(f"   feature 있는 슬라이드: {len(slide_df)}개 (case {slide_df['case_id'].nunique()}명)")

    clinical_final = pd.read_csv(CLINICAL_CSV_PATH)
    common_cases = sorted(
        set(rna_cptac.index) & set(clinical_final["case_id"]) & set(slide_df["case_id"])
    )
    print(f"6) RNA/clinical/WSI 전부 있는 공통 case: {len(common_cases)}명")
    if len(common_cases) == 0:
        raise RuntimeError("공통 case가 0명입니다.")

    clinical_idx = clinical_final.set_index("case_id")
    meta_records = []
    for case_id in common_cases:
        row = clinical_idx.loc[case_id]
        meta_records.append({
            "case_id": case_id,
            "site": "CPTAC",
            "is_female": 1.0 if str(row["sex"]).lower().startswith("f") else 0.0,
            "oncotree_code": "PAAD",
            "age": float(row["age_years"]),
            "survival_months": float(row["OS_time"]) / 30.44,
            "censorship": 1.0 - float(row["OS_event"]),
            "train": 1.0,
        })
    meta_df = pd.DataFrame(meta_records).set_index("case_id")

    gene_df = pd.DataFrame(index=rna_cptac.index)
    for symbol in rnaseq_genes:
        gene_df[f"{symbol}_rnaseq"] = rna_values[symbol]
    for symbol in cnv_genes:
        gene_df[f"{symbol}_cnv"] = cnv_values[symbol]
    for symbol in mut_genes:
        gene_df[f"{symbol}_mut"] = mut_values[symbol]
    gene_df = gene_df.loc[common_cases, gene_cols_ordered]

    slide_for_case = slide_df[slide_df["case_id"].isin(common_cases)]
    merged = slide_for_case.merge(meta_df.reset_index(), on="case_id", how="inner") \
                            .merge(gene_df.reset_index(), on="case_id", how="inner")
    merged = merged[["case_id", "slide_id"] + list(meta_df.columns) + gene_cols_ordered]
    print(f"7) 최종 행 수(슬라이드 단위): {len(merged)} (case {merged['case_id'].nunique()}명, "
          f"event={int(meta_df['censorship'].eq(0).sum())})")

    csv_dir = PORPOISE_ROOT / "datasets_csv_mutsig"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path = csv_dir / OUT_NAME
    tmp_csv = csv_dir / OUT_NAME.replace(".zip", "")
    merged.to_csv(tmp_csv, index=False)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_csv, arcname=tmp_csv.name)
    tmp_csv.unlink()
    print(f"8) 저장: {out_path}")


if __name__ == "__main__":
    main()
