"""
TCGA-PAAD / CPTAC-PDA RNA-seq + clinical(age/sex) 추출 스크립트.

data/raw/{TCGA,CPTAC}_RNA/<file_uuid>/*.rna_seq.augmented_star_gene_counts.tsv 와
data/raw/{TCGA,CPTAC}_clinic/clinical.tsv 를 case(환자) 단위로 정리해 Method.md 2~3절 스펙의
학습 입력 표(RNA feature matrix + clinical) 를 만든다.

WSI 존재 여부는 확인하지 않는다 — 모든 환자가 WSI를 갖고 있다고 가정하고, RNA + clinical(age/sex)
+ OS 레이블이 모두 있는 case만 최종 후보로 남긴다(실제 WSI 매칭은 별도로 data/dataset.py에서 수행).

절차:
    1. RNA tsv가 들어있는 폴더명은 GDC file UUID이고 case_id(barcode/submitter_id)를 담고 있지
       않으므로, GDC REST API(POST /files)로 file UUID -> case submitter_id / sample_type을
       조회해 data/raw/{DATASET}_RNA/file_case_map.csv 에 캐시한다(이미 있으면 네트워크 없이 재사용).
    2. sample_type == "Primary Tumor"인 파일만 사용한다(정상조직 등 제외). 한 case가 여러 tumor
       RNA 파일을 갖는 경우 TPM을 평균낸다.
    3. clinical.tsv 기반 case-level annotations.txt(entity_type=="case")는 TCGA-PAAD 코호트에
       섞여 있는 것으로 알려진 오분류 케이스(신경내분비종양/정상조직/비-PAAD 등) QC 플래그이므로
       기본적으로 제외한다(--keep-qc-flagged로 끌 수 있음).
    4. gene_type == protein_coding인 유전자만 사용하고 tpm_unstranded 값을 취한다.
    5. TCGA/CPTAC 각각의 protein-coding gene_id 교집합을 공통 유전자셋으로 쓰고, 각 cohort 내부
       z-score 정규화를 적용한다(Method.md 2절: "데이터셋 내부 z-score 정규화 후 사용").
    6. clinical.tsv에서 case당 age_years / sex만 사용한다 — race는 bias 우려로 제외.
       age_years는 demographic.age_at_index(TCGA)를 우선 쓰고, 없으면(CPTAC은 전부 결측)
       diagnoses.age_at_diagnosis(days)/365.25로 대체한다.
    7. data/os_labels_{tcga,cptac}.csv(OS_time/OS_event)와 inner join.

출력:
    data/rna_{tcga,cptac}.csv        case_id + <공통 유전자 ENSG id 컬럼> (cohort 내부 z-scored TPM)
    data/clinical_{tcga,cptac}.csv   case_id, dataset, age_years, sex, OS_time, OS_event
    data/common_genes.csv            gene_id, gene_name (TCGA∩CPTAC protein-coding 유전자 목록)

사용법:
    python -m data.extract_rna_clinical                  # tcga + cptac 모두
    python -m data.extract_rna_clinical --dataset cptac
    python -m data.extract_rna_clinical --keep-qc-flagged
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

RNA_ROOTS = {
    "tcga":  Path("data/raw/TCGA_RNA"),
    "cptac": Path("data/raw/CPTAC_RNA"),
}
CLINIC_ROOTS = {
    "tcga":  Path("data/raw/TCGA_clinic/clinical.tsv"),
    "cptac": Path("data/raw/CPTAC_clinic/clinical.tsv"),
}
OS_LABEL_PATHS = {
    "tcga":  Path("data/os_labels_tcga.csv"),
    "cptac": Path("data/os_labels_cptac.csv"),
}
OUT_RNA_PATHS = {
    "tcga":  Path("data/rna_tcga.csv"),
    "cptac": Path("data/rna_cptac.csv"),
}
OUT_CLINICAL_PATHS = {
    "tcga":  Path("data/clinical_tcga.csv"),
    "cptac": Path("data/clinical_cptac.csv"),
}
COMMON_GENES_PATH = Path("data/common_genes.csv")

NA_VALUES = ["'--"]
GDC_FILES_API = "https://api.gdc.cancer.gov/files"
GDC_BATCH_SIZE = 100
RNA_SUFFIX = ".rna_seq.augmented_star_gene_counts.tsv"


def _list_rna_files(root: Path) -> pd.DataFrame:
    """<root>/<file_uuid>/*.rna_seq.augmented_star_gene_counts.tsv 목록을 file_id, tsv_path로 정리."""
    records = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        tsvs = list(d.glob(f"*{RNA_SUFFIX}"))
        if tsvs:
            records.append({"file_id": d.name, "tsv_path": tsvs[0]})
    return pd.DataFrame(records)


def _query_gdc_file_case_map(file_ids: list) -> pd.DataFrame:
    """GDC REST API로 file UUID -> (case_id=submitter_id, sample_type)을 조회."""
    rows = []
    for i in range(0, len(file_ids), GDC_BATCH_SIZE):
        batch = file_ids[i : i + GDC_BATCH_SIZE]
        payload = {
            "filters": {"op": "in", "content": {"field": "file_id", "value": batch}},
            "fields": "file_id,cases.submitter_id,cases.samples.sample_type",
            "format": "JSON",
            "size": str(len(batch)),
        }
        resp = requests.post(GDC_FILES_API, json=payload, timeout=30)
        resp.raise_for_status()
        for hit in resp.json()["data"]["hits"]:
            case = hit["cases"][0]
            samples = case.get("samples") or [{}]
            rows.append({
                "file_id":     hit["file_id"],
                "case_id":     case["submitter_id"],
                "sample_type": samples[0].get("sample_type"),
            })
    return pd.DataFrame(rows)


def _load_or_query_file_case_map(dataset: str, root: Path, file_ids: list) -> pd.DataFrame:
    """file_case_map.csv 캐시를 재사용하고, 없는 file_id만 GDC API로 조회해 캐시에 추가."""
    cache_path = root / "file_case_map.csv"
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["file_id", "case_id", "sample_type"])

    missing = sorted(set(file_ids) - set(cached["file_id"]))
    if missing:
        print(f"[{dataset}] GDC API로 file_id -> case_id {len(missing)}건 조회 중 ...")
        fresh = _query_gdc_file_case_map(missing)
        cached = pd.concat([cached, fresh], ignore_index=True)
        cached.to_csv(cache_path, index=False)

    return cached[cached["file_id"].isin(file_ids)].reset_index(drop=True)


def _load_qc_flagged_cases(root: Path, clinical: pd.DataFrame) -> set:
    """case-level annotations.txt(entity_type=="case")에 잡힌 case_id(barcode) 집합.

    annotations.txt의 entity_id는 case UUID(clinical.tsv의 cases.case_id)이므로,
    clinical.tsv를 통해 barcode(cases.submitter_id)로 변환한다.
    """
    ann_paths = list(root.rglob("annotations.txt"))
    if not ann_paths:
        return set()

    ann = pd.concat(
        [pd.read_csv(p, sep="\t", dtype=str) for p in ann_paths],
        ignore_index=True,
    ).drop_duplicates(subset=["entity_id"])
    ann = ann[ann["entity_type"] == "case"]
    if ann.empty:
        return set()

    uuid_to_barcode = clinical.drop_duplicates("cases.case_id").set_index("cases.case_id")["cases.submitter_id"]
    flagged = ann["entity_id"].map(uuid_to_barcode).dropna()
    return set(flagged)


def _read_tpm(tsv_path: Path) -> pd.Series:
    """protein_coding 유전자만 남긴 gene_id -> tpm_unstranded Series."""
    df = pd.read_csv(tsv_path, sep="\t", skiprows=1)
    df = df[df["gene_type"] == "protein_coding"]
    return df.set_index("gene_id")["tpm_unstranded"].astype(float)


def _gene_id_to_name(tsv_path: Path) -> pd.Series:
    df = pd.read_csv(tsv_path, sep="\t", skiprows=1)
    df = df[df["gene_type"] == "protein_coding"]
    return df.set_index("gene_id")["gene_name"]


def _extract_age_years(clinical: pd.DataFrame) -> pd.Series:
    cols = ["cases.submitter_id", "demographic.age_at_index", "diagnoses.age_at_diagnosis"]
    cases = clinical[cols].groupby("cases.submitter_id", as_index=False).first()
    age_years = cases["demographic.age_at_index"].astype(float)
    fallback = (cases["diagnoses.age_at_diagnosis"].astype(float) / 365.25).round()
    age_years = age_years.fillna(fallback)
    return pd.Series(age_years.values, index=cases["cases.submitter_id"], name="age_years")


def extract_dataset(dataset: str, keep_qc_flagged: bool = False):
    rna_root = RNA_ROOTS[dataset]
    files = _list_rna_files(rna_root)
    print(f"[{dataset}] RNA tsv 폴더 {len(files)}개 발견")

    file_map = _load_or_query_file_case_map(dataset, rna_root, files["file_id"].tolist())
    merged = files.merge(file_map, on="file_id", how="inner")

    n_before_type = merged["case_id"].nunique()
    merged = merged[merged["sample_type"] == "Primary Tumor"]
    print(f"[{dataset}] sample_type == Primary Tumor 필터: case {n_before_type} -> {merged['case_id'].nunique()}")

    clinical = pd.read_csv(CLINIC_ROOTS[dataset], sep="\t", na_values=NA_VALUES)

    if not keep_qc_flagged:
        flagged = _load_qc_flagged_cases(rna_root, clinical)
        n_before = merged["case_id"].nunique()
        merged = merged[~merged["case_id"].isin(flagged)]
        n_after = merged["case_id"].nunique()
        if n_before != n_after:
            print(f"[{dataset}] QC 플래그(annotations.txt, 'does not meet study protocol' 등) case 제외: "
                  f"{n_before} -> {n_after}")

    gene_series_by_case, name_map = {}, None
    for case_id, group in merged.groupby("case_id"):
        series_list = [_read_tpm(p) for p in group["tsv_path"]]
        if name_map is None:
            name_map = _gene_id_to_name(group["tsv_path"].iloc[0])
        if len(series_list) > 1:
            print(f"[{dataset}] {case_id}: tumor RNA 파일 {len(series_list)}개 평균")
        gene_series_by_case[case_id] = pd.concat(series_list, axis=1).mean(axis=1)

    rna_df = pd.DataFrame(gene_series_by_case).T
    rna_df.index.name = "case_id"

    age_years = _extract_age_years(clinical)
    sex_cols = ["cases.submitter_id", "demographic.sex_at_birth"]
    sex = clinical[sex_cols].groupby("cases.submitter_id", as_index=False).first()
    sex = sex.set_index("cases.submitter_id")["demographic.sex_at_birth"]

    clinical_out = pd.DataFrame({"age_years": age_years, "sex": sex})
    clinical_out.index.name = "case_id"
    n_before_age = len(clinical_out)
    clinical_out = clinical_out.dropna(subset=["age_years", "sex"])
    if len(clinical_out) != n_before_age:
        print(f"[{dataset}] age_years/sex 결측으로 {n_before_age - len(clinical_out)}건 제외")

    os_df = pd.read_csv(OS_LABEL_PATHS[dataset]).set_index("case_id")[["OS_time", "OS_event"]]

    common_cases = sorted(set(rna_df.index) & set(clinical_out.index) & set(os_df.index))
    print(f"[{dataset}] RNA/clinical/OS 모두 존재하는 최종 case 수: {len(common_cases)}")

    rna_out = rna_df.loc[common_cases]
    clinical_final = clinical_out.loc[common_cases].join(os_df.loc[common_cases])
    clinical_final.insert(0, "dataset", dataset)
    clinical_final = clinical_final.reset_index()

    return rna_out, clinical_final, name_map


def main():
    parser = argparse.ArgumentParser(description="TCGA-PAAD / CPTAC-PDA RNA-seq + clinical(age/sex) 추출")
    parser.add_argument("--dataset", type=str, default="both", choices=["tcga", "cptac", "both"])
    parser.add_argument("--keep-qc-flagged", action="store_true",
                         help="TCGA annotations.txt(study protocol 미충족 등) 플래그 case도 포함")
    args = parser.parse_args()

    datasets = ["tcga", "cptac"] if args.dataset == "both" else [args.dataset]

    rna_by_ds, clinical_by_ds, name_map = {}, {}, None
    for ds in datasets:
        rna_out, clinical_out, ds_name_map = extract_dataset(ds, keep_qc_flagged=args.keep_qc_flagged)
        rna_by_ds[ds] = rna_out
        clinical_by_ds[ds] = clinical_out
        if name_map is None:
            name_map = ds_name_map

    if len(rna_by_ds) == 2:
        common_genes = sorted(set(rna_by_ds["tcga"].columns) & set(rna_by_ds["cptac"].columns))
        print(f"TCGA∩CPTAC 공통 protein-coding 유전자 수: {len(common_genes)}")
        gene_table = pd.DataFrame({"gene_id": common_genes, "gene_name": name_map.loc[common_genes].values})
        gene_table.to_csv(COMMON_GENES_PATH, index=False)
    else:
        (ds,) = rna_by_ds.keys()
        common_genes = sorted(rna_by_ds[ds].columns)

    for ds in datasets:
        rna_out = rna_by_ds[ds][common_genes]
        z = (rna_out - rna_out.mean()) / rna_out.std(ddof=0).replace(0, 1.0)
        z.index.name = "case_id"
        z.reset_index().to_csv(OUT_RNA_PATHS[ds], index=False)

        clinical_out = clinical_by_ds[ds]
        clinical_out.to_csv(OUT_CLINICAL_PATHS[ds], index=False)

        n_dead = int(clinical_out["OS_event"].sum())
        print(f"[{ds}] {len(clinical_out)} case -> {OUT_RNA_PATHS[ds]}, {OUT_CLINICAL_PATHS[ds]} "
              f"(event=Dead {n_dead}, censored=Alive {len(clinical_out) - n_dead})")


if __name__ == "__main__":
    main()
