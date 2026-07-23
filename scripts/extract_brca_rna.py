"""
TCGA-BRCA RNA-seq(STAR-counts) 다운로드 + 전처리 — `data/extract_rna_clinical.py`가 TCGA-PAAD에
적용한 것과 동일한 방법론(log2(fpkm_uq_unstranded+1), protein_coding 필터, 코호트 내부 z-score)을
그대로 재현한다. M4(WSI+RNA+Clinical)를 BRCA로 제대로 돌리기 위한 선행 작업.

TCGA-PAAD와 달리 로컬에 미리 받아둔 clinical.tsv/RNA tsv가 없으므로, GDC REST API로 직접
파일 목록을 조회하고(`data/raw_brca_rna_filelist.json`, 이미 조회해둠 — Primary Tumor 1111개
파일/1095 case) bulk data 엔드포인트(POST /data)로 배치 다운로드한다.

절차:
    1. GDC POST /data로 파일들을 배치(기본 100개씩) 다운로드해 tar.gz로 받고
       data/raw/TCGA_BRCA_RNA/<file_uuid>/<file_name>.tsv 로 풀어놓는다(PAAD와 동일한 디렉터리
       관례 — 나중에 extract_rna_clinical.py 계열 함수 재사용 가능성을 열어둔다).
    2. 파일별로 gene_type=="protein_coding"만 남기고 log2(fpkm_uq_unstranded+1)을 취한다
       (data/extract_rna_clinical.py::_read_tpm과 동일 로직).
    3. 한 case가 여러 tumor RNA 파일을 가지면 평균낸다.
    4. 코호트(BRCA) 내부에서 유전자별 z-score 정규화.
    5. data/brca_clinical.csv(case_id 목록)과 inner join.

주의: TCGA-PAAD/CPTAC에서 쓴 "literature_1500"(Bailey/Moffitt PDAC subtype + Cox+Stouffer
survival-optimized) 유전자셋은 췌장암 전용으로 큐레이션된 것이라 유방암에는 그대로 못 쓴다 —
여기서는 protein-coding 유전자 전체를 z-score해 저장해두고, 실제 학습 시 유전자 서브셋 선택은
별도로 정한다(예: 전체 사용 또는 고분산 유전자 상위 N개).

출력:
    data/rna_brca.csv        case_id + protein-coding gene_id 컬럼(z-scored log2 FPKM-UQ)
    data/brca_rna_genes.csv  gene_id, gene_name (protein-coding 유전자 목록)

사용법:
    python -m scripts.extract_brca_rna
"""
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FILELIST_PATH = Path("data/raw_brca_rna_filelist.json")
RAW_ROOT = Path("data/raw/TCGA_BRCA_RNA")
CLINICAL_PATH = Path("data/brca_clinical.csv")
OUT_RNA_PATH = Path("data/rna_brca.csv")
OUT_RAW_LOG2_PATH = Path("data/rna_brca_raw_log2.csv")
OUT_GENES_PATH = Path("data/brca_rna_genes.csv")

GDC_DATA_API = "https://api.gdc.cancer.gov/data"
BATCH_SIZE = 100


def _download_batch(file_ids: list[str]) -> None:
    req = urllib.request.Request(
        GDC_DATA_API, data=json.dumps({"ids": file_ids}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(RAW_ROOT)


def _download_all(files: list[dict]) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    remaining = [h for h in files if not any(RAW_ROOT.glob(f"{h['file_id']}/*.tsv"))]
    print(f"다운로드 대상: {len(remaining)}/{len(files)}개 (이미 존재하는 건 스킵)")
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        _download_batch([h["file_id"] for h in batch])
        print(f"  {min(i + BATCH_SIZE, len(remaining))}/{len(remaining)} 다운로드 완료")


def _read_fpkm_uq_log2(tsv_path: Path) -> pd.DataFrame:
    """data/extract_rna_clinical.py::_read_tpm과 동일 로직(2026-07-21 수정판)."""
    df = pd.read_csv(tsv_path, sep="\t", skiprows=1)
    df = df[df["gene_type"] == "protein_coding"]
    fpkm_uq = df.set_index("gene_id")["fpkm_uq_unstranded"].astype(float)
    gene_names = df.set_index("gene_id")["gene_name"]
    return pd.DataFrame({"value": np.log2(fpkm_uq + 1.0), "gene_name": gene_names})


def main():
    with open(FILELIST_PATH) as f:
        files = json.load(f)
    print(f"RNA 파일 목록: {len(files)}개 (Primary Tumor)")

    _download_all(files)

    case_series = {}
    gene_name_map = None
    for h in files:
        case_id = h["cases"][0]["submitter_id"]
        matches = list((RAW_ROOT / h["file_id"]).glob("*.tsv"))
        if not matches:
            continue
        parsed = _read_fpkm_uq_log2(matches[0])
        if gene_name_map is None:
            gene_name_map = parsed["gene_name"]
        case_series.setdefault(case_id, []).append(parsed["value"])

    print(f"RNA 파싱 완료 case 수: {len(case_series)}")

    # case당 여러 tumor 파일이면 평균
    rna_df = pd.DataFrame({
        case_id: pd.concat(series_list, axis=1).mean(axis=1)
        for case_id, series_list in case_series.items()
    }).T
    rna_df.index.name = "case_id"

    clinical = pd.read_csv(CLINICAL_PATH)
    rna_df = rna_df.loc[rna_df.index.intersection(clinical["case_id"])]
    print(f"임상 라벨과 join 후 case 수: {len(rna_df)}")

    # z-score 이전 원본(log2 FPKM-UQ+1) 값도 저장 — literature_1500 유전자 큐레이션이 BRCA엔
    # 안 맞아 대신 고분산 유전자를 뽑아야 하는데(scripts/select_brca_rna_genes.py), 이미
    # z-score된 데이터는 유전자별 분산이 전부 1로 맞춰져 있어 분산 순위를 매길 수 없다 —
    # 그래서 z-score 이전 원본이 별도로 필요하다.
    rna_df.reset_index().to_csv(OUT_RAW_LOG2_PATH, index=False)
    print(f"저장: {OUT_RAW_LOG2_PATH}  (z-score 이전 원본, 유전자 수: {rna_df.shape[1]})")

    # 코호트 내부 z-score
    z = (rna_df - rna_df.mean()) / rna_df.std(ddof=0).replace(0, 1.0)
    z.reset_index().to_csv(OUT_RNA_PATH, index=False)
    print(f"저장: {OUT_RNA_PATH}  (유전자 수: {z.shape[1]})")

    genes = gene_name_map.reset_index()
    genes.columns = ["gene_id", "gene_name"]
    genes = genes.drop_duplicates(subset="gene_id")
    genes.to_csv(OUT_GENES_PATH, index=False)
    print(f"저장: {OUT_GENES_PATH}  (protein-coding 유전자 {len(genes)}개)")


if __name__ == "__main__":
    main()
