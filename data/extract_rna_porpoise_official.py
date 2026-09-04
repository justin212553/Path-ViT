"""
sota/PORPOISE 공식 저장소가 배포한 TCGA-PAAD genomic CSV(datasets_csv_mutsig/tcga_paad_all_clean.csv.zip)
에서 RNA-seq 값을 그대로 뽑아 우리 파이프라인이 읽을 수 있는 형태로 저장.

[왜 필요한가] --rna-genes porpoise_sig(data/dataset.py::porpoise_signature_gene_ids)는 PORPOISE의
signatures.csv 유전자 "목록"만 재사용하고, 실제 발현값은 우리 자체 RNA-seq 추출 파이프라인
(data/extract_rna_clinical.py)으로 다시 계산했다. 사용자 지적: 같은 TCGA-PAAD 코호트인데 굳이
재추출할 이유가 없다 — PORPOISE가 이미 만들어 배포한 값(1553개 유전자, 이미 z-score된 상태)을
그대로 가져다 쓰는 게 "그들의 큐레이션이 진짜 더 나은가"를 검증하는 데 더 정확하다(우리 파이프라인의
정규화/스케일 차이가 섞이지 않음).

[한계] PORPOISE 공식 데이터는 TCGA-PAAD만 있고 CPTAC-PDA는 없다(원 논문이 다룬 적 없음) — 이
산출물로는 --dataset tcga 내부 5-fold CV만 가능하고 external(CPTAC) 평가는 불가능하다.

산출물: data/rna_tcga_porpoise_official.csv — case_id + 1553개 유전자 컬럼(컬럼명은 PORPOISE
원본의 "{SYMBOL}_rnaseq"에서 "_rnaseq" 접미사만 제거, 값은 PORPOISE 원본 그대로 재정규화 없음).
사용법: python -m data.extract_rna_porpoise_official
"""
import zipfile
import io
from pathlib import Path

import pandas as pd

SOURCE_ZIP = Path("sota/PORPOISE/datasets_csv_mutsig/tcga_paad_all_clean.csv.zip")
OUT_PATH   = Path("data/rna_tcga_porpoise_official.csv")


def main() -> None:
    with zipfile.ZipFile(SOURCE_ZIP) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))

    rnaseq_cols = [c for c in df.columns if c.endswith("_rnaseq")]
    df = df[["case_id"] + rnaseq_cols].drop_duplicates(subset="case_id", keep="first")
    df = df.rename(columns={c: c[: -len("_rnaseq")] for c in rnaseq_cols})
    df = df.sort_values("case_id").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"{OUT_PATH}: {df.shape[0]} cases x {df.shape[1] - 1} genes")


if __name__ == "__main__":
    main()
