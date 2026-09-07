"""
PORPOISE 아키텍처(자체 RNA로 재학습한 버전, scripts/prepare_porpoise_paad_data_ownrna.py)의
external validation cohort로 쓸 CPTAC-PDA genomic CSV를 만든다.

PORPOISE 공식 데이터엔 CPTAC이 아예 없어서(data/extract_rna_porpoise_official.py) 원래
불가능했던 평가인데, TCGA 쪽을 PORPOISE 원본 CSV 대신 우리 자체 RNA 파이프라인 산출물
(data/rna_tcga.csv/data/rna_cptac.csv)로 재학습하기로 했으므로 — TCGA/CPTAC이 완전히 동일한
파이프라인(log2(FPKM-UQ+1), protein-coding, 헤더 완전 일치 확인됨)을 쓰는 한 CPTAC도 같은
방식으로 만들 수 있다. 유전자는 전체 19,962개가 아니라 scripts/prepare_porpoise_paad_data_
ownrna.py와 동일하게 pdac_consistency_1500(data/dataset.py::pdac_consistency_gene_ids)으로
서브셋 — 사용자 지적(2026-09-06): 이미 이 프로젝트의 최종 후보 유전자셋이 있는데 새로
전체 유전자셋을 만들 이유가 없다. 이러면 PORPOISE 아키텍처와 우리 자체 --PORPOISE 레시피가
정확히 같은 RNA 입력을 쓰게 돼 아키텍처 차이만 순수하게 비교된다.

main.py는 안 거친다(study가 항상 "tcga_paad"로 고정돼 다른 CSV를 못 가리킴) — 이 CSV는
porpoise/eval_external.py가 직접 csv_path로 읽는다. train/val 구분이 없는 순수 external
평가용이라 fold split도 안 만든다(전체 환자를 매 (seed,fold) 체크포인트로 평가).

슬라이드->환자 매핑은 문자열 파싱이 아니라 data/dataset.py::_load_slide_index()가 쓰는
data/patches_cptac_uni2native/slide_index_task*.csv(파일명 파싱 대신 실제 검증된 매핑)를
그대로 재사용 — CPTAC 바코드는 TCGA처럼 위치 기반 파싱이 안 통한다(C3L-xxxxx 등, 프로젝트
전역에서 이미 이 인덱스 파일로만 매핑해옴).

[선행 조건] sbatch/extract_porpoise_style_features_cptac_hpc.sh 완료 —
data/porpoise_style_features/cptac/pt_files/*.pt 존재해야 함.

산출물: porpoise/datasets_csv/cptac_paad_external_clean.csv.zip
    (컬럼: case_id, slide_id, site, is_female, oncotree_code, age, survival_months,
     censorship, train + TCGA와 동일한 순서의 유전자 컬럼)

사용법:
    python -m scripts.prepare_porpoise_cptac_external_data
"""
from pathlib import Path
import zipfile

import pandas as pd

from data.dataset import _load_slide_index, pdac_consistency_gene_ids

PORPOISE_ROOT = Path("porpoise")
TRUE_RESNET50_PT_DIR = Path("data/porpoise_style_features/cptac/pt_files")
CPTAC_PATCHES_ROOT = Path("data/patches_cptac_uni2native")
OUT_NAME = "cptac_paad_external_clean.csv.zip"
N_GENES = 1500
# 2026-09-06: data.extract_rna_clinical.extract_dataset()가 읽는 원본 GDC RNA tsv 폴더
# (data/raw/CPTAC_RNA)가 HPC에 더 이상 없다(FileNotFoundError 실측) — scripts/prepare_
# porpoise_paad_data_ownrna.py와 동일하게 이미 산출된 캐시 CSV를 직접 읽는다.
RNA_CSV_PATH = Path("data/rna_cptac.csv")
CLINICAL_CSV_PATH = Path("data/clinical_cptac.csv")


def main():
    print(f"1) 캐시된 자체 RNA({RNA_CSV_PATH}, 이미 cohort-내부 z-score됨) + clinical({CLINICAL_CSV_PATH}) 로드 중...")
    rna_df = pd.read_csv(RNA_CSV_PATH).set_index("case_id")
    clinical_final = pd.read_csv(CLINICAL_CSV_PATH)
    print(f"   RNA(전체): {rna_df.shape[0]} cases x {rna_df.shape[1]} genes")

    gene_ids = pdac_consistency_gene_ids(top_n=N_GENES)
    missing = [g for g in gene_ids if g not in rna_df.columns]
    if missing:
        raise ValueError(f"pdac_consistency_1500 유전자 {len(missing)}개가 RNA 컬럼에 없음(앞 10개: {missing[:10]})")
    raw_log2 = rna_df[gene_ids]  # 변수명은 유지하지만 실제로는 이미 z-scored 값(위 주석 참조)
    print(f"   pdac_consistency_{N_GENES}로 서브셋: {raw_log2.shape[1]} genes (TCGA 쪽과 동일 함수로 뽑아 순서까지 동일)")

    print("2) true-ResNet50(PORPOISE 스펙) pt_files 스캔 중...")
    if not TRUE_RESNET50_PT_DIR.is_dir():
        raise FileNotFoundError(
            f"{TRUE_RESNET50_PT_DIR} 없음 — 먼저 sbatch/extract_porpoise_style_features_cptac_hpc.sh 완료 필요"
        )
    available_slide_ids = {p.stem for p in TRUE_RESNET50_PT_DIR.glob("*.pt")}
    print(f"   추출된 슬라이드 .pt: {len(available_slide_ids)}개")

    print("3) 슬라이드->환자 매핑(data/dataset.py::_load_slide_index, 문자열 파싱 아님)...")
    slide_index = _load_slide_index(CPTAC_PATCHES_ROOT)
    slide_index = slide_index[slide_index["slide_id"].isin(available_slide_ids)]
    slide_df = slide_index[["case_id", "slide_id"]].drop_duplicates()
    print(f"   true-ResNet50 feature 있는 슬라이드: {len(slide_df)}개 (case {slide_df['case_id'].nunique()}명)")

    common_cases = sorted(
        set(raw_log2.index) & set(clinical_final["case_id"]) & set(slide_df["case_id"])
    )
    print(f"4) RNA/clinical/WSI(true-ResNet50) 전부 있는 공통 case: {len(common_cases)}명")
    if len(common_cases) == 0:
        raise RuntimeError("공통 case가 0명 — RNA/clinical/WSI 세 출처의 case_id 포맷이 서로 다른지 확인 필요")

    clinical_idx = clinical_final.set_index("case_id")
    records = []
    for case_id in common_cases:
        row = clinical_idx.loc[case_id]
        records.append({
            "case_id": case_id,
            "site": "CPTAC",
            "is_female": 1.0 if str(row["sex"]).lower().startswith("f") else 0.0,
            "oncotree_code": "PAAD",
            "age": float(row["age_years"]),
            "survival_months": float(row["OS_time"]) / 30.44,
            "censorship": 1.0 - float(row["OS_event"]),  # PORPOISE 관례: censorship<1 == event 발생
            "train": 1.0,
        })
    clinical_out = pd.DataFrame(records)

    rna_out = raw_log2.loc[common_cases].reset_index()
    if rna_out.columns[0] != "case_id":
        rna_out = rna_out.rename(columns={rna_out.columns[0]: "case_id"})

    slide_for_case = slide_df[slide_df["case_id"].isin(common_cases)]
    merged = slide_for_case.merge(clinical_out, on="case_id", how="inner").merge(rna_out, on="case_id", how="inner")
    n_slides = len(merged)
    print(f"5) 최종 행 수(슬라이드 단위): {n_slides} (case {merged['case_id'].nunique()}명, "
          f"event={int(clinical_out['censorship'].eq(0).sum())})")

    csv_dir = PORPOISE_ROOT / "datasets_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path = csv_dir / OUT_NAME
    tmp_csv = csv_dir / OUT_NAME.replace(".zip", "")
    merged.to_csv(tmp_csv, index=False)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_csv, arcname=tmp_csv.name)
    tmp_csv.unlink()
    print(f"6) 저장: {out_path}")

    print("\nWSI feature 위치는 그대로 data/porpoise_style_features/cptac/pt_files 사용 "
          "(porpoise/eval_external.py가 직접 참조).")


if __name__ == "__main__":
    main()
