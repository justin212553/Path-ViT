"""
PORPOISE 공식 아키텍처를 "PORPOISE 원본 CSV"가 아니라 우리 자체 RNA 파이프라인
(data/extract_rna_clinical.py::extract_dataset)으로 재학습하기 위한 TCGA-PAAD genomic CSV
재구성 스크립트.

[왜 필요한가] scripts/prepare_porpoise_paad_data.py(2026-09-05)/기존 seed1,84 MMF 재현은
PORPOISE 원저자가 배포한 tcga_paad_all_clean.csv.zip(전처리 방식 비공개, 이 프로젝트 전체에서
계속 확인해온 self-contradiction)의 RNA 값을 그대로 썼다. CPTAC을 external cohort로 붙이려는데
PORPOISE 공식 데이터엔 CPTAC이 아예 없어서(data/extract_rna_porpoise_official.py 참조), 이
원본 CSV 유전자 컬럼에 CPTAC RNA를 억지로 매핑하면 유전자 ID/전처리 불일치로 인한 배치
이펙트가 "일반화 실패"처럼 보일 위험이 있다(사용자 결정, 2026-09-06: "자체 RNA로 하되 시드도
우리 프로젝트와 동일하게 5개로").

그래서 PORPOISE 아키텍처를 처음부터 우리 자체 RNA로 재학습한다 — 그것도 전체 19,962개 raw
유전자가 아니라, 지금 이 프로젝트의 최종 후보 유전자셋인 pdac_consistency_1500(JCI Insight
2025 5-데이터셋 교차분석, 우리 코호트 라벨 전혀 미참조라 leakage 없음, data/dataset.py::
pdac_consistency_gene_ids)을 그대로 쓴다(사용자 지적, 2026-09-06: "지금 쓰는게 PDAC
consistency니까 그거 그냥 그대로 쓰면 되지 않음?" — 전체 유전자셋을 새로 만들 이유가 없었다).
이러면 (1) TCGA/CPTAC RNA를 이미 정확히 같은 유전자·같은 전처리로 갖고 있어 새 추출이
필요 없고, (2) PORPOISE 아키텍처와 우리 자체 레시피(--PORPOISE, train.py)가 정확히 같은
RNA 입력을 쓰게 되어 "PORPOISE vs 우리 모델" 비교가 RNA 차이에 오염되지 않고 아키텍처
차이만 순수하게 비교된다. WSI feature(true-ResNet50, data/porpoise_style_features/tcga/
pt_files)는 그대로 재사용 — genomic 쪽만 바뀐다.

[주의 — 기존 산출물 덮어쓰기] porpoise/main.py는 study(=csv 파일명)를 split_dir 앞 2토큰에서만
뽑기 때문에(utils/utils.py::get_custom_exp_code, main.py 둘 다 '_'.join(split_dir.split('_')[:2])
-> 항상 "tcga_paad") split_dir을 뭐라고 짓든 genomic CSV 경로는 무조건
porpoise/datasets_csv/tcga_paad_all_clean.csv.zip 하나뿐이다. 즉 이 파일을 실제로 덮어써야
하고, 기존 PORPOISE-원본-RNA 결과(seed1/84, pooled C≈0.596~0.60)를 나중에 다시 참조할 수
있도록 최초 1회만 .official_backup.csv.zip으로 백업해둔다(porpoise/filter_available_slides.py의
.orig 백업 관례와 동일 패턴).

split_dir(폴드 분할 폴더)은 자유롭게 지을 수 있어(splits/5foldcv/<split_dir>/) "tcga_paad_ownrna"로
분리 — 원본 재현의 splits/5foldcv/tcga_paad/와 섞이지 않는다.

산출물:
    porpoise/datasets_csv/tcga_paad_all_clean.official_backup.csv.zip  (최초 1회, 원본 백업)
    porpoise/datasets_csv/tcga_paad_all_clean.csv.zip                  (덮어씀 — 자체 RNA 버전)
    porpoise/splits/5foldcv/tcga_paad_ownrna/splits_{0..4}.csv

사용법:
    python -m scripts.prepare_porpoise_paad_data_ownrna
"""
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import torch

from data.extract_rna_clinical import extract_dataset
from data.dataset import pdac_consistency_gene_ids

PORPOISE_ROOT = Path("porpoise")
TRUE_RESNET50_PT_DIR = Path("data/porpoise_style_features/tcga/pt_files")
N_FOLDS = 5
SPLIT_DIR_NAME = "tcga_paad_ownrna"
N_GENES = 1500


def main():
    print("1) 자체 RNA 파이프라인에서 raw log2(FPKM-UQ+1) TCGA RNA-seq 추출 중"
          "(캐시 재사용, PORPOISE 원본 CSV는 전혀 안 씀)...")
    raw_log2, clinical_final, _name_map = extract_dataset("tcga")
    print(f"   raw RNA-seq(전체): {raw_log2.shape[0]} cases x {raw_log2.shape[1]} genes")

    gene_ids = pdac_consistency_gene_ids(top_n=N_GENES)
    missing = [g for g in gene_ids if g not in raw_log2.columns]
    if missing:
        raise ValueError(f"pdac_consistency_1500 유전자 {len(missing)}개가 raw RNA 컬럼에 없음(앞 10개: {missing[:10]})")
    raw_log2 = raw_log2[gene_ids]
    print(f"   pdac_consistency_{N_GENES}로 서브셋: {raw_log2.shape[1]} genes")

    print("2) true-ResNet50(PORPOISE 스펙) pt_files 스캔 중...")
    if not TRUE_RESNET50_PT_DIR.is_dir():
        raise FileNotFoundError(
            f"{TRUE_RESNET50_PT_DIR} 없음 — 먼저 sbatch/extract_porpoise_style_features_hpc.sh 완료 필요"
        )
    slide_rows = []
    for pt_path in sorted(TRUE_RESNET50_PT_DIR.glob("*.pt")):
        slide_id = pt_path.stem
        parts = slide_id.split("-")
        if len(parts) < 3 or parts[0] != "TCGA":
            continue  # 방어적 필터 — 이 디렉터리는 TCGA만 있어야 하지만 혹시 몰라 확인
        case_id = "-".join(parts[:3])
        slide_rows.append({"case_id": case_id, "slide_id": slide_id})
    slide_df = pd.DataFrame(slide_rows)
    print(f"   true-ResNet50 슬라이드: {len(slide_df)}개 (case {slide_df['case_id'].nunique()}명)")

    common_cases = sorted(
        set(raw_log2.index) & set(clinical_final["case_id"]) & set(slide_df["case_id"])
    )
    print(f"3) RNA/clinical/WSI(true-ResNet50) 전부 있는 공통 case: {len(common_cases)}명")

    clinical_idx = clinical_final.set_index("case_id")
    records = []
    for case_id in common_cases:
        row = clinical_idx.loc[case_id]
        records.append({
            "case_id": case_id,
            "site": case_id.split("-")[1],
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
    print(f"4) 최종 행 수(슬라이드 단위): {n_slides} (case {merged['case_id'].nunique()}명)")

    csv_dir = PORPOISE_ROOT / "datasets_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    live_path = csv_dir / "tcga_paad_all_clean.csv.zip"
    backup_path = csv_dir / "tcga_paad_all_clean.official_backup.csv.zip"
    if live_path.exists() and not backup_path.exists():
        shutil.copy(live_path, backup_path)
        print(f"5) 기존(PORPOISE 원본 RNA 기반) CSV 백업: {backup_path}")
    elif backup_path.exists():
        print(f"5) 백업 이미 존재, 재백업 안 함: {backup_path}")

    tmp_csv = csv_dir / "tcga_paad_all_clean.csv"
    merged.to_csv(tmp_csv, index=False)
    with zipfile.ZipFile(live_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_csv, arcname="tcga_paad_all_clean.csv")
    tmp_csv.unlink()
    print(f"6) 자체 RNA 기반 CSV 저장(기존 파일 덮어씀): {live_path}")

    # porpoise/filter_available_slides.py는 자체적으로 tcga_paad_all_clean.orig.csv.zip이라는
    # 별도 백업을 만들어두고 "이미 있으면 재백업 안 함 -> 항상 그 .orig에서 필터링"하는 방식이라,
    # 방금 CSV를 통째로 갈아끼웠는데 이 .orig가 official-RNA 시절 걸로 남아 있으면 filter_
    # available_slides.py가 옛날(공식 RNA) 데이터를 기준으로 다시 필터링해버린다 — 지워서
    # 다음 실행 때 지금 막 만든 own-RNA CSV로 새로 백업/필터링되게 한다.
    stale_filter_orig = csv_dir / "tcga_paad_all_clean.orig.csv.zip"
    if stale_filter_orig.exists():
        stale_filter_orig.unlink()
        print(f"   (filter_available_slides.py용 캐시 백업 {stale_filter_orig} 삭제 — 다음 실행 때 own-RNA 기준으로 새로 생성됨)")

    print("7) 5-fold split 생성 중 (censorship 기준 stratified, case 단위)...")
    rng = np.random.RandomState(84)
    case_censorship = clinical_out.set_index("case_id")["censorship"]
    fold_of_case = {}
    for cens_val in sorted(case_censorship.unique()):
        group = case_censorship[case_censorship == cens_val].index.to_numpy()
        rng.shuffle(group)
        offset = rng.randint(0, N_FOLDS)
        for i, cid in enumerate(group):
            fold_of_case[cid] = (i + offset) % N_FOLDS

    split_dir = PORPOISE_ROOT / "splits" / "5foldcv" / SPLIT_DIR_NAME
    split_dir.mkdir(parents=True, exist_ok=True)
    all_cases = np.array(common_cases)
    for k in range(N_FOLDS):
        val_mask = np.array([fold_of_case[c] == k for c in all_cases])
        train_ids = all_cases[~val_mask]
        val_ids = all_cases[val_mask]
        n = max(len(train_ids), len(val_ids))
        train_col = list(train_ids) + [""] * (n - len(train_ids))
        val_col = list(val_ids) + [""] * (n - len(val_ids))
        pd.DataFrame({"train": train_col, "val": val_col}).to_csv(split_dir / f"splits_{k}.csv")
        print(f"   fold {k}: train={len(train_ids)} val={len(val_ids)}")

    print("\n완료. 실행 예시(기존 run 스크립트와 동일 data_root_dir 재사용, split_dir만 교체):")
    print("  python main.py --data_root_dir <기존 true-resnet50 data_root> "
          f"--which_splits 5foldcv --split_dir {SPLIT_DIR_NAME} "
          "--mode pathomic --model_type porpoise_mmf --bag_loss nll_surv --reg_type pathomic "
          "--fusion bilinear --gate_path --gate_omic --skip --dropinput 0.10 "
          "--results_dir ./results_ownrna_mmf --seed 84")


if __name__ == "__main__":
    main()
