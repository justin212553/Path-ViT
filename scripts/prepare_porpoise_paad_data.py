"""
공식 PORPOISE 저장소(sota/PORPOISE, mahmoodlab/PORPOISE 그대로 clone)에 TCGA-PAAD 데이터를
주입하기 위한 변환 스크립트 — 저희 재구현이 아니라 원 저자 코드 자체를 그대로 돌리기 위함
(사용자 지시: "성능 자체를 동일선상에 놓고 평가하기 위함").

원 논문(Chen et al. 2022 Cancer Cell) Table S2 기준 PAAD 결과 — MMF(WSI+유전체 융합)
c-index=0.653(p=1.69e-3, 유의), AMIL(WSI만)=0.580(p=0.230, 비유의),
SNN(유전체만)=0.593(p=0.056, 비유의) — 이 세션 내내 봐온 "WSI 추가가 비유의"와 반대 결과라
원 저자 파이프라인을 그대로 재현해 원인을 찾는 게 목적.

[CNV/mutation 없음 — 명시적 한계] 원 논문 CSV는 CNV(~2500컬럼)+RNA-seq+mutation을 섞어 쓰지만
저희는 genome-wide CNV/mutation 데이터가 없다. RNA-seq만으로 만든다 — 이건 원 설정에서
벗어난 부분이라 결과 해석 시 반드시 함께 언급해야 한다.

만드는 것:
    sota/PORPOISE/datasets_csv/tcga_paad_all_clean.csv.zip
        case_id, slide_id, site, is_female, oncotree_code, age, survival_months,
        censorship(=1-OS_event, PORPOISE 관례상 censorship<1이 "event 발생"),
        train(placeholder=1), <RNA-seq raw log2(FPKM-UQ+1) 컬럼들>
    sota/PORPOISE/inputs/tcga_paad_20x_features/pt_files/{slide_id}.pt
        저희 UNI2 feature(.pt, N x 1536)를 그대로 심볼릭 복사 — slide_id는 ".svs" 접미사 없이
        원본 슬라이드 폴더명 그대로 써서 PORPOISE 코드의 slide_id.rstrip('.svs') 버그(문자
        집합 기준 strip이라 접미사 제거가 아님)를 피한다.
    sota/PORPOISE/splits/5foldcv/tcga_paad/splits_{0..4}.csv
        censorship 기준 stratified 5-fold, 두 컬럼(train/val) 형식.

사용법:
    python -m scripts.prepare_porpoise_paad_data
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.extract_rna_clinical import extract_dataset

PORPOISE_ROOT = Path("sota/PORPOISE")
UNI2_TILES_ROOT = Path("data/patches_tcga/tiles")
CLINICAL_PATH = Path("data/clinical_tcga.csv")
N_FOLDS = 5
SEED = 84  # 이 프로젝트의 표준 2seed(84/126) 중 하나 재사용


def _tss(case_id: str) -> str:
    return case_id.split("-")[1]


def main():
    print("1) raw log2(FPKM-UQ+1) RNA-seq 추출 중 (캐시된 GDC 파일 재사용, 네트워크 호출 없음)...")
    raw_log2, _clinical_unused, _name_map = extract_dataset("tcga")
    print(f"   raw RNA-seq: {raw_log2.shape[0]} cases x {raw_log2.shape[1]} genes")

    clinical = pd.read_csv(CLINICAL_PATH)
    print(f"2) clinical: {len(clinical)} cases (age_years/sex/OS_time/OS_event)")

    print("3) 슬라이드 폴더 스캔 중 (features_uni2.pt 존재하는 것만)...")
    slide_rows = []
    for slide_dir in sorted(UNI2_TILES_ROOT.iterdir()):
        if not slide_dir.is_dir():
            continue
        feat_path = slide_dir / "features_uni2.pt"
        if not feat_path.exists():
            continue
        slide_id = slide_dir.name
        # TCGA case_id는 barcode 앞 3세그먼트(TCGA-XX-XXXX), slide_id는 그 뒤로 샘플/포션/UUID가 더 붙음.
        parts = slide_id.split("-")
        if len(parts) < 3 or parts[0] != "TCGA":
            continue  # CPTAC 슬라이드 등 TCGA가 아닌 것 제외
        case_id = "-".join(parts[:3])
        slide_rows.append({"case_id": case_id, "slide_id": slide_id, "feat_path": feat_path})
    slide_df = pd.DataFrame(slide_rows)
    print(f"   TCGA WSI 슬라이드: {len(slide_df)}개 (case {slide_df['case_id'].nunique()}명)")

    common_cases = sorted(
        set(raw_log2.index) & set(clinical["case_id"]) & set(slide_df["case_id"])
    )
    print(f"4) RNA/clinical/WSI 전부 있는 공통 case: {len(common_cases)}명")

    clinical_idx = clinical.set_index("case_id")
    records = []
    for case_id in common_cases:
        row = clinical_idx.loc[case_id]
        records.append({
            "case_id": case_id,
            "site": _tss(case_id),
            "is_female": 1.0 if str(row["sex"]).lower().startswith("f") else 0.0,
            "oncotree_code": "PAAD",
            "age": float(row["age_years"]),
            "survival_months": float(row["OS_time"]) / 30.44,  # TCGA 관례: day -> month
            "censorship": 1.0 - float(row["OS_event"]),  # PORPOISE 관례: censorship<1 == event 발생
            "train": 1.0,
        })
    clinical_out = pd.DataFrame(records)

    # RNA-seq 컬럼: 공통 case만, raw log2 그대로(PORPOISE가 자체적으로 fold-train 기준
    # StandardScaler를 적용하므로 여기서 z-score 하면 안 됨 — datasets/dataset_survival.py::get_scaler 참조).
    rna_out = raw_log2.loc[common_cases].reset_index().rename(columns={"index": "case_id", raw_log2.index.name or "case_id": "case_id"})
    if "case_id" not in rna_out.columns:
        rna_out.insert(0, "case_id", common_cases)

    # slide_id는 case당 여러 개(멀티 슬라이드) 가능 — PORPOISE 관례상 slide 단위로 행을 늘린다
    # (같은 case_id가 여러 행에 반복, genomic feature는 case 단위라 동일하게 반복됨).
    slide_for_case = slide_df[slide_df["case_id"].isin(common_cases)]
    merged = slide_for_case.merge(clinical_out, on="case_id", how="inner").merge(rna_out, on="case_id", how="inner")
    merged = merged.drop(columns=["feat_path"])
    n_slides = len(merged)
    print(f"5) 최종 행 수(슬라이드 단위, case당 여러 슬라이드 가능): {n_slides}")

    out_csv_dir = PORPOISE_ROOT / "datasets_csv"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_csv_dir / "tcga_paad_all_clean.csv"
    merged.to_csv(out_csv_path, index=False)
    import zipfile
    zip_path = out_csv_dir / "tcga_paad_all_clean.csv.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv_path, arcname="tcga_paad_all_clean.csv")
    out_csv_path.unlink()
    print(f"   저장: {zip_path}")

    print("6) WSI feature .pt 파일 복사 중 (pt_files/{slide_id}.pt, .svs 접미사 없음)...")
    pt_dir = PORPOISE_ROOT / "inputs" / "tcga_paad_20x_features" / "pt_files"
    pt_dir.mkdir(parents=True, exist_ok=True)
    for _, r in slide_for_case.iterrows():
        dst = pt_dir / f"{r['slide_id']}.pt"
        if not dst.exists():
            feat = torch.load(r["feat_path"], weights_only=True)
            torch.save(feat, dst)
    print(f"   저장: {pt_dir} ({len(slide_for_case)}개 .pt)")

    print("7) 5-fold split 생성 중 (censorship 기준 stratified, case 단위)...")
    rng = np.random.RandomState(SEED)
    case_censorship = clinical_out.set_index("case_id")["censorship"]
    fold_of_case = {}
    for cens_val in sorted(case_censorship.unique()):
        group = case_censorship[case_censorship == cens_val].index.to_numpy()
        rng.shuffle(group)
        offset = rng.randint(0, N_FOLDS)
        for i, cid in enumerate(group):
            fold_of_case[cid] = (i + offset) % N_FOLDS

    split_dir = PORPOISE_ROOT / "splits" / "5foldcv" / "tcga_paad"
    split_dir.mkdir(parents=True, exist_ok=True)
    all_cases = np.array(common_cases)
    for k in range(N_FOLDS):
        val_mask = np.array([fold_of_case[c] == k for c in all_cases])
        train_ids = all_cases[~val_mask]
        val_ids = all_cases[val_mask]
        n = max(len(train_ids), len(val_ids))
        train_col = list(train_ids) + [""] * (n - len(train_ids))
        val_col = list(val_ids) + [""] * (n - len(val_ids))
        split_df = pd.DataFrame({"train": train_col, "val": val_col})
        split_df.to_csv(split_dir / f"splits_{k}.csv")
        print(f"   fold {k}: train={len(train_ids)} val={len(val_ids)}")

    print("\n완료. 실행 예시:")
    print(f"  cd sota/PORPOISE && python main.py --which_splits 5foldcv --split_dir tcga_paad "
          f"--data_root_dir {(PORPOISE_ROOT / 'inputs').resolve()} --mode pathomic --reg_type pathomic "
          f"--model_type mm_attention_mil --results_dir results_paad_mmf")


if __name__ == "__main__":
    main()
