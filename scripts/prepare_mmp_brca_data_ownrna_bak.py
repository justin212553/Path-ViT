"""
MMP(Song et al., ICML 2024, mahmoodlab/MMP) 아키텍처를 이 연구의 BRCA 프로토콜 그대로
재현하기 위한 데이터 준비 스크립트.

배경: MMP의 실제 논문 백본이 UNI(version 1, uni_mass100k, 1024-dim)라서, 이 연구가 이미
Hugging Face(Dearcat/CPathPatchFeature)에서 받아 쓰고 있는 TCGA-BRCA UNI v1 patch feature를
그대로 재사용한다. WSI를 새로 추출할 필요가 없다(2026-09-14, 사용자 확인).

genomic 입력은 MMP 공식 4,241유전자 Hallmark 패널이 아니라 이 연구 자체 882유전자 BRCA
패널(data/rna_brca.csv, 이미 cohort-내부 z-score됨)을 그대로 쓴다 — MMP 공식 RNA
전처리(50개 MSigDB Hallmark 카테고리로 토큰화)를 재구현하려면 시간이 걸리고, "본 연구가 이미
검증한 패널을 그대로 먹인다"는 이 프로젝트의 기존 관례(PORPOISE own-RNA 변형과 동일 원칙)를
따른다. Clinical 정보는 MMP 원래 파이프라인엔 없는 modality라 이번에도 추가하지 않는다
(WSI+RNA 2-모달리티 비교, PAAD M6/M7과 같은 층위).

Split은 이 연구 자체 BRCA 프로토콜(scripts/brca_common.py::load_case_table_kfold, seed
84/126 x 5-fold internal + institution 기반 external holdout A2/AR/E9)을 그대로 재사용한다.
MMP 공식 5-fold split(TCGA_BRCA_overall_survival_k=0..4, dss_survival_days 기준)은 쓰지
않는다 — 우리 M1~M7과 동일한 case가 동일한 fold에 들어가야 공정한 비교가 되고, MMP 자체
label도 이 연구는 OS를 쓰므로(os_survival_days/os_censorship, main_survival.py의 기본값과
정확히 일치) target_col을 os로 맞춘다.

산출물:
  - {MMP_DATAROOT}/extracted_mag20x_patch256_fp/extracted-vit_large_patch16_224.dinov2.uni_mass100k/feats_h5/{slide_id}.h5
    ('features' key, UNI v1 patch feature 그대로 재포장, 좌표 정보는 쓰지 않음)
  - {MMP_ROOT}/data_csvs/rna/hallmarks/BRCAOWN/rna_clean.csv (case_id + 882유전자, 공식
    BRCA 폴더는 건드리지 않음)
  - {MMP_ROOT}/splits/survival/TCGA_BRCAOWN_k{fold}_seed{seed}/{train,val,test,external}.csv
    (10개 폴더, 폴더당 4개 csv. 폴더명이 "TCGA_BRCAOWN_..."인 이유는 main_survival.py가
    cancer_type을 split_dir.split('_')[1]로 유도하기 때문 — 'BRCAOWN'이 나오도록 맞춘 이름.
    external.csv는 A2/AR/E9 institution holdout으로, 매 fold 동일한 230명이지만 각 fold의
    train 전용 scaler/prototype을 적용해야 하므로 fold별 폴더 안에 따로 둔다)

사용법 (HPC, PORPOISE와 같은 방식으로 MMP 레포를 프로젝트 루트에 바로 clone해뒀다고 가정):
    python -m scripts.prepare_mmp_brca_data --mmp-root mmp/src --mmp-dataroot data/mmp_brca_dataroot
"""
import argparse
from pathlib import Path

import h5py
import pandas as pd
import torch

from scripts.brca_common import (
    EXTERNAL_TSS_MULTI,
    MANIFEST_PATH,
    RNA_ZSCORED_PATH,
    TILES_ROOT,
    load_case_table_kfold,
)

SEEDS = [84, 126]
N_FOLDS = 5
FEAT_NAME = "extracted-vit_large_patch16_224.dinov2.uni_mass100k"


def convert_wsi_features(mmp_dataroot: Path) -> None:
    print("1) UNI v1 patch feature(.pt) -> MMP 기대 h5 포맷 변환...")
    feat_dir = mmp_dataroot / "extracted_mag20x_patch256_fp" / FEAT_NAME / "feats_h5"
    feat_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    n_done, n_skip = 0, 0
    for slide_id in manifest["slide_id"]:
        out_path = feat_dir / f"{slide_id}.h5"
        if out_path.exists():
            n_skip += 1
            continue
        pt_path = TILES_ROOT / slide_id / "features_uni.pt"
        if not pt_path.exists():
            print(f"   경고: {pt_path} 없음, 건너뜀")
            continue
        features = torch.load(pt_path, weights_only=True).numpy()
        with h5py.File(out_path, "w") as f:
            f.create_dataset("features", data=features)
        n_done += 1
    print(f"   변환 {n_done}개, 이미 존재 {n_skip}개 (총 manifest 슬라이드 {len(manifest)}개)")


def write_rna_csv(mmp_root: Path) -> None:
    print("2) 자체 882유전자 BRCA RNA 패널을 MMP rna_clean.csv 포맷으로 저장...")
    # 주의: main_survival.py는 omics_dir을 split_dir 이름에서 유도한다
    # (cancer_type = split_dir.split('/')[-1].split('_')[1]) — 그래서 split_dir을
    # "TCGA_BRCAOWN_k{fold}_seed{seed}"로 지어서 cancer_type이 'BRCAOWN'이 되도록 맞췄고,
    # 여기서도 폴더명을 'BRCAOWN'으로 맞춰야 한다. 공식 'BRCA' 폴더(4,241유전자 Hallmark
    # 패널)는 그대로 남겨두고 건드리지 않는다 — 나중에 공식 재현과 비교하고 싶을 때를 위해.
    rna = pd.read_csv(RNA_ZSCORED_PATH)
    out_dir = mmp_root / "data_csvs" / "rna" / "hallmarks" / "BRCAOWN"
    out_dir.mkdir(parents=True, exist_ok=True)
    rna.to_csv(out_dir / "rna_clean.csv", index=False)
    print(f"   저장: {out_dir / 'rna_clean.csv'} ({rna.shape[0]}명 x {rna.shape[1] - 1}유전자)")


def _mmp_frame(case_table: pd.DataFrame, manifest: pd.DataFrame, split_value: str) -> pd.DataFrame:
    rows = case_table[case_table["split"] == split_value]
    # manifest에도 자체 OS_time/OS_event 컬럼이 있어서(용도가 다름), 이름 충돌을 피하려고
    # slide_id/case_id만 골라서 merge하고 라벨은 case_table(우리 프로토콜 기준) 쪽 값만 쓴다.
    merged = manifest[["case_id", "slide_id"]].merge(
        rows[["case_id", "OS_time", "OS_event"]], on="case_id", how="inner")
    merged = merged.rename(columns={"OS_time": "os_survival_days"})
    merged["os_censorship"] = 1 - merged["OS_event"]
    return merged[["case_id", "slide_id", "os_survival_days", "os_censorship"]]


def write_split_csvs(mmp_root: Path) -> None:
    print("3) 이 연구 자체 BRCA 프로토콜(seed 84/126 x 5-fold + A2/AR/E9 external)로 split csv 생성...")
    # split_dir 이름 "TCGA_BRCAOWN_k{fold}_seed{seed}"는 main_survival.py의
    # cancer_type = split_dir.split('_')[1] 유도 규칙에 맞춰 'BRCAOWN'이 나오도록 고른 이름이다
    # (write_rna_csv의 BRCAOWN 폴더와 반드시 짝이 맞아야 한다).
    #
    # external.csv는 각 fold의 split_dir "안에" 같이 넣는다(별도 디렉터리 아님) — 코드
    # 직접 확인 결과(main_survival.py::build_datasets, trainer.py::train), csv_splits.keys()를
    # 그대로 순회하며 각 split마다 validate_survival()을 돌리는 구조라, --split_names에
    # "external"을 추가하기만 하면 학습 스크립트가 한 번의 실행 안에서 A2/AR/E9 230명을
    # 자동으로 평가해준다. omics scaler와 PANTHER prototype 둘 다 train에서만 fit되고
    # external에는 적용만 되므로(build_datasets의 "if k=='train': scaler=..." 순서, 이미
    # --load_proto로 fold별 train 전용 prototype을 쓰고 있음) 별도 checkpoint 로딩 스크립트가
    # 필요 없다.
    manifest = pd.read_csv(MANIFEST_PATH)
    splits_root = mmp_root / "splits" / "survival"

    for seed in SEEDS:
        for fold in range(N_FOLDS):
            table = load_case_table_kfold(seed, fold, N_FOLDS, external_tss=EXTERNAL_TSS_MULTI)
            split_dir = splits_root / f"TCGA_BRCAOWN_k{fold}_seed{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            for split_value, fname in [("train", "train.csv"), ("val", "val.csv"),
                                        ("test", "test.csv"), ("external", "external.csv")]:
                df = _mmp_frame(table, manifest, split_value)
                df.to_csv(split_dir / fname, index=False)
            counts = table["split"].value_counts()
            print(f"   seed={seed} fold={fold}: train={counts.get('train', 0)} "
                  f"val={counts.get('val', 0)} test={counts.get('test', 0)} "
                  f"external={counts.get('external', 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmp-root", type=Path, required=True,
                     help="MMP repo의 src/ 디렉터리(예: mmp/src)")
    ap.add_argument("--mmp-dataroot", type=Path, required=True,
                     help="MMP가 --data_source로 참조할 dataroot(예: data/mmp_brca_dataroot)")
    args = ap.parse_args()

    convert_wsi_features(args.mmp_dataroot)
    write_rna_csv(args.mmp_root)
    write_split_csvs(args.mmp_root)
    print("\n완료. 다음 단계: sbatch/run_mmp_brca_prototype_array_hpc.sh 로 fold별 prototype 학습 후 "
          "sbatch/run_mmp_brca_survival_array_hpc.sh 로 학습.")


if __name__ == "__main__":
    main()
