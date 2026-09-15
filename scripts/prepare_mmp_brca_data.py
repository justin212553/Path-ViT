"""
MMP(Song et al., ICML 2024, mahmoodlab/MMP) 아키텍처를 이 연구의 BRCA 프로토콜 그대로
재현하기 위한 데이터 준비 스크립트.

2026-09-14(2차 수정): RNA를 이 연구 자체 882유전자 패널로 바꾸는 초안(scripts/
prepare_mmp_brca_data_ownrna_bak.py에 백업)을 뒤집었다 — MMP repo에는 실제로 BRCA용 공식
RNA 데이터가 이미 통째로 들어있다(data_csvs/rna/hallmarks/BRCA/rna_clean.csv, 1097명,
4169유전자, MSigDB Hallmark 50개 pathway). 이 연구 코호트 1095명 중 1093명이 그대로
커버된다(직접 대조 확인, 2026-09-14). PORPOISE 때 세운 원칙 그대로: 원저자가 공식적으로
문서화하고 공개한 설계 요소(MMP의 RNA 패널)는 우리 걸로 바꿔치기하지 않는다. WSI만 예외인데,
MMP의 백본이 정확히 이 연구가 이미 쓰는 UNI version 1(uni_mass100k, 1024-dim)이라 원저자
선택을 그대로 따르면서 재추출 없이 우리 feature를 재사용할 수 있기 때문이다(선택이 아니라
같은 것이라 바꿔치기가 아님).

정리하면:
  - WSI: 이 연구 자체 UNI v1 patch feature 재사용 (원저자와 동일 인코더라 문제 없음).
  - RNA: MMP 공식 BRCA Hallmark 패널 그대로 사용 (data_csvs/rna/hallmarks/BRCA/, 건드리지
    않음). 커버 안 되는 2명(1095명 중)은 이 비교에서만 제외한다.
  - Split: 이 연구 자체 프로토콜(scripts/brca_common.py::load_case_table_kfold, seed 84/126
    x 5-fold internal + institution 기반 external holdout A2/AR/E9)을 그대로 재사용한다.
    label도 이 연구는 OS를 쓰고 main_survival.py 기본값도 os_survival_days/os_censorship라
    그대로 일치한다.

split_dir 이름은 "TCGA_BRCA_k{fold}_seed{seed}"로 짓는다 — main_survival.py가 cancer_type을
split_dir.split('_')[1]로 유도하므로(코드 직접 확인) 'BRCA'가 나와야 공식 rna_clean.csv
폴더를 그대로 찾아간다.

산출물:
  - {MMP_DATAROOT}/extracted_mag20x_patch256_fp/extracted-vit_large_patch16_224.dinov2.uni_mass100k/feats_pt/{slide_id}.pt
    (복제 아님, data/patches_tcga_brca/tiles/{slide_id}/features_uni.pt를 가리키는 심볼릭 링크)
  - {MMP_ROOT}/splits/survival/TCGA_BRCA_k{fold}_seed{seed}/{train,val,test,external}.csv (10개
    폴더, 폴더당 4개 csv. external.csv는 A2/AR/E9 institution holdout 230명 중 공식 RNA
    데이터가 있는 환자만.)

MMP 공식 data_csvs/rna/hallmarks/BRCA/rna_clean.csv, signatures.csv류는 전혀 건드리지 않는다.

사용법 (HPC, PORPOISE와 같은 방식으로 MMP 레포를 프로젝트 루트에 바로 clone해뒀다고 가정):
    python -m scripts.prepare_mmp_brca_data --mmp-root mmp/src --mmp-dataroot data/mmp_brca_dataroot
"""
import argparse
from pathlib import Path

import pandas as pd

from scripts.brca_common import (
    EXTERNAL_TSS_MULTI,
    MANIFEST_PATH,
    TILES_ROOT,
    load_case_table_kfold,
)

SEEDS = [84, 126]
N_FOLDS = 5
FEAT_NAME = "extracted-vit_large_patch16_224.dinov2.uni_mass100k"


def convert_wsi_features(mmp_dataroot: Path) -> None:
    print("1) UNI v1 patch feature(.pt)를 MMP가 인식하는 feats_pt/ 디렉터리로 심볼릭 링크 연결...")
    # 2026-09-14: 데이터를 h5로 복제/변환할 필요가 없다는 걸 코드로 확인했다 —
    # WSISurvivalDataset/WSIPrototypeDataset 둘 다 data_source 디렉터리 이름이 "feats_h5"면
    # h5py로, "feats_pt"면 torch.load()로 읽도록 이미 분기가 있다(wsi_datasets/wsi_survival.py,
    # wsi_prototype.py 둘 다 `assert os.path.basename(src) in ['feats_h5', 'feats_pt']`).
    # 우리 features_uni.pt는 이미 (N_patches, 1024) 2D 텐서라 h5로 감쌀 이유가 전혀 없다 —
    # Hugging Face에서 받은 원본을 심볼릭 링크로 그대로 가리키기만 하면 된다(복제 없음, 즉시
    # 끝남). MMP 자신의 mmp.sh/clustering.sh 쉘 스크립트는 "feats_h5"를 하드코딩하지만, 우리는
    # 그 쉘 스크립트를 안 쓰고 sbatch에서 main_prototype.py/main_survival.py를 직접 호출하므로
    # --data_source에 feats_pt 경로를 그대로 넣으면 된다.
    feat_dir = mmp_dataroot / "extracted_mag20x_patch256_fp" / FEAT_NAME / "feats_pt"
    feat_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    n_done, n_skip, n_missing = 0, 0, 0
    for slide_id in manifest["slide_id"]:
        link_path = feat_dir / f"{slide_id}.pt"
        if link_path.exists() or link_path.is_symlink():
            n_skip += 1
            continue
        pt_path = (TILES_ROOT / slide_id / "features_uni.pt").resolve()
        if not pt_path.exists():
            print(f"   경고: {pt_path} 없음, 건너뜀")
            n_missing += 1
            continue
        link_path.symlink_to(pt_path)
        n_done += 1
    print(f"   심볼릭 링크 생성 {n_done}개, 이미 존재 {n_skip}개, 원본 없어서 건너뜀 {n_missing}개 "
          f"(총 manifest 슬라이드 {len(manifest)}개)")


def load_mmp_official_rna_patients(mmp_root: Path) -> set[str]:
    """MMP 공식 BRCA RNA 데이터가 커버하는 환자(patient-level case_id) 집합.

    'sample' 컬럼은 15자 TCGA sample barcode(예: TCGA-AR-A5QQ-01). -01(primary tumor)만
    남기고 앞 12자로 잘라 case_id와 비교 가능한 형태로 만든다. -11(정상조직)/-06(전이)은
    survival 예측에 안 쓴다(원저자 관례 그대로, 별도 확인 불필요 — 어차피 우리 case_table엔
    없는 샘플).
    """
    rna = pd.read_csv(mmp_root / "data_csvs" / "rna" / "hallmarks" / "BRCA" / "rna_clean.csv")
    primary = rna[rna["sample"].str.endswith("-01")]
    return set(primary["sample"].str[:12])


def _mmp_frame(case_table: pd.DataFrame, manifest: pd.DataFrame, split_value: str,
               rna_patients: set[str]) -> pd.DataFrame:
    # TCGA-PL-A8LV는 data/brca_clinical.csv 자체에 OS_time=-7(TCGA 임상 기록 오류로 알려진
    # 이상치)로 들어있다. 우리 자체 M1~M7 학습 코드는 이 값을 별도로 검증하지 않아 조용히
    # 넘어가지만, MMP(wsi_survival.py::validate_survival_dataset)는 survival_time >= 0을
    # strict하게 assert해서 죽는다. 값을 임의로 고치지 않고 이 한 명만 이 재현실험에서
    # 제외한다(2026-09-15).
    rows = case_table[(case_table["split"] == split_value) & (case_table["case_id"].isin(rna_patients))]
    rows = rows[rows["OS_time"] >= 0]
    # manifest에도 자체 OS_time/OS_event 컬럼이 있어서(용도가 다름), 이름 충돌을 피하려고
    # slide_id/case_id만 골라서 merge하고 라벨은 case_table(우리 프로토콜 기준) 쪽 값만 쓴다.
    merged = manifest[["case_id", "slide_id"]].merge(
        rows[["case_id", "OS_time", "OS_event"]], on="case_id", how="inner")
    merged = merged.rename(columns={"OS_time": "os_survival_days"})
    merged["os_censorship"] = 1 - merged["OS_event"]
    return merged[["case_id", "slide_id", "os_survival_days", "os_censorship"]]


def write_split_csvs(mmp_root: Path) -> None:
    print("2) 이 연구 자체 BRCA 프로토콜(seed 84/126 x 5-fold + A2/AR/E9 external)로 split csv 생성...")
    rna_patients = load_mmp_official_rna_patients(mmp_root)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["case_id"].isin(rna_patients)]
    n_missing = len(set(pd.read_csv(MANIFEST_PATH)["case_id"]) - rna_patients)
    print(f"   MMP 공식 RNA 커버 환자: {len(rna_patients)}명 (우리 코호트 중 RNA 없어서 빠지는 환자 {n_missing}명)")

    splits_root = mmp_root / "splits" / "survival"
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            table = load_case_table_kfold(seed, fold, N_FOLDS, external_tss=EXTERNAL_TSS_MULTI)
            split_dir = splits_root / f"TCGA_BRCA_k{fold}_seed{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            for split_value, fname in [("train", "train.csv"), ("val", "val.csv"),
                                        ("test", "test.csv"), ("external", "external.csv")]:
                df = _mmp_frame(table, manifest, split_value, rna_patients)
                df.to_csv(split_dir / fname, index=False)
            counts = table[table["case_id"].isin(rna_patients)]["split"].value_counts()
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
    write_split_csvs(args.mmp_root)
    print("\n완료. 다음 단계: sbatch/run_mmp_brca_prototype_array_hpc.sh 로 fold별 prototype 학습 후 "
          "sbatch/run_mmp_brca_survival_array_hpc.sh 로 학습.")


if __name__ == "__main__":
    main()
