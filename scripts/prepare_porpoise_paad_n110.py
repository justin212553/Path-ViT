"""
2026-10-10: PORPOISE 공식 아키텍처를 이 연구의 새 프로토콜과 "같은 코호트·같은 fold·같은 RNA 입력"으로
재학습하기 위한 데이터 준비(사용자 결정 C안 — 비교되는 것을 아키텍처 차이 하나로 줄인다).

- 코호트: data/cohort_n110.txt (TCGA 110 + CPTAC 136), 우리 M1~M7 fold-safe 배치와 동일
- RNA: pdac_consistency_1500, **raw log2**(data/rna_{tcga,cptac}_raw_log2.csv). PORPOISE는 fold의
  train split에서 StandardScaler를 직접 적합해 val과 external에 적용하므로(dataset_survival.py
  Generic_Split.get_scaler / eval_external.py가 같은 방식 재현), raw를 넣으면 우리 --fold-safe
  정규화(train 통계로 z-score, CPTAC도 TCGA train 통계)와 정확히 같아진다. 기존 own-RNA 판
  (prepare_porpoise_paad_data_ownrna.py)은 코호트 내 z-score된 값을 넣어 CPTAC 쪽이 달랐다
- split: data/dataset.py::_kfold_case_split을 우리 학습과 똑같이 호출(case_df = TCGA 110명,
  dataset/OS_event stratify). 2026-10-10 검증: 5 seeds x 5 folds 25개 전부에서 test 환자가 우리
  M4 예측 파일과 정확히 일치. PORPOISE split의 "train"=우리 train, "val"(=held-out 평가 대상)=우리
  test. 우리 val(체크포인트 선택용)은 PORPOISE에 주지 않는다 — 두 모델이 gradient를 받는 환자를
  같게 두기 위함(PORPOISE는 마지막 epoch을 쓰므로 선택용 집합이 필요 없음)
- WSI: PORPOISE 공식 스펙(ImageNet ResNet50, 3번째 블록) feature 그대로
  (data/porpoise_style_features/{tcga,cptac}/pt_files)

[주의] porpoise/main.py는 genomic CSV 경로를 split_dir 앞 두 토큰("tcga_paad")으로만 정하므로
porpoise/datasets_csv/tcga_paad_all_clean.csv.zip을 덮어써야 한다. 기존 파일은 최초 1회
.pre_n110_backup.csv.zip으로 백업한다. 다른 PORPOISE 변형(own-RNA 등)을 이후 다시 돌리려면
그 백업을 되돌려야 한다.

산출물:
    porpoise/datasets_csv/tcga_paad_all_clean.csv.zip           (덮어씀)
    porpoise/datasets_csv/cptac_paad_n110_external.csv.zip
    porpoise/splits/5foldcv_n110/tcga_paad_seed{seed}/splits_{fold}.csv

사용법(HPC, repo 루트): python -m scripts.prepare_porpoise_paad_n110
"""
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import _kfold_case_split, _load_slide_index, pdac_consistency_gene_ids

PORPOISE_ROOT = Path("porpoise")
COHORT = Path("data/cohort_n110.txt")
SEEDS = [84, 126, 42, 168, 210]
N_FOLDS = 5
WHICH_SPLITS = "5foldcv_n110"
SRC = {
    "tcga": dict(rna="data/rna_tcga_raw_log2.csv", clin="data/clinical_tcga.csv",
                 pt="data/porpoise_style_features/tcga/pt_files"),
    "cptac": dict(rna="data/rna_cptac_raw_log2.csv", clin="data/clinical_cptac.csv",
                  pt="data/porpoise_style_features/cptac/pt_files", patches="data/patches_cptac_uni2native"),
}


def slides_for(name, cases):
    pt_dir = Path(SRC[name]["pt"])
    if not pt_dir.is_dir():
        raise FileNotFoundError(f"{pt_dir} 없음 — PORPOISE 스펙 ResNet50 feature 추출이 먼저 필요")
    avail = {p.stem for p in pt_dir.glob("*.pt")}
    if name == "tcga":
        df = pd.DataFrame([{"case_id": "-".join(s.split("-")[:3]), "slide_id": s} for s in avail if s.startswith("TCGA")])
    else:
        idx = _load_slide_index(Path(SRC[name]["patches"]))
        df = idx[idx["slide_id"].isin(avail)][["case_id", "slide_id"]].drop_duplicates()
    return df[df["case_id"].isin(cases)]


def build(name, cases, genes):
    rna = pd.read_csv(SRC[name]["rna"]).set_index("case_id")
    missing = [g for g in genes if g not in rna.columns]
    if missing:
        raise ValueError(f"{name}: pdac_consistency_1500 중 {len(missing)}개 유전자가 raw RNA에 없음")
    clin = pd.read_csv(SRC[name]["clin"]).set_index("case_id")
    slides = slides_for(name, cases)
    have = sorted(set(cases) & set(rna.index) & set(clin.index) & set(slides["case_id"]))
    lost = sorted(set(cases) - set(have))
    print(f"   {name}: 코호트 {len(cases)}명 중 RNA·임상·ResNet50 feature가 모두 있는 환자 {len(have)}명"
          + (f" — 빠진 환자 {lost}" if lost else ""))
    meta = pd.DataFrame([{
        "case_id": c, "site": c.split("-")[1] if name == "tcga" else "CPTAC",
        "is_female": 1.0 if str(clin.loc[c, "sex"]).lower().startswith("f") else 0.0,
        "oncotree_code": "PAAD", "age": float(clin.loc[c, "age_years"]),
        "survival_months": float(clin.loc[c, "OS_time"]) / 30.44,
        "censorship": 1.0 - float(clin.loc[c, "OS_event"]), "train": 1.0} for c in have])
    expr = rna.loc[have, genes].reset_index().rename(columns={"index": "case_id"})
    out = slides[slides["case_id"].isin(have)].merge(meta, on="case_id").merge(expr, on="case_id")
    if out[genes].isna().any().any():
        raise ValueError(f"{name}: raw RNA에 NaN 있음")
    return out, have


def write_zip(df, path):
    tmp = path.with_suffix("")  # .csv
    df.to_csv(tmp, index=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp, arcname=tmp.name)
    tmp.unlink()


def main():
    cohort = set(COHORT.read_text().split())
    tcga_cases = sorted(c for c in cohort if c.startswith("TCGA"))
    cptac_cases = sorted(c for c in cohort if not c.startswith("TCGA"))
    genes = pdac_consistency_gene_ids(top_n=1500)

    print("1) genomic CSV 생성 (raw log2, pdac_consistency_1500)")
    tcga_df, tcga_have = build("tcga", tcga_cases, genes)
    cptac_df, _ = build("cptac", cptac_cases, genes)

    csv_dir = PORPOISE_ROOT / "datasets_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    live = csv_dir / "tcga_paad_all_clean.csv.zip"
    backup = csv_dir / "tcga_paad_all_clean.pre_n110_backup.csv.zip"
    if live.exists() and not backup.exists():
        shutil.copy(live, backup)
        print(f"   기존 {live.name} 백업: {backup}")
    write_zip(tcga_df, live)
    write_zip(cptac_df, csv_dir / "cptac_paad_n110_external.csv.zip")
    # filter_available_slides.py가 남겨둔 옛 .orig 백업이 있으면 그걸 기준으로 다시 필터링하므로 지운다
    stale = csv_dir / "tcga_paad_all_clean.orig.csv.zip"
    if stale.exists():
        stale.unlink()
    print(f"   TCGA {len(tcga_df)}행(슬라이드) / CPTAC {len(cptac_df)}행 저장")

    print("2) split 생성 — 우리 _kfold_case_split과 동일 배정")
    clin = pd.read_csv(SRC["tcga"]["clin"]).set_index("case_id")
    # 배정은 반드시 코호트 110명 전체로 계산(우리 학습과 같은 case_df) — feature가 없어 빠진 환자는
    # 배정 후 CSV에서만 제외해야 나머지 환자의 fold가 우리와 같게 유지된다
    case_df = pd.DataFrame({"dataset": "tcga", "OS_event": clin.loc[tcga_cases, "OS_event"].values}, index=tcga_cases)
    have = set(tcga_have)
    for seed in SEEDS:
        d = PORPOISE_ROOT / "splits" / WHICH_SPLITS / f"tcga_paad_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        for fold in range(N_FOLDS):
            sp = _kfold_case_split(case_df, seed, N_FOLDS, fold)
            tr = [c for c, v in sp.items() if v == "train" and c in have]
            te = [c for c, v in sp.items() if v == "test" and c in have]
            n = max(len(tr), len(te))
            pd.DataFrame({"train": tr + [""] * (n - len(tr)), "val": te + [""] * (n - len(te))}).to_csv(d / f"splits_{fold}.csv")
        print(f"   seed={seed}: 마지막 fold train/held-out {len(tr)}/{len(te)}")
    print("\n완료. 학습: sbatch sbatch/run_porpoise_n110_mmf_5seed_array_hpc.sh")


if __name__ == "__main__":
    main()
