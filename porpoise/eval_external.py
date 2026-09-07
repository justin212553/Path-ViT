"""
자체 RNA로 재학습한 PORPOISE MMF 체크포인트(scripts/prepare_porpoise_paad_data_ownrna.py로
학습, main.py 그대로 사용)를 CPTAC(scripts/prepare_porpoise_cptac_external_data.py 산출물)에
평가한다. PORPOISE 공식 코드엔 이 개념(TCGA 내부 5-fold CV 밖의 external cohort 평가) 자체가
없어서 새로 작성 — 알고리즘/모델은 원본 그대로(porpoise_datasets/dataset_survival.py,
models/model_porpoise.py, utils/core_utils.py::summary_survival을 그대로 재사용), 이 파일이
새로 하는 일은 "그 fold의 train-split 기준 StandardScaler를 재현해서 CPTAC에 적용 + forward
pass"뿐이다.

[핵심: genomic 정규화 재현] dataset_survival.py::Generic_Split.get_scaler()/apply_scaler()는
학습 시점에 fold의 train split에서 fit한 StandardScaler를 쓰는데, 이 scaler 객체 자체는 체크포인트에
저장되지 않는다(main.py도 안 함) — 하지만 get_scaler()는 train_split.genomic_features의 순수
결정론적 함수라, 그 fold의 splits_{fold}.csv로 train split을 다시 재구성하기만 하면 학습 때와
완전히 동일한 scaler를 재현할 수 있다. CPTAC의 genomic_features는 TCGA 학습에 쓰인 것과 반드시
같은 유전자 컬럼 순서로 맞춘 뒤(StandardScaler.transform은 컬럼명이 아니라 위치 기준) 이 scaler를
적용한다.

사용법(porpoise/ 디렉터리에서 실행 — main.py와 동일 관례):
    python eval_external.py --seed 84 --fold 0 \
        --results-dir results_ownrna_mmf --tcga-data-root data_root_true_resnet50 \
        --cptac-data-root ../data/porpoise_style_features/cptac \
        --out-csv ../.logs/porpoise_external_preds/cptac_ownrna_mmf_seed84_fold0.csv
"""
import argparse
import glob
import os

import pandas as pd
import torch

from porpoise_datasets.dataset_survival import Generic_MIL_Survival_Dataset, Generic_Split
from models.model_porpoise import PorpoiseMMF
from utils.core_utils import summary_survival
from utils.utils import get_split_loader


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--results-dir", type=str, required=True, help="main.py --results_dir으로 학습한 결과 루트")
    p.add_argument("--which-splits", type=str, default="5foldcv")
    p.add_argument("--tcga-csv", type=str, default="datasets_csv/tcga_paad_all_clean.csv.zip")
    p.add_argument("--tcga-split-dir", type=str, default="splits/5foldcv/tcga_paad_ownrna")
    p.add_argument("--tcga-data-root", type=str, required=True, help="main.py --data_root_dir와 동일 값(true-ResNet50)")
    p.add_argument("--cptac-csv", type=str, default="datasets_csv/cptac_paad_external_clean.csv.zip")
    p.add_argument("--cptac-data-root", type=str, default="../data/porpoise_style_features/cptac")
    p.add_argument("--out-csv", type=str, required=True)
    # 학습 때 쓴 것과 정확히 같은 모델 하이퍼파라미터여야 함(main.py 인자와 동일 이름/기본값) —
    # sbatch 학습 스크립트의 main.py 호출 인자를 그대로 옮겨 적을 것.
    p.add_argument("--fusion", type=str, default="bilinear")
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--gate_path", action="store_true", default=True)
    p.add_argument("--gate_omic", action="store_true", default=True)
    p.add_argument("--scale_dim1", type=int, default=8)
    p.add_argument("--scale_dim2", type=int, default=8)
    p.add_argument("--skip", action="store_true", default=True)
    p.add_argument("--dropinput", type=float, default=0.10)
    p.add_argument("--path_input_dim", type=int, default=1024)
    p.add_argument("--use_mlp", action="store_true", default=False)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 체크포인트 위치 탐색 — main.py/utils/utils.py::get_custom_exp_code의 param_code 문자열을
    #    손으로 재현하지 않고, 실제로 만들어진 디렉터리를 글롭으로 찾는다(scripts/
    #    pool_multiseed_kfold_preds.py::_find_pred_path와 동일한 이유 — 문자열 재현 실수 방지).
    pattern = os.path.join(args.results_dir, args.which_splits, "**", f"*_s{args.seed}", f"s_{args.fold}_checkpoint.pt")
    matches = sorted(glob.glob(pattern, recursive=True))
    if len(matches) != 1:
        raise FileNotFoundError(f"체크포인트 매칭 {len(matches)}개(정확히 1개여야 함): pattern={pattern} matches={matches}")
    ckpt_path = matches[0]
    print(f"[seed={args.seed} fold={args.fold}] checkpoint: {ckpt_path}")

    # 2) 그 fold의 TCGA train split을 재구성해서 train 전용 StandardScaler를 재현한다.
    tcga_dataset = Generic_MIL_Survival_Dataset(
        csv_path=args.tcga_csv, mode="pathomic", apply_sig=False,
        data_dir=os.path.join(args.tcga_data_root, "tcga_paad_20x_features"),
        shuffle=False, seed=args.seed, print_info=False, patient_strat=False,
        n_bins=4, label_col="survival_months", ignore=[],
    )
    all_splits = pd.read_csv(os.path.join(args.tcga_split_dir, f"splits_{args.fold}.csv"))
    train_split = tcga_dataset.get_split_from_df(all_splits=all_splits, split_key="train")
    scalers = train_split.get_scaler()
    omic_input_dim = train_split.genomic_features.shape[1]
    train_gene_order = list(train_split.genomic_features.columns)
    print(f"[seed={args.seed} fold={args.fold}] TCGA train n={len(train_split)}, omic_input_dim={omic_input_dim}")

    # 3) CPTAC 전체를 하나의 "split"으로 취급(train/val 구분 없음 — 순수 external 평가,
    #    이 코호트는 어떤 (seed,fold) 조합의 학습에도 등장한 적이 없으므로 held-out 원칙이
    #    자동으로 성립한다 — 우리 프로젝트의 다른 external 평가와 동일한 전제).
    cptac_dataset = Generic_MIL_Survival_Dataset(
        csv_path=args.cptac_csv, mode="pathomic", apply_sig=False,
        data_dir=args.cptac_data_root,
        shuffle=False, seed=0, print_info=False, patient_strat=False,
        n_bins=4, label_col="survival_months", ignore=[],
    )
    cptac_split = Generic_Split(
        cptac_dataset.slide_data, metadata=cptac_dataset.metadata, mode="pathomic",
        data_dir=args.cptac_data_root, label_col="survival_months",
        patient_dict=cptac_dataset.patient_dict, num_classes=cptac_dataset.num_classes,
    )
    print(f"[seed={args.seed} fold={args.fold}] CPTAC n={len(cptac_split)}")

    missing = set(train_gene_order) - set(cptac_split.genomic_features.columns)
    if missing:
        raise ValueError(
            f"TCGA 학습에 쓰인 유전자 {len(missing)}개가 CPTAC CSV에 없음(앞 10개: {sorted(missing)[:10]}) "
            "— scripts/prepare_porpoise_paad_data_ownrna.py와 scripts/prepare_porpoise_cptac_external_data.py가 "
            "같은 gene universe(data/extract_rna_clinical.py::extract_dataset)를 썼는지 확인 필요."
        )
    # StandardScaler.transform은 컬럼명이 아니라 위치(순서) 기준이므로 TCGA train과 정확히 같은
    # 순서로 재정렬한 뒤에만 scaler를 적용해야 한다.
    cptac_split.genomic_features = cptac_split.genomic_features[train_gene_order]
    cptac_split.apply_scaler(scalers=scalers)

    # 4) 모델 재구성(학습 때와 동일 하이퍼파라미터) + 체크포인트 로드.
    model = PorpoiseMMF(
        omic_input_dim=omic_input_dim, fusion=args.fusion, n_classes=args.n_classes,
        gate_path=args.gate_path, gate_omic=args.gate_omic,
        scale_dim1=args.scale_dim1, scale_dim2=args.scale_dim2,
        skip=args.skip, dropinput=args.dropinput,
        path_input_dim=args.path_input_dim, use_mlp=args.use_mlp,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    # 5) 전체 CPTAC 환자에 forward pass — 원본 utils/core_utils.py::summary_survival을 그대로
    #    재사용(위험도 계산 로직 원본과 100% 동일, 새 코드 없음).
    loader = get_split_loader(cptac_split, training=False, testing=False, weighted=False, mode="pathomic", batch_size=1)
    patient_results, raw_c_index = summary_survival(model, loader, args.n_classes)
    print(f"[seed={args.seed} fold={args.fold}] 이 체크포인트 단독 CPTAC c-index(참고용, 단일 실행): {raw_c_index:.4f}")

    rows = []
    for case_id, r in patient_results.items():
        rows.append({
            "case_id": case_id,
            "risk": float(r["risk"]),
            "OS_time": float(r["survival"]),  # 단위: 개월(survival_months) — 우리 프로젝트의 다른 OS_time(일 단위)과 단위가 다름, 이 eval 파이프라인 내부에서만 일관되게 쓰면 c-index/HR 계산엔 무관
            "OS_event": 1.0 - float(r["censorship"]),  # PORPOISE 관례(censorship<1==event) -> 우리 관례(1=사망) 로 뒤집음
        })
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"저장: {args.out_csv} ({len(out_df)}명)")


if __name__ == "__main__":
    main()
