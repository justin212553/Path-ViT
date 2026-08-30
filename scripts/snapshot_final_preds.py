"""
paper/results_table_pma_family_3seed_kfold_ci.md의 최종 결과표를 만든 실제 예측 CSV(internal
kfold_preds + external external_preds, 7개 모델 x seed{84,126} x 5fold)를 별도 폴더로 복사한다.

.logs/kfold_preds, .logs/external_preds는 지금까지의 온갖 ablation/hybrid/legacy 실험 CSV가
전부 섞여 있어(1000개 넘음) 최종표에 실제로 쓰인 파일만 골라내기 어렵다 — 이 스크립트가 정확히
그 파일들만(model_prefix + FINAL_MODELS 태그 + seed 84/126 + fold 0~4) 골라
paper/final_preds_snapshot/{kfold_preds,external_preds}/로 복사한다. paired bootstrap delta
비교(scripts/paired_bootstrap_delta.py) 등 사후 분석이 이 스냅샷 폴더만 보고 재현 가능하게
하기 위함.

파일 경로 결정 로직은 scripts/pool_multiseed_kfold_preds.py::_load_seed_predictions,
scripts/pool_multiseed_external_preds.py::_load_run_predictions와 동일한 fallback 관례를 따른다
(model_prefix 자체에 _FOLD{f}OF{n} 접미사가 이미 붙어 저장되는 경우가 대부분).

사용법: python scripts/snapshot_final_preds.py
"""
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [84, 126]
N_FOLDS = 5

# 2026-08-21 결과표 최종 채택 태그 (paper/results_table_pma_family_3seed_kfold_ci.md 기준)
FINAL_MODELS = {
    "M1": "M1_POOL_uni2native_SS_DISP",
    "M2": "M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN",
    "M3": "PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP",
    "M4": "PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD",
    "M5": "M5_STG_R_RAWLIN",
    "M6": "M6_INT1500",
    "M7": "M7_INT1500_STG_R_COX_ADD",
}


def _resolve(pred_dir: Path, dataset: str, model: str, seed: int, fold: int) -> Path | None:
    for name in (
        f"{dataset}_{model}_FOLD{fold}OF{N_FOLDS}_seed{seed}_fold{fold}of{N_FOLDS}.csv",
        f"{dataset}_{model}_seed{seed}_fold{fold}of{N_FOLDS}.csv",
    ):
        p = pred_dir / name
        if p.exists():
            return p
    return None


def _copy_all(src_dir: Path, dst_dir: Path, dataset: str) -> tuple[int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_missing = 0, 0
    for tag, model in FINAL_MODELS.items():
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                src = _resolve(src_dir, dataset, model, seed, fold)
                if src is None:
                    print(f"  [MISSING] {tag} seed={seed} fold={fold}: {dataset}_{model}_..._seed{seed}_fold{fold}of{N_FOLDS}.csv")
                    n_missing += 1
                    continue
                shutil.copy2(src, dst_dir / src.name)
                n_ok += 1
    return n_ok, n_missing


def main():
    out_root = _ROOT / "paper" / "final_preds_snapshot"

    print("=== internal (kfold_preds, tcga) ===")
    ok, missing = _copy_all(_ROOT / ".logs" / "kfold_preds", out_root / "kfold_preds", "tcga")
    print(f"  복사 {ok}개 / 누락 {missing}개 (기대: {len(FINAL_MODELS)}*{len(SEEDS)}*{N_FOLDS}={len(FINAL_MODELS) * len(SEEDS) * N_FOLDS})")

    print("=== external (external_preds, cptac) ===")
    ok, missing = _copy_all(_ROOT / ".logs" / "external_preds", out_root / "external_preds", "cptac")
    print(f"  복사 {ok}개 / 누락 {missing}개 (기대: {len(FINAL_MODELS)}*{len(SEEDS)}*{N_FOLDS}={len(FINAL_MODELS) * len(SEEDS) * N_FOLDS})")

    print(f"\n스냅샷 위치: {out_root}")


if __name__ == "__main__":
    main()
