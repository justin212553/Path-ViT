"""
2026-10-10: PORPOISE·MMP 공식 코드 재현 결과(pkl)를 이 프로젝트 예측 CSV 형식
(.logs/{kfold_preds,external_preds}/{dataset}_{TAG}_seed{s}_fold{f}of{n}.csv, 컬럼
case_id,risk,OS_time,OS_event)으로 내보낸다. 그러면 analyze_*.py와 paired_bootstrap_delta.py가
우리 모델과 같은 환자 단위로 baseline을 바로 짝지어 비교할 수 있다. HPC에서 실행:

    cd /pub/wonseukl/Path-ViT/
    python scripts/export_baseline_preds.py porpoise --seeds 84,126,42,168,210
    python scripts/export_baseline_preds.py mmp --seeds 84,126,42,168,210

- porpoise: porpoise/results_official_geneset_mmf/**/*_s{seed}/split_latest_val_{fold}_results.pkl
  (공식 5-fold split의 held-out = internal) -> kfold_preds/tcga_PORPOISE_OFFICIALGENE_MMF_*.csv.
  external(CPTAC)은 eval_porpoise_official_geneset_*_external_cptac*_hpc.sh가 이미 같은 형식으로
  external_preds/cptac_PORPOISE_OFFICIALGENE_MMF_*.csv에 쓰므로 여기서는 건드리지 않는다.
- mmp: mmp/src/results/BRCA_INSTCV_officialRNA_seed{s}_k{f}::PANTHER_default::{feat}/**/{test,external}_results.pkl
  -> kfold_preds/brca_MMP_INSTCV_*.csv (test) / external_preds/brca_MMP_INSTCV_*.csv (external)

극성: 두 baseline 모두 censorship(1=검열)을 쓰므로 OS_event = 1 − censorship로 바꾼다
(pool_porpoise_official_kfold.py, pool_mmp_brca_kfold.py와 같은 변환). PORPOISE/MMP는
best/final 구분 없이 마지막 epoch 하나만 저장하므로 _FINALEPOCH_ 파일은 만들지 않는다 —
분석할 때 include_final_epoch=False로 불러야 한다.
"""
import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from pool_mmp_brca_kfold import _load_dump  # noqa: E402

PORPOISE_RESULTS = _ROOT / "porpoise" / "results_official_geneset_mmf"
MMP_RESULTS = _ROOT / "mmp" / "src" / "results"
MMP_FEAT = "extracted-vit_large_patch16_224.dinov2.uni_mass100k"
N_FOLDS = 5


def _write(path: Path, preds: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "risk", "OS_time", "OS_event"])
        for cid, (r, t, e) in preds.items():
            w.writerow([cid, r, t, int(round(e))])


def export_porpoise(seeds, out_root: Path):
    import pickle
    for seed in seeds:
        for fold in range(N_FOLDS):
            hits = sorted(PORPOISE_RESULTS.glob(f"**/*_s{seed}/split_latest_val_{fold}_results.pkl"))
            if len(hits) != 1:
                raise FileNotFoundError(f"PORPOISE seed={seed} fold={fold}: pkl {len(hits)}개(1개여야 함) — {hits}")
            with open(hits[0], "rb") as f:
                res = pickle.load(f)
            preds = {cid: (float(d["risk"]), float(d["survival"]), 1.0 - float(d["censorship"])) for cid, d in res.items()}
            out = out_root / "kfold_preds" / f"tcga_PORPOISE_OFFICIALGENE_MMF_seed{seed}_fold{fold}of{N_FOLDS}.csv"
            _write(out, preds)
            print(f"  {out.name}: {len(preds)}명")


def export_mmp(seeds, out_root: Path):
    for seed in seeds:
        for fold in range(N_FOLDS):
            for split, sub in (("test", "kfold_preds"), ("external", "external_preds")):
                preds = _load_dump(MMP_RESULTS, "BRCA_INSTCV_officialRNA", MMP_FEAT, seed, fold, split)
                out = out_root / sub / f"brca_MMP_INSTCV_seed{seed}_fold{fold}of{N_FOLDS}.csv"
                _write(out, preds)
                print(f"  {sub}/{out.name}: {len(preds)}명")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["porpoise", "mmp"])
    ap.add_argument("--seeds", type=str, default="84,126,42,168,210")
    ap.add_argument("--out-root", type=str, default=str(_ROOT / ".logs"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    (export_porpoise if args.which == "porpoise" else export_mmp)(seeds, Path(args.out_root))


if __name__ == "__main__":
    main()
