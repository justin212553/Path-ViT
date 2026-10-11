"""
2026-10-10: PORPOISE·MMP 공식 코드 재현 결과(pkl)를 이 프로젝트 예측 CSV 형식
(.logs/{kfold_preds,external_preds}/{dataset}_{TAG}_seed{s}_fold{f}of{n}.csv, 컬럼
case_id,risk,OS_time,OS_event)으로 내보낸다. 그러면 analyze_*.py와 paired_bootstrap_delta.py가
우리 모델과 같은 환자 단위로 baseline을 바로 짝지어 비교할 수 있다. HPC에서 실행:

    cd /pub/wonseukl/Path-ViT/
    python scripts/export_baseline_preds.py porpoise --seeds 84,126,42,168,210
    python scripts/export_baseline_preds.py mmp --seeds 84,126,42,168,210

- porpoise: porpoise/results_n110_mmf/**/*_s{seed}/split_latest_val_{fold}_results.pkl
  (우리와 같은 fold 배정의 held-out = internal) -> kfold_preds/tcga_PORPOISE_N110_MMF_*.csv.
  external(CPTAC)은 sbatch/eval_porpoise_n110_mmf_external_cptac_5seed_hpc.sh가 이미 같은 형식으로
  external_preds/cptac_PORPOISE_N110_MMF_*.csv에 쓰므로 여기서는 건드리지 않는다.
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

PORPOISE_RESULTS = _ROOT / "porpoise" / "results_n110_mmf"  # C안: 같은 코호트·fold·RNA (prepare_porpoise_paad_n110.py)
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
            out = out_root / "kfold_preds" / f"tcga_PORPOISE_N110_MMF_seed{seed}_fold{fold}of{N_FOLDS}.csv"
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


def check(which, seeds, out_root: Path):
    """2026-10-10: export 전에 25개 (seed, fold) 전부의 단계별 산출물 유무를 한 표로 보여주고, 다시 돌릴
    array index와 sbatch 명령을 출력한다(export는 첫 누락에서 멈춰 어디가 비었는지 안 보이므로)."""
    print(f"=== {which} 상태 (array index = seed_idx*5 + fold) ===")
    if which == "porpoise":
        root = PORPOISE_RESULTS
        print(f"results 루트: {root} ({'있음' if root.is_dir() else '없음 — 학습 잡이 시작조차 안 됐거나 경로가 다름'})")
        cols = ["checkpoint", "internal pkl", "external csv"]
    else:
        root = MMP_RESULTS
        print(f"results 루트: {root} ({'있음' if root.is_dir() else '없음'})")
        cols = ["prototype", "test pkl", "external pkl"]
    print(f"{'idx':>3} {'seed':>4} {'fold':>4}  " + "  ".join(f"{c:<13}" for c in cols))
    bad = {c: [] for c in cols}
    for si, seed in enumerate(seeds):
        for fold in range(N_FOLDS):
            idx = si * N_FOLDS + fold
            if which == "porpoise":
                st = [any(root.glob(f"**/*_s{seed}/s_{fold}_checkpoint.pt")),
                      any(root.glob(f"**/*_s{seed}/split_latest_val_{fold}_results.pkl")),
                      (out_root / "external_preds" / f"cptac_PORPOISE_N110_MMF_seed{seed}_fold{fold}of{N_FOLDS}.csv").exists()]
            else:
                split_dir = _ROOT / "mmp" / "src" / "splits" / "survival" / f"TCGA_BRCA_k{fold}_seed{seed}_instcv"
                top = root / f"BRCA_INSTCV_officialRNA_seed{seed}_k{fold}::PANTHER_default::{MMP_FEAT}"
                st = [any((split_dir / "prototypes").glob("prototypes_c16_*.pkl")),
                      any(top.glob("**/test_results.pkl")), any(top.glob("**/external_results.pkl"))]
            for c, ok in zip(cols, st):
                if not ok:
                    bad[c].append((idx, seed, fold))
            print(f"{idx:>3} {seed:>4} {fold:>4}  " + "  ".join(f"{'OK' if ok else '--':<13}" for ok in st))
    if root.is_dir():
        print(f"\n(results 루트 바로 아래 항목 일부: {sorted(p.name for p in root.iterdir())[:6]})")
    print()
    if not any(bad.values()):
        print("전부 OK — export 해도 됨.")
        return
    if which == "porpoise":
        # main.py는 한 실행이 5 fold를 순차 처리하므로 재학습은 seed 단위(array index = seed_idx)
        tr = sorted({seeds.index(s) for _, s, _ in bad["checkpoint"] + bad["internal pkl"]})
        ev = sorted({i for i, _, _ in bad["external csv"]})
        if tr:
            print(f"재학습 필요 seed: {[seeds[i] for i in tr]} — "
                  f"T=$(sbatch --parsable --array={','.join(map(str, tr))} sbatch/run_porpoise_n110_mmf_5seed_array_hpc.sh)")
            print(f"  그 뒤 external 평가: sbatch --dependency=afterok:$T --array={','.join(map(str, ev))} "
                  "sbatch/eval_porpoise_n110_mmf_external_cptac_5seed_hpc.sh")
        elif ev:
            print(f"external 평가만 다시: sbatch --array={','.join(map(str, ev))} sbatch/eval_porpoise_n110_mmf_external_cptac_5seed_hpc.sh")
    else:
        pr = sorted({i for i, _, _ in bad["prototype"]})
        sv = sorted({i for i, _, _ in bad["test pkl"] + bad["external pkl"]})
        if pr:
            print(f"prototype 다시: P=$(sbatch --parsable --array={','.join(map(str, pr))} sbatch/run_mmp_brca_instcv_prototype_array_hpc.sh)")
            print(f"  그 뒤 학습: sbatch --dependency=afterok:$P --array={','.join(map(str, sv))} sbatch/run_mmp_brca_instcv_survival_array_hpc.sh")
        elif sv:
            print(f"학습만 다시: sbatch --array={','.join(map(str, sv))} sbatch/run_mmp_brca_instcv_survival_array_hpc.sh")
    print("다시 내기 전에 해당 로그(.logs/run_porpoise_n110_* / eval_porpoise_n110_* / run_mmp_brca_instcv_*_<idx>.log)의 "
          "마지막 줄로 실패 원인을 먼저 볼 것 — 같은 원인이면 재제출해도 또 실패한다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["porpoise", "mmp"])
    ap.add_argument("--seeds", type=str, default="84,126,42,168,210")
    ap.add_argument("--out-root", type=str, default=str(_ROOT / ".logs"))
    ap.add_argument("--check", action="store_true", help="export하지 않고 25개 (seed, fold) 단계별 상태와 재제출 명령만 출력")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.check:
        check(args.which, seeds, Path(args.out_root))
        return
    (export_porpoise if args.which == "porpoise" else export_mmp)(seeds, Path(args.out_root))


if __name__ == "__main__":
    main()
