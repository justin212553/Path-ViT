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
PORPOISE_TAG = "PORPOISE_N110_MMF"
PORPOISE_SBATCH = ("run_porpoise_n110_mmf_5seed_array_hpc.sh", "eval_porpoise_n110_mmf_external_cptac_5seed_hpc.sh")
# 2026-10-11: 시스템 수준 비교용 — 공식 입력(datasets_csv_mutsig), 프로토콜만 우리와 동일. which="porpoise_official"
PORPOISE_VARIANTS = {
    "porpoise": ("results_n110_mmf", "PORPOISE_N110_MMF", PORPOISE_SBATCH),
    "porpoise_official": ("results_official_n110_mmf", "PORPOISE_OFFICIAL_N110_MMF",
                          ("run_porpoise_official_n110_mmf_5seed_array_hpc.sh",
                           "eval_porpoise_official_n110_mmf_external_cptac_5seed_hpc.sh")),
}
MMP_RESULTS = _ROOT / "mmp" / "src" / "results"
MMP_FEAT = "extracted-vit_large_patch16_224.dinov2.uni_mass100k"
# 2026-10-22: --qc(OS_time <= 0 제외 재실행)면 아래 세 값이 QC판으로 바뀐다(main 참조)
MMP_PREFIX = "BRCA_INSTCV_officialRNA"
MMP_OUTTAG = "MMP_INSTCV"
MMP_SPLIT_SUFFIX = "instcv"
N_FOLDS = 5


def _write(path: Path, preds: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "risk", "OS_time", "OS_event"])
        for cid, (r, t, e) in preds.items():
            w.writerow([cid, r, t, int(round(e))])


def export_porpoise(seeds, out_root: Path):
    """2026-10-11: 첫 누락에서 멈추지 않고 가능한 것은 전부 내보낸 뒤, 문제 조합을 원인과 함께 모아 출력."""
    import pickle
    problems = []
    for seed in seeds:
        for fold in range(N_FOLDS):
            hits = sorted(PORPOISE_RESULTS.glob(f"**/*_s{seed}/split_latest_val_{fold}_results.pkl"))
            if len(hits) != 1:
                problems.append((seed, fold, f"pkl {len(hits)}개(1개여야 함)", [str(h) for h in hits]))
                continue
            with open(hits[0], "rb") as f:
                res = pickle.load(f)
            preds = {cid: (float(d["risk"]), float(d["survival"]), 1.0 - float(d["censorship"])) for cid, d in res.items()}
            out = out_root / "kfold_preds" / f"tcga_{PORPOISE_TAG}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
            _write(out, preds)
            print(f"  {out.name}: {len(preds)}명")
    _report(problems)


def export_mmp(seeds, out_root: Path):
    problems = []
    for seed in seeds:
        for fold in range(N_FOLDS):
            for split, sub in (("test", "kfold_preds"), ("external", "external_preds")):
                try:
                    preds = _load_dump(MMP_RESULTS, MMP_PREFIX, MMP_FEAT, seed, fold, split)
                except FileNotFoundError as ex:
                    problems.append((seed, fold, f"{split} pkl 없음", [str(ex)]))
                    continue
                out = out_root / sub / f"brca_{MMP_OUTTAG}_seed{seed}_fold{fold}of{N_FOLDS}.csv"
                _write(out, preds)
                print(f"  {sub}/{out.name}: {len(preds)}명")
    _report(problems)


def _report(problems):
    if not problems:
        print("\n전부 내보냄.")
        return
    print(f"\n!! 내보내지 못한 조합 {len(problems)}개 (array index = seed_idx*5 + fold):")
    for seed, fold, why, detail in problems:
        print(f"   seed={seed} fold={fold}: {why}")
        for d in detail:
            print(f"      {d}")
    print("pkl이 2개 이상이면 중단됐던 옛 실행의 폴더가 남은 것 — 둘 중 최신(수정 시각) 폴더가 맞는지 확인 후 옛 것을 지우고 다시 export.")
    sys.exit(1)


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
                      (out_root / "external_preds" / f"cptac_{PORPOISE_TAG}_seed{seed}_fold{fold}of{N_FOLDS}.csv").exists()]
            else:
                split_dir = _ROOT / "mmp" / "src" / "splits" / "survival" / f"TCGA_BRCA_k{fold}_seed{seed}_{MMP_SPLIT_SUFFIX}"
                top = root / f"{MMP_PREFIX}_seed{seed}_k{fold}::PANTHER_default::{MMP_FEAT}"
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
        # 학습도 fold 단위 array(seed_idx*5+fold) — 2026-10-10 preemption 대응으로 쪼갬
        tr = sorted({i for i, _, _ in bad["checkpoint"] + bad["internal pkl"]})
        ev = sorted({i for i, _, _ in bad["external csv"]})
        if tr:
            print(f"재학습: T=$(sbatch --parsable --array={','.join(map(str, tr))} sbatch/{PORPOISE_SBATCH[0]})")
            print(f"  그 뒤 external 평가: sbatch --dependency=aftercorr:$T --array={','.join(map(str, tr))} "
                  f"sbatch/{PORPOISE_SBATCH[1]}")
            rest = [i for i in ev if i not in tr]
            if rest:
                print(f"  (학습은 됐고 external만 빈 것: sbatch --array={','.join(map(str, rest))} "
                      f"sbatch/{PORPOISE_SBATCH[1]})")
        elif ev:
            print(f"external 평가만 다시: sbatch --array={','.join(map(str, ev))} sbatch/{PORPOISE_SBATCH[1]}")
    else:
        pr = sorted({i for i, _, _ in bad["prototype"]})
        sv = sorted({i for i, _, _ in bad["test pkl"] + bad["external pkl"]})
        if pr:
            print(f"prototype 다시: P=$(sbatch --parsable --array={','.join(map(str, pr))} sbatch/run_mmp_brca_instcv_prototype_array_hpc.sh)")
            print(f"  그 뒤 학습: sbatch --dependency=aftercorr:$P --array={','.join(map(str, sv))} sbatch/run_mmp_brca_instcv_survival_array_hpc.sh")
        elif sv:
            print(f"학습만 다시: sbatch --array={','.join(map(str, sv))} sbatch/run_mmp_brca_instcv_survival_array_hpc.sh")
    print("다시 내기 전에 해당 로그(.logs/run_porpoise_n110_* / eval_porpoise_n110_* / run_mmp_brca_instcv_*_<idx>.log)의 "
          "마지막 줄로 실패 원인을 먼저 볼 것 — 같은 원인이면 재제출해도 또 실패한다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["porpoise", "porpoise_official", "mmp"])
    ap.add_argument("--seeds", type=str, default="84,126,42,168,210")
    ap.add_argument("--out-root", type=str, default=str(_ROOT / ".logs"))
    ap.add_argument("--qc", action="store_true", help="2026-10-22: OS_time <= 0 제외 재실행 결과(results_*_qc, BRCA_INSTCVQC_*) — 태그에 _QC / INSTCVQC")
    ap.add_argument("--check", action="store_true", help="export하지 않고 25개 (seed, fold) 단계별 상태와 재제출 명령만 출력")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    which = args.which
    if which.startswith("porpoise"):
        global PORPOISE_RESULTS, PORPOISE_TAG, PORPOISE_SBATCH
        rdir, PORPOISE_TAG, PORPOISE_SBATCH = PORPOISE_VARIANTS[which]
        if args.qc:
            rdir, PORPOISE_TAG = rdir + "_qc", PORPOISE_TAG.replace("N110", "N110_QC")
        PORPOISE_RESULTS = _ROOT / "porpoise" / rdir
        which = "porpoise"
    args.which = which
    if which == "mmp" and args.qc:
        global MMP_PREFIX, MMP_OUTTAG, MMP_SPLIT_SUFFIX
        MMP_PREFIX, MMP_OUTTAG, MMP_SPLIT_SUFFIX = "BRCA_INSTCVQC_officialRNA", "MMP_INSTCVQC", "instcvqc"
    if args.check:
        check(args.which, seeds, Path(args.out_root))
        return
    (export_porpoise if args.which == "porpoise" else export_mmp)(seeds, Path(args.out_root))


if __name__ == "__main__":
    main()
