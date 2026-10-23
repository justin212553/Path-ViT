"""
2026-10-22: OS_time <= 0 제외(QC) 재실행을 HPC 밖(Kaggle·Vast.ai·로컬)에서 돌리는 공용 실행기.
HPC free-gpu가 1주째 안 돌아 옮겼다. 모델 명령은 sbatch/fs_paad_array.sh, sbatch/fs_brca_array.sh의
QC=1 경로를 그대로 옮긴 것이다(같은 플래그, 같은 group-ts, 같은 출력 파일명).

- 작업 단위: (cohort, model, seed, fold). 5 seeds x 5 folds = 모델당 25개.
- 이어 돌리기: 성공한 작업마다 .logs/compute_done/<job>.done을 남기고, 다음 실행에서 건너뛴다.
- 병렬: --workers N개를 동시에 띄운다. GPU가 여러 장이면 CUDA_VISIBLE_DEVICES를 돌려 가며 배정.
- 시간 예산: --budget-hours를 넘기면 새 작업을 시작하지 않고, 돌고 있는 것만 마친 뒤 끝낸다
  (Kaggle 12시간 세션이 강제 종료되기 전에 깔끔하게 끝내 출력이 저장되게).
- --mirror DIR: 작업이 끝날 때마다 예측 CSV·done 표시·작업 로그를 DIR로 복사(Kaggle /kaggle/working/out).

사용 예:
  python -m scripts.compute.qc_jobs --cohort paad --models M1 M3 --list
  python -m scripts.compute.qc_jobs --cohort paad --models M1 M2 M3 M4 M3SA M4SA M4NOMUT --part 1/3 --workers 2
  python -m scripts.compute.qc_jobs --cohort brca --models M1 M2 M3 M4 M3SA M4SA --workers 4
모델 이름은 분석 산출물의 옛 번호(원고 번호 대응은 paper/TODO_reviewer_feedback.md 끝의 표).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (84, 126, 42, 168, 210)
N_FOLDS = 5
DONE_DIR = ROOT / ".logs" / "compute_done"
JOB_LOG_DIR = ROOT / ".logs" / "compute_jobs"

# ---- PAAD: sbatch/fs_paad_array.sh 그대로 (QC=1) ----
_WSI = "--cluster-pool --backbone uni2native --patch-keep-frac 0.8"
_RNA = "--rna-genes pdac_consistency_1500"
_CLIN = "--clinical-staging --clinical-margin --clinical-lr-mult 100"
PAAD = {
    "M1": f"train.py --M1 {_WSI}",
    "M2": f"train.py --M2 {_WSI} {_CLIN}",
    "M3": f"train.py --PMA {_WSI} --no-clinical {_RNA} --use-cnv",
    "M4": f"train.py --PMA {_WSI} --combine-mode cox_add {_RNA} --use-cnv {_CLIN} --clinical-mutation",
    "M5": "train_light.py --M5 --raw-linear --clinical-margin --clinical-staging",
    "M6": f"train_light.py --M6 {_RNA} --use-cnv",
    "M7": f"train_light.py --M7 {_RNA} --use-cnv --combine-mode cox_add {_CLIN} --clinical-mutation",
    "M3SA": f"train.py --PMA {_WSI} --self-attn-fusion --no-clinical {_RNA} --use-cnv",
    "M4SA": f"train.py --PMA {_WSI} --self-attn-fusion --combine-mode cox_add {_RNA} --use-cnv {_CLIN} --clinical-mutation",
    "M3NOCNV": f"train.py --PMA {_WSI} --no-clinical {_RNA}",
    "M4NOCNV": f"train.py --PMA {_WSI} --combine-mode cox_add {_RNA} {_CLIN} --clinical-mutation",
    "M6NOCNV": f"train_light.py --M6 {_RNA}",
    "M7NOCNV": f"train_light.py --M7 {_RNA} --combine-mode cox_add {_CLIN} --clinical-mutation",
    "M4NOMUT": f"train.py --PMA {_WSI} --combine-mode cox_add {_RNA} --use-cnv {_CLIN}",
    "M4FIXC": f"train.py --PMA {_WSI} --fs-fixed-centroids --combine-mode cox_add {_RNA} --use-cnv {_CLIN} --clinical-mutation",
    "M4SAFIXC": f"train.py --PMA {_WSI} --fs-fixed-centroids --self-attn-fusion --combine-mode cox_add {_RNA} --use-cnv {_CLIN} --clinical-mutation",
}

# ---- BRCA: sbatch/fs_brca_array.sh 그대로 (QC=1 -> instcvqc) ----
_PMA = "scripts.train_brca_m4 --gene-selection consistency --cluster-pool --rna-aux-weight 0"
BRCA = {
    "M1": "scripts.train_brca_m1_m2",
    "M2": "scripts.train_brca_m1_m2 --clinical --clinical-staging --clinical-lr-mult 100",
    "M3": f"{_PMA} --no-clinical",
    "M4": f"{_PMA} --clinical-staging --clinical-lr-mult 100",
    "M5": "scripts.train_brca_m5 --clinical-staging",
    "M6": "scripts.train_brca_m6 --gene-selection consistency",
    "M7": "scripts.train_brca_m7 --gene-selection consistency --clinical-staging --clinical-lr-mult 100",
    "M3SA": f"{_PMA} --self-attn-fusion --no-clinical",
    "M4SA": f"{_PMA} --self-attn-fusion --clinical-staging --clinical-lr-mult 100",
    "M2REC": "scripts.train_brca_m1_m2 --clinical --clinical-staging --clinical-receptors --clinical-lr-mult 100",
    "M4REC": f"{_PMA} --clinical-staging --clinical-receptors --clinical-lr-mult 100",
}


def command(cohort, model, seed, fold):
    if cohort == "paad":
        common = (f"--dataset tcga --external --seed {seed} --fold {fold} --n-folds {N_FOLDS} "
                  f"--fold-safe --restrict-cohort-file data/cohort_n110.txt --exclude-nonpositive-os "
                  f"--surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 --group-ts 0929fs_paad_{model}_qc")
        return [sys.executable, "-u", f"./{PAAD[model].split()[0]}"] + PAAD[model].split()[1:] + common.split()
    common = (f"--seed {seed} --fold {fold} --n-folds {N_FOLDS} --external-tss instcvqc --fold-safe "
              f"--surv-loss both --nll-n-bins 4 --nll-cox-weight 1.0 --group-ts 0929fs_brca_{model}_qc")
    return [sys.executable, "-u", "-m"] + BRCA[model].split() + common.split()


def job_name(cohort, model, seed, fold):
    return f"{cohort}_{model}_qc_seed{seed}_fold{fold}"


def all_jobs(cohort, models):
    # 모델을 바깥 루프로 두면 한 모델이 먼저 25개 다 끝난다(중간에 끊겨도 완성된 모델이 남음)
    return [(cohort, m, s, f) for m in models for s in SEEDS for f in range(N_FOLDS)]


def mirror_outputs(dst: Path):
    for sub in ("kfold_preds", "external_preds", "compute_done", "compute_jobs"):
        src = ROOT / ".logs" / sub
        if src.is_dir():
            out = dst / ".logs" / sub
            out.mkdir(parents=True, exist_ok=True)
            for p in src.iterdir():
                q = out / p.name
                if p.is_file() and (not q.exists() or q.stat().st_mtime < p.stat().st_mtime):
                    shutil.copy2(p, q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=["paad", "brca"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--part", default="1/1", help="k/n — 남은 작업을 n등분해 k번째만(번갈아 배정)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=None, help="GPU 수(기본: 자동 감지)")
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--mirror", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="실행하지 않고 남은 작업만 출력")
    args = ap.parse_args()

    table = PAAD if args.cohort == "paad" else BRCA
    bad = [m for m in args.models if m not in table]
    if bad:
        sys.exit(f"알 수 없는 모델 {bad} — 가능: {sorted(table)}")
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # 전체 목록에서 먼저 나눈 뒤 끝난 것을 뺀다 — 그래야 파트를 다시 돌려도 배정이 바뀌지 않는다
    k, n = (int(x) for x in args.part.split("/"))
    jobs = all_jobs(args.cohort, args.models)[k - 1::n]
    jobs = [j for j in jobs if not (DONE_DIR / f"{job_name(*j)}.done").exists()]
    print(f"남은 작업 {len(jobs)}개 (part {args.part})", flush=True)
    if args.list:
        for j in jobs:
            print("  ", job_name(*j))
        return

    n_gpu = args.gpus
    if n_gpu is None:
        try:
            import torch
            n_gpu = max(torch.cuda.device_count(), 1)
        except Exception:
            n_gpu = 1
    env_base = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WANDB_MODE="offline")
    t0 = time.time()
    running = {}  # Popen -> (job, log handle, start)
    queue = list(jobs)
    slot = 0
    failed = []
    while queue or running:
        over = args.budget_hours is not None and (time.time() - t0) / 3600 > args.budget_hours
        while queue and len(running) < args.workers and not over:
            job = queue.pop(0)
            name = job_name(*job)
            env = dict(env_base, CUDA_VISIBLE_DEVICES=str(slot % n_gpu))
            slot += 1
            log = open(JOB_LOG_DIR / f"{name}.log", "w", encoding="utf-8")
            cmd = command(*job)
            log.write(" ".join(cmd) + "\n")
            log.flush()
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[p] = (job, log, time.time())
            print(f"[{time.strftime('%H:%M:%S')}] start {name} (gpu {env['CUDA_VISIBLE_DEVICES']})", flush=True)
        if over and queue:
            print(f"시간 예산 {args.budget_hours}h 도달 — 새 작업 {len(queue)}개는 다음 실행으로", flush=True)
            queue = []
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            job, log, st = running.pop(p)
            log.close()
            name = job_name(*job)
            mins = (time.time() - st) / 60
            if p.returncode == 0:
                (DONE_DIR / f"{name}.done").write_text(f"{mins:.1f} min\n")
                print(f"[{time.strftime('%H:%M:%S')}] done  {name} ({mins:.1f} min)", flush=True)
            else:
                failed.append(name)
                print(f"[{time.strftime('%H:%M:%S')}] FAIL  {name} rc={p.returncode} — 로그 .logs/compute_jobs/{name}.log", flush=True)
            if args.mirror:
                mirror_outputs(args.mirror)
    print(f"끝: {(time.time() - t0) / 3600:.2f}h, 실패 {len(failed)}개 {failed}", flush=True)
    if args.mirror:
        mirror_outputs(args.mirror)


if __name__ == "__main__":
    main()
