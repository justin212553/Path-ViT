"""
check_ksens_status.py — 2026-10-05 K 민감도 재실험(fold-safe, 5 seeds x 5 folds) 상태 확인.
PAAD/BRCA M4를 K=5, K=20으로 다시 돌린 4개 배치(sbatch/submit_fs_batch.sh paad|brca M4K5 M4K20)
를 한 번에 본다. HPC에서:

    cd /pub/wonseukl/Path-ViT/
    python scripts/check_ksens_status.py            # 제출 전 점검 + 큐 상태 + 예측 완료 여부
    python scripts/check_ksens_status.py --preflight-only   # 제출 전 점검만

1) 제출 전 점검: HPC의 sbatch/train 스크립트가 M4K5/M4K20을 아는 최신본인지
   (옛 파일로 제출하면 "unknown MODEL=M4K5"로 25개 task가 전부 즉시 실패)
2) 큐 상태: squeue에서 FS-PAAD-M4K*/FS-BRCA-M4K* 잡의 상태별 task 수
3) 예측 완료 여부: check_phase2_status.py의 표 로직을 그대로 재사용(CHECKS만 이 목록으로 교체)

K=11 기준(M4)은 이미 완료 확인됨(_FS_COHN110 / _INSTCV_FS)이라 여기서는 보지 않는다.
"""
import argparse
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_phase2_status as status  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_SEEDS5 = [84, 126, 42, 168, 210]

KSENS_CHECKS = [
    ("PAAD M4 K=5",  ["CNV_SS_STG_R_MUT_CLUSTERPOOL_FS_K5_COHN110_COX_ADD"],  _SEEDS5, 5),
    ("PAAD M4 K=20", ["CNV_SS_STG_R_MUT_CLUSTERPOOL_FS_K20_COHN110_COX_ADD"], _SEEDS5, 5),
    ("BRCA M4 K=5",  ["BRCA_PMA_CONS882_STG_SS_CLUSTERPOOL_CLR100", "INSTCV_FS_K5"],  _SEEDS5, 5),
    ("BRCA M4 K=20", ["BRCA_PMA_CONS882_STG_SS_CLUSTERPOOL_CLR100", "INSTCV_FS_K20"], _SEEDS5, 5),
]

# (파일, 그 파일에 반드시 있어야 하는 문자열들)
PREFLIGHT = [
    ("sbatch/fs_paad_array.sh", ["M4K5)", "M4K20)", "--cluster-k 5", "--cluster-k 20"]),
    ("sbatch/fs_brca_array.sh", ["M4K5)", "M4K20)", "--cluster-k 5", "--cluster-k 20"]),
    ("sbatch/submit_fs_batch.sh", ["PAAD_EXTRA=(", "M4K5", "BRCA_EXTRA=("]),
    ("train.py", ["--cluster-k", "fit_fold_safe_centroids"]),
    ("scripts/train_brca_m4.py", ["--cluster-k", "_K{args.cluster_k}"]),
]


def preflight() -> bool:
    print("=== 1) 제출 전 점검 (스크립트가 최신본인지) ===")
    ok = True
    for rel, needles in PREFLIGHT:
        path = _ROOT / rel
        if not path.exists():
            print(f"  [MISSING] {rel}")
            ok = False
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lacking = [n for n in needles if n not in text]
        if lacking:
            print(f"  [OLD]     {rel} — 없음: {lacking}  → 로컬 최신본으로 교체할 것")
            ok = False
        else:
            print(f"  [OK]      {rel}")
    print(f"  -> {'제출 가능' if ok else '제출하지 말 것 — 위 파일부터 옮기기'}")
    return ok


def queue_status():
    print("\n=== 2) 큐 상태 (squeue) ===")
    if shutil.which("squeue") is None:
        print("  squeue 없음 — HPC가 아닌 곳에서 실행 중, 건너뜀")
        return
    out = subprocess.run(["squeue", "--me", "-h", "-r", "-o", "%j|%T|%R"],
                         capture_output=True, text=True).stdout
    rows = [line.split("|") for line in out.splitlines() if "-M4K" in line.split("|")[0]]
    if not rows:
        print("  K 민감도 잡이 큐에 없음 — 아직 제출 전이거나 전부 끝남(아래 3번으로 판단)")
        return
    counts = Counter((name, state) for name, state, _ in rows)
    for name in sorted({n for n, _ in counts}):
        states = ", ".join(f"{s}={c}" for (n, s), c in sorted(counts.items()) if n == name)
        print(f"  {name:<16} {states}")
    held = [r for r in rows if "user env" in r[2].lower() or "held" in r[2].lower()]
    if held:
        print(f"  !! held task {len(held)}개 — 'user env retrieval failed'면 scontrol release로 풀 것")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preflight-only", action="store_true")
    args, rest = p.parse_known_args()

    ok = preflight()
    if args.preflight_only:
        sys.exit(0 if ok else 1)
    queue_status()

    print("\n=== 3) 예측 완료 여부 ===")
    status.CHECKS = KSENS_CHECKS
    sys.argv = [sys.argv[0], *rest]  # --root, --only 등은 check_phase2_status로 그대로 전달
    status.main()


if __name__ == "__main__":
    main()
