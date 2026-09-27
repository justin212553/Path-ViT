"""
check_phase2_status.py — 2026-09-24 페이퍼 리뷰 대응 Phase 2 ablation 14개(A: M4-SA/M3-SA x2
cohort, B: K5/K20/mean-pool x2 cohort, C: M4/M7 5-seed 확장 x2 cohort)의 현재 상태를 한 번에
전부 보여준다. 빠진 것만이 아니라 이미 끝난 것도 전부 한 표에 찍는다 — check_pred_completeness.py
를 태그별로 하나씩 돌리는 걸 대신해주는 고정 목록판.

scripts/check_pred_completeness.py의 scan/tags_matching/check_tag를 그대로 재사용한다(같은
파싱 로직 중복 방지). HPC에서 그대로 돌리면 된다:

    cd /pub/wonseukl/Path-ViT/
    python scripts/check_phase2_status.py

특정 항목만 보고 싶으면 --only로 라벨 일부를 필터링(부분 문자열):

    python scripts/check_phase2_status.py --only PAAD --only "M4-SA"

새 ablation을 추가하고 싶으면 아래 CHECKS 리스트에 한 줄 추가하면 된다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pred_completeness import scan, tags_matching, check_tag  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_SUBDIRS = ["kfold_preds", "external_preds"]
_SEEDS5 = [84, 126, 42, 168, 210]
_SEEDSX = [42, 168, 210]  # 이번에 새로 추가한 시드만(원래 있던 84/126 baseline은 Phase 2 대상 아님)

# (라벨, AND로 다 걸려야 하는 태그 부분문자열 리스트, 기대 시드, n_folds)
CHECKS = [
    # --- A) co-attention 효과 분리(M4-SA/M3-SA), 5-seed ---
    ("PAAD M4-SA",   ["STG_R_MUT_SELFATTNFUSION_CLUSTERPOOL_COX_ADD"], _SEEDS5, 5),
    ("PAAD M3-SA",   ["SELFATTNFUSION_CLUSTERPOOL_NOCLINICAL"],        _SEEDS5, 5),
    ("BRCA M4-SA",   ["CONS882_SELFATTNFUSION_STG_SS_AUX_CLUSTERPOOL_CLR100"], _SEEDS5, 5),
    ("BRCA M3-SA",   ["NOCLINICAL_SELFATTNFUSION_SS_AUX_CLUSTERPOOL"], _SEEDS5, 5),
    # --- B) K 민감도 + mean-pool, 1-seed(84) ---
    ("PAAD K=5",       ["CENTROIDS", "UNI2NATIVE_K5"],  [84], 5),
    ("PAAD K=20",      ["CENTROIDS", "UNI2NATIVE_K20"], [84], 5),
    ("PAAD mean-pool", ["STG_R_MUT_NOCOATTN_CLUSTERPOOL_COX_ADD"], [84], 5),
    ("BRCA K=5",       ["CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_BRCA_UNI_K5"],  [84], 5),
    ("BRCA K=20",      ["CLUSTERPOOL_CENTROIDSCLUSTER_CENTROIDS_BRCA_UNI_K20"], [84], 5),
    ("BRCA mean-pool", ["CONS882_NOCOATTN_STG_SS_AUX_CLUSTERPOOL_CLR100"], [84], 5),
    # --- C) M4/M7 5-seed 확장(84,126은 이미 있던 것 — 여기선 새로 추가한 3개만 확인) ---
    ("PAAD M4 (+3 seed)", ["STG_R_MUT_CLUSTERPOOL_COX_ADD"],        _SEEDSX, 5),
    ("PAAD M7 (+3 seed)", ["M7_PDACCONS1500_CNV_STG_R_MUT_COX_ADD"], _SEEDSX, 5),
    ("BRCA M4 (+3 seed)", ["CONS882_STG_SS_AUX_CLUSTERPOOL_CLR100"], _SEEDSX, 5),
    ("BRCA M7 (+3 seed)", ["BRCA_M7_CONS882_STG_NLLSURV4_NLLCOX1_CLR100"], _SEEDSX, 5),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default=str(_ROOT / ".logs"))
    p.add_argument("--only", action="append", default=[],
                   help="라벨에 이 부분 문자열이 포함된 항목만 본다(여러 번 주면 OR). 생략하면 전체.")
    p.add_argument("--require-checkpoint", choices=["best", "final", "both"], default="both")
    args = p.parse_args()

    root = Path(args.root)
    scanned, unmatched = scan(root, _SUBDIRS)

    checks = CHECKS
    if args.only:
        checks = [c for c in checks if any(o.lower() in c[0].lower() for o in args.only)]

    print(f"root={root}")
    print(f"{'label':<20} {'kfold(best/final)':<22} {'external(best/final)':<24} status")
    print("-" * 90)

    n_ok, n_bad = 0, 0
    bad_rows = []
    for label, patterns, seeds, n_folds in checks:
        # 2026-09-27(버그 수정): PAAD는 내부 예측이 tcga_ 접두, 외부 예측이 cptac_ 접두로 서로
        # 다른 태그 문자열이라(BRCA는 둘 다 brca_로 같음), 매칭된 태그를 kfold_preds/
        # external_preds 양쪽에 그대로 다 들이대면 "이 서브디렉토리엔 원래 존재할 수 없는 태그"가
        # 0/expected로 잡혀 거짓 누락이 된다. 서브디렉토리별로 각자 안에 있는 태그만 따로
        # 매칭해야 한다.
        expected_total = len(seeds) * n_folds
        combined_missing = {}  # sd -> ckpt -> {seed: folds}
        cell = {}
        row_ok = True
        any_matched = False
        for sd in _SUBDIRS:
            matched_sd = tags_matching(list(scanned[sd].keys()), patterns)
            if not matched_sd:
                cell[sd] = f"0/{expected_total}|0/{expected_total}"
                continue
            any_matched = True
            best = final = 0
            for tag in matched_sd:
                rep = check_tag(scanned[sd][tag], seeds, n_folds)
                for ckpt in ("BEST", "FINAL"):
                    info = rep.get(ckpt)
                    if info is None:
                        continue
                    combined_missing.setdefault(sd, {}).setdefault(ckpt, {})
                    for s, folds in info["missing"].items():
                        combined_missing[sd][ckpt].setdefault(s, set()).update(folds)
                    if ckpt == "BEST":
                        best += info["present"]
                    else:
                        final += info["present"]
            cell[sd] = f"{best}/{expected_total}|{final}/{expected_total}"
            for ckpt in ("BEST", "FINAL"):
                if args.require_checkpoint != "both" and ckpt.lower() != args.require_checkpoint:
                    continue
                if combined_missing.get(sd, {}).get(ckpt):
                    row_ok = False
        if not any_matched:
            print(f"{label:<20} {'0/' + str(expected_total):<22} {'0/' + str(expected_total):<24} "
                  f"NOT STARTED (tag 0개 발견)")
            n_bad += 1
            bad_rows.append((label, None, seeds, n_folds, {}))
            continue
        status = "OK" if row_ok else "INCOMPLETE"
        print(f"{label:<20} {cell.get('kfold_preds', '-'):<22} {cell.get('external_preds', '-'):<24} {status}")
        if row_ok:
            n_ok += 1
        else:
            n_bad += 1
            bad_rows.append((label, True, seeds, n_folds, combined_missing))

    print("-" * 90)
    print(f"OK={n_ok}  INCOMPLETE/NOT STARTED={n_bad}  (총 {len(checks)}개 확인)")

    if bad_rows:
        print()
        print("=== INCOMPLETE 상세 ===")
        for label, matched, seeds, n_folds, missing in bad_rows:
            print(f"[{label}]")
            if matched is None:
                print(f"  태그 자체가 안 걸림 — 선행 단계(예: fit_clusters) 미완료 또는 잡 미제출 가능성")
                continue
            for sd in _SUBDIRS:
                for ckpt in ("BEST", "FINAL"):
                    m = missing.get(sd, {}).get(ckpt)
                    if not m:
                        continue
                    for s in sorted(m):
                        print(f"  {sd}/{ckpt}: seed={s} missing fold={sorted(m[s])}")

    if unmatched.get("kfold_preds") or unmatched.get("external_preds"):
        print()
        print(f"(참고: 이 파일명 규칙에 안 맞아 건너뛴 구버전 파일이 있음 — "
              f"kfold {len(unmatched.get('kfold_preds', []))}개, "
              f"external {len(unmatched.get('external_preds', []))}개, 정상)")

    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
