"""
check_pred_completeness.py — .logs/{kfold,external}_preds 안에 특정 모델 태그의 예측 CSV가
seed x fold 조합 전부 존재하는지 확인한다. HPC에서 로컬로 파일을 옮기기 전에, 또는 sbatch 잡이
끝난 직후 무결성 확인용으로 쓴다.

두 가지 모드:
  1. 발견 모드(--seeds 생략): --tag 패턴에 걸리는 모든 실제 태그를 찾아, 각 태그 자체에 존재하는
     seed 집합을 기준으로 fold 누락만 보고한다(태그 자체가 통째로 없는 경우는 애초에 안 잡힘).
  2. 검증 모드(--seeds 지정): --tag 패턴에 걸리는 태그마다, 주어진 --seeds x --n-folds 전체 조합이
     다 있는지 확인한다. --tag가 아예 하나도 안 걸리면 "0개 발견, 전부 누락"으로 보고한다 —
     fit_clusters가 안 끝나서 학습 자체가 시작도 못 한 경우(PAAD K5/K20)처럼 통째로 빠진 경우를
     잡아내기 위한 모드.

파일명 규칙: <tag>_[FINALEPOCH_]seed<S>_fold<f>of<n>.csv
PAAD(tcga_*/cptac_*) 계열은 tag 중간에 _FOLD<f>OF<n>이 중복으로 한 번 더 박혀 있어(train.py/
train_light.py 고유의 tag 조립 방식, BRCA에는 없음) 이를 벗겨내야 진짜 모델 단위로 묶인다.

사용 예:
  # 발견 모드 — SELFATTNFUSION이 들어간 태그가 뭐가 있고 각각 뭐가 비었는지
  python scripts/check_pred_completeness.py --tag SELFATTNFUSION

  # 검증 모드 — M4-SA(PAAD)가 5시드(84,126,42,168,210) x 5fold 전부 있는지, 없으면 실패(exit 1)
  python scripts/check_pred_completeness.py --tag SELFATTNFUSION --tag CLUSTERPOOL_COX_ADD \\
      --seeds 84,126,42,168,210 --n-folds 5

  # 특정 잡의 SLURM --array 재제출 인덱스까지 뽑기(그 잡의 SEEDS bash 배열 순서를 --seed-order로)
  python scripts/check_pred_completeness.py --tag SELFATTNFUSION_CLUSTERPOOL_COX_ADD \\
      --seeds 84,126 --n-folds 5 --seed-order 84,126

  # PAAD K5가 통째로 안 돌았는지 확인(0개 발견이면 fit_clusters부터 다시 봐야 함)
  python scripts/check_pred_completeness.py --tag CENTROIDS --tag K5 --seeds 84 --n-folds 5

  # JSON으로 뽑아서 다른 스크립트/나(어시스턴트)가 파싱
  python scripts/check_pred_completeness.py --tag SELFATTNFUSION --seeds 84,126,42,168,210 --n-folds 5 --json
"""
import argparse
import fnmatch
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FILENAME_RE = re.compile(r"^(?P<tag>.+?)_(?P<ckpt>FINALEPOCH_)?seed(?P<seed>\d+)_fold(?P<fold>\d+)of(?P<nfolds>\d+)\.csv$")
_PAAD_FOLD_ARTIFACT_RE = re.compile(r"_FOLD\d+OF\d+$")


def parse_filename(name: str):
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    tag = _PAAD_FOLD_ARTIFACT_RE.sub("", m.group("tag"))
    ckpt = "FINAL" if m.group("ckpt") else "BEST"
    return tag, ckpt, int(m.group("seed")), int(m.group("fold")), int(m.group("nfolds"))


def scan(root: Path, subdirs: list[str]):
    """반환: {subdir: {tag: {ckpt: {(seed, fold, nfolds)}}}}, 그리고 파싱 실패 파일 목록."""
    out = {sd: defaultdict(lambda: defaultdict(set)) for sd in subdirs}
    unmatched = defaultdict(list)
    for sd in subdirs:
        d = root / sd
        if not d.is_dir():
            continue
        for f in d.glob("*.csv"):
            parsed = parse_filename(f.name)
            if parsed is None:
                unmatched[sd].append(f.name)
                continue
            tag, ckpt, seed, fold, nfolds = parsed
            out[sd][tag][ckpt].add((seed, fold, nfolds))
    return out, unmatched


def tags_matching(all_tags, patterns: list[str]):
    """patterns가 비어 있으면 전부. 아니면 각 패턴이 fnmatch 또는 부분 문자열로 걸리는 태그만
    (전부 다 걸려야 하는 AND가 아니라, 하나라도 걸리면 되는 OR가 아니라 -- 아래처럼 전부(AND) 만족
    해야 매칭. 여러 --tag를 준 건 "이 문자열들 전부 포함하는 태그만" 의미로 쓴다)."""
    if not patterns:
        return list(all_tags)
    matched = []
    for tag in all_tags:
        if all((p in tag) or fnmatch.fnmatch(tag, f"*{p}*") for p in patterns):
            matched.append(tag)
    return matched


def check_tag(entries_by_ckpt, expected_seeds, n_folds):
    """entries_by_ckpt: {ckpt: {(seed,fold,nfolds)}}. expected_seeds가 None이면 발견 모드
    (그 태그 안에 실제로 나타난 seed 전체를 기준으로 fold 누락만 본다). 아니면 검증 모드."""
    report = {}
    for ckpt, entries in entries_by_ckpt.items():
        present = {(s, f) for s, f, _n in entries}
        seeds = expected_seeds if expected_seeds is not None else sorted({s for s, f in present})
        missing = {}
        for s in seeds:
            miss_folds = [f for f in range(n_folds) if (s, f) not in present]
            if miss_folds:
                missing[s] = miss_folds
        total_expected = len(seeds) * n_folds
        total_present = sum(1 for s in seeds for f in range(n_folds) if (s, f) in present)
        report[ckpt] = {
            "seeds_checked": seeds,
            "missing": missing,
            "present": total_present,
            "expected": total_expected,
        }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default=str(_ROOT / ".logs"),
                   help="기본 .logs(레포 루트 기준). kfold_preds/external_preds가 이 아래 있다고 가정.")
    p.add_argument("--subdirs", type=str, default="kfold_preds,external_preds",
                   help="쉼표로 구분. 기본 둘 다 확인.")
    p.add_argument("--tag", action="append", default=[],
                   help="태그에 포함돼야 할 부분 문자열. 여러 번 줄 수 있고, 전부(AND) 만족하는 "
                        "태그만 검사 대상이 된다. 생략하면 발견된 모든 태그를 본다.")
    p.add_argument("--seeds", type=str, default=None,
                   help="쉼표로 구분된 정수 시드 목록. 주면 검증 모드(이 시드 전체 x --n-folds가 "
                        "다 있는지 확인, --tag가 하나도 안 걸리면 0/전체로 보고). 생략하면 발견 "
                        "모드(그 태그에 실제 존재하는 시드 기준으로 fold 누락만 본다).")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed-order", type=str, default=None,
                   help="쉼표로 구분된 정수 시드 목록 — 특정 sbatch 스크립트의 SEEDS bash 배열 "
                        "순서 그대로. 주면 누락된 (seed,fold)마다 SLURM --array 인덱스"
                        "(=seed_order.index(seed)*n_folds+fold)를 같이 계산해 재제출 커맨드를 "
                        "찍어준다. --seeds에 이 스크립트가 다루는 시드만 있어야 인덱스가 맞는다.")
    p.add_argument("--require-checkpoint", choices=["best", "final", "both"], default="both",
                   help="완결 여부(exit code)를 판단할 때 어느 체크포인트를 기준으로 볼지. "
                        "기본 both(둘 다 완전해야 완결).")
    p.add_argument("--json", action="store_true", help="사람이 읽을 리포트 대신 JSON 한 덩어리로 출력.")
    args = p.parse_args()

    root = Path(args.root)
    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    expected_seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    seed_order = [int(x) for x in args.seed_order.split(",")] if args.seed_order else None

    scanned, unmatched = scan(root, subdirs)

    all_tags = sorted(set().union(*[set(scanned[sd].keys()) for sd in subdirs])) if subdirs else []
    matched_tags = tags_matching(all_tags, args.tag)

    result = {"root": str(root), "subdirs": subdirs, "tag_patterns": args.tag,
              "expected_seeds": expected_seeds, "n_folds": args.n_folds, "tags": {}}
    overall_ok = True

    if args.seeds is not None and not matched_tags:
        # 검증 모드인데 패턴에 걸리는 태그가 하나도 없음 — 통째로 안 돈 경우(예: fit_clusters
        # 미완료로 학습 자체가 시작도 못 한 케이스)를 명시적으로 보고한다.
        overall_ok = False
        result["tags"]["<no tag matched>"] = {
            sd: {"BEST": {"present": 0, "expected": len(expected_seeds) * args.n_folds,
                          "missing": {s: list(range(args.n_folds)) for s in expected_seeds}}}
            for sd in subdirs
        }

    for tag in matched_tags:
        result["tags"][tag] = {}
        for sd in subdirs:
            entries_by_ckpt = scanned[sd].get(tag, {})
            rep = check_tag(entries_by_ckpt, expected_seeds, args.n_folds)
            result["tags"][tag][sd] = rep
            for ckpt, info in rep.items():
                if args.require_checkpoint != "both" and ckpt.lower() != args.require_checkpoint:
                    continue
                if info["missing"]:
                    overall_ok = False

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if overall_ok else 1)

    # ---- 사람이 읽는 리포트 ----
    print(f"root={root}  subdirs={subdirs}  n_folds={args.n_folds}")
    if args.tag:
        print(f"tag patterns (AND): {args.tag}")
    print(f"mode: {'검증(expected seeds=' + str(expected_seeds) + ')' if expected_seeds is not None else '발견'}")
    print()

    if args.seeds is not None and not matched_tags:
        print(f"!! 패턴 {args.tag}에 걸리는 태그가 0개입니다 — 해당 실험이 통째로 안 돌았을 "
              f"가능성이 높습니다(예: 선행 단계 미완료로 FileNotFoundError 즉시 종료, 잡 미제출 등).")
        print(f"   기대: seeds={expected_seeds} x n_folds={args.n_folds} = "
              f"{len(expected_seeds) * args.n_folds}개, 실제: 0개")
        print()

    for tag, per_sd in result["tags"].items():
        if tag == "<no tag matched>":
            continue
        print(f"[tag] {tag}")
        for sd in subdirs:
            rep = per_sd.get(sd, {})
            for ckpt in ["BEST", "FINAL"]:
                info = rep.get(ckpt)
                if info is None:
                    print(f"  {sd}/{ckpt}: (파일 없음)")
                    continue
                pct = (100 * info["present"] // info["expected"]) if info["expected"] else 0
                status = "OK" if not info["missing"] else "MISSING"
                print(f"  {sd}/{ckpt}: {info['present']}/{info['expected']} ({pct}%) -> {status}")
                for s, folds in sorted(info["missing"].items()):
                    print(f"      seed={s} missing fold={folds}")
        if seed_order is not None:
            idxs = set()
            for sd in subdirs:
                for ckpt in ["BEST", "FINAL"]:
                    info = per_sd.get(sd, {}).get(ckpt)
                    if not info:
                        continue
                    for s, folds in info["missing"].items():
                        if s not in seed_order:
                            continue
                        for f in folds:
                            idxs.add(seed_order.index(s) * args.n_folds + f)
            if idxs:
                print(f"  -> 재제출: sbatch --array={','.join(str(i) for i in sorted(idxs))} <이 태그를 만든 sbatch 스크립트>")
        print()

    for sd, names in unmatched.items():
        if names:
            print(f"!! {sd}에 파일명 패턴이 안 맞아 건너뛴 파일 {len(names)}개 (예: {names[:3]})")

    print()
    print("전체 결과:", "OK" if overall_ok else "INCOMPLETE")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
