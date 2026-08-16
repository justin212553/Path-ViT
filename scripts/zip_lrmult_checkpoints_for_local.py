"""
--clinical-lr-mult/--rna-lr-mult 4개 조합(M2 concat, M3, M4 cox_add, M4 concat, 전부
seed42 5-fold)의 checkpoint를 HPC에서 로컬로 옮기기 위한 압축 스크립트 — scripts/
zip_uni2official_features_for_hpc.py와 반대 방향(로컬->HPC가 아니라 HPC->로컬)이지만
매커니즘은 동일(프로젝트 루트 기준 상대경로 그대로 담아서 로컬에서 풀면 정확히 같은
위치에 복원됨).

2026-08-15: external eval이 HPC의 checkpoint를 필요로 하는데, 로컬엔 fold1만(파일럿 때
직접 학습) 있고 나머지 fold(0/2/3/4)는 예측 CSV만 다운로드돼 있어 로컬에서 eval을 못
돌리는 상황 — 체크포인트 자체를 옮겨와서 로컬에서 --eval-external-ckpt를 직접 돌리기 위함.

담는 4개 모델 태그:
  M2_uni2_STG_R_DISP_NOVIT_CLR20                      (concat, clinical-lr-mult 20x)
  M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_RLR20  (M3=WSI+RNA, rna-lr-mult 20x)
  M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_CLR20_RLR20  (M4 cox_add, 둘 다 20x)
  M4_uni2_INT1500_SS_AUX_STG_R_DISP_NOVIT_CLR20_RLR20          (M4 concat, 둘 다 20x)

이미 로컬에 있는 fold1도 같이 담기지만(중복, 무해) 어차피 무압축이라 손해가 크지 않다.

사용법(HPC에서, 프로젝트 루트에서):
    python -m scripts.zip_lrmult_checkpoints_for_local

로컬에서 받은 뒤:
    cd <프로젝트 루트> && unzip -o lrmult_checkpoints_for_local.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CKPT_DIR = _ROOT / "models" / "checkpoint"

MODEL_TAG_GLOBS = [
    "survival_tcga_uni2_seed42_*M2_uni2_STG_R_DISP_NOVIT_CLR20_FOLD*OF5_best_clinical.pt",
    "survival_tcga_uni2_seed42_*M4_uni2_INT1500_SS_AUX_NOCLINICAL_DISP_NOVIT_RLR20_FOLD*OF5_best_clinical_rna.pt",
    "survival_tcga_uni2_seed42_*M4_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD_NOVIT_CLR20_RLR20_FOLD*OF5_best_clinical_rna.pt",
    "survival_tcga_uni2_seed42_*M4_uni2_INT1500_SS_AUX_STG_R_DISP_NOVIT_CLR20_RLR20_FOLD*OF5_best_clinical_rna.pt",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="lrmult_checkpoints_for_local.zip",
                         help="프로젝트 루트 기준 출력 zip 경로(기본: 루트 바로 아래)")
    parser.add_argument("--compress", action="store_true",
                         help="deflate 압축 사용(기본은 무압축 ZIP_STORED — 체크포인트는 이미 조밀한 "
                              "float 텐서라 압축해도 별로 안 줄고 시간만 오래 걸림)")
    args = parser.parse_args()
    out_path = _ROOT / args.out

    files = []
    for pattern in MODEL_TAG_GLOBS:
        matches = sorted(CKPT_DIR.glob(pattern))
        print(f"{pattern}: {len(matches)}개 발견")
        files.extend(matches)

    if not files:
        print("담을 체크포인트를 하나도 못 찾았습니다 — 태그/경로를 확인하세요.")
        return

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"총 {len(files)}개 파일, {total_bytes / 1e9:.2f} GB — 압축 시작... "
          f"({'deflate' if args.compress else '무압축(ZIP_STORED)'})")

    compression = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(out_path, "w", compression=compression, allowZip64=True) as zf:
        for i, f in enumerate(files, 1):
            arcname = f.relative_to(_ROOT).as_posix()
            zf.write(f, arcname)
            print(f"  [{i}/{len(files)}] {arcname}")

    print(f"완료: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    print("로컬로 전송 후: cd <프로젝트 루트> && unzip -o lrmult_checkpoints_for_local.zip")


if __name__ == "__main__":
    main()
